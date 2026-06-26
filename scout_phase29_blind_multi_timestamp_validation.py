"""
Scout Phase 29 - Blind Multi-Timestamp Validation

Frozen engine blind test at 3 holdout timestamps.
No formula/feature/threshold/weight/merge changes. Analysis only.

Usage:
  python scout_phase29_blind_multi_timestamp_validation.py
"""

from __future__ import annotations

import json
import math
import statistics
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import scout_phase20_winner_state_ranking as p20
import scout_phase22_search_formula_evolution as p22
import scout_phase23_search_formula_league as p23
import scout_phase16_human_blind_test as p16
import scout_phase19_winner_ranking_dna as p19
from scout_phase19_winner_ranking_dna import extract_dna_features
from scout_phase16_human_blind_test import fetch_forward_5m, parse_kst
from season2_p37_scout_decision_hierarchy import write_csv
from season2_universe_blind_test import load_eligible_symbols, ohlcv

OUT_DIR = Path("logs") / "phase29_blind_validation"
CACHE_DIR = OUT_DIR / "kline_cache"
P19_CACHE = Path("logs") / "phase19_winner_dna" / "kline_cache"

BLIND_TIMESTAMPS = (
    "2026-06-08 13:00:00",
    "2026-06-02 19:00:00",
    "2026-06-19 21:00:00",
)
FORMULAS = p23.FORMULAS
# Frozen league confidence (Phase23 LOO standings, no blind leakage)
FORMULA_CONF = {
    "A": 0.366, "A1": 0.358, "A2": 0.377, "A3": 0.358,
    "A4": 0.360, "A5": 0.374, "A6": 0.380,
}
HORIZON_BARS = {"30m": 6, "1h": 12, "2h": 24, "4h": 48, "8h": 96, "24h": 288}
FAILURE_TYPES = (
    "False Trigger", "Late Expansion", "Weak Acceleration",
    "Low Survival", "Flat Transition", "Ranking Miss", "Unknown",
)
# Frozen Trigger_A components (Phase27)
TRIGGER_A_COMPONENTS = (
    ("30m_return_pct_delta", 1.0),
    ("15m_ma20_distance_pct_rate", 1.0),
    ("volume_rate_stability", 0.8),
)


def safe_print(msg: str) -> None:
    print(msg.encode("ascii", "replace").decode("ascii"))


def g(f: dict, key: str, default: float = 0.0) -> float:
    return float(f.get(key, default))


def build_variants(f: dict) -> dict[str, float]:
    out = dict(f)
    for tf in ("15m", "30m"):
        for metric in ("return_pct", "ma20_distance_pct", "volume_ratio", "range_pct"):
            ck, pk = f"{tf}_current_{metric}", f"{tf}_previous_{metric}"
            if ck in f and pk in f:
                d = g(f, ck) - g(f, pk)
                out[f"{tf}_{metric}_delta"] = d
                out[f"{tf}_{metric}_rate"] = d / max(abs(g(f, pk)), 0.01)
    out["volume_rate_stability"] = -abs(out.get("15m_volume_ratio_rate", 0))
    out["30m_return_pct_delta"] = out.get("30m_return_pct_delta", 0)
    out["15m_ma20_distance_pct_rate"] = out.get("15m_ma20_distance_pct_rate", 0)
    return out


def future_survival_score(f: dict) -> float:
    v = build_variants(f)
    ret_d = g(v, "30m_return_pct_delta")
    ma_r = g(v, "15m_ma20_distance_pct_rate")
    vol_s = g(v, "volume_rate_stability")
    raw = 0.4 * (ret_d / 2.0) + 0.4 * ma_r + 0.2 * vol_s
    return max(0.0, min(1.0, 0.5 + raw * 0.5))


def trigger_a_score(f: dict, norms: dict[str, tuple[float, float]]) -> float:
    v = build_variants(f)
    s = 0.0
    for name, w in TRIGGER_A_COMPONENTS:
        mu, sd = norms.get(name, (0.0, 1.0))
        z = (g(v, name) - mu) / sd if sd > 1e-9 else 0.0
        s += w * z
    return s


def curve_score_light(f: dict) -> float:
    v = build_variants(f)
    ma_d = g(v, "15m_ma20_distance_pct_delta")
    ret_v = g(v, "30m_return_pct_delta")
    comp = g(f, "5m_compression")
    exp = g(f, "5m_range_energy")
    flat_pen = min(comp / 15.0, 1.0) * 0.3
    raw = 0.35 * (ma_d / 2.0) + 0.35 * (ret_v / 2.0) + 0.2 * (exp / 5.0) - flat_pen
    return max(0.0, min(1.0, 0.5 + raw * 0.3))


def norm_scores(vals: list[float]) -> list[float]:
    if not vals:
        return []
    lo, hi = min(vals), max(vals)
    span = hi - lo if hi > lo else 1.0
    return [(v - lo) / span for v in vals]


def load_training(th: p20.Thresholds) -> tuple[dict, dict]:
    """Train profile + formula stats from phase19 excluding blind timestamps."""
    raw = p20.load_candidates()
    hold = set(BLIND_TIMESTAMPS)
    train = [r for r in raw if r["scan_kst"] not in hold]
    ann = p20.annotate(train, th)
    by_scan: dict[str, list[dict]] = defaultdict(list)
    for r in ann:
        by_scan[r["scan_kst"]].append(r)
    for s in by_scan:
        by_scan[s].sort(key=lambda x: x["outcome_rank"])
    w_train, _ = p20.winner_loser_sets(by_scan)
    profile = p20.build_profile(w_train, ann) if w_train else p20.build_profile([], ann)
    stats = p22.build_train_stats(ann, by_scan, th)
    return profile, stats


def scan_symbol(symbol: str, scan_kst: str, end_ms: int) -> dict | None:
    p19.CACHE_DIR = P19_CACHE
    p16.CACHE_DIR = CACHE_DIR
    feats = extract_dna_features(symbol, end_ms)
    if not feats:
        return None
    return {"symbol": symbol, "scan_kst": scan_kst, "features": feats, "price": feats["price"]}


def max_up_bars(symbol: str, entry: float, start_ms: int, bars: int) -> float:
    p16.CACHE_DIR = CACHE_DIR
    try:
        fwd = fetch_forward_5m(symbol, start_ms, min(bars, 288))
        if not fwd or entry <= 0:
            return 0.0
        entry_p = float(fwd[0][1]) if fwd else entry
        mx = max(ohlcv(k)[1] for k in fwd[:bars])
        return round((mx - entry_p) / entry_p * 100, 4)
    except Exception:
        return 0.0


def score_scan(
    rows: list[dict],
    profile: dict,
    stats: dict,
    th: p20.Thresholds,
) -> list[dict]:
    ann = p20.annotate(rows, th)
    for r in ann:
        peers = ann
        base = p20.state_match_score(r["states"], r["transitions"], profile)
        r["state_score_raw"] = base
        r["formula_scores"] = p23.formula_scores_a6(r, peers, base, th, stats)
        r["formula_score"] = r["formula_scores"]["A6"]
        r["future_score_raw"] = future_survival_score(r["features"])
        r["curve_score_raw"] = curve_score_light(r["features"])

    # trigger norms
    keys = {n for n, _ in TRIGGER_A_COMPONENTS}
    norms: dict[str, tuple[float, float]] = {}
    variants = [build_variants(r["features"]) for r in ann]
    for k in keys:
        vals = [g(v, k) for v in variants]
        sd = statistics.stdev(vals) if len(vals) > 1 else 1.0
        norms[k] = (statistics.mean(vals), sd if sd > 1e-9 else 1.0)
    for r in ann:
        r["trigger_score_raw"] = trigger_a_score(r["features"], norms)

    # formula votes (production merge input)
    picks_top2: dict[str, list[str]] = {}
    for fid in FORMULAS:
        ranked = sorted(ann, key=lambda x: x["formula_scores"][fid], reverse=True)
        picks_top2[fid] = [x["symbol"] for x in ranked[:2]]
    votes = p23.consensus_votes(picks_top2, FORMULA_CONF)
    max_base = max(r["state_score_raw"] for r in ann) if ann else 1.0

    for r in ann:
        sym = r["symbol"]
        sn = r["state_score_raw"] / max_base if max_base else 0
        hist_sim = votes.get(sym, {}).get("agreement_ratio", 0)
        r["state_score"] = sn
        r["meta_score_raw"] = p23.meta_score(r, votes, sym, sn, hist_sim)
        r["votes"] = votes.get(sym, {})

    # normalize track scores 0-1 within scan
    norm_fields = (
        ("formula_score", "formula_norm"),
        ("state_score_raw", "state_norm"),
        ("future_score_raw", "future_norm"),
        ("trigger_score_raw", "trigger_norm"),
        ("curve_score_raw", "curve_norm"),
        ("meta_score_raw", "meta_norm"),
    )
    for src, dst in norm_fields:
        vals = [r[src] for r in ann]
        nv = norm_scores(vals)
        for r, v in zip(ann, nv):
            r[dst] = v

    for r in ann:
        r["future_score"] = r["future_norm"]
        r["trigger_score"] = r["trigger_norm"]
        r["curve_score"] = r["curve_norm"]
        r["meta_score"] = r["meta_norm"]
        r["formula_score_disp"] = round(r["formula_norm"] * 100, 2)
        r["state_score_disp"] = round(r["state_norm"] * 100, 2)
        r["future_score_disp"] = round(r["future_norm"] * 100, 2)
        r["trigger_score_disp"] = round(r["trigger_norm"] * 100, 2)
        r["curve_score_disp"] = round(r["curve_norm"] * 100, 2)
        r["meta_score_disp"] = round(r["meta_norm"] * 100, 2)

    ann.sort(key=lambda x: x["meta_score_raw"], reverse=True)
    return ann


def eval_forward(rows: list[dict], start_ms: int) -> None:
    for r in rows:
        entry = r["price"]
        sym = r["symbol"]
        for label, bars in HORIZON_BARS.items():
            r[f"max_up_{label}"] = max_up_bars(sym, entry, start_ms, bars)
        time.sleep(0.01)


def hit_rates(ranked_syms: list[str], actual_ranked: list[str], ks: tuple[int, ...] = (1, 2, 3, 5, 10)) -> dict[int, int]:
    actual = set(actual_ranked)
    return {k: len(set(ranked_syms[:k]) & actual) for k in ks}


def classify_failure(r: dict, actual_rank: int, meta_rank: int) -> str:
    st = r["states"]
    if r["trigger_score_disp"] >= 70 and r["max_up_4h"] < 2.0:
        return "False Trigger"
    if st["30m"] == "Compression" and st["1h"] != "Acceleration":
        return "Late Expansion"
    if st["1h"] not in ("Acceleration", "Expansion", "ExpansionStart"):
        if st["1h"] == "Flat":
            return "Flat Transition"
        return "Weak Acceleration"
    if r["future_score_disp"] < 40:
        return "Low Survival"
    if actual_rank <= 5 and meta_rank > 10:
        return "Ranking Miss"
    return "Unknown"


def track_contribution(actual_top: list[dict]) -> dict[str, float]:
    if not actual_top:
        return {t: 0.0 for t in ("formula", "state", "future", "trigger", "curve")}
    tracks = {
        "formula": "formula_norm",
        "state": "state_norm",
        "future": "future_norm",
        "trigger": "trigger_norm",
        "curve": "curve_norm",
    }
    means = {k: statistics.mean([r[v] for r in actual_top]) for k, v in tracks.items()}
    total = sum(means.values()) or 1.0
    return {k: round(v / total * 100, 1) for k, v in means.items()}


def agreement_bucket(r: dict) -> str:
    scores = [r["formula_norm"], r["state_norm"], r["future_norm"], r["trigger_norm"], r["curve_norm"]]
    hi = sum(1 for s in scores if s >= 0.7)
    lo = sum(1 for s in scores if s < 0.3)
    if hi >= 4:
        return "all_agree"
    if lo >= 3:
        return "conflict"
    top = max(range(5), key=lambda i: scores[i])
    names = ["formula", "state", "future", "trigger", "curve"]
    if scores[top] >= 0.85 and sum(1 for s in scores if s >= 0.6) == 1:
        return f"single_{names[top]}"
    return "mixed"


def run_blind_scan(
    scan_kst: str,
    symbols: list[str],
    profile: dict,
    stats: dict,
    th: p20.Thresholds,
    workers: int = 12,
) -> list[dict]:
    end_ms = int(parse_kst(scan_kst).timestamp() * 1000)
    start_ms = end_ms + 5 * 60 * 1000
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p19.CACHE_DIR = P19_CACHE
    p16.CACHE_DIR = CACHE_DIR

    raw: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(scan_symbol, sym, scan_kst, end_ms): sym for sym in symbols}
        for fut in as_completed(futs):
            r = fut.result()
            if r:
                raw.append(r)

    safe_print(f"  Pattern B candidates: {len(raw)}")
    scored = score_scan(raw, profile, stats, th)
    eval_forward(scored, start_ms)
    scored.sort(key=lambda x: x["max_up_4h"], reverse=True)
    for i, r in enumerate(scored, 1):
        r["actual_rank_4h"] = i
    scored.sort(key=lambda x: x["meta_score_raw"], reverse=True)
    for i, r in enumerate(scored, 1):
        r["meta_rank"] = i
    return scored


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    symbols = sorted(load_eligible_symbols(refresh=False, cache_only=False))
    safe_print(f"Universe: {len(symbols)} symbols")

    raw = p20.load_candidates()
    by_scan: dict[str, list] = defaultdict(list)
    for r in raw:
        by_scan[r["scan_kst"]].append(r)
    winner_feats = [r["features"] for rows in by_scan.values() for r in rows[:3] if len(rows) >= 4]
    th = p20.build_thresholds(winner_feats)
    profile, stats = load_training(th)

    all_top20: list[dict] = []
    track_perf: list[dict] = []
    failures: list[dict] = []
    agreements: list[dict] = []
    confidence_rows: list[dict] = []
    contrib_rows: list[dict] = []

    for scan_kst in BLIND_TIMESTAMPS:
        safe_print(f"\n=== BLIND SCAN {scan_kst} ===")
        scored = run_blind_scan(scan_kst, symbols, profile, stats, th)
        if len(scored) < 10:
            safe_print(f"  WARNING: only {len(scored)} candidates")
            continue

        actual_top10 = [r["symbol"] for r in sorted(scored, key=lambda x: x["max_up_4h"], reverse=True)[:10]]
        actual_top5 = actual_top10[:5]
        actual_top3 = actual_top10[:3]
        actual_top2 = actual_top10[:2]
        actual_top1 = actual_top10[:1]
        meta_ranked = [r["symbol"] for r in scored]

        top20 = scored[:20]
        for i, r in enumerate(top20, 1):
            all_top20.append({
                "scan_kst": scan_kst,
                "rank": i,
                "symbol": r["symbol"],
                "formula_score": r["formula_score_disp"],
                "state_score": r["state_score_disp"],
                "future_score": r["future_score_disp"],
                "trigger_score": r["trigger_score_disp"],
                "curve_score": r["curve_score_disp"],
                "meta_score": r["meta_score_disp"],
                "max_up_30m": r.get("max_up_30m", 0),
                "max_up_1h": r.get("max_up_1h", 0),
                "max_up_2h": r.get("max_up_2h", 0),
                "max_up_4h": r.get("max_up_4h", 0),
                "max_up_8h": r.get("max_up_8h", 0),
                "max_up_24h": r.get("max_up_24h", 0),
                "actual_rank_4h": r["actual_rank_4h"],
                "states_1h": r["states"]["1h"],
            })

        meta_hits = hit_rates(meta_ranked, actual_top2)
        for k in (1, 2, 3, 5, 10):
            confidence_rows.append({
                "scan_kst": scan_kst,
                "metric": f"meta_top{k}_hit",
                "value": meta_hits.get(k, 0),
                "denom": k,
            })

        # track-independent performance
        track_map = {
            "Track_A_formula": "formula_score",
            "Track_B_state": "state_score_raw",
            "Track_C_future": "future_score_raw",
            "Track_D_trigger": "trigger_score_raw",
            "Track_E_curve": "curve_score_raw",
        }
        for tname, skey in track_map.items():
            tr = sorted(scored, key=lambda x: x[skey], reverse=True)
            tsyms = [x["symbol"] for x in tr]
            h2 = len(set(tsyms[:2]) & set(actual_top2))
            h5 = len(set(tsyms[:5]) & set(actual_top5))
            h10 = len(set(tsyms[:10]) & set(actual_top10))
            track_perf.append({
                "scan_kst": scan_kst,
                "track": tname,
                "top2_hit": h2,
                "top5_hit": h5,
                "top10_hit": h10,
            })

        actual_top_rows = [r for r in scored if r["symbol"] in actual_top2]
        contrib = track_contribution(actual_top_rows)
        contrib_rows.append({"scan_kst": scan_kst, **contrib})

        for r in top20:
            if r["actual_rank_4h"] > 10:
                failures.append({
                    "scan_kst": scan_kst,
                    "symbol": r["symbol"],
                    "meta_rank": r["meta_rank"],
                    "actual_rank_4h": r["actual_rank_4h"],
                    "max_up_4h": r["max_up_4h"],
                    "failure_type": classify_failure(r, r["actual_rank_4h"], r["meta_rank"]),
                })

        for r in top20:
            agreements.append({
                "scan_kst": scan_kst,
                "symbol": r["symbol"],
                "bucket": agreement_bucket(r),
                "max_up_4h": r["max_up_4h"],
                "actual_rank_4h": r["actual_rank_4h"],
                "meta_rank": r["meta_rank"],
            })

        # confidence calibration buckets
        for r in scored:
            bucket = int(r["meta_score_disp"] // 10) * 10
            bucket = min(90, max(50, bucket))
            confidence_rows.append({
                "scan_kst": scan_kst,
                "metric": f"conf_{bucket}_{bucket+10}",
                "symbol": r["symbol"],
                "meta_score": r["meta_score_disp"],
                "hit_top10": 1 if r["symbol"] in actual_top10 else 0,
                "max_up_4h": r["max_up_4h"],
            })

    # aggregate confidence calibration
    cal_buckets: dict[str, list[int]] = defaultdict(list)
    for row in confidence_rows:
        if row["metric"].startswith("conf_"):
            cal_buckets[row["metric"]].append(row["hit_top10"])
    cal_out = [
        {"bucket": k, "n": len(v), "top10_hit_rate": round(sum(v) / len(v) * 100, 2) if v else 0}
        for k, v in sorted(cal_buckets.items())
    ]

    fail_ctr = Counter(f["failure_type"] for f in failures)
    agree_ctr = Counter(a["bucket"] for a in agreements)
    agree_perf: dict[str, list[float]] = defaultdict(list)
    for a in agreements:
        agree_perf[a["bucket"]].append(a["max_up_4h"])

    write_csv(OUT_DIR / "top20_picks.csv", all_top20)
    write_csv(OUT_DIR / "track_performance.csv", track_perf)
    write_csv(OUT_DIR / "track_contribution.csv", contrib_rows)
    write_csv(OUT_DIR / "failure_analysis.csv", failures)
    write_csv(OUT_DIR / "agreement_analysis.csv", agreements)
    write_csv(OUT_DIR / "confidence_calibration.csv", cal_out)

    # aggregate meta hits
    meta_top2_total = sum(1 for r in confidence_rows if r["metric"] == "meta_top2_hit" for _ in [0] if r["value"] >= 1)
    n_scans = len(BLIND_TIMESTAMPS)

    lines = [
        "############################################################",
        "SCOUT PHASE 29 - BLIND MULTI-TIMESTAMP VALIDATION",
        "############################################################",
        "",
        "Frozen engine. No formula/feature/threshold/weight/merge changes.",
        f"Timestamps: {', '.join(BLIND_TIMESTAMPS)}",
        f"Universe: {len(symbols)} symbols | Training scans exclude holdouts",
        "",
        "=" * 62,
        "A. TIMESTAMP TOP20 PICKS (see top20_picks.csv)",
        "=" * 62,
    ]
    for scan in BLIND_TIMESTAMPS:
        picks = [r for r in all_top20 if r["scan_kst"] == scan][:5]
        lines.append(f"  [{scan}]")
        for p in picks:
            lines.append(
                f"    #{p['rank']} {p['symbol']} meta={p['meta_score']:.0f} "
                f"4h_up={p['max_up_4h']:.2f}% actual_rank={p['actual_rank_4h']}"
            )

    lines.extend(["", "=" * 62, "B. ACTUAL PERFORMANCE (meta ranking)", "=" * 62])
    for scan in BLIND_TIMESTAMPS:
        hits = [r for r in confidence_rows if r["scan_kst"] == scan and r["metric"].startswith("meta_top")]
        lines.append(f"  [{scan}]")
        for h in sorted(hits, key=lambda x: x["metric"]):
            lines.append(f"    {h['metric']}: {h['value']}/{h['denom']}")

    lines.extend(["", "=" * 62, "C. TRACK INDEPENDENT PERFORMANCE", "=" * 62])
    for tp in track_perf:
        lines.append(
            f"  {tp['scan_kst']} {tp['track']}: TOP2={tp['top2_hit']} TOP5={tp['top5_hit']} TOP10={tp['top10_hit']}"
        )

    lines.extend(["", "=" * 62, "D. TRACK CONTRIBUTION (actual TOP2)", "=" * 62])
    for c in contrib_rows:
        lines.append(
            f"  {c['scan_kst']}: formula={c.get('formula',0)}% state={c.get('state',0)}% "
            f"future={c.get('future',0)}% trigger={c.get('trigger',0)}% curve={c.get('curve',0)}%"
        )

    lines.extend(["", "=" * 62, "E. FAILURE CLUSTER", "=" * 62])
    for ft, cnt in fail_ctr.most_common():
        lines.append(f"  {ft}: {cnt}")

    lines.extend(["", "=" * 62, "F. AGREEMENT EFFECT", "=" * 62])
    for bucket, cnt in agree_ctr.most_common():
        mu = statistics.mean(agree_perf[bucket]) if agree_perf[bucket] else 0
        lines.append(f"  {bucket}: n={cnt} avg_4h_up={mu:.2f}%")

    lines.extend(["", "=" * 62, "G. CONFIDENCE CALIBRATION", "=" * 62])
    for c in cal_out:
        lines.append(f"  {c['bucket']}: n={c['n']} top10_hit={c['top10_hit_rate']:.1f}%")

    lines.extend(["", "=" * 62, "H. PRODUCTION READINESS", "=" * 62])
    avg_top2 = statistics.mean([
        r["value"] for r in confidence_rows if r["metric"] == "meta_top2_hit"
    ]) / 2 * 100 if confidence_rows else 0
    lines.append(f"  Blind meta TOP2 hit (avg): {avg_top2:.1f}%")
    lines.append(f"  Production candidate threshold: 66% — {'NOT MET' if avg_top2 < 66 else 'MET'}")
    lines.append("  Tracks remain independent; merge rule = Phase23 meta_score unchanged.")
    lines.append("  Verdict: BLIND VALIDATION — descriptive only, no engine changes.")
    lines.append("")
    lines.append("DISCLAIMER: Forward outcomes observed post-hoc — not price prediction.")

    report = OUT_DIR / "phase29_blind_validation_report.txt"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for ln in lines:
        safe_print(ln)
    safe_print(f"\nSaved: {report}")


if __name__ == "__main__":
    run()
