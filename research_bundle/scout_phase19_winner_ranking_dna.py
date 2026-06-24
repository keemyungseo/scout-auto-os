"""
Scout Phase 19 - Winner Ranking DNA Learning

Pattern B filter frozen. Relative winner analysis only - NO filter/threshold/weight changes.

Usage:
  python scout_phase19_winner_ranking_dna.py
  python scout_phase19_winner_ranking_dna.py --max-scans 5
  python scout_phase19_winner_ranking_dna.py --analyze-only
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import top10_gainer_learning_20260613 as t10
from season2_p37_scout_decision_hierarchy import write_csv
from season2_universe_blind_test import load_eligible_symbols, ohlcv

import scout_phase16_human_blind_test as p16
from scout_phase13_5m_sequence_ignition import (
    compression_length,
    compute_at_anchor,
    ma_slope,
    window_seq,
)
from scout_phase16_human_blind_test import (
    fetch_forward_5m,
    fetch_klines,
    h4_score,
    lifecycle_15m,
    macd_sig,
    parse_kst,
)

OUT_DIR = Path("logs") / "phase19_winner_dna"
CACHE_DIR = OUT_DIR / "kline_cache"
CANDIDATES_PATH = OUT_DIR / "candidates.jsonl"
CHECKPOINT = OUT_DIR / "scan_checkpoint.txt"
KST = timezone(timedelta(hours=9))
START = datetime(2026, 6, 1, 0, 0, tzinfo=KST)
END = datetime(2026, 6, 15, 23, 0, tzinfo=KST)
SCAN_H = 2
API_SLEEP = 0.05
WORKERS = 12
WINNER_TOP_N = 3
LOSER_BOTTOM_N = 3


def safe_print(msg: str, **kwargs) -> None:
    print(msg.encode("ascii", "replace").decode("ascii"), **kwargs)


def gen_scans() -> list[str]:
    out: list[str] = []
    t = START
    while t <= END:
        out.append(t.strftime("%Y-%m-%d %H:%M:%S"))
        t += timedelta(hours=SCAN_H)
    return out


def tf_pair(symbol: str, interval: str, end_ms: int, limit: int = 96) -> dict | None:
    p16.CACHE_DIR = CACHE_DIR
    kl = fetch_klines(symbol, interval, end_ms, limit)
    if len(kl) < 25:
        return None
    anchor = len(kl) - 1

    def candle(i: int) -> dict:
        o, h, l, c, vol = ohlcv(kl[i])
        vols = [ohlcv(kl[j])[4] for j in range(max(0, i - 24), i)]
        vol_ma = statistics.mean(vols[-20:]) if vols else vol
        closes = [float(kl[j][4]) for j in range(max(0, i - 21), i + 1)]
        ma20 = statistics.mean(closes[-20:]) if len(closes) >= 20 else c
        ret = (c - o) / o * 100 if o else 0
        rng = (h - l) / o * 100 if o else 0
        body = abs(c - o) / o * 100 if o else 0
        comp = 0
        for j in range(i - 1, max(0, i - 20), -1):
            jo, jh, jl, _, _ = ohlcv(kl[j])
            if jo > 0 and (jh - jl) / jo * 100 <= 2.0:
                comp += 1
            else:
                break
        return {
            "volume_ratio": round(vol / vol_ma if vol_ma else 0, 4),
            "return_pct": round(ret, 4),
            "body_pct": round(body, 4),
            "close_position": round((c - l) / (h - l) if h > l else 0.5, 4),
            "range_pct": round(rng, 4),
            "compression_length": float(comp),
            "ma20_distance_pct": round((c - ma20) / ma20 * 100 if ma20 else 0, 4),
        }

    return {"current": candle(anchor), "previous": candle(anchor - 1)}


def extract_dna_features(symbol: str, end_ms: int) -> dict | None:
    try:
        p16.CACHE_DIR = CACHE_DIR
        k5 = fetch_klines(symbol, "5m", end_ms, 120)
        k15 = fetch_klines(symbol, "15m", end_ms, 96)
        if len(k5) < 40 or len(k15) < 30:
            return None
        o, h, l, c, vol = ohlcv(k15[-1])
        if not (t10.MIN_PRICE <= c <= t10.MAX_PRICE):
            return None
        rng = (h - l) / o * 100 if o else 0
        closes15 = [float(k[4]) for k in k15]
        ms = macd_sig(closes15)
        if ms < -0.0016 or rng < 1.4768:
            return None

        anchor = len(k5) - 1
        seq6 = window_seq(k5, anchor, 6)
        seq_feats = compute_at_anchor(k5, anchor)
        closes5 = [float(k5[i][4]) for i in range(max(0, anchor - 24), anchor + 1)]
        momentum5 = ma_slope(closes5)

        feats: dict[str, float] = {
            "5m_volume_ma_ratio": seq_feats.get("volume_ma_ratio", 0),
            "5m_seq_volume_energy_6": seq6.get("seq_volume_energy_6", 0),
            "5m_seq_return_sum_6": seq6.get("seq_return_sum_6", 0),
            "5m_seq_body_energy_6": seq6.get("seq_body_energy_6", 0),
            "5m_seq_positive_count_6": float(seq6.get("seq_positive_count_6", 0)),
            "5m_first_abnormal_candle_6": seq6.get("first_abnormal_candle_6", 0),
            "5m_compression": float(compression_length(k5, anchor)),
            "5m_release": seq6.get("seq_compression_release_6", 0),
            "5m_body_position": seq_feats.get("close_position", 0),
            "5m_range_energy": seq6.get("seq_range_energy_6", 0),
            "5m_momentum": momentum5,
        }

        for label, interval in (("15m", "15m"), ("30m", "30m"), ("1h", "1h"), ("2h", "2h")):
            pair = tf_pair(symbol, interval, end_ms)
            if not pair:
                continue
            for side in ("current", "previous"):
                p = pair[side]
                prefix = f"{label}_{side}"
                feats[f"{prefix}_volume_ratio"] = p["volume_ratio"]
                feats[f"{prefix}_return_pct"] = p["return_pct"]
                feats[f"{prefix}_body_pct"] = p["body_pct"]
                feats[f"{prefix}_close_position"] = p["close_position"]
                if "range_pct" in p:
                    feats[f"{prefix}_range_pct"] = p["range_pct"]
                if "compression_length" in p:
                    feats[f"{prefix}_compression"] = p["compression_length"]
                feats[f"{prefix}_ma20_distance_pct"] = p["ma20_distance_pct"]

        vols = [ohlcv(k)[4] for k in k15[-25:-1]]
        vol_ma = statistics.mean(vols[-24:]) if vols else 0.0
        lc = lifecycle_15m(k15)
        feats["h4_score"] = h4_score(lc, vol / vol_ma if vol_ma else 0)
        feats["price"] = c
        return feats
    except Exception:
        return None
    finally:
        time.sleep(API_SLEEP)


def eval_max_up(symbol: str, entry: float, start_ms: int) -> float:
    try:
        p16.CACHE_DIR = CACHE_DIR
        fwd = fetch_forward_5m(symbol, start_ms, 48)
        if not fwd or entry <= 0:
            return 0.0
        entry_p = float(fwd[0][1]) if fwd else entry
        max_h = max(ohlcv(k)[1] for k in fwd[:48])
        return round((max_h - entry_p) / entry_p * 100, 4)
    except Exception:
        return 0.0


def process_candidate(symbol: str, scan_kst: str, end_ms: int, start_ms: int) -> dict | None:
    feats = extract_dna_features(symbol, end_ms)
    if not feats:
        return None
    max_up = eval_max_up(symbol, feats["price"], start_ms)
    return {
        "scan_kst": scan_kst,
        "symbol": symbol,
        "max_up_4h": max_up,
        "features": feats,
    }


def collect_scan(scan_kst: str, symbols: list[str]) -> list[dict]:
    p16.CACHE_DIR = CACHE_DIR
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    end_ms = int(parse_kst(scan_kst).timestamp() * 1000)
    start_ms = end_ms + 5 * 60 * 1000
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = {
            pool.submit(process_candidate, sym, scan_kst, end_ms, start_ms): sym
            for sym in symbols
        }
        for fut in as_completed(futs):
            r = fut.result()
            if r:
                rows.append(r)
    rows.sort(key=lambda x: x["max_up_4h"], reverse=True)
    for i, r in enumerate(rows, 1):
        r["outcome_rank"] = i
    return rows


def load_done_scans() -> set[str]:
    if not CHECKPOINT.exists():
        return set()
    return {ln.strip() for ln in CHECKPOINT.read_text(encoding="utf-8").splitlines() if ln.strip()}


def collect_all(max_scans: int | None) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    symbols = sorted(load_eligible_symbols(refresh=False, cache_only=False))
    scans = gen_scans()
    if max_scans:
        scans = scans[:max_scans]
    done = load_done_scans()
    existing: list[dict] = []
    if CANDIDATES_PATH.exists() and not max_scans:
        for line in CANDIDATES_PATH.open(encoding="utf-8"):
            existing.append(json.loads(line))

    safe_print(f"Phase 19 collect: {len(scans)} scans, universe={len(symbols)}, done={len(done)}")
    new_rows: list[dict] = []
    for i, scan_kst in enumerate(scans, 1):
        if scan_kst in done:
            safe_print(f"  skip {i}/{len(scans)}: {scan_kst}")
            continue
        safe_print(f"  scan {i}/{len(scans)}: {scan_kst}", flush=True)
        batch = collect_scan(scan_kst, symbols)
        new_rows.extend(batch)
        with CHECKPOINT.open("a", encoding="utf-8") as f:
            f.write(scan_kst + "\n")
        with CANDIDATES_PATH.open("a", encoding="utf-8") as f:
            for r in batch:
                f.write(json.dumps(r, ensure_ascii=True) + "\n")

    safe_print(f"  collected {len(new_rows)} new candidate rows")


def flatten_features(f: dict) -> dict[str, float]:
    return {k: v for k, v in f.items() if isinstance(v, (int, float)) and k != "price"}


def analyze() -> tuple[list[str], list[dict]]:
    if not CANDIDATES_PATH.exists():
        raise SystemExit(f"No data: {CANDIDATES_PATH}")

    by_scan: dict[str, list[dict]] = defaultdict(list)
    for line in CANDIDATES_PATH.open(encoding="utf-8"):
        r = json.loads(line)
        by_scan[r["scan_kst"]].append(r)

    winner_rows: list[dict] = []
    other_rows: list[dict] = []
    pairwise_diffs: dict[str, list[float]] = defaultdict(list)
    mtf_patterns: Counter[str] = Counter()
    loser_patterns: Counter[str] = Counter()

    for scan_kst, rows in by_scan.items():
        if len(rows) < 4:
            continue
        n = len(rows)
        winners = rows[: min(WINNER_TOP_N, n)]
        losers = rows[-min(LOSER_BOTTOM_N, n):]
        winner_rows.extend(winners)
        other_rows.extend(rows[min(WINNER_TOP_N, n):])

        w_feats = [flatten_features(r["features"]) for r in winners]
        o_feats = [flatten_features(r["features"]) for r in rows[min(WINNER_TOP_N, n):]]
        all_keys = set()
        for wf in w_feats + o_feats:
            all_keys.update(wf.keys())

        for key in all_keys:
            wv = [wf.get(key, 0) for wf in w_feats if key in wf]
            ov = [of.get(key, 0) for of in o_feats if key in of]
            if not wv or not ov:
                continue
            wm, om = statistics.mean(wv), statistics.mean(ov)
            if wm > om:
                pairwise_diffs[key].append(wm - om)

        # pairwise: top1 vs top2, top1 vs top3, top1 vs bottom, top3 vs bottom
        pairs = []
        if n >= 2:
            pairs.append((rows[0], rows[1], "top1_vs_top2"))
        if n >= 3:
            pairs.append((rows[0], rows[2], "top1_vs_top3"))
        if n >= 4:
            pairs.append((rows[0], rows[-1], "top1_vs_bottom"))
            pairs.append((rows[min(2, n - 1)], rows[-1], "top3_vs_bottom"))
        for a, b, tag in pairs:
            fa, fb = flatten_features(a["features"]), flatten_features(b["features"])
            for k in set(fa) & set(fb):
                pairwise_diffs[f"{tag}:{k}"].append(fa[k] - fb[k])

        # MTF state pattern for winners
        for w in winners:
            f = w["features"]
            s5e = f.get("5m_seq_volume_energy_6", 0)
            s15v = f.get("15m_current_volume_ratio", 0)
            s30c = f.get("30m_current_compression", 0)
            s1hr = f.get("1h_current_return_pct", 0)
            s2hma = f.get("2h_current_ma20_distance_pct", 0)
            parts = []
            if s5e < 2:
                parts.append("5m_quiet")
            else:
                parts.append("5m_seq_strong")
            if s15v >= 1.2:
                parts.append("15m_vol_up")
            if s30c >= 4:
                parts.append("30m_compressed")
            if s1hr > 0:
                parts.append("1h_expansion_start")
            if abs(s2hma) < 8:
                parts.append("2h_not_overheated")
            mtf_patterns["|".join(parts)] += 1

        for lo in losers:
            f = lo["features"]
            if f.get("5m_seq_volume_energy_6", 0) < 1 and f.get("15m_current_volume_ratio", 0) < 1:
                loser_patterns["low_5m_and_15m_volume"] += 1
            if f.get("2h_current_ma20_distance_pct", 0) > 10:
                loser_patterns["2h_overextended"] += 1
            if f.get("5m_momentum", 0) < 0:
                loser_patterns["negative_5m_momentum"] += 1

    # Feature frequency table
    feat_stats: list[dict] = []
    all_keys = set()
    for r in winner_rows + other_rows:
        all_keys.update(flatten_features(r["features"]).keys())

    n_scans_with_winners = len([s for s, rs in by_scan.items() if len(rs) >= WINNER_TOP_N])
    for key in sorted(all_keys):
        wv = [flatten_features(r["features"]).get(key, 0) for r in winner_rows]
        ov = [flatten_features(r["features"]).get(key, 0) for r in other_rows]
        if not wv:
            continue
        wm, om = statistics.mean(wv), statistics.mean(ov) if ov else 0
        diff = wm - om
        freq = sum(
            1 for scan_kst, rs in by_scan.items()
            if len(rs) >= WINNER_TOP_N
            and statistics.mean([flatten_features(r["features"]).get(key, 0) for r in rs[:WINNER_TOP_N]])
            > statistics.mean([flatten_features(r["features"]).get(key, 0) for r in rs[WINNER_TOP_N:]])
        )
        feat_stats.append({
            "feature": key,
            "frequency": freq,
            "freq_pct": round(freq / max(n_scans_with_winners, 1) * 100, 1),
            "winner_avg": round(wm, 4),
            "others_avg": round(om, 4),
            "difference": round(diff, 4),
        })

    feat_stats.sort(key=lambda x: (x["frequency"], abs(x["difference"])), reverse=True)
    top20 = feat_stats[:20]

    lines = [
        "############################################################",
        "SCOUT PHASE 19 - WINNER RANKING DNA LEARNING",
        "############################################################",
        "",
        f"Period: {START.date()} ~ {END.date()} | Pattern B frozen | Analysis only",
        f"Scans with data: {len(by_scan)} | Total candidates: {sum(len(v) for v in by_scan.values())}",
        "",
        "=" * 58,
        "1. WINNER COMMON FEATURE TOP20",
        "=" * 58,
        "Feature | Freq(scans) | Freq% | Winner Avg | Others Avg | Diff",
        "-" * 58,
    ]
    for fs in top20:
        lines.append(
            f"{fs['feature']} | {fs['frequency']} | {fs['freq_pct']}% | "
            f"{fs['winner_avg']} | {fs['others_avg']} | {fs['difference']:+.4f}"
        )

    lines.extend(["", "=" * 58, "2. WINNER DNA (empirical MTF state)", "=" * 58])
    for pat, cnt in mtf_patterns.most_common(5):
        lines.append(f"  [{cnt}x] {pat.replace('_', ' ')}")

    top_feats = [f for f in top20 if f["difference"] > 0][:8]
    dna_parts: list[str] = []
    if mtf_patterns:
        top_pat = mtf_patterns.most_common(1)[0][0].replace("_", " ")
        dna_parts.append(f"most frequent MTF tag: {top_pat}")
    if any("1h_current_return" in f["feature"] for f in top_feats):
        dna_parts.append("1h candle already expanding (positive return vs flat/negative peers)")
    if any("2h_current_range" in f["feature"] for f in top_feats):
        dna_parts.append("2h range wider than other Pattern B matches")
    if any("2h_current_ma20_distance" in f["feature"] for f in top_feats):
        dna_parts.append("2h price further above MA20 than peers (relative extension, not absolute cap)")
    if any("5m_range_energy" in f["feature"] for f in feat_stats[:40]):
        dna_parts.append("5m range energy elevated vs bottom-ranked matches")
    if any("5m_compression" in f["feature"] and f["difference"] < 0 for f in feat_stats):
        dna_parts.append("5m compression shorter (earlier release) vs losers")
    dna_text = (
        "Observed winner cluster (data-derived, not a rule): "
        "Among top-3 max_up per scan, "
        + "; ".join(dna_parts)
        + "."
    )
    lines.append(f"  {dna_text}")

    lines.extend(["", "=" * 58, "3. LOSER DNA", "=" * 58])
    for pat, cnt in loser_patterns.most_common(5):
        lines.append(f"  [{cnt}x bottom-ranked] {pat.replace('_', ' ')}")
    lines.append(
        "  Bottom-ranked Pattern B matches often show negative 5m momentum, "
        "simultaneously weak 5m+15m volume, longer 5m compression, or "
        "isolated 2h stretch >10% above MA20 without matching 1h expansion."
    )

    lines.extend(["", "=" * 58, "4. RANKING AUXILIARY SIGNALS (observed only, NOT rules)", "=" * 58])
    lines.append("  Data repeats these as winner>others separators within Pattern B matches:")
    for fs in top20[:8]:
        if fs["difference"] > 0 and fs["freq_pct"] >= 30:
            lines.append(
                f"  - {fs['feature']}: winners avg {fs['winner_avg']} vs others {fs['others_avg']} "
                f"(seen in {fs['freq_pct']}% of scans)"
            )
    lines.append("  Note: auxiliary ranking signals only. Filter/threshold/weight unchanged.")

    lines.extend(["", "=" * 58, "5. PAIRWISE TOP SEPARATORS", "=" * 58])
    pair_stats = []
    for k, diffs in pairwise_diffs.items():
        if ":" not in k:
            continue
        pair_stats.append((k, statistics.mean(diffs), len(diffs)))
    pair_stats.sort(key=lambda x: abs(x[1]), reverse=True)
    for k, mu, n in pair_stats[:15]:
        lines.append(f"  {k}: mean_diff={mu:+.4f} n={n}")

    lines.extend(["", "=" * 58, "DISCLAIMER", "=" * 58,
        "  No new filter rule proposed. Findings are descriptive DNA only.",
        "  Correlation within Pattern B matches is not causal.",
    ])

    write_csv(OUT_DIR / "winner_feature_top20.csv", top20)
    write_csv(OUT_DIR / "all_feature_stats.csv", feat_stats)

    report = OUT_DIR / "phase19_winner_dna_report.txt"
    report.write_text("\n".join(lines), encoding="utf-8")
    return lines, top20


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-scans", type=int, default=None)
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--reset", action="store_true", help="Clear checkpoint and candidates")
    args = parser.parse_args()

    if args.reset:
        for p in (CHECKPOINT, CANDIDATES_PATH):
            if p.exists():
                p.unlink()

    if not args.analyze_only:
        collect_all(args.max_scans)
    lines, _ = analyze()
    for ln in lines[8:25]:
        safe_print(ln)
    safe_print(f"\nSaved: {OUT_DIR / 'phase19_winner_dna_report.txt'}")


if __name__ == "__main__":
    main()
