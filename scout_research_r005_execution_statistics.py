"""
Scout Research R005 — Execution Statistics Engine

A6 frozen search (TOP2/TOP5/TOP7). No formula/reject/penalty/feature changes.
Large-sample execution probability from historical forward paths (5m candles).

Usage:
  python scout_research_r005_execution_statistics.py
  python scout_research_r005_execution_statistics.py --no-b001
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import scout_phase19_winner_ranking_dna as p19
import scout_phase20_winner_state_ranking as p20
import scout_phase22_search_formula_evolution as p22
import scout_phase23_search_formula_league as p23
from season2_p37_scout_decision_hierarchy import write_csv
from season2_universe_blind_test import ohlcv

import scout_phase16_human_blind_test as p16
from season2_scout_mission import mission_summary_lines

OUT_DIR = Path("logs") / "research_r005_execution_statistics"
KST = timezone(timedelta(hours=9))

HORIZON_BARS = {"30m": 6, "1h": 12, "2h": 24, "3h": 36, "4h": 48}
PO_THRESHOLDS = (2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0)
PEAK_BUCKETS = (30, 60, 90, 120, 150, 180, 240)
A6_BINS = ((0, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, float("inf")))
STATE_KEYS = ("Acceleration", "ExpansionStart", "TrendAlive")


def safe_print(msg: str) -> None:
    print(msg.encode("ascii", "replace").decode("ascii"))


def parse_kst(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=KST)


def pctile(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (k - f) * (s[c] - s[f])


def state_label(states: dict) -> str:
    return f"1h={states.get('1h', '?')}|2h={states.get('2h', '?')}"


def a6_bin(score: float) -> str:
    for lo, hi in A6_BINS:
        if lo <= score < hi:
            if hi == float("inf"):
                return "7+"
            return f"{lo:g}~{hi:g}"
    return "unknown"


def horizon_stats(chunk: list[list], entry: float) -> dict:
    if not chunk or entry <= 0:
        return {"max_return": 0.0, "min_return": 0.0, "final_return": 0.0}
    max_h = max(ohlcv(k)[1] for k in chunk)
    min_l = min(ohlcv(k)[2] for k in chunk)
    close = float(chunk[-1][4])
    return {
        "max_return": round((max_h - entry) / entry * 100, 4),
        "min_return": round((min_l - entry) / entry * 100, 4),
        "final_return": round((close - entry) / entry * 100, 4),
    }


def compute_execution_path(symbol: str, scan_kst: str, entry_hint: float) -> dict:
    """Forward execution path from 5m candles (post-search only)."""
    p16.CACHE_DIR = p19.CACHE_DIR
    start_ms = int(parse_kst(scan_kst).timestamp() * 1000)
    try:
        fwd = p16.fetch_forward_5m(symbol, start_ms, 48)
    except Exception:
        return {}
    if not fwd:
        return {}
    entry = float(fwd[0][1]) if fwd else entry_hint
    if entry <= 0:
        return {}

    out: dict = {
        "search_price": round(entry, 8),
        "mfe_pct": 0.0,
        "mae_pct": 0.0,
        "peak_return_pct": 0.0,
        "peak_minutes": 0,
        "peak_time_kst": "",
        "drawdown_after_peak_pct": 0.0,
    }

    for label, n in HORIZON_BARS.items():
        hs = horizon_stats(fwd[:n], entry)
        out[f"{label}_max_return"] = hs["max_return"]
        out[f"{label}_min_return"] = hs["min_return"]
        out[f"{label}_final_return"] = hs["final_return"]

    # Full 4h path: MFE, MAE, peak, post-peak drawdown
    max_h, max_i, max_t = entry, 0, start_ms
    worst_mae = 0.0
    for i, k in enumerate(fwd[:48]):
        h, l = ohlcv(k)[1], ohlcv(k)[2]
        mfe_i = (h - entry) / entry * 100
        mae_i = (entry - l) / entry * 100
        if mfe_i > (max_h - entry) / entry * 100:
            max_h, max_i, max_t = h, i, int(k[0])
        worst_mae = max(worst_mae, mae_i)

    peak_ret = (max_h - entry) / entry * 100
    post_peak_low = max_h
    for k in fwd[max_i:48]:
        post_peak_low = min(post_peak_low, ohlcv(k)[2])
    dd_after = (max_h - post_peak_low) / max_h * 100 if max_h > 0 else 0.0

    peak_dt = datetime.fromtimestamp(max_t / 1000, tz=KST)
    out.update({
        "mfe_pct": round(peak_ret, 4),
        "mae_pct": round(worst_mae, 4),
        "peak_return_pct": round(peak_ret, 4),
        "peak_minutes": int((max_t - start_ms) / 60000),
        "peak_time_kst": peak_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "drawdown_after_peak_pct": round(dd_after, 4),
    })
    return out


def loo_a6_scan_rows(min_per_scan: int = 7) -> list[dict]:
    raw: list[dict] = []
    for line in p19.CANDIDATES_PATH.open(encoding="utf-8"):
        raw.append(json.loads(line))
    by_scan: dict[str, list[dict]] = defaultdict(list)
    for r in raw:
        by_scan[r["scan_kst"]].append(r)
    for s in by_scan:
        by_scan[s].sort(key=lambda x: x.get("outcome_rank", 999))

    winner_feats = [r["features"] for rows in by_scan.values() for r in rows[:3] if len(rows) >= 4]
    th = p20.build_thresholds(winner_feats)
    annotated_all = p20.annotate(raw, th)

    scan_rows: list[dict] = []
    for scan in sorted(by_scan.keys()):
        rows = [r for r in annotated_all if r["scan_kst"] == scan]
        if len(rows) < min_per_scan:
            continue
        train = [r for r in annotated_all if r["scan_kst"] != scan]
        train_by: dict[str, list[dict]] = defaultdict(list)
        for r in train:
            train_by[r["scan_kst"]].append(r)
        w_train, _ = p20.winner_loser_sets(train_by)
        profile = p20.build_profile(w_train, train) if w_train else p20.build_profile([], train)
        stats = p22.build_train_stats(train, train_by, th)

        for r in rows:
            base = p20.state_match_score(r["states"], r["transitions"], profile)
            r["a6"] = p23.formula_scores_a6(r, rows, base, th, stats)["A6"]

        ranked = sorted(rows, key=lambda x: x["a6"], reverse=True)
        for i, r in enumerate(ranked, 1):
            r["a6_rank"] = i

        scan_rows.append({
            "source": "loo_phase19",
            "scan_kst": scan,
            "top2": ranked[:2],
            "top5": ranked[:5],
            "top7": ranked[:7],
        })
    return scan_rows


def flatten_picks(scan_rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    for sr in scan_rows:
        for tier in ("top2", "top5", "top7"):
            for r in sr[tier]:
                out.append({
                    "source": sr["source"],
                    "search_time": sr["scan_kst"],
                    "tier": tier,
                    "a6_rank": r["a6_rank"],
                    "symbol": r["symbol"],
                    "a6_score": round(r["a6"], 4),
                    "state": state_label(r["states"]),
                    "state_1h": r["states"].get("1h", ""),
                    "state_2h": r["states"].get("2h", ""),
                    "search_price_hint": r["features"].get("price", 0),
                    "_row": r,
                })
    return out


def load_b001_scan_row() -> dict | None:
    path = Path("logs") / "blind_test_b001" / "search_picks.csv"
    if not path.exists():
        return None
    by_tier: dict[str, list[dict]] = defaultdict(list)
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_tier[row["group"]].append(row)
    if not by_tier.get("top7"):
        return None

    scan_kst = "2026-06-16 17:00:00"
    picks: dict[str, list[dict]] = {}
    for tier in ("top2", "top5", "top7"):
        picks[tier] = []
        for row in by_tier.get(tier, []):
            reason = row.get("reason", "")
            st_1h, st_2h = "?", "?"
            if "1h=" in reason and "2h=" in reason:
                part = reason.split("|")[0]
                for piece in part.split():
                    if piece.startswith("1h="):
                        st_1h = piece.split("=", 1)[1]
                    if piece.startswith("2h="):
                        st_2h = piece.split("=", 1)[1]
            picks[tier].append({
                "symbol": row["symbol"],
                "a6": float(row["a6"]),
                "a6_rank": int(row["rank"]),
                "states": {"1h": st_1h, "2h": st_2h},
                "features": {"price": float(row.get("fwd_search_price", 0))},
            })
    return {"source": "blind_b001", "scan_kst": scan_kst, **picks}


def enrich_candidates(picks: list[dict]) -> list[dict]:
    enriched: list[dict] = []
    fwd_cache: dict[tuple[str, str], dict] = {}
    total = len(picks)
    for i, p in enumerate(picks, 1):
        key = (p["search_time"], p["symbol"])
        if key not in fwd_cache:
            fwd_cache[key] = compute_execution_path(
                p["symbol"], p["search_time"], p["search_price_hint"],
            )
        fwd = fwd_cache[key]
        row = {k: v for k, v in p.items() if k != "_row"}
        row.update(fwd)
        row["a6_bin"] = a6_bin(p["a6_score"])
        enriched.append(row)
        if i % 200 == 0:
            safe_print(f"  forward stats {i}/{total}")
    return enriched


def tier_subset(rows: list[dict], tier: str) -> list[dict]:
    return [r for r in rows if r["tier"] == tier and r.get("mfe_pct") is not None]


def dist_stats(vals: list[float]) -> dict:
    if not vals:
        return {}
    return {
        "n": len(vals),
        "mean": round(statistics.mean(vals), 4),
        "median": round(statistics.median(vals), 4),
        "std": round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0,
        "p25": round(pctile(vals, 25), 4),
        "p50": round(pctile(vals, 50), 4),
        "p75": round(pctile(vals, 75), 4),
        "p90": round(pctile(vals, 90), 4),
        "p95": round(pctile(vals, 95), 4),
        "p99": round(pctile(vals, 99), 4),
    }


def po_rates(rows: list[dict], field: str = "mfe_pct") -> dict[str, float]:
    if not rows:
        return {f"po_{int(t)}": 0.0 for t in PO_THRESHOLDS}
    n = len(rows)
    return {
        f"po_{int(t)}": round(sum(1 for r in rows if r.get(field, 0) >= t) / n * 100, 2)
        for t in PO_THRESHOLDS
    }


def peak_distribution(rows: list[dict]) -> dict:
    peaks = [r["peak_minutes"] for r in rows if "peak_minutes" in r]
    if not peaks:
        return {}
    bucket_ctr = Counter()
    for m in peaks:
        placed = False
        for b in PEAK_BUCKETS:
            if m <= b:
                bucket_ctr[b] = bucket_ctr.get(b, 0) + 1
                placed = True
                break
        if not placed:
            bucket_ctr[999] = bucket_ctr.get(999, 0) + 1
    n = len(peaks)
    cdf: dict[str, float] = {}
    cum = 0
    for b in PEAK_BUCKETS:
        cum += bucket_ctr.get(b, 0)
        cdf[f"<={b}m"] = round(cum / n * 100, 2)
    return {
        "n": n,
        "mean_min": round(statistics.mean(peaks), 1),
        "median_min": round(statistics.median(peaks), 1),
        "buckets": {f"<={b}m": bucket_ctr.get(b, 0) for b in PEAK_BUCKETS},
        "cdf_pct": cdf,
    }


def mae_distribution(rows: list[dict]) -> dict:
    maes = [r["mae_pct"] for r in rows if "mae_pct" in r]
    if not maes:
        return {}
    return {
        "n": len(maes),
        "mean": round(statistics.mean(maes), 4),
        "median": round(statistics.median(maes), 4),
        "p90": round(pctile(maes, 90), 4),
        "p95": round(pctile(maes, 95), 4),
        "p99": round(pctile(maes, 99), 4),
        "worst": round(max(maes), 4),
    }


def execution_rule_suggestion(rows: list[dict]) -> dict:
    maxs = [r["mfe_pct"] for r in rows]
    finals = [r.get("4h_final_return", 0) for r in rows]
    peaks = [r["peak_minutes"] for r in rows]
    maes = [r["mae_pct"] for r in rows]
    if not maxs:
        return {}
    return {
        "expected_max_return_mean": round(statistics.mean(maxs), 4),
        "expected_max_return_median": round(statistics.median(maxs), 4),
        "recommended_tp_70": round(pctile(maxs, 70), 4),
        "recommended_tp_80": round(pctile(maxs, 80), 4),
        "recommended_tp_85": round(pctile(maxs, 85), 4),
        "recommended_tp_90": round(pctile(maxs, 90), 4),
        "expected_peak_mean_min": round(statistics.mean(peaks), 1),
        "expected_peak_median_min": round(statistics.median(peaks), 1),
        "recommended_hold_min": round(statistics.median(peaks), 1),
        "recommended_stop_loss_mae95": round(pctile(maes, 95), 4),
        "expected_ev_final_mean": round(statistics.mean(finals), 4),
        "expected_ev_opportunity_mean": round(statistics.mean(maxs), 4),
    }


def conditioned_block(rows: list[dict], label: str) -> dict:
    maxs = [r["mfe_pct"] for r in rows]
    if not maxs:
        return {"label": label, "n": 0}
    po = po_rates(rows)
    mae = mae_distribution(rows)
    peak = peak_distribution(rows)
    return {
        "label": label,
        "n": len(rows),
        **dist_stats(maxs),
        **po,
        "peak_median_min": peak.get("median_min", 0),
        "mae_p95": mae.get("p95", 0),
    }


def conditional_probability(rows: list[dict]) -> list[dict]:
    """Report 7: predefined condition combos on full candidate pool (top7 deduped)."""
    seen: set[tuple[str, str]] = set()
    pool: list[dict] = []
    for r in rows:
        key = (r["search_time"], r["symbol"])
        if key in seen:
            continue
        seen.add(key)
        pool.append(r)

    conditions = [
        ("A6>6", lambda r: r["a6_score"] > 6),
        ("A6>5", lambda r: r["a6_score"] > 5),
        ("TrendAlive", lambda r: r.get("state_2h") == "TrendAlive"),
        ("Acceleration", lambda r: r.get("state_1h") == "Acceleration"),
        ("ExpansionStart", lambda r: r.get("state_1h") == "ExpansionStart"),
        ("A6>6 AND TrendAlive", lambda r: r["a6_score"] > 6 and r.get("state_2h") == "TrendAlive"),
        ("A6>6 AND Acceleration", lambda r: r["a6_score"] > 6 and r.get("state_1h") == "Acceleration"),
        ("A6>5 AND TrendAlive", lambda r: r["a6_score"] > 5 and r.get("state_2h") == "TrendAlive"),
        ("Acceleration AND TrendAlive", lambda r: r.get("state_1h") == "Acceleration" and r.get("state_2h") == "TrendAlive"),
    ]

    out: list[dict] = []
    for name, pred in conditions:
        sub = [r for r in pool if pred(r)]
        if not sub:
            out.append({"condition": name, "n": 0})
            continue
        maxs = [r["mfe_pct"] for r in sub]
        maes = [r["mae_pct"] for r in sub]
        peaks = [r["peak_minutes"] for r in sub]
        out.append({
            "condition": name,
            "n": len(sub),
            "po_3_pct": round(sum(1 for r in sub if r["mfe_pct"] >= 3) / len(sub) * 100, 2),
            "po_5_pct": round(sum(1 for r in sub if r["mfe_pct"] >= 5) / len(sub) * 100, 2),
            "po_7_pct": round(sum(1 for r in sub if r["mfe_pct"] >= 7) / len(sub) * 100, 2),
            "avg_max_pct": round(statistics.mean(maxs), 4),
            "median_max_pct": round(statistics.median(maxs), 4),
            "peak_median_min": round(statistics.median(peaks), 1),
            "mae_p95": round(pctile(maes, 95), 4),
        })
    return out


def format_report1(tiers: dict[str, list[dict]]) -> str:
    lines = ["REPORT 1 — Execution Statistics (4h MFE = max_return)", "=" * 60]
    for tier, rows in tiers.items():
        vals = [r["mfe_pct"] for r in rows]
        ds = dist_stats(vals)
        lines.append(f"\n[{tier.upper()}] n={ds.get('n', 0)}")
        if not ds:
            lines.append("  (no data)")
            continue
        lines.append(f"  Mean Max Return:   {ds['mean']:.4f}%")
        lines.append(f"  Median Max Return: {ds['median']:.4f}%")
        lines.append(f"  Std:               {ds['std']:.4f}")
        lines.append(f"  25%: {ds['p25']:.4f}%  50%: {ds['p50']:.4f}%  75%: {ds['p75']:.4f}%")
        lines.append(f"  90%: {ds['p90']:.4f}%  95%: {ds['p95']:.4f}%  99%: {ds['p99']:.4f}%")
    return "\n".join(lines)


def format_report2(tiers: dict[str, list[dict]]) -> str:
    lines = ["REPORT 2 — Positive Opportunity (MFE threshold reach %)", "=" * 60]
    for tier, rows in tiers.items():
        po = po_rates(rows)
        lines.append(f"\n[{tier.upper()}] n={len(rows)}")
        for t in PO_THRESHOLDS:
            lines.append(f"  +{int(t)}%: {po[f'po_{int(t)}']:.2f}%")
    return "\n".join(lines)


def format_report3(tiers: dict[str, list[dict]]) -> str:
    lines = ["REPORT 3 — Peak Time Distribution", "=" * 60]
    for tier, rows in tiers.items():
        pd = peak_distribution(rows)
        lines.append(f"\n[{tier.upper()}] n={pd.get('n', 0)}")
        if not pd:
            continue
        lines.append(f"  Mean: {pd['mean_min']} min | Median: {pd['median_min']} min")
        lines.append("  Bucket counts (first bucket <=Xm containing peak):")
        for b in PEAK_BUCKETS:
            lines.append(f"    <={b}m: {pd['buckets'].get(f'<={b}m', 0)}")
        lines.append("  CDF:")
        for k, v in pd["cdf_pct"].items():
            lines.append(f"    {k}: {v:.2f}%")
    return "\n".join(lines)


def format_report4(tiers: dict[str, list[dict]]) -> str:
    lines = ["REPORT 4 — MAE Distribution", "=" * 60]
    for tier, rows in tiers.items():
        md = mae_distribution(rows)
        lines.append(f"\n[{tier.upper()}] n={md.get('n', 0)}")
        if not md:
            continue
        lines.append(
            f"  Mean: {md['mean']:.4f}% | Median: {md['median']:.4f}% | "
            f"90%: {md['p90']:.4f}% | 95%: {md['p95']:.4f}% | "
            f"99%: {md['p99']:.4f}% | Worst: {md['worst']:.4f}%"
        )
    return "\n".join(lines)


def format_report5(tiers: dict[str, list[dict]]) -> str:
    lines = ["REPORT 5 — Execution Rule Suggestion", "=" * 60]
    for tier, rows in tiers.items():
        er = execution_rule_suggestion(rows)
        lines.append(f"\n[{tier.upper()}] n={len(rows)}")
        if not er:
            continue
        lines.append(f"  Expected Max Return — Mean: {er['expected_max_return_mean']:.4f}% | Median: {er['expected_max_return_median']:.4f}%")
        lines.append(f"  Recommended TP(70%): {er['recommended_tp_70']:.4f}%")
        lines.append(f"  Recommended TP(80%): {er['recommended_tp_80']:.4f}%")
        lines.append(f"  Recommended TP(85%): {er['recommended_tp_85']:.4f}%")
        lines.append(f"  Recommended TP(90%): {er['recommended_tp_90']:.4f}%")
        lines.append(f"  Expected Peak — Mean: {er['expected_peak_mean_min']:.0f} min | Median: {er['expected_peak_median_min']:.0f} min")
        lines.append(f"  Recommended Hold Time (median peak): {er['recommended_hold_min']:.0f} min")
        lines.append(f"  Recommended Stop Loss (MAE 95%): -{er['recommended_stop_loss_mae95']:.4f}%")
        lines.append(f"  Expected EV (final 4h mean): {er['expected_ev_final_mean']:.4f}%")
        lines.append(f"  Expected EV (opportunity / mean MFE): {er['expected_ev_opportunity_mean']:.4f}%")
    return "\n".join(lines)


def format_report6(all_rows: list[dict]) -> str:
    lines = ["REPORT 6 — Conditioned Statistics", "=" * 60]
    pool = tier_subset(all_rows, "top7")
    lines.append("\n--- A6 Score Bins (TOP7 pool) ---")
    for lo, hi in A6_BINS:
        label = "7+" if hi == float("inf") else f"{lo:g}~{hi:g}"
        sub = [r for r in pool if a6_bin(r["a6_score"]) == label]
        blk = conditioned_block(sub, label)
        lines.append(
            f"  [{label}] n={blk['n']} mean_max={blk.get('mean', 0):.2f}% "
            f"med={blk.get('median', 0):.2f}% po+3={blk.get('po_3', 0):.1f}%"
        )

    lines.append("\n--- State Conditions (TOP7 pool) ---")
    for st in STATE_KEYS:
        if st == "TrendAlive":
            sub = [r for r in pool if r.get("state_2h") == st]
        else:
            sub = [r for r in pool if r.get("state_1h") == st]
        blk = conditioned_block(sub, st)
        lines.append(
            f"  [{st}] n={blk['n']} mean_max={blk.get('mean', 0):.2f}% "
            f"po+3={blk.get('po_3', 0):.1f}% po+5={blk.get('po_5', 0):.1f}% "
            f"peak_med={blk.get('peak_median_min', 0):.0f}m mae95={blk.get('mae_p95', 0):.2f}%"
        )
    return "\n".join(lines)


def format_report7(cond_rows: list[dict]) -> str:
    lines = ["REPORT 7 — Conditional Probability", "=" * 60]
    for r in cond_rows:
        if r.get("n", 0) == 0:
            lines.append(f"\n[{r['condition']}] n=0")
            continue
        lines.append(f"\n[{r['condition']}] n={r['n']}")
        lines.append(
            f"  PO +3%: {r['po_3_pct']:.1f}% | +5%: {r['po_5_pct']:.1f}% | +7%: {r['po_7_pct']:.1f}%"
        )
        lines.append(
            f"  Avg Max: {r['avg_max_pct']:.2f}% | Median Max: {r['median_max_pct']:.2f}% | "
            f"Peak Med: {r['peak_median_min']:.0f}m | MAE95: {r['mae_p95']:.2f}%"
        )
    return "\n".join(lines)


def candidate_csv_rows(rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    for r in rows:
        if not r.get("search_price"):
            continue
        row = {
            "source": r["source"],
            "search_time": r["search_time"],
            "tier": r["tier"],
            "a6_rank": r["a6_rank"],
            "symbol": r["symbol"],
            "a6_score": r["a6_score"],
            "a6_bin": r["a6_bin"],
            "state": r["state"],
            "state_1h": r["state_1h"],
            "state_2h": r["state_2h"],
            "search_price": r["search_price"],
            "peak_minutes": r.get("peak_minutes", 0),
            "peak_return_pct": r.get("peak_return_pct", 0),
            "peak_time_kst": r.get("peak_time_kst", ""),
            "drawdown_after_peak_pct": r.get("drawdown_after_peak_pct", 0),
            "mfe_pct": r.get("mfe_pct", 0),
            "mae_pct": r.get("mae_pct", 0),
        }
        for label in HORIZON_BARS:
            row[f"{label}_max_return"] = r.get(f"{label}_max_return", 0)
            row[f"{label}_min_return"] = r.get(f"{label}_min_return", 0)
            row[f"{label}_final_return"] = r.get(f"{label}_final_return", 0)
        out.append(row)
    return out


def run(include_b001: bool = True) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p19.CACHE_DIR = Path("logs") / "phase19_winner_dna" / "kline_cache"
    p16.CACHE_DIR = p19.CACHE_DIR

    safe_print("R005 LOO A6 scan aggregation...")
    scan_rows = loo_a6_scan_rows()
    n_loo = len(scan_rows)
    if include_b001:
        b001 = load_b001_scan_row()
        if b001:
            scan_rows.append(b001)
            safe_print("R005 included Blind Test B001 scan")

    picks = flatten_picks(scan_rows)
    safe_print(f"R005 computing forward execution stats for {len(picks)} pick-rows...")
    enriched = enrich_candidates(picks)
    valid = [r for r in enriched if r.get("search_price")]
    safe_print(f"R005 valid execution records: {len(valid)}")

    tiers = {
        "top2": tier_subset(valid, "top2"),
        "top5": tier_subset(valid, "top5"),
        "top7": tier_subset(valid, "top7"),
    }

    cond_prob = conditional_probability(valid)
    r1 = format_report1(tiers)
    r2 = format_report2(tiers)
    r3 = format_report3(tiers)
    r4 = format_report4(tiers)
    r5 = format_report5(tiers)
    r6 = format_report6(valid)
    r7 = format_report7(cond_prob)

    header = [
        "############################################################",
        "SCOUT RESEARCH R005 — EXECUTION STATISTICS ENGINE",
        "############################################################",
        "",
        f"A6 frozen | LOO scans: {n_loo} | B001: {'yes' if include_b001 and any(r['source']=='blind_b001' for r in valid) else 'no'}",
        f"Total pick-rows: {len(valid)} | Unique (scan,symbol) in top7: {len({(r['search_time'],r['symbol']) for r in tier_subset(valid,'top7')})}",
        "Forward: 5m candles post search_time | Formula unchanged",
        "",
    ]
    master = "\n".join(header + [r1, "", r2, "", r3, "", r4, "", r5, "", r6, "", r7, ""])
    master += "\n\n" + "\n".join(mission_summary_lines())

    (OUT_DIR / "research_r005_report.txt").write_text(master, encoding="utf-8")
    (OUT_DIR / "report_01_execution_statistics.txt").write_text(r1, encoding="utf-8")
    (OUT_DIR / "report_02_positive_opportunity.txt").write_text(r2, encoding="utf-8")
    (OUT_DIR / "report_03_peak_time.txt").write_text(r3, encoding="utf-8")
    (OUT_DIR / "report_04_mae_distribution.txt").write_text(r4, encoding="utf-8")
    (OUT_DIR / "report_05_execution_rules.txt").write_text(r5, encoding="utf-8")
    (OUT_DIR / "report_06_conditioned.txt").write_text(r6, encoding="utf-8")
    (OUT_DIR / "report_07_conditional_probability.txt").write_text(r7, encoding="utf-8")

    write_csv(OUT_DIR / "candidates_execution.csv", candidate_csv_rows(valid))
    write_csv(OUT_DIR / "tier_summary.csv", [
        {"tier": t, **dist_stats([r["mfe_pct"] for r in rows]), **po_rates(rows)}
        for t, rows in tiers.items()
    ])
    write_csv(OUT_DIR / "execution_rules.csv", [
        {"tier": t, **execution_rule_suggestion(rows)} for t, rows in tiers.items()
    ])
    write_csv(OUT_DIR / "conditional_probability.csv", cond_prob)

    # Conditioned CSV blocks
    pool = tier_subset(valid, "top7")
    cond_rows: list[dict] = []
    for lo, hi in A6_BINS:
        label = "7+" if hi == float("inf") else f"{lo:g}~{hi:g}"
        sub = [r for r in pool if lo <= r["a6_score"] < hi]
        cond_rows.append({"kind": "a6_bin", **conditioned_block(sub, label)})
    for st in STATE_KEYS:
        if st == "TrendAlive":
            sub = [r for r in pool if r.get("state_2h") == st]
        else:
            sub = [r for r in pool if r.get("state_1h") == st]
        cond_rows.append({"kind": "state", **conditioned_block(sub, st)})
    write_csv(OUT_DIR / "conditioned_statistics.csv", cond_rows)

    safe_print(master)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-b001", action="store_true", help="Exclude B001 blind scan")
    args = ap.parse_args()
    run(include_b001=not args.no_b001)


if __name__ == "__main__":
    main()
