"""
R005 LOO hit-rate report — A6 frozen, Phase19 candidates.

Per-scan Top2/5/7 vs actual max_up_4h outcomes.
Outputs: logs/research_r005_loo_hit_report/

Usage:
  python scout_research_r005_loo_hit_report.py
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import scout_phase16_human_blind_test as p16
import scout_phase19_winner_ranking_dna as p19
import scout_phase20_winner_state_ranking as p20
import scout_phase22_search_formula_evolution as p22
import scout_phase23_search_formula_league as p23
from season2_p37_scout_decision_hierarchy import write_csv
from season2_universe_blind_test import ohlcv

OUT_DIR = Path("logs") / "research_r005_loo_hit_report"
KST = timezone(timedelta(hours=9))
TIERS = ("top2", "top5", "top7")
TIER_N = {"top2": 2, "top5": 5, "top7": 7}


def safe_print(msg: str) -> None:
    print(msg.encode("ascii", "replace").decode("ascii"))


def parse_kst(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=KST)


def forward_final_return(symbol: str, scan_kst: str) -> float | None:
    """4h close return from scan_time entry (5m forward cache)."""
    start_ms = int(parse_kst(scan_kst).timestamp() * 1000)
    try:
        fwd = p16.fetch_forward_5m(symbol, start_ms, 48)
    except Exception:
        return None
    if not fwd:
        return None
    entry_p = float(fwd[0][1])
    if entry_p <= 0:
        return None
    close_4h = ohlcv(fwd[min(47, len(fwd) - 1)])[3]
    return round((close_4h - entry_p) / entry_p * 100, 4)


def loo_a6_scans(min_per_scan: int = 7) -> list[dict]:
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
        by_max = sorted(rows, key=lambda x: x["max_up_4h"], reverse=True)

        picks = {t: ranked[: TIER_N[t]] for t in TIERS}
        actual = {
            t: {by_max[i]["symbol"] for i in range(TIER_N[t])}
            for t in TIERS
        }

        scan_rows.append({
            "scan_kst": scan,
            "n_candidates": len(rows),
            "picks": picks,
            "actual": actual,
        })
    return scan_rows


def tier_metrics(picks: list[dict], actual_top: set[str], scan_kst: str, n: int) -> dict:
    syms = [p["symbol"] for p in picks]
    overlap = len(set(syms) & actual_top)
    max_vals = [p["max_up_4h"] for p in picks]
    finals: list[float] = []
    for p in picks:
        fr = forward_final_return(p["symbol"], scan_kst)
        if fr is not None:
            finals.append(fr)

    return {
        "topn_overlap": overlap,
        "topn_overlap_pct": round(overlap / n * 100, 2) if n else 0.0,
        "max_gt0_pct": round(sum(1 for v in max_vals if v > 0) / n * 100, 2),
        "max_ge3_pct": round(sum(1 for v in max_vals if v >= 3.0) / n * 100, 2),
        "max_ge4_pct": round(sum(1 for v in max_vals if v >= 4.0) / n * 100, 2),
        "final_gt0_pct": round(sum(1 for v in finals if v > 0) / len(finals) * 100, 2) if finals else None,
        "avg_max_up_4h": round(statistics.mean(max_vals), 4) if max_vals else 0.0,
    }


def aggregate(rows: list[dict], tier: str) -> dict:
    n = TIER_N[tier]
    keys = ("topn_overlap_pct", "max_gt0_pct", "max_ge3_pct", "max_ge4_pct", "final_gt0_pct")
    acc: dict[str, list[float]] = {k: [] for k in keys}
    overlaps = 0
    for r in rows:
        m = r[tier]
        overlaps += m["topn_overlap"]
        for k in keys:
            v = m.get(k)
            if v is not None:
                acc[k].append(v)
    total_slots = len(rows) * n
    return {
        "n_scans": len(rows),
        "slots": total_slots,
        "topn_overlap_total": overlaps,
        "topn_overlap_slot_pct": round(overlaps / total_slots * 100, 2) if total_slots else 0.0,
        **{f"avg_{k}": round(statistics.mean(acc[k]), 2) if acc[k] else None for k in keys},
        "median_max_ge3_pct": round(statistics.median(acc["max_ge3_pct"]), 2) if acc["max_ge3_pct"] else None,
        "median_max_ge4_pct": round(statistics.median(acc["max_ge4_pct"]), 2) if acc["max_ge4_pct"] else None,
    }


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p19.CACHE_DIR = Path("logs") / "phase19_winner_dna" / "kline_cache"
    p16.CACHE_DIR = p19.CACHE_DIR

    safe_print("R005 LOO hit report — loading scans...")
    scan_rows = loo_a6_scans()
    safe_print(f"LOO scans with >=7 candidates: {len(scan_rows)}")

    per_scan: list[dict] = []
    for i, sr in enumerate(scan_rows, 1):
        row: dict = {"scan_kst": sr["scan_kst"], "n_candidates": sr["n_candidates"]}
        for tier in TIERS:
            m = tier_metrics(sr["picks"][tier], sr["actual"][tier], sr["scan_kst"], TIER_N[tier])
            row[tier] = m
            for k, v in m.items():
                row[f"{tier}_{k}"] = v
        per_scan.append(row)
        if i % 20 == 0:
            safe_print(f"  processed {i}/{len(scan_rows)} scans (forward final)...")

    agg = {tier: aggregate(per_scan, tier) for tier in TIERS}

    lines = [
        "############################################################",
        "R005 LOO HIT RATE REPORT — A6 frozen (Phase19)",
        "############################################################",
        "",
        f"Source: {p19.CANDIDATES_PATH}",
        f"LOO scans (min 7 Pattern-B candidates): {len(scan_rows)}",
        "Train: all scans except holdout | Rank: A6 | Truth: max_up_4h per scan",
        "final>0: 4h close return from scan_time (5m forward cache)",
        "",
        "=== DEFINITIONS ===",
        "  topn_overlap: A6 pick symbols in actual top-N by max_up_4h (slot hit %)",
        "  max_gt0:      pick max_up_4h > 0",
        "  max_ge3:      pick max_up_4h >= 3%",
        "  max_ge4:      pick max_up_4h >= 4%  <-- production target",
        "  final_gt0:    4h final return > 0 from scan entry",
        "",
    ]

    for tier in TIERS:
        a = agg[tier]
        lines.extend([
            f"=== AGGREGATE {tier.upper()} ({a['n_scans']} scans, {a['slots']} slots) ===",
            f"  Actual-TopN overlap (slot-weighted): {a['topn_overlap_slot_pct']}%",
            f"  Avg per-scan max>0 hit:                {a['avg_max_gt0_pct']}%",
            f"  Avg per-scan max>=3% hit:              {a['avg_max_ge3_pct']}%",
            f"  Avg per-scan max>=4% hit:              {a['avg_max_ge4_pct']}%",
            f"  Median per-scan max>=3% hit:           {a['median_max_ge3_pct']}%",
            f"  Median per-scan max>=4% hit:           {a['median_max_ge4_pct']}%",
            f"  Avg per-scan final>0 hit:              {a['avg_final_gt0_pct']}%",
            "",
        ])

    lines.extend([
        "=== B001 REFERENCE (single holdout, not in LOO aggregate) ===",
        "  2026-06-16 17:00 | Top5 max>=3%: 100% | actual-TopN overlap: 40%",
        "  See logs/blind_test_b001/blind_test_b001_report.txt",
        "",
        "=== INTERPRETATION ===",
        "  Compare avg/median max>=4% across tiers for +4% production goal.",
        "  Low actual-TopN overlap with high max>=3% means ranking finds movers, not exact winners.",
        "",
    ])

    report = "\n".join(lines)
    (OUT_DIR / "r005_loo_hit_report.txt").write_text(report, encoding="utf-8")
    write_csv(OUT_DIR / "per_scan_hit_rates.csv", per_scan)
    write_csv(OUT_DIR / "aggregate_summary.csv", [
        {"tier": t, **agg[t]} for t in TIERS
    ])
    safe_print(report)


if __name__ == "__main__":
    run()
