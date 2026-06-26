"""
Scout Research R004 — Candidate EV (Top5 Positive Opportunity)

No formula/feature/layer changes. LOO A6 blind on Phase19 outcomes.
Positive Opportunity = max_up_4h reached threshold (final price irrelevant).

Usage:
  python scout_research_r004_candidate_ev.py
"""

from __future__ import annotations

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

OUT_DIR = Path("logs") / "research_r004_candidate_ev"
KST = timezone(timedelta(hours=9))
PO_THRESHOLDS = (3.0, 5.0, 7.0)


def safe_print(msg: str) -> None:
    print(msg.encode("ascii", "replace").decode("ascii"))


def parse_kst(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=KST)


def forward_detail(symbol: str, scan_kst: str, entry_price: float) -> dict:
    """Peak time + final 4h return from 5m forward (post-search only)."""
    p16.CACHE_DIR = p19.CACHE_DIR
    start_ms = int(parse_kst(scan_kst).timestamp() * 1000)
    try:
        fwd = p16.fetch_forward_5m(symbol, start_ms, 48)
    except Exception:
        return {}
    if not fwd or entry_price <= 0:
        return {}
    entry_p = float(fwd[0][1])
    max_h, max_t = entry_p, start_ms
    for k in fwd[:48]:
        h = ohlcv(k)[1]
        if h > max_h:
            max_h, max_t = h, int(k[0])
    close_4h = ohlcv(fwd[min(47, len(fwd) - 1)])[3]
    peak_min = int((max_t - start_ms) / 60000)
    max_ret = (max_h - entry_p) / entry_p * 100
    return {
        "max_return_pct": round(max_ret, 4),
        "peak_minutes": peak_min,
        "final_4h_return_pct": round((close_4h - entry_p) / entry_p * 100, 4),
        "hit_3": max_ret >= 3.0,
        "hit_5": max_ret >= 5.0,
        "hit_7": max_ret >= 7.0,
    }


def peak_bucket(m: int) -> str:
    if m <= 30:
        return "30m"
    if m <= 60:
        return "1h"
    if m <= 120:
        return "2h"
    if m <= 180:
        return "3h"
    return "4h"


def loo_a6_scans() -> list[dict]:
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
        if len(rows) < 7:
            continue
        train = [r for r in annotated_all if r["scan_kst"] != scan]
        train_by: dict[str, list[dict]] = defaultdict(list)
        for r in train:
            train_by[r["scan_kst"]].append(r)
        w_train, _ = p20.winner_loser_sets(train_by)
        profile = p20.build_profile(w_train, train) if w_train else p20.build_profile([], train)
        stats = p22.build_train_stats(train, train_by, th)

        for r in rows:
            peers = rows
            base = p20.state_match_score(r["states"], r["transitions"], profile)
            r["a6"] = p23.formula_scores_a6(r, peers, base, th, stats)["A6"]

        ranked = sorted(rows, key=lambda x: x["a6"], reverse=True)
        for i, r in enumerate(ranked, 1):
            r["a6_rank"] = i

        by_max = sorted(rows, key=lambda x: x["max_up_4h"], reverse=True)
        actual_top2 = {by_max[0]["symbol"], by_max[1]["symbol"]}

        scan_rows.append({
            "scan_kst": scan,
            "actual_top2": sorted(actual_top2),
            "top2": ranked[:2],
            "top5": ranked[:5],
            "top7": ranked[:7],
            "top2_hit": len({r["symbol"] for r in ranked[:2]} & actual_top2),
        })
    return scan_rows


def po_rate(picks: list[dict], thr: float) -> float:
    if not picks:
        return 0.0
    return sum(1 for p in picks if p["max_up_4h"] >= thr) / len(picks) * 100


def aggregate_picks(scan_rows: list[dict], key: str) -> list[dict]:
    out: list[dict] = []
    for sr in scan_rows:
        for r in sr[key]:
            out.append({**r, "scan_kst": sr["scan_kst"]})
    return out


def summarize_picks(picks: list[dict], use_max_field: str = "max_up_4h") -> dict:
    vals = [p[use_max_field] for p in picks]
    if not vals:
        return {}
    return {
        "n": len(picks),
        "po_3_pct": round(po_rate(picks, 3.0), 2),
        "po_5_pct": round(po_rate(picks, 5.0), 2),
        "po_7_pct": round(po_rate(picks, 7.0), 2),
        "loss_opp_3_pct": round(100 - po_rate(picks, 3.0), 2),
        "avg_max_return": round(statistics.mean(vals), 4),
        "median_max_return": round(statistics.median(vals), 4),
    }


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p19.CACHE_DIR = Path("logs") / "phase19_winner_dna" / "kline_cache"

    safe_print("R004 LOO A6 scan aggregation...")
    scan_rows = loo_a6_scans()
    n_scans = len(scan_rows)

    top2_all = aggregate_picks(scan_rows, "top2")
    top5_all = aggregate_picks(scan_rows, "top5")
    top7_all = aggregate_picks(scan_rows, "top7")

    top2_hit_pct = sum(sr["top2_hit"] for sr in scan_rows) / (2 * n_scans) * 100
    t5 = summarize_picks(top5_all)
    t7 = summarize_picks(top7_all)
    t2 = summarize_picks(top2_all)

    # Peak/final detail on sample: last 30 scans (holdout-style) to limit API
    sample_scans = scan_rows[-30:]
    detail_rows: list[dict] = []
    safe_print(f"R004 forward detail sample ({len(sample_scans)} scans)...")
    for sr in sample_scans:
        for tier, picks in (("top5", sr["top5"]), ("top7", sr["top7"])):
            for r in picks:
                entry = r["features"].get("price", 0)
                d = forward_detail(r["symbol"], sr["scan_kst"], entry)
                if d:
                    detail_rows.append({
                        "scan_kst": sr["scan_kst"], "tier": tier,
                        "symbol": r["symbol"], "a6_rank": r["a6_rank"],
                        **d,
                    })

    peak_dist = Counter(peak_bucket(r["peak_minutes"]) for r in detail_rows if r["tier"] == "top5")
    avg_peak_top5 = statistics.mean([r["peak_minutes"] for r in detail_rows if r["tier"] == "top5"]) if detail_rows else 0
    final_pos_top5 = sum(1 for r in detail_rows if r["tier"] == "top5" and r["final_4h_return_pct"] > 0)
    final_n_top5 = sum(1 for r in detail_rows if r["tier"] == "top5")

    # B001 reference
    b001_path = Path("logs") / "blind_test_b001" / "search_picks.csv"
    b001_note = ""
    if b001_path.exists():
        import csv
        with b001_path.open(encoding="utf-8") as f:
            b1 = list(csv.DictReader(f))
        t5b = [r for r in b1 if r["group"] == "top5"]
        if t5b:
            b001_note = (
                f"B001 single (2026-06-16): Top5 PO +3%="
                f"{sum(1 for r in t5b if float(r['fwd_max_return_pct'])>=3)/len(t5b)*100:.0f}% "
                f"avg_max={statistics.mean(float(r['fwd_max_return_pct']) for r in t5b):.1f}%"
            )

    can_hit_90 = t5["po_3_pct"] >= 90

    lines = [
        "############################################################",
        "SCOUT RESEARCH R004 — CANDIDATE EV (Top5 Positive Opportunity)",
        "############################################################",
        "",
        f"LOO A6 blind | {n_scans} scans | Formula frozen | max_up_4h = opportunity proxy",
        "",
        "=" * 62,
        "CORE KPI",
        "=" * 62,
        f"  Top2 Hit %:                    {top2_hit_pct:.1f}%",
        f"  Top5 PO Rate (+3%):            {t5['po_3_pct']:.1f}%",
        f"  Top5 PO Rate (+5%):            {t5['po_5_pct']:.1f}%",
        f"  Top5 PO Rate (+7%):            {t5['po_7_pct']:.1f}%",
        f"  Top5 Loss Opportunity (<3%):   {t5['loss_opp_3_pct']:.1f}%",
        f"  Top5 Avg Max Return:           {t5['avg_max_return']:.2f}%",
        f"  Top5 Median Max Return:        {t5['median_max_return']:.2f}%",
        f"  Top7 PO Rate (+3%):            {t7['po_3_pct']:.1f}%",
        f"  Top7 PO Rate (+5%):            {t7['po_5_pct']:.1f}%",
        "",
        "=" * 62,
        "TOP2 SUMMARY",
        "=" * 62,
        f"  Avg max return: {t2['avg_max_return']:.2f}% | PO +5%: {t2['po_5_pct']:.1f}%",
        "",
        "=" * 62,
        "PEAK TIME (Top5, last-30-scan forward sample)",
        "=" * 62,
        f"  Avg peak minutes: {avg_peak_top5:.0f}",
        f"  Distribution: {dict(peak_dist)}",
        f"  Final 4h positive (sample): {final_pos_top5}/{final_n_top5}",
        "",
        "=" * 62,
        "KEY QUESTIONS",
        "=" * 62,
        f"  1. Top5 provides tradeable opportunity? "
        f"{'YES' if t5['po_3_pct'] >= 70 else 'PARTIAL' if t5['po_3_pct'] >= 50 else 'NO'} "
        f"(+3% rate {t5['po_3_pct']:.1f}%)",
        f"  2. Top5 EV positive (avg max)? {'YES' if t5['avg_max_return'] > 5 else 'MARGINAL' if t5['avg_max_return'] > 3 else 'NO'} "
        f"(avg {t5['avg_max_return']:.2f}%)",
        f"  3. Top5 PO >= 90%? {'YES' if can_hit_90 else 'NO'} ({t5['po_3_pct']:.1f}% at +3%)",
        f"  4. Top7 for loss avoidance? PO +3%={t7['po_3_pct']:.1f}% vs Top5 {t5['po_3_pct']:.1f}% "
        f"(Top7 wider but lower PO)",
        "",
        f"  {b001_note}",
        "",
        "=" * 62,
        "VERDICT",
        "=" * 62,
    ]

    if t5["po_3_pct"] >= 80 and t5["avg_max_return"] >= 5:
        verdict = "TOP5_EV_STRONG — prioritize opportunity KPI alongside Top2"
    elif t5["po_3_pct"] >= 60:
        verdict = "TOP5_EV_PARTIAL — Top2 miss acceptable if exit at peak; 90% PO not met"
    else:
        verdict = "STOP — Top5 opportunity insufficient for dual KPI"

    lines.append(f"  {verdict}")
    lines.append("")
    lines.extend(mission_summary_lines())

    report = "\n".join(lines)
    (OUT_DIR / "research_r004_report.txt").write_text(report, encoding="utf-8")
    safe_print(report)

    write_csv(OUT_DIR / "scan_summary.csv", [
        {
            "scan_kst": sr["scan_kst"],
            "top2_hit": sr["top2_hit"],
            "actual_top2": "|".join(sr["actual_top2"]),
            "pick_top2": "|".join(r["symbol"] for r in sr["top2"]),
            "top5_po3": sum(1 for r in sr["top5"] if r["max_up_4h"] >= 3),
            "top5_avg_max": round(statistics.mean([r["max_up_4h"] for r in sr["top5"]]), 4),
        }
        for sr in scan_rows
    ])
    write_csv(OUT_DIR / "kpi_summary.csv", [
        {"tier": "top2", **t2, "top2_hit_pct": round(top2_hit_pct, 2)},
        {"tier": "top5", **t5},
        {"tier": "top7", **t7},
    ])
    write_csv(OUT_DIR / "forward_detail_sample.csv", detail_rows)


if __name__ == "__main__":
    run()
