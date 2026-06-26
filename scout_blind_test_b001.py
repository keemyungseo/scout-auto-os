"""
Scout Blind Test B001 — Real-world A6 inference at fixed timestamp.

Search: 2026-06-16 17:00 KST | Eval forward: 17:00-21:00
A6 frozen. No reject/rerank. No post-search data in ranking.

Usage:
  python scout_blind_test_b001.py
"""

from __future__ import annotations

import json
import statistics
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import scout_phase19_winner_ranking_dna as p19
import scout_phase20_winner_state_ranking as p20
import scout_phase22_search_formula_evolution as p22
import scout_phase23_search_formula_league as p23
from season2_p37_scout_decision_hierarchy import write_csv
from season2_universe_blind_test import load_eligible_symbols, ohlcv

import scout_phase16_human_blind_test as p16

OUT_DIR = Path("logs") / "blind_test_b001"
SCAN_KST = "2026-06-16 17:00:00"
EVAL_END_KST = "2026-06-16 21:00:00"
KST = timezone(timedelta(hours=9))
WORKERS = 12


def safe_print(msg: str) -> None:
    print(msg.encode("ascii", "replace").decode("ascii"))


def load_train_before(cutoff: str) -> tuple[list[dict], dict[str, list[dict]]]:
    raw: list[dict] = []
    for line in p19.CANDIDATES_PATH.open(encoding="utf-8"):
        r = json.loads(line)
        if r["scan_kst"] < cutoff:
            raw.append(r)
    by_scan: dict[str, list[dict]] = defaultdict(list)
    for r in raw:
        by_scan[r["scan_kst"]].append(r)
    for s in by_scan:
        by_scan[s].sort(key=lambda x: x.get("outcome_rank", 999))
    winner_feats = [r["features"] for rows in by_scan.values() for r in rows[:3] if len(rows) >= 4]
    th = p20.build_thresholds(winner_feats)
    annotated = p20.annotate(raw, th)
    ann_by: dict[str, list[dict]] = defaultdict(list)
    for r in annotated:
        ann_by[r["scan_kst"]].append(r)
    return annotated, ann_by


def brief_reason(row: dict) -> str:
    st = row["states"]
    return (
        f"1h={st['1h']} 2h={st['2h']} | "
        f"1h_rng={row['features'].get('1h_current_range_pct', 0):.1f}% "
        f"30m_ret={row['features'].get('30m_current_return_pct', 0):.1f}%"
    )


def eval_forward(symbol: str, entry_price: float, start_ms: int) -> dict:
    p16.CACHE_DIR = OUT_DIR / "kline_cache"
    fwd = p16.fetch_forward_5m(symbol, start_ms, 48)
    if not fwd or entry_price <= 0:
        return {}
    entry_p = float(fwd[0][1])
    scan_dt = datetime.fromtimestamp(start_ms / 1000, tz=KST)
    max_h, max_t_ms = entry_p, start_ms
    for k in fwd[:48]:
        h = ohlcv(k)[1]
        if h > max_h:
            max_h, max_t_ms = h, int(k[0])
    close_4h = ohlcv(fwd[min(47, len(fwd) - 1)])[3]
    peak_min = int((max_t_ms - start_ms) / 60000)
    peak_dt = datetime.fromtimestamp(max_t_ms / 1000, tz=KST)
    return {
        "search_price": round(entry_p, 8),
        "max_price_4h": round(max_h, 8),
        "max_return_pct": round((max_h - entry_p) / entry_p * 100, 4),
        "peak_minutes": peak_min,
        "peak_time_kst": peak_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "final_4h_return_pct": round((close_4h - entry_p) / entry_p * 100, 4),
    }


def peak_bucket(minutes: int) -> str:
    if minutes <= 30:
        return "30m"
    if minutes <= 60:
        return "1h"
    if minutes <= 120:
        return "2h"
    if minutes <= 180:
        return "3h"
    return "4h"


def summarize_group(rows: list[dict]) -> dict:
    rets = [r["fwd"]["max_return_pct"] for r in rows if r.get("fwd")]
    peaks = [r["fwd"]["peak_minutes"] for r in rows if r.get("fwd")]
    if not rets:
        return {}
    return {
        "avg_max_return": round(statistics.mean(rets), 4),
        "median_max_return": round(statistics.median(rets), 4),
        "best_return": round(max(rets), 4),
        "worst_return": round(min(rets), 4),
        "avg_peak_minutes": round(statistics.mean(peaks), 1) if peaks else 0,
    }


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p19.CACHE_DIR = OUT_DIR / "kline_cache"
    p16.CACHE_DIR = OUT_DIR / "kline_cache"

    scan_dt = p16.parse_kst(SCAN_KST)
    end_ms = int(scan_dt.timestamp() * 1000)
    start_ms = end_ms
    eval_end_ms = int(p16.parse_kst(EVAL_END_KST).timestamp() * 1000)

    safe_print(f"B001 SEARCH {SCAN_KST} KST — A6 frozen, no future data in ranking")

    train, train_by = load_train_before(SCAN_KST)
    winner_feats = [r["features"] for rows in train_by.values() for r in rows[:3] if len(rows) >= 4]
    th = p20.build_thresholds(winner_feats)
    w_train, _ = p20.winner_loser_sets(train_by)
    profile = p20.build_profile(w_train, train) if w_train else p20.build_profile([], train)
    stats = p22.build_train_stats(train, train_by, th)

    symbols = sorted(load_eligible_symbols(refresh=False, cache_only=False))
    safe_print(f"Universe: {len(symbols)} symbols")

    candidates: list[dict] = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = {
            pool.submit(p19.process_candidate, sym, SCAN_KST, end_ms, start_ms): sym
            for sym in symbols
        }
        done = 0
        for fut in as_completed(futs):
            done += 1
            row = fut.result()
            if row:
                candidates.append(row)
            if done % 100 == 0:
                safe_print(f"  scanned {done}/{len(symbols)} matches={len(candidates)}")

    safe_print(f"Pattern B matches: {len(candidates)}")
    if len(candidates) < 7:
        raise SystemExit("Insufficient candidates for Top7")

    annotated = p20.annotate(candidates, th)
    for r in annotated:
        base = p20.state_match_score(r["states"], r["transitions"], profile)
        r["a6"] = p23.formula_scores_a6(r, annotated, base, th, stats)["A6"]
        r["base_score"] = base

    ranked = sorted(annotated, key=lambda x: x["a6"], reverse=True)
    for i, r in enumerate(ranked, 1):
        r["a6_rank"] = i

    # Forward eval for ALL candidates (universe truth) then picks only
    safe_print("Forward evaluation 17:00-21:00...")
    for r in ranked:
        r["fwd"] = eval_forward(r["symbol"], r["features"]["price"], start_ms)
        time.sleep(0.02)

    by_outcome = sorted(ranked, key=lambda x: x["fwd"].get("max_return_pct", 0), reverse=True)
    actual_top2 = {r["symbol"] for r in by_outcome[:2]}

    picks = {
        "top2": ranked[:2],
        "top5": ranked[:5],
        "top7": ranked[:7],
    }

    # Reports
    lines = [
        "========================",
        "BLIND TEST B001",
        f"Search: {SCAN_KST} | Eval: {SCAN_KST} -> {EVAL_END_KST}",
        "Formula: A6 frozen | No reject/rerank",
        "========================",
        "",
        "=== SEARCH RESULTS (pre-outcome) ===",
    ]
    for group_name, group_rows in picks.items():
        lines.append(f"\n--- {group_name.upper()} ---")
        lines.append(f"{'Rank':<5} {'Symbol':<16} {'A6 Score':>10}  Reason")
        for r in group_rows:
            lines.append(
                f"{r['a6_rank']:<5} {r['symbol']:<16} {r['a6']:>10.4f}  {brief_reason(r)}"
            )

    for label, group_rows in [("TOP2", picks["top2"]), ("TOP5", picks["top5"]), ("TOP7", picks["top7"])]:
        lines.extend(["", f"=== {label} FORWARD REPORT ==="])
        lines.append(
            f"{'Rank':<5} {'Symbol':<14} {'SearchPx':>12} {'MaxPx':>12} "
            f"{'MaxRet%':>8} {'PeakMin':>8} {'Final4h%':>9}"
        )
        for r in group_rows:
            f = r["fwd"]
            lines.append(
                f"{r['a6_rank']:<5} {r['symbol']:<14} {f['search_price']:>12} {f['max_price_4h']:>12} "
                f"{f['max_return_pct']:>7.2f}% {f['peak_minutes']:>7} {f['final_4h_return_pct']:>8.2f}%"
            )

    lines.append("\n=== SUMMARY ===")
    for label, group_rows in [("Top2", picks["top2"]), ("Top5", picks["top5"]), ("Top7", picks["top7"])]:
        s = summarize_group(group_rows)
        lines.append(
            f"{label}: avg_max={s['avg_max_return']:.2f}% med={s['median_max_return']:.2f}% "
            f"best={s['best_return']:.2f}% worst={s['worst_return']:.2f}% "
            f"avg_peak={s['avg_peak_minutes']:.0f}min"
        )

    pick2_syms = {r["symbol"] for r in picks["top2"]}
    hits = len(pick2_syms & actual_top2)
    top5_pos = [r["fwd"]["max_return_pct"] for r in picks["top5"]]
    top7_loss = sum(1 for r in picks["top7"] if r["fwd"]["final_4h_return_pct"] < 0)
    peak_ctr = Counter(peak_bucket(r["fwd"]["peak_minutes"]) for r in picks["top7"])

    lines.extend([
        "",
        "=== ADDITIONAL ANALYSIS ===",
        f"1. Top2 was actual Top2? {hits}/2 hits | actual winners: {', '.join(sorted(actual_top2))}",
        f"2. Top5 EV+: avg_max={statistics.mean(top5_pos):.2f}% | all positive max: {all(x>0 for x in top5_pos)}",
        f"3. Top7 loss count (final<0): {top7_loss}/7",
        f"4. Peak timing (Top7): {dict(peak_ctr)}",
        "",
        "========================",
        "TOP2",
    ])
    for r in picks["top2"]:
        f = r["fwd"]
        lines.append(f"  {r['symbol']}  max={f['max_return_pct']:.2f}%  peak={f['peak_minutes']}min")
    t5s = summarize_group(picks["top5"])
    t7s = summarize_group(picks["top7"])
    lines.extend([
        "------------------------",
        "TOP5",
        f"  avg_max={t5s['avg_max_return']:.2f}%  avg_peak={t5s['avg_peak_minutes']:.0f}min",
        "------------------------",
        "TOP7",
        f"  avg_max={t7s['avg_max_return']:.2f}%  avg_peak={t7s['avg_peak_minutes']:.0f}min",
        "------------------------",
        "CONCLUSION: A6 blind EV & peak timing snapshot (single observation)",
        "========================",
    ])

    report = "\n".join(lines)
    (OUT_DIR / "blind_test_b001_report.txt").write_text(report, encoding="utf-8")
    safe_print(report)

    write_csv(OUT_DIR / "search_picks.csv", [
        {
            "group": g, "rank": r["a6_rank"], "symbol": r["symbol"], "a6": round(r["a6"], 4),
            "reason": brief_reason(r),
            **{f"fwd_{k}": v for k, v in r["fwd"].items()},
        }
        for g, rows in picks.items() for r in rows
    ])
    write_csv(OUT_DIR / "universe_forward.csv", [
        {
            "a6_rank": r["a6_rank"], "symbol": r["symbol"],
            "max_return_pct": r["fwd"].get("max_return_pct", 0),
            "outcome_rank_by_max": i,
        }
        for i, r in enumerate(by_outcome, 1)
    ])


if __name__ == "__main__":
    run()
