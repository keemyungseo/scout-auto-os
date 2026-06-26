"""
Scout Phase 18 - Generalization Blind Validation

Replays frozen Phase16 SCOUT logic on independent timestamps.
NO filter/threshold/weight/ranking changes.

Usage:
  python scout_phase18_generalization_validation.py --full
  python scout_phase18_generalization_validation.py --scan-kst "2026-06-01 05:00:00"
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from season2_p37_scout_decision_hierarchy import write_csv
from season2_universe_blind_test import load_eligible_symbols, ohlcv

import scout_phase16_human_blind_test as p16
from scout_phase16_human_blind_test import (
    Candidate,
    fetch_forward_5m,
    find_missed,
    parse_kst,
    scan_universe,
)

OUT_DIR = Path("logs") / "phase18_generalization"
HIT_7 = 7.0
FP_THRESHOLD = 5.0

SCAN_TIMES = [
    "2026-06-01 05:00:00", "2026-06-01 11:00:00", "2026-06-01 17:00:00", "2026-06-01 23:00:00",
    "2026-06-02 05:00:00", "2026-06-02 11:00:00", "2026-06-02 17:00:00", "2026-06-02 23:00:00",
    "2026-06-03 05:00:00", "2026-06-03 11:00:00", "2026-06-03 17:00:00", "2026-06-03 23:00:00",
]


def safe_print(msg: str) -> None:
    print(msg.encode("ascii", "replace").decode("ascii"))


def eval_forward_full(symbol: str, entry_price: float, start_ms: int) -> dict:
    empty = {"return_4h": 0.0, "max_up_4h": 0.0, "mdd_4h": 0.0, "entry_price": entry_price}
    for attempt in range(5):
        try:
            fwd = fetch_forward_5m(symbol, start_ms, 48)
            if entry_price <= 0 or not fwd:
                return empty
            chunk = fwd[:48]
            entry = float(chunk[0][1]) if chunk else entry_price
            if entry <= 0:
                entry = entry_price
            max_h = max(ohlcv(k)[1] for k in chunk)
            min_l = min(ohlcv(k)[2] for k in chunk)
            close = float(chunk[-1][4])
            return {
                "entry_price": round(entry, 8),
                "return_4h": round((close - entry) / entry * 100, 4),
                "max_up_4h": round((max_h - entry) / entry * 100, 4),
                "mdd_4h": round((min_l - entry) / entry * 100, 4),
            }
        except Exception:
            if attempt < 4:
                time.sleep(2 * (attempt + 1))
                continue
            return empty
    return empty


def miss_category(missed_sym: str, matches: list[Candidate], match_syms: set[str], mc: Candidate | None) -> str:
    if missed_sym not in match_syms:
        return "filter_miss"
    if not mc:
        return "data_unavailable"
    if mc.rank_scout > 2:
        return "ranking_miss"
    return "ranking_miss"


def diagnose_failure(missed_sym: str, matches: list[Candidate], match_syms: set[str], mc: Candidate | None) -> str:
    cat = miss_category(missed_sym, matches, match_syms, mc)
    if cat == "filter_miss":
        return "filter_miss (not in Pattern B matches)"
    if cat == "data_unavailable":
        return "data_unavailable"
    if not mc:
        return "ranking_miss"
    parts = ["ranking_miss"]
    if mc.seq_5m.get("seq_volume_energy_6", 0) < 1.5:
        parts.append("low volE6")
    if mc.volume_state in ("Low_Volume", "Normal"):
        parts.append("volume weak")
    if mc.rank_h4 > 10:
        parts.append("low H4 rank")
    if mc.momentum_state == "Weak":
        parts.append("MTF weak")
    return "; ".join(parts)


def scan_diagnosis(top2: list[dict], missed: list[dict]) -> dict:
    hits = [t for t in top2 if t.get("hit_7pct")]
    seq_ok = any(t.get("volE6", 0) >= 1.5 for t in top2) and len(hits) >= 1
    vol_ok = any(t.get("volume_state") in ("Volume_Surge", "Low_Volume_Energy_Build") for t in top2)
    mtf_ok = any(t.get("momentum_state") in ("Accelerating_Up", "Sequence_Positive") for t in top2)
    lc_ok = any(t.get("h4_rank", 99) <= 10 for t in top2)
    return {
        "sequence_worked": seq_ok,
        "volume_worked": vol_ok,
        "mtf_worked": mtf_ok,
        "lifecycle_worked": lc_ok,
    }


def run_single_scan(scan_kst: str, universe_count: int) -> dict:
    p16.CACHE_DIR = OUT_DIR / "kline_cache"
    p16.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    end_ms = int(parse_kst(scan_kst).timestamp() * 1000)
    start_ms = end_ms + 5 * 60 * 1000

    safe_print(f"  scan: {scan_kst}")
    matches = scan_universe(end_ms)
    top10 = sorted(matches, key=lambda x: x.explosion_score, reverse=True)[:10]
    top2 = top10[:2]
    picked = {c.symbol for c in top2}

    eval_rows: list[dict] = []
    for c in matches:
        fwd = eval_forward_full(c.symbol, c.price, start_ms)
        c.forward = fwd
        eval_rows.append({"symbol": c.symbol, "rank_scout": c.rank_scout, **fwd})
        time.sleep(0.18)

    eval_rows.sort(key=lambda x: x["max_up_4h"], reverse=True)
    for i, r in enumerate(eval_rows, 1):
        r["actual_rank"] = i

    match_syms = {c.symbol for c in matches}
    top2_results = []
    for c in top2:
        ar = next((r for r in eval_rows if r["symbol"] == c.symbol), {})
        top2_results.append({
            "symbol": c.symbol,
            "scout_rank": c.rank_scout,
            "h4_rank": c.rank_h4,
            "scout_score": c.explosion_score,
            "volE6": c.seq_5m.get("seq_volume_energy_6", 0),
            "volume_state": c.volume_state,
            "momentum_state": c.momentum_state,
            "return_4h": ar.get("return_4h", 0),
            "max_up_4h": ar.get("max_up_4h", 0),
            "mdd_4h": ar.get("mdd_4h", 0),
            "actual_rank": ar.get("actual_rank"),
            "hit_7pct": ar.get("max_up_4h", 0) >= HIT_7,
        })

    missed_raw = find_missed(eval_rows, picked)
    missed_diag = []
    for m in missed_raw[:5]:
        mc = next((x for x in matches if x.symbol == m["symbol"]), None)
        missed_diag.append({
            "symbol": m["symbol"],
            "max_up_4h": m["max_up_4h"],
            "scout_rank": m["rank_scout"],
            "category": miss_category(m["symbol"], matches, match_syms, mc),
            "reason": diagnose_failure(m["symbol"], matches, match_syms, mc),
        })

    diag = scan_diagnosis(top2_results, missed_diag)
    top2_hits = sum(1 for r in top2_results if r["hit_7pct"])
    fp_count = sum(1 for r in top2_results if r["max_up_4h"] < FP_THRESHOLD)

    return {
        "scan_kst": scan_kst,
        "universe_count": universe_count,
        "match_count": len(matches),
        "top10": [
            {
                "symbol": c.symbol,
                "scout_rank": c.rank_scout,
                "h4_rank": c.rank_h4,
                "scout_score": c.explosion_score,
                "volE6": c.seq_5m.get("seq_volume_energy_6"),
            }
            for c in top10
        ],
        "top2": top2_results,
        "top2_hit_7_count": top2_hits,
        "top2_any_hit": top2_hits >= 1,
        "false_positive_top2": fp_count,
        "missed": missed_diag,
        "diagnosis": diag,
        "eval_rows": eval_rows,
    }


def model_judgment(
    top2_hit_rate: float,
    scan_any_hit_rate: float,
    avg_max_up: float,
    missed_total: int,
    fp_total: int,
) -> str:
    if top2_hit_rate >= 35 and scan_any_hit_rate >= 50 and avg_max_up >= 6:
        return "KEEP"
    if top2_hit_rate >= 20 or avg_max_up >= 5:
        return "MODIFY"
    return "DISCARD"


def build_full_report(results: list[dict]) -> tuple[list[str], list[dict]]:
    all_top2 = [r for s in results for r in s["top2"]]
    all_returns = [r["return_4h"] for r in all_top2]
    all_maxups = [r["max_up_4h"] for r in all_top2]
    all_mdds = [r["mdd_4h"] for r in all_top2]
    all_ranks = [r.get("actual_rank", 99) for r in all_top2 if r.get("actual_rank")]
    n_scans = len(results)
    n_picks = len(all_top2)

    top2_hit_rate = sum(1 for r in all_top2 if r["hit_7pct"]) / max(n_picks, 1) * 100
    scan_any_hit = sum(1 for s in results if s["top2_any_hit"]) / max(n_scans, 1) * 100
    fp_total = sum(s["false_positive_top2"] for s in results)
    missed_total = sum(len(s["missed"]) for s in results)
    ret_win = sum(1 for r in all_top2 if r["return_4h"] > 0) / max(n_picks, 1) * 100

    scan_scores = []
    for s in results:
        avg_mu = statistics.mean([t["max_up_4h"] for t in s["top2"]]) if s["top2"] else 0
        scan_scores.append((s["scan_kst"], avg_mu, s["top2_hit_7_count"]))
    best_scan = max(scan_scores, key=lambda x: x[1])
    worst_scan = min(scan_scores, key=lambda x: x[1])

    verdict = model_judgment(top2_hit_rate, scan_any_hit, statistics.mean(all_maxups), missed_total, fp_total)

    lines = [
        "###############################################################",
        "SCOUT PHASE 18 - FULL GENERALIZATION BLIND VALIDATION",
        "###############################################################",
        "",
        "Frozen Phase16 SCOUT. NO filter/threshold/weight changes.",
        f"Total scans: {n_scans} | Eval: scan + 4h forward",
        "",
        "=" * 58,
        "FINAL SUMMARY",
        "=" * 58,
        f"  Total scans: {n_scans}",
        f"  TOP2 hit rate (>=7% max_up): {top2_hit_rate:.1f}%",
        f"  At least one TOP2 hit per scan: {scan_any_hit:.1f}%",
        f"  Avg TOP2 max_up: {statistics.mean(all_maxups):.2f}%",
        f"  Median TOP2 max_up: {statistics.median(all_maxups):.2f}%",
        f"  Avg TOP2 return: {statistics.mean(all_returns):.2f}%",
        f"  Median TOP2 return: {statistics.median(all_returns):.2f}%",
        f"  Win rate (return > 0): {ret_win:.1f}%",
        f"  Avg MDD: {statistics.mean(all_mdds):.2f}%",
        f"  Worst MDD: {min(all_mdds):.2f}%",
        f"  Actual rank avg: {statistics.mean(all_ranks):.1f}" if all_ranks else "  Actual rank avg: n/a",
        f"  Actual rank median: {statistics.median(all_ranks):.1f}" if all_ranks else "  Actual rank median: n/a",
        f"  Missed winners (>=7% outside TOP2): {missed_total}",
        f"  False positives (TOP2 max_up <5%): {fp_total}",
        f"  Best scan: {best_scan[0]} (avg max_up {best_scan[1]:.2f}%)",
        f"  Worst scan: {worst_scan[0]} (avg max_up {worst_scan[1]:.2f}%)",
        "",
        "=" * 58,
        "MODEL JUDGMENT",
        "=" * 58,
        f"  Decision: {verdict}",
        f"  Evidence: hit_rate={top2_hit_rate:.1f}% scan_any_hit={scan_any_hit:.1f}% "
        f"avg_max_up={statistics.mean(all_maxups):.2f}% missed={missed_total} fp={fp_total}",
        "",
        f"ONE LINE: Phase16 SCOUT generalization verdict: {verdict}",
        "",
        "=" * 58,
        "PER-SCAN DETAIL",
        "=" * 58,
    ]

    for s in results:
        d = s["diagnosis"]
        lines.append(f"\n--- {s['scan_kst']} ---")
        lines.append(f"  universe={s['universe_count']} matches={s['match_count']}")
        lines.append("  TOP10 Scout:")
        for t in s["top10"]:
            lines.append(
                f"    #{t['scout_rank']} {t['symbol']} score={t['scout_score']} "
                f"h4=#{t['h4_rank']} volE6={t['volE6']}"
            )
        lines.append("  TOP2 picks:")
        for t in s["top2"]:
            lines.append(
                f"    {t['symbol']} max_up={t['max_up_4h']}% ret={t['return_4h']}% "
                f"mdd={t['mdd_4h']}% rank=#{t['actual_rank']} hit7={t['hit_7pct']}"
            )
        if s["missed"]:
            lines.append("  Missed winners:")
            for m in s["missed"]:
                lines.append(f"    {m['symbol']} max_up={m['max_up_4h']}% [{m['category']}] {m['reason']}")
        lines.append(
            f"  Diagnosis: seq={d['sequence_worked']} vol={d['volume_worked']} "
            f"mtf={d['mtf_worked']} lifecycle={d['lifecycle_worked']}"
        )

    csv_rows: list[dict] = []
    for s in results:
        for t in s["top2"]:
            csv_rows.append({
                "scan_kst": s["scan_kst"],
                "universe_count": s["universe_count"],
                "match_count": s["match_count"],
                "symbol": t["symbol"],
                "scout_rank": t["scout_rank"],
                "h4_rank": t["h4_rank"],
                "scout_score": t["scout_score"],
                "volE6": t["volE6"],
                "return_4h_pct": t["return_4h"],
                "max_up_4h_pct": t["max_up_4h"],
                "mdd_4h_pct": t["mdd_4h"],
                "actual_rank": t["actual_rank"],
                "hit7": t["hit_7pct"],
                "scan_any_hit": s["top2_any_hit"],
            })

    return lines, csv_rows


def run(scans: list[str], full: bool = False, resume: bool = False) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    universe_count = len(load_eligible_symbols(refresh=False, cache_only=False))
    checkpoint = OUT_DIR / "full_run_checkpoint.jsonl"

    done: dict[str, dict] = {}
    if resume and checkpoint.exists():
        for line in checkpoint.open(encoding="utf-8"):
            rec = json.loads(line)
            done[rec["scan_kst"]] = rec

    safe_print(f"Phase 18: {len(scans)} blind scans | universe={universe_count} | resume={len(done)}")
    if full and not resume and checkpoint.exists():
        checkpoint.unlink()

    results: list[dict] = []
    for scan_kst in scans:
        if scan_kst in done:
            safe_print(f"  skip cached: {scan_kst}")
            results.append(done[scan_kst])
            continue
        result = run_single_scan(scan_kst, universe_count)
        results.append(result)
        with checkpoint.open("a", encoding="utf-8") as f:
            slim = {k: v for k, v in result.items() if k != "eval_rows"}
            f.write(json.dumps(slim, ensure_ascii=True) + "\n")

    lines, csv_rows = build_full_report(results)

    if full:
        report = OUT_DIR / "phase18_full_12scan_report.txt"
        csv_path = OUT_DIR / "phase18_full_12scan_results.csv"
    else:
        report = OUT_DIR / "phase18_generalization_report.txt"
        csv_path = OUT_DIR / "all_top2_picks.csv"

    report.write_text("\n".join(lines), encoding="utf-8")
    write_csv(csv_path, csv_rows)

    with (OUT_DIR / "scan_results.jsonl").open("w", encoding="utf-8") as f:
        for s in results:
            slim = {k: v for k, v in s.items() if k != "eval_rows"}
            f.write(json.dumps(slim, ensure_ascii=True) + "\n")

    for line in lines[8:22]:
        safe_print(line)
    safe_print(f"Saved: {report}")
    safe_print(f"Saved: {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Run all 12 scans, full report")
    parser.add_argument("--resume", action="store_true", help="Resume full run from checkpoint")
    parser.add_argument("--max-scans", type=int, default=None)
    parser.add_argument("--scan-kst", type=str, default=None)
    args = parser.parse_args()

    if args.scan_kst:
        run([args.scan_kst], full=False)
    elif args.full:
        run(SCAN_TIMES.copy(), full=True, resume=args.resume)
    else:
        scans = SCAN_TIMES.copy()
        if args.max_scans:
            scans = scans[: args.max_scans]
        run(scans, full=False)


if __name__ == "__main__":
    main()
