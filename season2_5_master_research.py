"""
Scout Season2.5 - Master Research System

Order: Regime -> Universe -> Feature -> Final Scout
Evaluation: Tail-first (Extreme Winner > Average Return)
STRICT Blind Validation | NO process variables | NO composite scores

Usage:
  python season2_5_master_research.py --phase regime
  python season2_5_master_research.py --phase universe
  python season2_5_master_research.py --phase all --cache-only
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
import urllib.error
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import top10_gainer_learning_20260613 as t10

from season2_p37_scout_decision_hierarchy import pf, write_csv
from season2_universe_blind_test import (
    CACHE_DIR,
    KST,
    MIN_UNIVERSE,
    RANDOM_SEED,
    REFERENCE_RANDOM_BASELINE,
    START_KST,
    END_KST,
    UNIVERSE_NAMES,
    assign_universe_members,
    load_cached_scan_times,
    mean_field,
    percentile,
    random_top2,
)

LOGS_DIR = Path("logs")
OUT_DIR = LOGS_DIR / "season2_5"
REGIME_CACHE = OUT_DIR / "regime_cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REGIME_CACHE.mkdir(parents=True, exist_ok=True)

MIN_SCANS_PASS = 5  # repeated superiority requires >= 5 scans


@dataclass
class TailMetrics:
    label: str
    scan_count: int
    candidate_count_avg: float
    avg_return: float
    median_return: float
    win_rate: float
    max_return: float
    min_return: float
    top5_return: float
    top10_return: float
    rate_10pct_plus: float
    rate_5pct_plus: float
    top5pct_avg: float
    tail_ratio: float
    extreme_winner_ratio: float
    extreme_count: int
    variance: float
    vs_reference_avg: float
    passed: str
    recommended: str


REGIME_DEFS = {
    "R1": "BTC_uptrend_24h_gt_3pct",
    "R2": "BTC_sideways_24h_within_2pct",
    "R3": "BTC_spike_2h_gt_2pct",
    "R4": "BTC_crash_recovery_24h_down_2h_flat",
    "R5": "Alt_strength_median_alt_minus_btc_gt_2pct",
    "R6": "Market_volume_surge_btc_vol_growth_gt_20pct",
    "R7": "Funding_neutral_DATA_UNAVAILABLE",
    "R8": "High_volatility_btc_atr_gt_2pct",
}


def load_scan_snapshots() -> list[tuple[str, datetime, list[dict]]]:
    rows: list[tuple[str, datetime, list[dict]]] = []
    for scan_kst, scan_dt in load_cached_scan_times():
        path = CACHE_DIR / f"{scan_kst.replace(' ', '_').replace(':', '')}.json"
        if not path.exists():
            continue
        symbols = json.loads(path.read_text(encoding="utf-8")).get("symbols", [])
        if len(symbols) >= MIN_UNIVERSE:
            rows.append((scan_kst, scan_dt, symbols))
    return rows


def fetch_btc_metrics(scan_dt: datetime, refresh: bool) -> dict | None:
    scan_kst = scan_dt.strftime("%Y-%m-%d %H:%M:%S")
    cache_path = REGIME_CACHE / f"{scan_kst.replace(' ', '_').replace(':', '')}.json"
    if cache_path.exists() and not refresh:
        return json.loads(cache_path.read_text(encoding="utf-8"))

    try:
        end_ms = int(scan_dt.timestamp() * 1000)
        klines = t10.fetch_klines_before("BTCUSDT", t10.INTERVAL_2H, end_ms, t10.RANKING_KLINES_2H)
        if len(klines) < t10.RANKING_KLINES_2H:
            return None
        ranking = t10.compute_24h_ranking(klines)
        if ranking is None:
            return None
        prev_24 = klines[-(t10.CANDLES_24H_2H + 1) : -1]
        prev_close_2h = float(klines[-2][4])
        price = ranking["price_at_scan"]
        ret_2h = t10.close_return_percent(price, prev_close_2h)
        vol_recent = sum(float(c[5]) for c in prev_24[-6:])
        vol_prior = sum(float(c[5]) for c in prev_24[-12:-6]) if len(prev_24) >= 12 else vol_recent
        vol_growth = (vol_recent / vol_prior - 1.0) * 100 if vol_prior > 0 else 0.0
        atr = t10.average_true_range_percent(prev_24, price)
        data = {
            "scan_time_kst": scan_kst,
            "btc_return_24h": ranking["return_24h_percent"],
            "btc_return_2h": ret_2h,
            "btc_volume_growth_pct": vol_growth,
            "btc_atr_pct": atr,
        }
        cache_path.write_text(json.dumps(data), encoding="utf-8")
        time.sleep(t10.API_SLEEP_SEC)
        return data
    except (urllib.error.HTTPError, urllib.error.URLError, OSError):
        return None


def classify_regimes(btc: dict | None, alt_snaps: list[dict]) -> list[str]:
    """Return all matching regime tags for this scan."""
    tags: list[str] = []
    if btc:
        r24 = btc["btc_return_24h"]
        r2 = btc["btc_return_2h"]
        vg = btc["btc_volume_growth_pct"]
        atr = btc["btc_atr_pct"]
        if r24 > 3.0:
            tags.append("R1")
        if -2.0 <= r24 <= 2.0:
            tags.append("R2")
        if r2 > 2.0 and r24 > 1.0:
            tags.append("R3")
        if r24 < -3.0 and abs(r2) < 1.0:
            tags.append("R4")
        if vg > 20.0:
            tags.append("R6")
        if atr > 2.0:
            tags.append("R8")

    if alt_snaps:
        alt_med = statistics.median(s["return_24h_percent"] for s in alt_snaps)
        btc_r = btc["btc_return_24h"] if btc else 0.0
        if alt_med - btc_r > 2.0:
            tags.append("R5")

    # R7: no historical funding in local dataset
    return tags or ["R0_unknown"]


def compute_tail_metrics(
    label: str,
    scan_records: list[dict],
    pool_returns: list[float],
    reference_avg: float,
    reference_rate_5: float,
) -> TailMetrics:
    if not scan_records:
        return TailMetrics(
            label=label, scan_count=0, candidate_count_avg=0,
            avg_return=0, median_return=0, win_rate=0, max_return=0, min_return=0,
            top5_return=0, top10_return=0, rate_10pct_plus=0, rate_5pct_plus=0,
            top5pct_avg=0, tail_ratio=0, extreme_winner_ratio=0, extreme_count=0,
            variance=0, vs_reference_avg=-reference_avg, passed="no", recommended="no",
        )

    scan_avgs = [r["avg_return"] for r in scan_records]
    pick_returns = [r["top1_return"] for r in scan_records] + [r["top2_return"] for r in scan_records]
    p50 = percentile(pool_returns, 50) if pool_returns else 1.0
    p95 = percentile(pool_returns, 95) if pool_returns else 0.0
    p90 = percentile(pool_returns, 90) if pool_returns else 0.0
    p95_pool = [r for r in pool_returns if r >= p95] if pool_returns else []

    avg = statistics.mean(scan_avgs)
    rate_10 = sum(1 for r in pick_returns if r >= 10.0) / len(pick_returns) if pick_returns else 0
    rate_5 = sum(1 for r in pick_returns if r >= 5.0) / len(pick_returns) if pick_returns else 0
    tail_ratio = (p95 / p50) if p50 > 0.01 else 0.0
    extreme_winner_ratio = statistics.mean([r["extreme_in_pool"] for r in scan_records])

    passed = (
        avg > reference_avg
        and sum(1 for a in scan_avgs if a > 0) / len(scan_avgs) >= 0.5
        and len(scan_avgs) >= MIN_SCANS_PASS
        and rate_5 >= reference_rate_5
    )
    recommended = "yes" if passed else "no"

    return TailMetrics(
        label=label,
        scan_count=len(scan_records),
        candidate_count_avg=round(statistics.mean(r["candidate_count"] for r in scan_records), 1),
        avg_return=round(avg, 4),
        median_return=round(statistics.median(scan_avgs), 4),
        win_rate=round(sum(1 for a in scan_avgs if a > 0) / len(scan_avgs), 4),
        max_return=round(max(scan_avgs), 4),
        min_return=round(min(scan_avgs), 4),
        top5_return=round(percentile(scan_avgs, 95), 4),
        top10_return=round(percentile(scan_avgs, 90), 4),
        rate_10pct_plus=round(rate_10, 4),
        rate_5pct_plus=round(rate_5, 4),
        top5pct_avg=round(statistics.mean(p95_pool) if p95_pool else 0, 4),
        tail_ratio=round(tail_ratio, 4),
        extreme_winner_ratio=round(extreme_winner_ratio, 4),
        extreme_count=sum(1 for r in pick_returns if r >= 10.0),
        variance=round(statistics.pvariance(scan_avgs), 4) if len(scan_avgs) > 1 else 0,
        vs_reference_avg=round(avg - reference_avg, 4),
        passed="yes" if passed else "no",
        recommended=recommended,
    )


def tail_metrics_to_row(rank: int, m: TailMetrics, kind: str) -> dict:
    return {
        "rank": rank,
        "kind": kind,
        "label": m.label,
        "scan_count": m.scan_count,
        "candidate_count_avg": m.candidate_count_avg,
        "avg_return_pct": m.avg_return,
        "vs_reference_random_pct": m.vs_reference_avg,
        "median_return_pct": m.median_return,
        "win_rate": m.win_rate,
        "max_return_pct": m.max_return,
        "min_return_pct": m.min_return,
        "top5_scan_return_pct": m.top5_return,
        "top10_scan_return_pct": m.top10_return,
        "rate_10pct_plus": m.rate_10pct_plus,
        "rate_5pct_plus": m.rate_5pct_plus,
        "top5pct_pool_avg": m.top5pct_avg,
        "tail_ratio": m.tail_ratio,
        "extreme_winner_ratio": m.extreme_winner_ratio,
        "extreme_count_10pct_plus": m.extreme_count,
        "variance": m.variance,
        "passed": m.passed,
        "recommended": m.recommended,
        "reference_random_baseline_pct": REFERENCE_RANDOM_BASELINE,
        "learning_recommendation": "NO_ACTION",
    }


def run_random_top2_blind(
    scans: list[tuple[str, datetime, list[dict]]],
    member_fn,
    label: str,
    refresh_btc: bool,
) -> tuple[list[dict], list[float], TailMetrics]:
    """Generic random Top2 blind test. member_fn(symbols) -> candidate list."""
    scan_rows: list[dict] = []
    pool_returns: list[float] = []

    for scan_idx, (scan_kst, scan_dt, symbols) in enumerate(scans):
        btc = fetch_btc_metrics(scan_dt, refresh_btc)
        regimes = classify_regimes(btc, symbols)
        members = member_fn(symbols, scan_kst, scan_dt, regimes)
        if len(members) < MIN_UNIVERSE:
            continue

        for m in members:
            pool_returns.append(m["forward_2h_pct"])

        rng = random.Random(RANDOM_SEED + scan_idx)
        picked = random_top2(members, rng)
        top1 = picked[0]["forward_2h_pct"]
        top2 = picked[1]["forward_2h_pct"] if len(picked) > 1 else top1
        avg = (top1 + top2) / 2
        extreme_in_pool = any(m["forward_2h_pct"] >= 5.0 for m in members)

        scan_rows.append({
            "scan_time_kst": scan_kst,
            "regime": "|".join(regimes),
            "universe": label,
            "candidate_count": len(members),
            "top1": picked[0]["symbol"],
            "top2": picked[1]["symbol"] if len(picked) > 1 else "",
            "top1_return": top1,
            "top2_return": top2,
            "avg_return": avg,
            "extreme_in_pool": 1 if extreme_in_pool else 0,
        })

    ref_rows = scan_rows  # baseline from same scan set
    ref_picks = [r["top1_return"] for r in ref_rows] + [r["top2_return"] for r in ref_rows if r["top2_return"]]
    ref_rate_5 = sum(1 for r in ref_picks if r >= 5.0) / len(ref_picks) if ref_picks else 0
    ref_avg = statistics.mean([r["avg_return"] for r in ref_rows]) if ref_rows else REFERENCE_RANDOM_BASELINE

    metrics = compute_tail_metrics(label, scan_rows, pool_returns, ref_avg, ref_rate_5)
    return scan_rows, pool_returns, metrics


def phase_regime(scans: list[tuple[str, datetime, list[dict]]], refresh_btc: bool) -> tuple[list[TailMetrics], list[dict]]:
    all_detail: list[dict] = []
    by_regime_scans: dict[str, list[dict]] = {k: [] for k in REGIME_DEFS}
    by_regime_scans["R0_unknown"] = []
    by_regime_pool: dict[str, list[float]] = {k: [] for k in REGIME_DEFS}
    by_regime_pool["R0_unknown"] = []

    for scan_idx, (scan_kst, scan_dt, symbols) in enumerate(scans):
        btc = fetch_btc_metrics(scan_dt, refresh_btc)
        regimes = classify_regimes(btc, symbols)
        rng = random.Random(RANDOM_SEED + scan_idx)
        picked = random_top2(symbols, rng)
        if len(picked) < 2:
            continue
        top1, top2 = picked[0], picked[1]
        avg = (top1["forward_2h_pct"] + top2["forward_2h_pct"]) / 2
        extreme = any(s["forward_2h_pct"] >= 5.0 for s in symbols)

        row = {
            "scan_time_kst": scan_kst,
            "regime": "|".join(regimes),
            "universe": "FULL",
            "candidate_count": len(symbols),
            "top1": top1["symbol"],
            "top2": top2["symbol"],
            "top1_return": top1["forward_2h_pct"],
            "top2_return": top2["forward_2h_pct"],
            "avg_return": avg,
            "extreme_in_pool": 1 if extreme else 0,
        }
        all_detail.append(row)

        for tag in regimes:
            if tag not in by_regime_scans:
                by_regime_scans[tag] = []
                by_regime_pool[tag] = []
            by_regime_scans[tag].append(row)
            by_regime_pool[tag].extend(s["forward_2h_pct"] for s in symbols)

    ref_avg = statistics.mean(r["avg_return"] for r in all_detail) if all_detail else REFERENCE_RANDOM_BASELINE
    ref_picks = [r["top1_return"] for r in all_detail] + [r["top2_return"] for r in all_detail]
    ref_rate_5 = sum(1 for r in ref_picks if r >= 5.0) / len(ref_picks) if ref_picks else 0

    results: list[TailMetrics] = []
    for rid, name in REGIME_DEFS.items():
        m = compute_tail_metrics(rid, by_regime_scans.get(rid, []), by_regime_pool.get(rid, []), ref_avg, ref_rate_5)
        m.label = f"{rid}_{name}"
        results.append(m)

    results.sort(key=lambda x: (x.rate_5pct_plus, x.rate_10pct_plus, x.avg_return), reverse=True)
    return results, all_detail


def phase_universe(scans: list[tuple[str, datetime, list[dict]]], refresh_btc: bool) -> tuple[list[TailMetrics], list[dict]]:
    all_detail: list[dict] = []
    results: list[TailMetrics] = []

    ref_rows: list[dict] = []
    for scan_idx, (scan_kst, scan_dt, symbols) in enumerate(scans):
        rng = random.Random(RANDOM_SEED + scan_idx)
        picked = random_top2(symbols, rng)
        if len(picked) >= 2:
            ref_rows.append({
                "avg_return": (picked[0]["forward_2h_pct"] + picked[1]["forward_2h_pct"]) / 2,
                "top1_return": picked[0]["forward_2h_pct"],
                "top2_return": picked[1]["forward_2h_pct"],
            })
    ref_avg = statistics.mean(r["avg_return"] for r in ref_rows) if ref_rows else REFERENCE_RANDOM_BASELINE
    ref_picks = [r["top1_return"] for r in ref_rows] + [r["top2_return"] for r in ref_rows]
    ref_rate_5 = sum(1 for r in ref_picks if r >= 5.0) / len(ref_picks) if ref_picks else 0

    for uid, uname in UNIVERSE_NAMES.items():
        scan_rows: list[dict] = []
        pool: list[float] = []
        for scan_idx, (scan_kst, scan_dt, symbols) in enumerate(scans):
            btc = fetch_btc_metrics(scan_dt, refresh_btc)
            regimes = classify_regimes(btc, symbols)
            universes = assign_universe_members(symbols)
            members = universes.get(uid, [])
            if len(members) < MIN_UNIVERSE:
                continue
            pool.extend(m["forward_2h_pct"] for m in members)
            rng = random.Random(RANDOM_SEED + scan_idx + ord(uid))
            picked = random_top2(members, rng)
            top1 = picked[0]
            top2 = picked[1] if len(picked) > 1 else top1
            avg = (top1["forward_2h_pct"] + top2["forward_2h_pct"]) / 2
            row = {
                "scan_time_kst": scan_kst,
                "regime": "|".join(regimes),
                "universe": uid,
                "candidate_count": len(members),
                "top1": top1["symbol"],
                "top2": top2["symbol"],
                "top1_return": top1["forward_2h_pct"],
                "top2_return": top2["forward_2h_pct"],
                "avg_return": avg,
                "extreme_in_pool": 1 if any(m["forward_2h_pct"] >= 5.0 for m in members) else 0,
            }
            scan_rows.append(row)
            all_detail.append(row)

        m = compute_tail_metrics(uid, scan_rows, pool, ref_avg, ref_rate_5)
        m.label = uname
        results.append(m)

    results.sort(key=lambda x: (x.rate_5pct_plus, x.rate_10pct_plus, x.avg_return), reverse=True)
    return results, all_detail


def write_report(
    regime_results: list[TailMetrics],
    universe_results: list[TailMetrics],
    scan_count: int,
    data_notes: list[str],
) -> None:
    lines = [
        "===== SCOUT SEASON2.5 MASTER RESEARCH REPORT =====",
        "",
        "Mission: Extreme Value Search (2h Top2 tail performance)",
        f"Period: {START_KST.strftime('%Y-%m-%d')} ~ {END_KST.strftime('%Y-%m-%d')} KST | 2h interval",
        f"Reference Random baseline (prior): {REFERENCE_RANDOM_BASELINE}%",
        f"Valid cached scans analyzed: {scan_count}",
        "",
        "=== FALSIFIED (do not revisit) ===",
        "- Season1 Process (Scout +0.60% vs Random +0.92%)",
        "- Season2 single price/volume features (0 adopted)",
        "- Composite Scout Score / Belief / Sync / Narrative",
        "",
        "=== Research order ===",
        "1 Regime -> 2 Universe -> 3 Feature -> 4 Final Scout",
        "Feature research BLOCKED until Regime + Universe pass.",
        "",
        "=== Regime Ranking (tail-first) ===",
    ]
    for i, r in enumerate(regime_results, 1):
        lines.append(
            f"{i}. {r.label} | +5% rate={r.rate_5pct_plus:.1%} +10% rate={r.rate_10pct_plus:.1%} "
            f"avg={r.avg_return}% scans={r.scan_count} pass={r.passed}"
        )

    lines.extend(["", "=== Universe Ranking (tail-first) ==="])
    for i, r in enumerate(universe_results, 1):
        lines.append(
            f"{i}. {r.label} | +5% rate={r.rate_5pct_plus:.1%} avg={r.avg_return}% "
            f"extreme_pool={r.extreme_winner_ratio:.1%} scans={r.scan_count} pass={r.passed}"
        )

    passed_r = [r for r in regime_results if r.recommended == "yes"]
    passed_u = [r for r in universe_results if r.recommended == "yes"]

    lines.extend(["", "=== Post-analysis (8 questions) ==="])
    lines.append(f"1. Why success? {'None yet' if not passed_r and not passed_u else 'See passed rows'}")
    lines.append("2. Why failure? Top-gainer pool exhaustion; forward_2h measurement; small sample")
    lines.append(f"3. Regime cause? Top regime by +5% rate: {regime_results[0].label if regime_results else 'N/A'}")
    lines.append(f"4. Universe cause? Top universe by +5% rate: {universe_results[0].label if universe_results else 'N/A'}")
    lines.append("5. Feature cause? N/A - feature phase not started")
    lines.append(f"6. Data shortage? YES - only {scan_count} scans cached (need >= {MIN_SCANS_PASS} per bucket)")
    lines.append("7. API/cache? Binance 418 rate limit blocked full Jun6-15 collection")
    lines.append(f"8. Sample sufficient? NO if scan_count < {MIN_SCANS_PASS}")

    lines.extend(["", "=== Decision ==="])
    if passed_u:
        lines.append(f"Proceed to Feature research inside universe: {passed_u[0].label}")
    elif passed_r:
        lines.append(f"Proceed to Universe research filtered by regime: {passed_r[0].label}")
    else:
        lines.append("NO Regime/Universe passed. Feature research remains BLOCKED.")
        lines.append("Next: rebuild kline cache Jun6-15, re-run with true forward_2h.")

    if data_notes:
        lines.extend(["", "=== Data notes ==="] + data_notes)
    lines.append("")
    lines.append("Learning recommendation: NO_ACTION")

    (OUT_DIR / "master_research_report.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scout Season2.5 Master Research")
    parser.add_argument("--phase", choices=["regime", "universe", "all"], default="all")
    parser.add_argument("--refresh-btc", action="store_true")
    args = parser.parse_args()

    scans = load_scan_snapshots()
    if not scans:
        raise SystemExit(
            "No cached scan snapshots. Run season2_universe_blind_test.py first "
            "(or wait for API rate limit to clear)."
        )

    data_notes = [
        f"Analyzing {len(scans)} cached scan(s) from {CACHE_DIR}",
        "Tail metrics prioritized: +10% rate > +5% rate > top5% avg > tail ratio > avg return",
        "R7 Funding Neutral: no historical funding data - excluded from scoring",
    ]

    regime_results: list[TailMetrics] = []
    universe_results: list[TailMetrics] = []
    all_rows: list[dict] = []

    if args.phase in ("regime", "all"):
        print("Phase 1: Regime blind test (Random Top2, FULL universe)...")
        regime_results, regime_rows = phase_regime(scans, args.refresh_btc)
        all_rows.extend(regime_rows)
        write_csv(
            OUT_DIR / "regime_ranking_table.csv",
            [tail_metrics_to_row(i, r, "regime") for i, r in enumerate(regime_results, 1)],
        )

    if args.phase in ("universe", "all"):
        print("Phase 2: Universe blind test (Random Top2 per universe)...")
        universe_results, uni_rows = phase_universe(scans, args.refresh_btc)
        all_rows.extend(uni_rows)
        write_csv(
            OUT_DIR / "universe_ranking_table.csv",
            [tail_metrics_to_row(i, r, "universe") for i, r in enumerate(universe_results, 1)],
        )

    if all_rows:
        detail = []
        for r in all_rows:
            detail.append({
                "scan_time_kst": r["scan_time_kst"],
                "regime": r.get("regime", ""),
                "universe": r.get("universe", ""),
                "candidate_count": r["candidate_count"],
                "top1": r["top1"],
                "top2": r["top2"],
                "avg_return": round(r["avg_return"], 4),
                "median_return": round(r["avg_return"], 4),
                "top1_return": round(r["top1_return"], 4),
                "top2_return": round(r["top2_return"], 4),
                "extreme_in_pool": r["extreme_in_pool"],
                "learning_recommendation": "NO_ACTION",
            })
        write_csv(OUT_DIR / "scan_records.csv", detail)

    write_report(regime_results, universe_results, len(scans), data_notes)

    print("")
    print("Season2.5 Master Research complete")
    print(f"Scans: {len(scans)} | Outputs: {OUT_DIR}")
    if regime_results:
        top = regime_results[0]
        print(f"Top Regime (tail): {top.label} +5%={top.rate_5pct_plus:.1%} avg={top.avg_return}%")
    if universe_results:
        top = universe_results[0]
        print(f"Top Universe (tail): {top.label} +5%={top.rate_5pct_plus:.1%} avg={top.avg_return}%")
    passed = [r for r in universe_results + regime_results if r.recommended == "yes"]
    print(f"Passed buckets: {len(passed)} | Feature phase: {'OPEN' if passed else 'BLOCKED'}")


if __name__ == "__main__":
    main()
