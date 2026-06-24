"""
Scout Season2.1 - Universe Blind Test

Find which candidate universe produces the best random Top2 2h forward returns.
NO feature scoring. NO process variables. Random Top2 within each universe only.

Usage:
  python season2_universe_blind_test.py
  python season2_universe_blind_test.py --max-scans 5
  python season2_universe_blind_test.py --refresh-cache
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import top10_gainer_learning_20260613 as t10

from season2_p37_scout_decision_hierarchy import pf, write_csv

LOGS_DIR = Path("logs")
OUT_DIR = LOGS_DIR / "universe_research"
CACHE_DIR = OUT_DIR / "snapshots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

KST = timezone(timedelta(hours=9))
START_KST = datetime(2026, 6, 1, 9, 0, tzinfo=KST)
END_KST = datetime(2026, 6, 15, 9, 0, tzinfo=KST)
SCAN_INTERVAL_H = 2
RANDOM_SEED = 42
REFERENCE_RANDOM_BASELINE = 0.9214  # prior top10-universe blind loop
UNIVERSE_SIZE = 100
MIN_UNIVERSE = 2

SCAN_HOURS_FALLBACK = (9, 11, 13, 15, 17, 19, 21, 23)


def gen_scan_times() -> list[tuple[str, datetime]]:
    times: list[tuple[str, datetime]] = []
    t = START_KST
    while t <= END_KST:
        times.append((t.strftime("%Y-%m-%d %H:%M:%S"), t))
        t += timedelta(hours=SCAN_INTERVAL_H)
    return times


def ohlcv(kline: list) -> tuple[float, float, float, float, float]:
    return (
        float(kline[1]),
        float(kline[2]),
        float(kline[3]),
        float(kline[4]),
        float(kline[5]),
    )


def forward_return_2h(entry: float, forward_klines: list[list], scan_dt: datetime) -> float | None:
    if not forward_klines:
        return None
    open_p = float(forward_klines[0][1])
    close_p = float(forward_klines[0][4])
    base = entry if entry > 0 else open_p
    if base <= 0:
        return None
    # 2h forward candle move (T0 open -> T0+2h close)
    return (close_p - open_p) / base * 100


def forward_drawdown_2h(entry: float, forward_klines: list[list], scan_dt: datetime) -> float:
    if entry <= 0 or not forward_klines:
        return 0.0
    target = scan_dt + timedelta(hours=2)
    min_low = entry
    for candle in forward_klines:
        if t10.kline_close_dt(candle) > target:
            break
        _, _, low, _, _ = ohlcv(candle)
        min_low = min(min_low, low)
    return max(0.0, (entry - min_low) / entry * 100)


def build_light_snapshot(
    symbol: str,
    scan_kst: str,
    scan_dt: datetime,
    klines: list[list],
) -> dict | None:
    if len(klines) < t10.RANKING_KLINES_2H:
        return None

    ranking = t10.compute_24h_ranking(klines)
    if ranking is None:
        return None

    price = ranking["price_at_scan"]
    signal = klines[-1]
    open_p, high_p, low_p, close_p, vol = ohlcv(signal)

    prev_24 = klines[-(t10.CANDLES_24H_2H + 1) : -1]
    if len(prev_24) < t10.CANDLES_24H_2H:
        return None

    high_24h = max(ohlcv(c)[1] for c in prev_24)
    low_24h = min(ohlcv(c)[2] for c in prev_24)
    quote_vol_24h = sum(ohlcv(c)[4] * ohlcv(c)[3] for c in prev_24)

    vol_recent = sum(ohlcv(c)[4] for c in prev_24[-6:])
    vol_prior = sum(ohlcv(c)[4] for c in prev_24[-12:-6]) if len(prev_24) >= 12 else vol_recent
    vol_growth = (vol_recent / vol_prior - 1.0) * 100 if vol_prior > 0 else 0.0

    prev_close_2h = float(klines[-2][4]) if len(klines) >= 2 else price
    return_2h = t10.close_return_percent(price, prev_close_2h)

    drawdown_from_high = (price - high_24h) / high_24h * 100 if high_24h > 0 else 0.0
    position_24h = t10.position_in_range(price, low_24h, high_24h)
    highest_close_24 = max(float(c[4]) for c in prev_24)

    atr_pct = t10.average_true_range_percent(prev_24, price)
    recent_3_vol = sum(ohlcv(c)[4] for c in klines[-4:-1]) / 3 if len(klines) >= 4 else vol
    prior_3_vol = sum(ohlcv(c)[4] for c in klines[-7:-4]) / 3 if len(klines) >= 7 else recent_3_vol
    vol_accel = recent_3_vol / prior_3_vol if prior_3_vol > 0 else 0.0

    pre6 = t10.compression_flags(klines[:-1], min(6, len(klines) - 1))

    return {
        "symbol": symbol,
        "scan_time_kst": scan_kst,
        "entry_price": price,
        "return_24h_percent": ranking["return_24h_percent"],
        "return_2h_percent": return_2h,
        "quote_volume_24h": quote_vol_24h,
        "volume_growth_24h_pct": vol_growth,
        "drawdown_from_24h_high_pct": drawdown_from_high,
        "position_24h_percent": position_24h,
        "atr_percent": atr_pct,
        "volume_acceleration_ratio": vol_accel,
        "pre6_tight_range": pre6.get("tight_range", False),
        "pre6_volume_contraction": pre6.get("volume_contraction", False),
        "near_24h_high": position_24h >= 88 and close_p <= highest_close_24 * 1.01,
        "break_24h_high": close_p > highest_close_24,
    }


def fetch_one_symbol(
    symbol: str,
    scan_kst: str,
    scan_dt: datetime,
    end_ms: int,
) -> dict | None:
    try:
        klines = t10.fetch_klines_before(symbol, t10.INTERVAL_2H, end_ms, t10.RANKING_KLINES_2H)
        snap = build_light_snapshot(symbol, scan_kst, scan_dt, klines)
        if snap is None:
            return None
        fwd = t10.fetch_klines_forward(symbol, end_ms, end_ms + 4 * t10.INTERVAL_2H_MS)
        ret2 = forward_return_2h(snap["entry_price"], fwd, scan_dt)
        if ret2 is None:
            return None
        snap["forward_2h_pct"] = ret2
        snap["forward_drawdown_2h_pct"] = forward_drawdown_2h(snap["entry_price"], fwd, scan_dt)
        time.sleep(0.05)
        return snap
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError):
        time.sleep(0.2)
        return None


def load_or_build_scan_snapshots(
    scan_kst: str,
    scan_dt: datetime,
    eligible: set[str],
    refresh: bool,
    workers: int = 8,
) -> list[dict]:
    cache_path = CACHE_DIR / f"{scan_kst.replace(' ', '_').replace(':', '')}.json"
    if cache_path.exists() and not refresh:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        return data.get("symbols", [])

    end_ms = int(scan_dt.timestamp() * 1000)
    symbols = sorted(eligible)
    snapshots: list[dict] = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(fetch_one_symbol, sym, scan_kst, scan_dt, end_ms): sym
            for sym in symbols
        }
        done = 0
        for fut in as_completed(futures):
            done += 1
            snap = fut.result()
            if snap:
                snapshots.append(snap)
            if done % 80 == 0:
                print(f"    {scan_kst}: {done}/{len(symbols)} fetched, {len(snapshots)} valid")
            if done % 20 == 0:
                time.sleep(0.5)

    cache_path.write_text(
        json.dumps({"scan_time_kst": scan_kst, "symbols": snapshots}, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"    {scan_kst}: cached {len(snapshots)} symbols")
    return snapshots


def assign_universe_members(all_snaps: list[dict]) -> dict[str, list[dict]]:
    if not all_snaps:
        return {k: [] for k in "ABCDEFGH"}

    by_quote_vol = sorted(all_snaps, key=lambda s: s["quote_volume_24h"], reverse=True)
    by_vol_growth = sorted(all_snaps, key=lambda s: s["volume_growth_24h_pct"], reverse=True)
    by_ret_2h = sorted(all_snaps, key=lambda s: s["return_2h_percent"], reverse=True)
    by_atr = sorted(all_snaps, key=lambda s: s["atr_percent"], reverse=True)
    ret_24h_median = statistics.median(s["return_24h_percent"] for s in all_snaps)

    rank_2h = {s["symbol"]: i + 1 for i, s in enumerate(by_ret_2h)}

    uni_a = set(s["symbol"] for s in by_quote_vol[:UNIVERSE_SIZE])
    uni_b = set(s["symbol"] for s in by_vol_growth[:UNIVERSE_SIZE])
    uni_c = {
        s["symbol"] for s in by_ret_2h
        if 20 <= rank_2h[s["symbol"]] <= 60 and rank_2h[s["symbol"]] > 10
    }
    uni_d = {
        s["symbol"] for s in all_snaps
        if -12.0 <= s["drawdown_from_24h_high_pct"] <= -3.0
    }
    uni_e = set(s["symbol"] for s in by_atr[:UNIVERSE_SIZE])
    uni_f = {
        s["symbol"] for s in all_snaps
        if s["pre6_tight_range"] and s["volume_acceleration_ratio"] >= 1.0
    }
    uni_g = {
        s["symbol"] for s in all_snaps
        if s["near_24h_high"] and not s["break_24h_high"]
    }
    uni_h = {
        s["symbol"] for s in all_snaps
        if s["return_24h_percent"] <= ret_24h_median and s["volume_acceleration_ratio"] >= 1.1
    }

    lookup = {s["symbol"]: s for s in all_snaps}
    names = {
        "A": uni_a,
        "B": uni_b,
        "C": uni_c,
        "D": uni_d,
        "E": uni_e,
        "F": uni_f,
        "G": uni_g,
        "H": uni_h,
    }
    return {
        key: [lookup[s] for s in syms if s in lookup]
        for key, syms in names.items()
    }


def random_top2(members: list[dict], rng: random.Random) -> list[dict]:
    if len(members) <= 2:
        return members[:]
    return rng.sample(members, 2)


def mean_field(rows: list[dict], field: str) -> float:
    if not rows:
        return 0.0
    return statistics.mean(r[field] for r in rows)


def percentile(vals: list[float], pct: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    idx = int(len(s) * pct / 100)
    idx = min(idx, len(s) - 1)
    return s[idx]


@dataclass
class UniverseResult:
    universe_id: str
    universe_name: str
    scan_count: int
    avg_return: float
    vs_random: float
    win_rate: float
    top10_pct: float
    median: float
    max_ret: float
    min_ret: float
    variance: float
    extreme_winner_ratio: float
    top5pct_ratio: float
    avg_drawdown: float
    recommended: str


UNIVERSE_NAMES = {
    "A": "A_top100_quote_volume",
    "B": "B_top100_volume_growth",
    "C": "C_2h_return_rank_20_60",
    "D": "D_pullback_3_12pct_from_24h_high",
    "E": "E_top100_atr",
    "F": "F_6h_box_volume_increase",
    "G": "G_near_24h_high_pre_breakout",
    "H": "H_low_24h_return_volume_increase",
}


def load_cached_scan_times() -> list[tuple[str, datetime]]:
    rows: list[tuple[str, datetime]] = []
    for path in sorted(CACHE_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        scan_kst = data.get("scan_time_kst", "")
        if not scan_kst or len(data.get("symbols", [])) < MIN_UNIVERSE:
            continue
        scan_dt = datetime.strptime(scan_kst, "%Y-%m-%d %H:%M:%S").replace(tzinfo=KST)
        if START_KST <= scan_dt <= END_KST:
            rows.append((scan_kst, scan_dt))
    return rows


def load_eligible_symbols(refresh: bool, cache_only: bool) -> set[str]:
    if cache_only:
        symbols: set[str] = set()
        for path in CACHE_DIR.glob("*.json"):
            data = json.loads(path.read_text(encoding="utf-8"))
            for s in data.get("symbols", []):
                symbols.add(s["symbol"])
        return symbols
    try:
        return t10.get_eligible_symbols()
    except (urllib.error.HTTPError, urllib.error.URLError, OSError) as exc:
        print(f"  exchangeInfo failed ({exc}); using cached symbol union")
        return load_eligible_symbols(refresh=True, cache_only=True)


def run_universe_blind_test(
    max_scans: int | None,
    refresh: bool,
    workers: int,
    start_date: str | None,
    end_date: str | None,
    cache_only: bool,
) -> tuple[list[UniverseResult], list[dict], dict[str, list[float]]]:
    eligible = load_eligible_symbols(refresh, cache_only)
    if not eligible:
        raise SystemExit("No eligible symbols")

    if cache_only:
        scan_times = load_cached_scan_times()
    else:
        scan_times = gen_scan_times()
        if start_date:
            scan_times = [(s, d) for s, d in scan_times if s[:10] >= start_date]
        if end_date:
            scan_times = [(s, d) for s, d in scan_times if s[:10] <= end_date]
    if max_scans:
        scan_times = scan_times[:max_scans]

    per_scan_returns: dict[str, list[float]] = {k: [] for k in UNIVERSE_NAMES}
    per_scan_drawdowns: dict[str, list[float]] = {k: [] for k in UNIVERSE_NAMES}
    all_member_returns: dict[str, list[float]] = {k: [] for k in UNIVERSE_NAMES}
    extreme_flags: dict[str, list[bool]] = {k: [] for k in UNIVERSE_NAMES}
    scan_detail_rows: list[dict] = []
    rng_base = random.Random(RANDOM_SEED)

    for scan_idx, (scan_kst, scan_dt) in enumerate(scan_times):
        print(f"Scan {scan_idx + 1}/{len(scan_times)}: {scan_kst}")
        snapshots = load_or_build_scan_snapshots(scan_kst, scan_dt, eligible, refresh, workers)
        if cache_only:
            cache_path = CACHE_DIR / f"{scan_kst.replace(' ', '_').replace(':', '')}.json"
            if cache_path.exists():
                snapshots = json.loads(cache_path.read_text(encoding="utf-8")).get("symbols", [])
        if len(snapshots) < 20:
            print(f"  skip - only {len(snapshots)} symbols with forward data")
            continue

        universes = assign_universe_members(snapshots)
        rng = random.Random(RANDOM_SEED + scan_idx)

        for uid, members in universes.items():
            if len(members) < MIN_UNIVERSE:
                continue

            for m in members:
                all_member_returns[uid].append(m["forward_2h_pct"])

            best_in_uni = max(m["forward_2h_pct"] for m in members)
            extreme_flags[uid].append(best_in_uni >= 5.0)

            picked = random_top2(members, rng)
            top2_ret = mean_field(picked, "forward_2h_pct")
            top2_dd = mean_field(picked, "forward_drawdown_2h_pct")
            per_scan_returns[uid].append(top2_ret)
            per_scan_drawdowns[uid].append(top2_dd)

            for rank, item in enumerate(picked, 1):
                scan_detail_rows.append({
                    "scan_time_kst": scan_kst,
                    "universe_id": uid,
                    "universe_name": UNIVERSE_NAMES[uid],
                    "rank": rank,
                    "symbol": item["symbol"],
                    "entry_price": round(item["entry_price"], 6),
                    "forward_2h_pct": round(item["forward_2h_pct"], 4),
                    "forward_drawdown_2h_pct": round(item["forward_drawdown_2h_pct"], 4),
                    "universe_size": len(members),
                    "selection_method": "random_top2",
                    "learning_recommendation": "NO_ACTION",
                })

    results: list[UniverseResult] = []
    for uid, name in UNIVERSE_NAMES.items():
        rets = per_scan_returns[uid]
        if not rets:
            results.append(UniverseResult(
                universe_id=uid, universe_name=name, scan_count=0,
                avg_return=0, vs_random=-REFERENCE_RANDOM_BASELINE, win_rate=0,
                top10_pct=0, median=0, max_ret=0, min_ret=0, variance=0,
                extreme_winner_ratio=0, top5pct_ratio=0, avg_drawdown=0,
                recommended="no",
            ))
            continue

        pool = all_member_returns[uid]
        p95 = percentile(pool, 95) if pool else 0
        top5_count = sum(1 for v in pool if v >= p95) if pool else 0
        top5_ratio = top5_count / len(pool) if pool else 0

        avg = statistics.mean(rets)
        recommended = "yes" if avg > REFERENCE_RANDOM_BASELINE and sum(1 for r in rets if r > 0) / len(rets) >= 0.5 else "no"

        results.append(UniverseResult(
            universe_id=uid,
            universe_name=name,
            scan_count=len(rets),
            avg_return=round(avg, 4),
            vs_random=round(avg - REFERENCE_RANDOM_BASELINE, 4),
            win_rate=round(sum(1 for r in rets if r > 0) / len(rets), 4),
            top10_pct=round(percentile(rets, 90), 4),
            median=round(statistics.median(rets), 4),
            max_ret=round(max(rets), 4),
            min_ret=round(min(rets), 4),
            variance=round(statistics.pvariance(rets), 4) if len(rets) > 1 else 0.0,
            extreme_winner_ratio=round(statistics.mean(extreme_flags[uid]), 4) if extreme_flags[uid] else 0,
            top5pct_ratio=round(top5_ratio, 4),
            avg_drawdown=round(statistics.mean(per_scan_drawdowns[uid]), 4) if per_scan_drawdowns[uid] else 0,
            recommended=recommended,
        ))

    results.sort(key=lambda r: r.avg_return, reverse=True)
    return results, scan_detail_rows, per_scan_returns


def write_outputs(results: list[UniverseResult], scan_detail_rows: list[dict]) -> None:
    ranking_rows = [{
        "rank": i + 1,
        "universe_id": r.universe_id,
        "universe_name": r.universe_name,
        "avg_return_pct": r.avg_return,
        "vs_reference_random_pct": r.vs_random,
        "win_rate": r.win_rate,
        "top10pct_scan_return": r.top10_pct,
        "median_scan_return": r.median,
        "max_scan_return": r.max_ret,
        "min_scan_return": r.min_ret,
        "variance": r.variance,
        "extreme_winner_ratio": r.extreme_winner_ratio,
        "top5pct_member_ratio": r.top5pct_ratio,
        "avg_drawdown_pct": r.avg_drawdown,
        "scan_count": r.scan_count,
        "recommended": r.recommended,
        "reference_random_baseline_pct": REFERENCE_RANDOM_BASELINE,
        "learning_recommendation": "NO_ACTION",
    } for i, r in enumerate(results)]

    write_csv(OUT_DIR / "universe_ranking_table.csv", ranking_rows)
    write_csv(OUT_DIR / "universe_scan_details.csv", scan_detail_rows)

    analysis_rows = [{
        "universe_id": r.universe_id,
        "universe_name": r.universe_name,
        "mean_return": r.avg_return,
        "variance": r.variance,
        "extreme_winner_ratio": r.extreme_winner_ratio,
        "top5pct_member_ratio": r.top5pct_ratio,
        "avg_drawdown": r.avg_drawdown,
        "learning_recommendation": "NO_ACTION",
    } for r in results]
    write_csv(OUT_DIR / "universe_extended_analysis.csv", analysis_rows)

    lines = [
        "===== SCOUT SEASON2.1 — UNIVERSE BLIND TEST =====",
        "",
        f"Period: {START_KST.strftime('%Y-%m-%d %H:%M')} ~ {END_KST.strftime('%Y-%m-%d %H:%M')} KST",
        "Method: Random Top2 within each universe (no feature scoring)",
        f"Reference Random baseline (prior top10 universe): {REFERENCE_RANDOM_BASELINE}%",
        "",
        "=== Universe Ranking ===",
        "",
    ]
    for i, r in enumerate(results, 1):
        lines.append(
            f"{i}. {r.universe_name} | avg={r.avg_return}% vs_ref={r.vs_random:+.4f}% "
            f"win={r.win_rate:.1%} scans={r.scan_count} recommend={r.recommended}"
        )

    passed = [r for r in results if r.recommended == "yes"]
    lines.extend([
        "",
        f"Universes beating reference Random ({REFERENCE_RANDOM_BASELINE}%): {len(passed)}",
        "",
        "=== Conclusion ===",
    ])
    if passed:
        best = passed[0]
        lines.append(
            f"Best universe: {best.universe_name} ({best.avg_return}% avg Top2 random return)"
        )
        lines.append("Feature research may resume ONLY inside this universe.")
    else:
        lines.append("NO universe beat reference Random. All universes rejected.")
        lines.append("Do NOT resume feature research until universe or data quality improves.")

    lines.extend([
        "",
        "=== Data notes ===",
        "- Snapshots cached under logs/universe_research/snapshots/",
        "- forward_2h from first 2h candle close after scan (live klines)",
        "- Jun 1-5 scans run if API available; cache speeds reruns",
        "",
        "Learning recommendation: NO_ACTION",
    ])
    (OUT_DIR / "universe_research_report.txt").write_text("\n".join(lines), encoding="utf-8")


def print_summary(results: list[UniverseResult]) -> None:
    print("")
    print("###############################################################")
    print("Universe Blind Test Summary (Season2.1)")
    print("###############################################################")
    print(f"Reference Random baseline: {REFERENCE_RANDOM_BASELINE}%")
    print("")
    print(f"{'Rank':<5} {'Universe':<40} {'AvgRet%':>8} {'vsRef':>8} {'WinRate':>8} {'Scans':>6} {'Rec':>4}")
    print("-" * 85)
    for i, r in enumerate(results, 1):
        print(
            f"{i:<5} {r.universe_name:<40} {r.avg_return:>8.4f} {r.vs_random:>+8.4f} "
            f"{r.win_rate:>8.1%} {r.scan_count:>6} {r.recommended:>4}"
        )
    passed = [r for r in results if r.recommended == "yes"]
    print("")
    if passed:
        print(f"Recommended: {passed[0].universe_name} ({passed[0].avg_return}%)")
    else:
        print("Recommended: NONE - all universes below reference Random")
    print(f"Outputs: {OUT_DIR}")
    print("###############################################################")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scout Season2.1 Universe Blind Test")
    parser.add_argument("--max-scans", type=int, default=None, help="Limit scans for testing")
    parser.add_argument("--refresh-cache", action="store_true", help="Rebuild snapshot cache")
    parser.add_argument("--workers", type=int, default=8, help="Parallel API workers")
    parser.add_argument("--start-date", default="2026-06-06", help="First scan date (KST)")
    parser.add_argument("--end-date", default="2026-06-15", help="Last scan date (KST)")
    parser.add_argument("--cache-only", action="store_true", help="Analyze cached snapshots only")
    args = parser.parse_args()

    results, details, _ = run_universe_blind_test(
        args.max_scans,
        args.refresh_cache,
        args.workers,
        args.start_date,
        args.end_date,
        args.cache_only,
    )
    write_outputs(results, details)
    print_summary(results)


if __name__ == "__main__":
    main()
