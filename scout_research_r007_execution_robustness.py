"""
Scout Research R007 — Execution Robustness Research

A6 frozen Top1 picks. Plateau / robust execution rules (not peak EV).
Monte Carlo + capital simulation. No formula changes.

Usage:
  python scout_research_r007_execution_robustness.py
  python scout_research_r007_execution_robustness.py --mc-runs 500
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import scout_phase19_winner_ranking_dna as p19
from season2_p37_scout_decision_hierarchy import write_csv

import scout_phase16_human_blind_test as p16
from scout_research_r005_execution_statistics import load_b001_scan_row, loo_a6_scan_rows, parse_kst, safe_print
from scout_research_r006_pilot_execution_engine import Bar, load_forward_bars, simulate
from season2_scout_mission import mission_summary_lines

OUT_DIR = Path("logs") / "research_r007_execution_robustness"

TP_GRID = (10, 12, 14, 15, 16, 18, 20)
SL_GRID = (6, 8, 10, 12)
HOLD_GRID = (30, 45, 60, 75, 90, 105, 120)
MC_TRADE_COUNTS = (1000, 5000, 10000)
MC_RUNS_DEFAULT = 500
CAPITAL_LEVELS = (100, 1000, 10000)
RUIN_THRESHOLD = 0.50
TRADES_PER_YEAR = 4380


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


def load_top1_trades() -> list[tuple[dict, list[Bar]]]:
    scan_rows = loo_a6_scan_rows()
    b001 = load_b001_scan_row()
    if b001:
        scan_rows.append(b001)
    out: list[tuple[dict, list[Bar]]] = []
    for sr in sorted(scan_rows, key=lambda x: x["scan_kst"]):
        if not sr.get("top2"):
            continue
        r = sr["top2"][0]
        bars = load_forward_bars(r["symbol"], sr["scan_kst"])
        if not bars:
            continue
        out.append(({
            "search_time": sr["scan_kst"],
            "symbol": r["symbol"],
            "a6_score": round(r["a6"], 4),
            "month": parse_kst(sr["scan_kst"]).strftime("%Y-%m"),
        }, bars))
    return out


def annualized_log_return(returns: list[float], trades_per_year: int = TRADES_PER_YEAR) -> float:
    if not returns:
        return 0.0
    logs = [math.log(1 + r / 100) for r in returns if r > -99]
    if not logs:
        return 0.0
    return (math.exp(statistics.mean(logs) * trades_per_year) - 1) * 100


def hold_bars(mins: int) -> int:
    return max(1, mins // 5)


def longest_losing_streak(returns: list[float]) -> int:
    best = cur = 0
    for r in returns:
        if r <= 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def equity_curve(returns: list[float], compound: bool = True, initial: float = 100.0) -> list[float]:
    eq = initial
    curve = [eq]
    for r in returns:
        if compound:
            eq *= 1 + r / 100
        else:
            eq += initial * r / 100
        curve.append(eq)
    return curve


def max_drawdown_from_curve(curve: list[float]) -> float:
    peak = curve[0]
    mdd = 0.0
    for v in curve:
        peak = max(peak, v)
        mdd = max(mdd, (peak - v) / peak * 100 if peak > 0 else 0)
    return mdd


def sharpe_proxy(returns: list[float]) -> float:
    if len(returns) < 2:
        return 0.0
    std = statistics.stdev(returns)
    return statistics.mean(returns) / std if std > 0 else 0.0


def sortino_proxy(returns: list[float]) -> float:
    downs = [r for r in returns if r < 0]
    if not downs:
        return sharpe_proxy(returns)
    ds = statistics.stdev(downs) if len(downs) > 1 else abs(downs[0])
    return statistics.mean(returns) / ds if ds > 0 else 0.0


def rule_metrics(returns: list[float], compound: bool = True) -> dict:
    if not returns:
        return {}
    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r <= 0]
    ev = statistics.mean(returns)
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    pf = gross_win / gross_loss if gross_loss > 0 else 99.0
    curve = equity_curve(returns, compound=True)
    return {
        "trade_count": len(returns),
        "win_rate": round(len(wins) / len(returns) * 100, 2),
        "loss_rate": round(len(losses) / len(returns) * 100, 2),
        "ev": round(ev, 4),
        "profit_factor": round(pf, 4),
        "avg_return": round(ev, 4),
        "median_return": round(statistics.median(returns), 4),
        "sharpe_proxy": round(sharpe_proxy(returns), 4),
        "sortino_proxy": round(sortino_proxy(returns), 4),
        "max_dd": round(max_drawdown_from_curve(curve), 4),
        "longest_losing_streak": longest_losing_streak(returns),
        "return_std": round(statistics.stdev(returns), 4) if len(returns) > 1 else 0.0,
    }


def grid_index(grid: tuple, value) -> int:
    return grid.index(value)


def neighbor_coords(tp: int, sl: int, hold: int) -> list[tuple[int, int, int]]:
    ti = grid_index(TP_GRID, tp)
    si = grid_index(SL_GRID, sl)
    hi = grid_index(HOLD_GRID, hold)
    out: list[tuple[int, int, int]] = []
    for dt in (-1, 0, 1):
        for ds in (-1, 0, 1):
            for dh in (-1, 0, 1):
                if dt == ds == dh == 0:
                    continue
                nt, ns, nh = ti + dt, si + ds, hi + dh
                if 0 <= nt < len(TP_GRID) and 0 <= ns < len(SL_GRID) and 0 <= nh < len(HOLD_GRID):
                    out.append((TP_GRID[nt], SL_GRID[ns], HOLD_GRID[nh]))
    return out


def robust_score(tp: int, sl: int, hold: int, ev_map: dict[tuple[int, int, int], float], self_ev: float, pf: float, max_dd: float) -> dict:
    neighbors = neighbor_coords(tp, sl, hold)
    neighbor_evs = [ev_map[n] for n in neighbors if n in ev_map]
    if not neighbor_evs:
        neighbor_evs = [self_ev]
    mean_n = statistics.mean(neighbor_evs)
    std_n = statistics.stdev(neighbor_evs) if len(neighbor_evs) > 1 else 0.0
    min_n = min(neighbor_evs)
    peak_gap = max(0.0, self_ev - mean_n)
    peak_only = self_ev >= max(neighbor_evs) and peak_gap > 0.4

    score = (
        mean_n * 0.35
        + min_n * 0.25
        - std_n * 0.40
        - peak_gap * 0.50
        - (1.5 if peak_only else 0.0)
        + min(pf, 5) * 0.15
        - max_dd * 0.03
    )
    return {
        "robust_score": round(score, 4),
        "neighbor_mean_ev": round(mean_n, 4),
        "neighbor_min_ev": round(min_n, 4),
        "neighbor_std_ev": round(std_n, 4),
        "peak_gap": round(peak_gap, 4),
        "peak_only_flag": int(peak_only),
    }


def build_grid(trades: list[tuple[dict, list[Bar]]]) -> tuple[dict[tuple[int, int, int], list[float]], list[dict]]:
    returns_map: dict[tuple[int, int, int], list[float]] = {}
    trade_meta: list[dict] = []
    total = len(TP_GRID) * len(SL_GRID) * len(HOLD_GRID)
    done = 0
    for tp in TP_GRID:
        for sl in SL_GRID:
            for hold in HOLD_GRID:
                key = (tp, sl, hold)
                hb = hold_bars(hold)
                rets: list[float] = []
                for meta, bars in trades:
                    r = simulate(bars, 0, tp_pct=tp, sl_pct=sl, hold_bars=hb)
                    if r:
                        rets.append(r.return_pct)
                returns_map[key] = rets
                done += 1
                if done % 40 == 0:
                    safe_print(f"  grid {done}/{total}")
    for meta, _ in trades:
        trade_meta.append(meta)
    return returns_map, trade_meta


def evaluate_grid(returns_map: dict[tuple[int, int, int], list[float]]) -> list[dict]:
    ev_map = {k: statistics.mean(v) for k, v in returns_map.items() if v}
    rows: list[dict] = []
    for (tp, sl, hold), rets in returns_map.items():
        m = rule_metrics(rets)
        if not m:
            continue
        rs = robust_score(tp, sl, hold, ev_map, m["ev"], m["profit_factor"], m["max_dd"])
        curve = equity_curve(rets, compound=True)
        rows.append({
            "tp": tp, "sl": sl, "hold_minutes": hold,
            **m, **rs,
            "final_equity_100": round(curve[-1], 4),
        })
    rows.sort(key=lambda x: (x["robust_score"], x["neighbor_min_ev"], x["ev"]), reverse=True)
    return rows


def monte_carlo(returns: list[float], n_trades: int, n_runs: int, initial: float = 100.0) -> dict:
    if not returns:
        return {}
    total_rets: list[float] = []
    total_additive: list[float] = []
    max_dds: list[float] = []
    ruins: list[int] = []
    streaks: list[int] = []
    ev_per_trade: list[float] = []

    for _ in range(n_runs):
        sample = [random.choice(returns) for _ in range(n_trades)]
        curve = equity_curve(sample, compound=True, initial=initial)
        total_rets.append((curve[-1] - initial) / initial * 100)
        total_additive.append(sum(sample))
        ev_per_trade.append(statistics.mean(sample))
        max_dds.append(max_drawdown_from_curve(curve))
        streaks.append(longest_losing_streak(sample))
        ruins.append(1 if min(curve) < initial * RUIN_THRESHOLD else 0)

    return {
        "n_trades": n_trades,
        "n_runs": n_runs,
        "median_ev_per_trade": round(statistics.median(ev_per_trade), 4),
        "median_total_return_compound_pct": round(statistics.median(total_rets), 4),
        "worst_5pct_total_return_compound": round(pctile(total_rets, 5), 4),
        "worst_1pct_total_return_compound": round(pctile(total_rets, 1), 4),
        "median_total_return_additive_pct": round(statistics.median(total_additive), 4),
        "worst_5pct_total_return_additive": round(pctile(total_additive, 5), 4),
        "median_max_dd": round(statistics.median(max_dds), 4),
        "worst_5pct_max_dd": round(pctile(max_dds, 95), 4),
        "prob_ruin": round(sum(ruins) / len(ruins) * 100, 4),
        "median_losing_streak": round(statistics.median(streaks), 1),
        "worst_losing_streak": max(streaks),
    }


def capital_simulation(returns: list[float], trade_meta: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for initial in CAPITAL_LEVELS:
        for compound in (True, False):
            curve = equity_curve(returns, compound=compound, initial=float(initial))
            rows.append({
                "initial_capital": initial,
                "compounding": compound,
                "final_equity": round(curve[-1], 4),
                "total_return_pct": round((curve[-1] - initial) / initial * 100, 4),
                "max_dd": round(max_drawdown_from_curve(curve), 4),
            })
    return rows


def monthly_returns(returns: list[float], trade_meta: list[dict]) -> list[dict]:
    by_month: dict[str, list[float]] = defaultdict(list)
    for meta, r in zip(trade_meta, returns):
        by_month[meta["month"]].append(r)
    rows: list[dict] = []
    for month in sorted(by_month.keys()):
        rets = by_month[month]
        additive = sum(rets)
        rows.append({
            "month": month,
            "trades": len(rets),
            "month_return_additive_pct": round(additive, 4),
            "month_return_avg_x_trades": round(statistics.mean(rets) * len(rets), 4),
            "avg_trade_ev": round(statistics.mean(rets), 4),
        })
    return rows


def histogram_buckets(monthly: list[dict]) -> list[dict]:
    vals = [r["month_return_additive_pct"] for r in monthly]
    if not vals:
        return []
    lo, hi = min(vals), max(vals)
    step = max(1.0, (hi - lo) / 10)
    buckets: dict[str, int] = defaultdict(int)
    for v in vals:
        b = int((v - lo) / step) if step > 0 else 0
        label = f"{lo + b * step:.1f}~{lo + (b + 1) * step:.1f}"
        buckets[label] += 1
    return [{"bucket": k, "count": v} for k, v in sorted(buckets.items())]


def drawdown_curve_series(returns: list[float], initial: float = 100.0) -> list[dict]:
    curve = equity_curve(returns, compound=True, initial=initial)
    peak = curve[0]
    rows: list[dict] = []
    for i, eq in enumerate(curve):
        peak = max(peak, eq)
        dd = (peak - eq) / peak * 100 if peak > 0 else 0
        rows.append({"trade_idx": i, "equity": round(eq, 4), "drawdown_pct": round(dd, 4)})
    return rows


def capital_curve_series(returns: list[float], initial: float = 100.0) -> list[dict]:
    curve = equity_curve(returns, compound=True, initial=initial)
    return [{"trade_idx": i, "equity": round(v, 4)} for i, v in enumerate(curve)]


def run(mc_runs: int = MC_RUNS_DEFAULT) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p19.CACHE_DIR = Path("logs") / "phase19_winner_dna" / "kline_cache"
    p16.CACHE_DIR = p19.CACHE_DIR
    random.seed(42)

    safe_print("R007 loading Top1 trade paths...")
    trades = load_top1_trades()
    n_trades = len(trades)
    n_scans = len({m["search_time"] for m, _ in trades})
    safe_print(f"R007 {n_trades} Top1 paths from {n_scans} scans")

    safe_print("R007 building grid (196 rules)...")
    returns_map, trade_meta = build_grid(trades)
    grid_rows = evaluate_grid(returns_map)
    top10 = grid_rows[:10]

    safe_print("R007 Monte Carlo on Top10 robust rules...")
    mc_rows: list[dict] = []
    for rank, rule in enumerate(top10, 1):
        key = (rule["tp"], rule["sl"], rule["hold_minutes"])
        rets = returns_map[key]
        for n in MC_TRADE_COUNTS:
            mc = monte_carlo(rets, n, mc_runs)
            mc_rows.append({
                "rank": rank,
                "tp": rule["tp"], "sl": rule["sl"], "hold_minutes": rule["hold_minutes"],
                **mc,
            })

    recommended = top10[0]
    rec_key = (recommended["tp"], recommended["sl"], recommended["hold_minutes"])
    rec_returns = returns_map[rec_key]
    cap_rows = capital_simulation(rec_returns, trade_meta)
    monthly = monthly_returns(rec_returns, trade_meta)
    hist = histogram_buckets(monthly)
    cap_curve = capital_curve_series(rec_returns, initial=100.0)
    dd_curve = drawdown_curve_series(rec_returns, initial=100.0)

    monthly_ev = statistics.mean([m["month_return_additive_pct"] for m in monthly]) if monthly else 0.0
    hist_trades_per_month = statistics.mean([m["trades"] for m in monthly]) if monthly else n_trades
    pilot_json = {
        "version": "r007_pilot_v1",
        "search_formula": "A6_frozen",
        "search_tier": "top1",
        "entry": "immediate",
        "tp": recommended["tp"],
        "sl": recommended["sl"],
        "hold": recommended["hold_minutes"],
        "ev": recommended["ev"],
        "pf": recommended["profit_factor"],
        "wr": recommended["win_rate"],
        "expected_month_return": round(monthly_ev, 4),
        "expected_month_return_additive": round(monthly_ev, 4),
        "avg_trades_per_month": round(hist_trades_per_month, 1),
        "max_dd": recommended["max_dd"],
        "robust_score": recommended["robust_score"],
        "neighbor_min_ev": recommended["neighbor_min_ev"],
        "peak_only_flag": recommended["peak_only_flag"],
        "sample_trades": n_trades,
        "sample_scans": n_scans,
    }
    (OUT_DIR / "recommended_pilot_rule.json").write_text(json.dumps(pilot_json, indent=2), encoding="utf-8")

    # Reference candidate from R006
    ref_key = (15, 10, 60)
    ref_row = next((r for r in grid_rows if (r["tp"], r["sl"], r["hold_minutes"]) == ref_key), None)

    lines = [
        "############################################################",
        "SCOUT RESEARCH R007 — EXECUTION ROBUSTNESS",
        "############################################################",
        "",
        f"A6 frozen Top1 | entry=immediate | trades={n_trades} | scans={n_scans}",
        f"Grid: TP{list(TP_GRID)} x SL{list(SL_GRID)} x Hold{list(HOLD_GRID)} = {len(grid_rows)} rules",
        "",
        "=" * 62,
        "REFERENCE (R006 candidate TP15/SL10/Hold60 on Top5)",
        "=" * 62,
    ]
    if ref_row:
        lines.append(
            f"  Top1 replay: EV={ref_row['ev']}% WR={ref_row['win_rate']}% PF={ref_row['profit_factor']} "
            f"robust={ref_row['robust_score']} rank=#{grid_rows.index(ref_row)+1}/{len(grid_rows)}"
        )
    else:
        lines.append("  (not in grid)")

    lines.extend(["", "=" * 62, "TOP10 ROBUST RULES", "=" * 62])
    for i, r in enumerate(top10, 1):
        lines.append(
            f"  #{i} TP{r['tp']} SL{r['sl']} Hold{r['hold_minutes']}m | "
            f"robust={r['robust_score']} ev={r['ev']}% wr={r['win_rate']}% pf={r['profit_factor']} "
            f"dd={r['max_dd']}% | nmin_ev={r['neighbor_min_ev']}% peak_only={r['peak_only_flag']}"
        )

    lines.extend(["", "=" * 62, "RECOMMENDED PILOT RULE", "=" * 62, json.dumps(pilot_json, indent=2)])

    lines.extend(["", "=" * 62, f"MONTE CARLO (#1 rule, {mc_runs} runs)", "=" * 62])
    for row in [r for r in mc_rows if r["rank"] == 1]:
        lines.append(
            f"  n={row['n_trades']}: ev/trade={row['median_ev_per_trade']}% "
            f"total_add={row['median_total_return_additive_pct']}% "
            f"worst5%_add={row['worst_5pct_total_return_additive']}% "
            f"ruin={row['prob_ruin']}% med_DD={row['median_max_dd']}%"
        )

    lines.extend(["", "=" * 62, "CAPITAL SIMULATION (#1 rule)", "=" * 62])
    for c in cap_rows:
        lines.append(
            f"  init={c['initial_capital']} compound={c['compounding']} "
            f"final={c['final_equity']} ret={c['total_return_pct']}% dd={c['max_dd']}%"
        )

    report = "\n".join(lines) + "\n\n" + "\n".join(mission_summary_lines())
    (OUT_DIR / "research_r007_report.txt").write_text(report, encoding="utf-8")

    write_csv(OUT_DIR / "grid_all_rules.csv", grid_rows)
    write_csv(OUT_DIR / "top10_robust_rules.csv", top10)
    write_csv(OUT_DIR / "monte_carlo_top10.csv", mc_rows)
    write_csv(OUT_DIR / "capital_simulation.csv", cap_rows)
    write_csv(OUT_DIR / "monthly_returns.csv", monthly)
    write_csv(OUT_DIR / "monthly_return_histogram.csv", hist)
    write_csv(OUT_DIR / "capital_curve.csv", cap_curve)
    write_csv(OUT_DIR / "drawdown_curve.csv", dd_curve)

    safe_print(report)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc-runs", type=int, default=MC_RUNS_DEFAULT)
    args = ap.parse_args()
    run(mc_runs=args.mc_runs)


if __name__ == "__main__":
    main()
