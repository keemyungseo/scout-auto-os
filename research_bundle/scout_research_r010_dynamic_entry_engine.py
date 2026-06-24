"""
Scout Research R010 — Dynamic Entry Engine

A6 frozen candidates. Compare entry methods A–E with unified R009-B exit.
Blind forward-only. No formula/re-rank changes.

Usage:
  python scout_research_r010_dynamic_entry_engine.py
  python scout_research_r010_dynamic_entry_engine.py --tier top5
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import scout_phase19_winner_ranking_dna as p19
from season2_p37_scout_decision_hierarchy import write_csv

import scout_phase16_human_blind_test as p16
from scout_research_r005_execution_statistics import safe_print
from scout_research_r006_pilot_execution_engine import Bar, load_forward_bars, unique_top_picks
from scout_research_r008_exit_engine import state_snapshot
from scout_research_r009_dynamic_exit_engine import efr_from_bar, load_trades, MAX_HOLD_BARS
from season2_scout_mission import mission_summary_lines

OUT_DIR = Path("logs") / "research_r010_dynamic_entry"

ENTRY_A = [(0, 1.0)]
ENTRY_B = [(0, 0.5), (1, 0.5)]
ENTRY_C = [(0, 0.3), (1, 0.3), (2, 0.4)]
ENTRY_D = [(0, 0.2), (1, 0.4), (2, 0.4)]


@dataclass
class EntryResult:
    method: str
    symbol: str
    search_time: str
    avg_entry_price: float
    immediate_price: float
    return_pct: float
    final_return_pct: float
    max_return_pct: float
    mae_pct: float
    mfe_pct: float
    hold_minutes: int
    exit_reason: str
    entry_efficiency: float
    entry_capture: float
    cost_improvement_pct: float
    opportunity_loss_pct: float
    upside_remaining_pct: float
    fills: list[tuple[int, float]] = field(default_factory=list)
    a6_score: float = 0.0


def mfe_mae_from(bars: list[Bar], start_i: int, ref_px: float) -> tuple[float, float]:
    if ref_px <= 0 or start_i >= len(bars):
        return 0.0, 0.0
    max_h, min_l = ref_px, ref_px
    for b in bars[start_i:]:
        max_h = max(max_h, b.h)
        min_l = min(min_l, b.l)
    return (
        (max_h - ref_px) / ref_px * 100,
        (ref_px - min_l) / ref_px * 100,
    )


def build_dynamic_fills(bars: list[Bar], meta: dict) -> list[tuple[int, float]]:
    """Entry E: 30% initial, add on improving EFR + state (max 100% within 30m)."""
    fills: list[tuple[int, float]] = [(0, 0.3)]
    alloc = 0.3
    prev_efr = efr_from_bar(bars, 0, 12)
    for j in range(1, min(7, len(bars))):
        snap = state_snapshot(bars, j, 0)
        efr = efr_from_bar(bars, j, 12)
        improved = efr > prev_efr * 1.05
        state_ok = snap.trend_alive and (snap.acceleration or not snap.volume_weak)
        if improved and state_ok and alloc < 1.0:
            add = min(0.15, 1.0 - alloc)
            if add > 0.01:
                fills.append((j, add))
                alloc += add
        if not snap.trend_alive and snap.momentum_weak:
            break
        prev_efr = efr
    return fills


def find_exit_bar(bars: list[Bar], start_i: int, avg_entry: float) -> tuple[int, float, str]:
    """R009 Track B exit logic from start_i using avg_entry reference."""
    end_i = min(start_i + MAX_HOLD_BARS - 1, len(bars) - 1)
    sl_px = avg_entry * 0.90
    for j in range(start_i + 1, end_i + 1):
        b = bars[j]
        current_ret = (b.c - avg_entry) / avg_entry * 100
        e30 = efr_from_bar(bars, j, 6)
        e60 = efr_from_bar(bars, j, 12)
        snap = state_snapshot(bars, j, 0)
        if b.l <= sl_px:
            return j, -10.0, "protective_sl"
        if current_ret > 0.5 and e30 < current_ret and e60 < current_ret:
            return j, current_ret, "efr_exit"
        if current_ret > 1.0 and not snap.trend_alive and snap.momentum_weak:
            return j, current_ret, "state_exhaustion"
    ret = (bars[end_i].c - avg_entry) / avg_entry * 100
    return end_i, ret, "max_hold"


def simulate_entry(
    bars: list[Bar],
    meta: dict,
    method: str,
    fills: list[tuple[int, float]],
) -> EntryResult | None:
    if not bars or not fills:
        return None
    immediate_px = bars[0].o
    if immediate_px <= 0:
        return None

    total_w = sum(w for _, w in fills)
    if abs(total_w - 1.0) > 0.01 and total_w < 0.99:
        fills = [(i, w / total_w) for i, w in fills]

    cost = 0.0
    alloc = 0.0
    last_fill_i = 0
    for bar_i, w in sorted(fills, key=lambda x: x[0]):
        if bar_i >= len(bars):
            continue
        cost += w * bars[bar_i].o
        alloc += w
        last_fill_i = max(last_fill_i, bar_i)

    if alloc <= 0:
        return None
    avg_entry = cost / alloc
    exit_i, ret, reason = find_exit_bar(bars, last_fill_i, avg_entry)
    mfe, mae = mfe_mae_from(bars, 0, immediate_px)
    mfe_avg, mae_avg = mfe_mae_from(bars, last_fill_i, avg_entry)
    final_ret = (bars[min(len(bars) - 1, MAX_HOLD_BARS - 1)].c - avg_entry) / avg_entry * 100

    imm_mfe, _ = mfe_mae_from(bars, 0, immediate_px)
    eff = min(100.0, ret / imm_mfe * 100) if imm_mfe > 0 and ret > 0 else 0.0
    capture = min(100.0, ret / mfe_avg * 100) if mfe_avg > 0 and ret > 0 else 0.0
    cost_imp = (immediate_px - avg_entry) / immediate_px * 100 if immediate_px else 0.0
    opp_loss = max(0.0, imm_mfe - ret)
    upside_rem = max(0.0, mfe_avg - ret)

    return EntryResult(
        method=method,
        symbol=meta["symbol"],
        search_time=meta["search_time"],
        avg_entry_price=round(avg_entry, 8),
        immediate_price=round(immediate_px, 8),
        return_pct=round(ret, 4),
        final_return_pct=round(final_ret, 4),
        max_return_pct=round(mfe_avg, 4),
        mae_pct=round(mae_avg, 4),
        mfe_pct=round(mfe_avg, 4),
        hold_minutes=(exit_i - 0) * 5,
        exit_reason=reason,
        entry_efficiency=round(eff, 4),
        entry_capture=round(capture, 4),
        cost_improvement_pct=round(cost_imp, 4),
        opportunity_loss_pct=round(opp_loss, 4),
        upside_remaining_pct=round(upside_rem, 4),
        fills=fills,
        a6_score=meta.get("a6_score", 0),
    )


def method_stats(results: list[EntryResult]) -> dict:
    if not results:
        return {}
    rets = [r.return_pct for r in results]
    wins = [x for x in rets if x > 0]
    losses = [x for x in rets if x <= 0]
    gw, gl = sum(wins), abs(sum(losses))
    pf = gw / gl if gl > 0 else 99.0
    eq, peak, mdd = 100.0, 100.0, 0.0
    for r in rets:
        eq *= 1 + r / 100
        peak = max(peak, eq)
        mdd = max(mdd, (peak - eq) / peak * 100)
    daily: dict[str, float] = defaultdict(float)
    for r in results:
        daily[r.search_time[:10]] += r.return_pct
    daily_vals = list(daily.values())
    std = statistics.stdev(rets) if len(rets) > 1 else 0.0
    return {
        "n": len(results),
        "avg_entry_price": round(statistics.mean([r.avg_entry_price for r in results]), 8),
        "avg_max_return": round(statistics.mean([r.max_return_pct for r in results]), 4),
        "avg_final_return": round(statistics.mean([r.final_return_pct for r in results]), 4),
        "avg_mae": round(statistics.mean([r.mae_pct for r in results]), 4),
        "avg_mfe": round(statistics.mean([r.mfe_pct for r in results]), 4),
        "ev": round(statistics.mean(rets), 4),
        "win_rate": round(len(wins) / len(rets) * 100, 2),
        "profit_factor": round(pf, 4),
        "max_dd": round(mdd, 4),
        "avg_hold_min": round(statistics.mean([r.hold_minutes for r in results]), 1),
        "avg_entry_efficiency": round(statistics.mean([r.entry_efficiency for r in results]), 4),
        "avg_entry_capture": round(statistics.mean([r.entry_capture for r in results]), 4),
        "avg_cost_improvement": round(statistics.mean([r.cost_improvement_pct for r in results]), 4),
        "avg_opportunity_loss": round(statistics.mean([r.opportunity_loss_pct for r in results]), 4),
        "avg_upside_remaining": round(statistics.mean([r.upside_remaining_pct for r in results]), 4),
        "total_daily_return": round(sum(daily_vals), 4),
        "avg_daily_return": round(statistics.mean(daily_vals), 4) if daily_vals else 0,
        "sharpe_proxy": round(statistics.mean(rets) / std, 4) if std > 0 else 0,
    }


def special_research(
    imm: list[EntryResult],
    delayed: list[EntryResult],
    method_delayed: str,
) -> dict:
    wait_better = []
    imm_mandatory = []
    delay_loses = []
    by_key = {(r.symbol, r.search_time): r for r in delayed}

    for a in imm:
        key = (a.symbol, a.search_time)
        d = by_key.get(key)
        if not d:
            continue
        if d.cost_improvement_pct > 0.1 and d.return_pct > a.return_pct:
            wait_better.append({
                "symbol": a.symbol, "search_time": a.search_time,
                "imm_ret": a.return_pct, "delayed_ret": d.return_pct,
                "cost_improvement": d.cost_improvement_pct,
            })
        if a.mfe_pct > 5 and d.opportunity_loss_pct > 2 and a.return_pct > d.return_pct + 1:
            imm_mandatory.append({
                "symbol": a.symbol, "search_time": a.search_time,
                "imm_ret": a.return_pct, "delayed_ret": d.return_pct,
                "opp_loss": d.opportunity_loss_pct,
            })
        if d.opportunity_loss_pct > 1.0:
            delay_loses.append({
                "symbol": a.symbol, "search_time": a.search_time,
                "opp_loss": d.opportunity_loss_pct,
            })

    return {
        "waiting_improved_n": len(wait_better),
        "immediate_mandatory_n": len(imm_mandatory),
        "delay_opportunity_loss_n": len(delay_loses),
        "delay_opportunity_loss_rate": round(len(delay_loses) / max(1, len(imm)) * 100, 2),
        "waiting_improved": wait_better[:20],
        "immediate_mandatory": imm_mandatory[:20],
        "delay_opportunity_loss": delay_loses[:20],
    }


def result_row(r: EntryResult) -> dict:
    return {
        "method": r.method,
        "symbol": r.symbol,
        "search_time": r.search_time,
        "avg_entry_price": r.avg_entry_price,
        "immediate_price": r.immediate_price,
        "return_pct": r.return_pct,
        "final_return_pct": r.final_return_pct,
        "max_return_pct": r.max_return_pct,
        "mae_pct": r.mae_pct,
        "mfe_pct": r.mfe_pct,
        "hold_minutes": r.hold_minutes,
        "exit_reason": r.exit_reason,
        "entry_efficiency": r.entry_efficiency,
        "entry_capture": r.entry_capture,
        "cost_improvement_pct": r.cost_improvement_pct,
        "opportunity_loss_pct": r.opportunity_loss_pct,
        "upside_remaining_pct": r.upside_remaining_pct,
        "fills": str(r.fills),
        "a6_score": r.a6_score,
    }


def recommend_entry(stats_by_method: dict[str, dict]) -> dict:
    scored = []
    for method, st in stats_by_method.items():
        score = (
            st.get("avg_daily_return", 0) * 0.45
            + st.get("ev", 0) * 0.30
            + st.get("avg_entry_efficiency", 0) * 0.10
            - st.get("avg_opportunity_loss", 0) * 0.15
        )
        scored.append((score, method, st))
    scored.sort(reverse=True)
    _, best_method, best = scored[0]
    fills_spec = {
        "A_immediate_100": {"fills": [{"delay_min": 0, "weight": 1.0}]},
        "B_50_50": {"fills": [{"delay_min": 0, "weight": 0.5}, {"delay_min": 5, "weight": 0.5}]},
        "C_30_30_40": {"fills": [
            {"delay_min": 0, "weight": 0.3}, {"delay_min": 5, "weight": 0.3}, {"delay_min": 10, "weight": 0.4},
        ]},
        "D_20_40_40": {"fills": [
            {"delay_min": 0, "weight": 0.2}, {"delay_min": 5, "weight": 0.4}, {"delay_min": 10, "weight": 0.4},
        ]},
        "E_dynamic_state": {
            "initial_weight": 0.3, "add_rule": "EFR+state improve", "max_window_min": 30,
        },
    }
    return {
        "version": "r010_entry_v1",
        "search_formula": "A6_frozen",
        "exit_engine": "R009_track_B_efr_state",
        "recommended_method": best_method,
        "allocation": fills_spec.get(best_method, {}),
        "expected_ev": best.get("ev"),
        "win_rate": best.get("win_rate"),
        "profit_factor": best.get("profit_factor"),
        "avg_daily_return": best.get("avg_daily_return"),
        "avg_entry_efficiency": best.get("avg_entry_efficiency"),
        "avg_cost_improvement": best.get("avg_cost_improvement"),
        "avg_opportunity_loss": best.get("avg_opportunity_loss"),
        "max_dd": best.get("max_dd"),
        "compatible_with": ["R011_sizing", "R012_slots", "R013_portfolio", "R014_reverse", "R015_review"],
    }


def run(tier: str = "top5") -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p19.CACHE_DIR = Path("logs") / "phase19_winner_dna" / "kline_cache"
    p16.CACHE_DIR = p19.CACHE_DIR

    safe_print(f"R010 loading {tier} trades...")
    trades = load_trades(tier)
    n = len(trades)
    safe_print(f"R010 {n} trades | entry methods A–E...")

    methods = {
        "A_immediate_100": ENTRY_A,
        "B_50_50": ENTRY_B,
        "C_30_30_40": ENTRY_C,
        "D_20_40_40": ENTRY_D,
    }
    all_results: dict[str, list[EntryResult]] = {k: [] for k in methods}
    all_results["E_dynamic_state"] = []

    for meta, bars in trades:
        for name, fills in methods.items():
            r = simulate_entry(bars, meta, name, fills)
            if r:
                all_results[name].append(r)
        dyn_fills = build_dynamic_fills(bars, meta)
        r = simulate_entry(bars, meta, "E_dynamic_state", dyn_fills)
        if r:
            all_results["E_dynamic_state"].append(r)

    stats = {m: method_stats(rs) for m, rs in all_results.items()}
    special_b = special_research(all_results["A_immediate_100"], all_results["B_50_50"], "B")
    special_c = special_research(all_results["A_immediate_100"], all_results["C_30_30_40"], "C")
    engine = recommend_entry(stats)

    (OUT_DIR / "recommended_entry_engine.json").write_text(json.dumps(engine, indent=2), encoding="utf-8")

    lines = [
        "############################################################",
        "SCOUT RESEARCH R010 — DYNAMIC ENTRY ENGINE",
        "############################################################",
        "",
        f"A6 frozen | tier={tier} | trades={n} | exit=R009-B unified",
        "",
        "=" * 62,
        "REPORT 1 — Entry Method Comparison",
        "=" * 62,
    ]
    for m, st in sorted(stats.items(), key=lambda x: x[1].get("avg_daily_return", 0), reverse=True):
        lines.append(
            f"  {m}: ev={st.get('ev')}% wr={st.get('win_rate')}% pf={st.get('profit_factor')} "
            f"daily={st.get('avg_daily_return')}% eff={st.get('avg_entry_efficiency')}% "
            f"cost_imp={st.get('avg_cost_improvement')}% opp_loss={st.get('avg_opportunity_loss')}%"
        )

    lines.extend(["", "=" * 62, "REPORT 2 — Split Entry Statistics", "=" * 62])
    for m in ("B_50_50", "C_30_30_40", "D_20_40_40"):
        st = stats[m]
        lines.append(
            f"  {m}: avg_entry={st.get('avg_entry_price')} cost_imp={st.get('avg_cost_improvement')}% "
            f"ev={st.get('ev')}% capture={st.get('avg_entry_capture')}%"
        )

    lines.extend(["", "=" * 62, "REPORT 3 — Dynamic Entry (E)", "=" * 62])
    st_e = stats["E_dynamic_state"]
    lines.append(
        f"  E_dynamic: ev={st_e.get('ev')}% wr={st_e.get('win_rate')}% "
        f"eff={st_e.get('avg_entry_efficiency')}% hold={st_e.get('avg_hold_min')}m"
    )

    lines.extend(["", "=" * 62, "SPECIAL RESEARCH", "=" * 62])
    lines.append(f"  Waiting improved (vs A): B={special_b['waiting_improved_n']} C={special_c['waiting_improved_n']}")
    lines.append(f"  Immediate mandatory: B={special_b['immediate_mandatory_n']} C={special_c['immediate_mandatory_n']}")
    lines.append(f"  Delay opportunity loss rate (B): {special_b['delay_opportunity_loss_rate']}%")

    best_ex = sorted(all_results[engine["recommended_method"]], key=lambda x: x.return_pct, reverse=True)[:10]
    worst_ex = sorted(all_results[engine["recommended_method"]], key=lambda x: x.return_pct)[:10]

    lines.extend(["", "=" * 62, "REPORT 5 — Recommended Entry Engine", "=" * 62])
    lines.append(json.dumps(engine, indent=2))

    report = "\n".join(lines) + "\n\n" + "\n".join(mission_summary_lines())
    (OUT_DIR / "research_r010_report.txt").write_text(report, encoding="utf-8")

    write_csv(OUT_DIR / "report_01_method_comparison.csv", [{"method": m, **st} for m, st in stats.items()])
    write_csv(OUT_DIR / "report_02_split_entry.csv", [
        {"method": m, **stats[m]} for m in ("B_50_50", "C_30_30_40", "D_20_40_40", "E_dynamic_state")
    ])
    for m, rs in all_results.items():
        write_csv(OUT_DIR / f"trades_{m}.csv", [result_row(r) for r in rs])
    write_csv(OUT_DIR / "report_04_best_cases.csv", [result_row(r) for r in best_ex])
    write_csv(OUT_DIR / "report_04_worst_cases.csv", [result_row(r) for r in worst_ex])
    write_csv(OUT_DIR / "special_waiting_improved.csv", special_b["waiting_improved"] + special_c["waiting_improved"])
    write_csv(OUT_DIR / "special_immediate_mandatory.csv", special_b["immediate_mandatory"])
    write_csv(OUT_DIR / "special_delay_opportunity_loss.csv", special_b["delay_opportunity_loss"])

    safe_print(report)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="top5", choices=("top1", "top2", "top5", "top7"))
    args = ap.parse_args()
    run(tier=args.tier)


if __name__ == "__main__":
    main()
