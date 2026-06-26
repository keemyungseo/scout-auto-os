"""
Scout Research R012 — Meta Rotation Engine

Fixed: A6 + Immediate Entry + R009-B Exit + 2 Slots + Top5 universe.
Track A: hold until exit then re-search.
Track B: 5m meta rotation vs CurrentRemainingEV / NewEntryEV.

Usage:
  python scout_research_r012_meta_rotation.py
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import scout_phase19_winner_ranking_dna as p19
from season2_p37_scout_decision_hierarchy import write_csv

import scout_phase16_human_blind_test as p16
from scout_research_r005_execution_statistics import parse_kst, safe_print
from scout_research_r008_exit_engine import state_snapshot
from scout_research_r009_dynamic_exit_engine import efr_from_bar
from scout_research_r011_position_engine import (
    CatalogTrade,
    OpenPos,
    build_catalog,
    partial_return,
    simulate_portfolio,
)
from season2_scout_mission import mission_summary_lines

OUT_DIR = Path("logs") / "research_r012_meta_rotation"
SLOTS = 2
TOP_N = 5
FEE_PCT = 0.05
SLIPPAGE_PCT = 0.10
THRESHOLDS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)


@dataclass
class SwitchEvent:
    at_time: str
    from_symbol: str
    to_symbol: str
    threshold: float
    benefit_est: float
    current_rem_ev: float
    new_entry_ev: float
    switch_cost: float
    old_ret_at_switch: float
    old_counterfactual: float
    new_actual_ret: float
    success: bool


def build_top5_catalog(full: dict[str, list[CatalogTrade]]) -> dict[str, list[CatalogTrade]]:
    return {k: v[:TOP_N] for k, v in full.items() if v}


def hist_prior(catalog: dict[str, list[CatalogTrade]], a6: float) -> float:
    rets = [t.return_pct for trades in catalog.values() for t in trades if abs(t.a6_score - a6) < 1.0]
    return statistics.mean(rets) if rets else 4.0


def peak_capture_so_far(trade: CatalogTrade, at_dt: datetime) -> float:
    ret = partial_return(trade, at_dt)
    if trade.return_pct <= 0 or ret <= 0:
        return 0.0
    return min(100.0, ret / trade.return_pct * 100)


def current_remaining_ev(trade: CatalogTrade, at_dt: datetime) -> float:
    mins = int((at_dt - trade.entry_dt).total_seconds() / 60)
    bar_i = min(len(trade.bars) - 1, max(0, mins // 5))
    snap = state_snapshot(trade.bars, bar_i, 0)
    efr = efr_from_bar(trade.bars, bar_i, 12)
    ret = partial_return(trade, at_dt)
    remaining = max(0.0, trade.return_pct - ret)
    pc = peak_capture_so_far(trade, at_dt)
    state_sc = (
        (2.0 if snap.trend_alive else 0.0)
        + (2.0 if snap.acceleration else 0.0)
        + (1.0 if not snap.volume_weak else 0.0)
        + (1.0 if not snap.momentum_weak else 0.0)
        + snap.vol_ratio * 0.5
    )
    return (
        0.22 * efr
        + 0.18 * trade.a6_score
        + 0.15 * state_sc
        + 0.15 * remaining
        + 0.10 * max(0.0, 10.0 - pc * 0.1)
        + 0.10 * max(0.0, ret)
        - 0.10 * max(0.0, -ret)
    )


def new_entry_ev(cand: CatalogTrade, prior: float, at_dt: datetime | None = None) -> float:
    bar_i = 0
    if at_dt and at_dt > cand.entry_dt:
        mins = int((at_dt - cand.entry_dt).total_seconds() / 60)
        bar_i = min(len(cand.bars) - 1, max(0, mins // 5))
    snap = state_snapshot(cand.bars, bar_i, 0)
    efr = efr_from_bar(cand.bars, bar_i, 12)
    state_sc = (
        (2.0 if snap.trend_alive else 0.0)
        + (2.0 if snap.acceleration else 0.0)
        + (1.0 if not snap.volume_weak else 0.0)
        + (1.0 if not snap.momentum_weak else 0.0)
    )
    return 0.32 * cand.a6_score + 0.28 * efr + 0.20 * state_sc + 0.20 * prior


def switch_cost(trade: CatalogTrade, at_dt: datetime) -> float:
    ret_now = partial_return(trade, at_dt)
    forgone = max(0.0, trade.return_pct - ret_now)
    opp = forgone * 0.4
    return FEE_PCT * 2 + SLIPPAGE_PCT + opp


def counterfactual_hold(trade: CatalogTrade, from_dt: datetime) -> float:
    return trade.return_pct - partial_return(trade, from_dt)


@dataclass
class LivePos:
    trade: CatalogTrade
    weight: float
    entry_dt: datetime
    scan_origin: str


def simulate_track_b(
    catalog: dict[str, list[CatalogTrade]],
    threshold: float,
) -> tuple[dict, list[SwitchEvent], list[dict]]:
    scans = sorted(catalog.keys())
    prior_map = {t.symbol + t.scan_kst: hist_prior(catalog, t.a6_score) for trades in catalog.values() for t in trades}
    slots: list[LivePos | None] = [None] * SLOTS
    daily_pnl: dict[str, float] = defaultdict(float)
    realized: list[float] = []
    hold_times: list[int] = []
    switches: list[SwitchEvent] = []
    hold_decisions = {"correct_hold": 0, "late_hold": 0, "wrong_switch": 0, "success_switch": 0}
    equity, peak_eq, mdd = 100.0, 100.0, 0.0
    switch_count = 0
    trade_count = 0

    def realize(ret: float, weight: float, day: str, fee: float = 0.0) -> None:
        nonlocal equity, peak_eq, mdd
        contrib = weight * 0.5 * ret / 100
        fc = fee * weight * 0.5
        daily_pnl[day] += (contrib - fc) * 100
        equity *= 1 + contrib - fc
        peak_eq = max(peak_eq, equity)
        mdd = max(mdd, (peak_eq - equity) / peak_eq * 100 if peak_eq else 0)
        realized.append(ret)

    def close_live(pos: LivePos, at_dt: datetime, partial: bool) -> float:
        nonlocal trade_count
        if partial:
            ret = partial_return(pos.trade, at_dt)
            hold_times.append(int((at_dt - pos.entry_dt).total_seconds() / 60))
        else:
            ret = pos.trade.return_pct
            hold_times.append(pos.trade.hold_minutes)
        realize(ret, pos.weight, at_dt.strftime("%Y-%m-%d"), FEE_PCT)
        trade_count += 1
        return ret

    def best_new_candidate(at_dt: datetime, held_syms: set[str], scan_kst: str) -> CatalogTrade | None:
        cands = [c for c in catalog.get(scan_kst, []) if c.symbol not in held_syms]
        if not cands:
            return None
        scored = []
        for c in cands:
            pr = prior_map.get(c.symbol + c.scan_kst, hist_prior(catalog, c.a6_score))
            scored.append((new_entry_ev(c, pr, at_dt), c))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]

    def rotation_check(at_dt: datetime, scan_kst: str) -> None:
        nonlocal switch_count
        held = {p.trade.symbol for p in slots if p}
        for si in range(SLOTS):
            pos = slots[si]
            if not pos:
                continue
            if at_dt >= pos.trade.exit_dt:
                continue
            new_c = best_new_candidate(at_dt, held, scan_kst)
            if not new_c or new_c.symbol == pos.trade.symbol:
                hold_decisions["correct_hold"] += 1
                continue
            cur_ev = current_remaining_ev(pos.trade, at_dt)
            pr = prior_map.get(new_c.symbol + new_c.scan_kst, hist_prior(catalog, new_c.a6_score))
            new_ev = new_entry_ev(new_c, pr, at_dt)
            cost = switch_cost(pos.trade, at_dt)
            benefit = new_ev - cur_ev - cost
            cf_hold = counterfactual_hold(pos.trade, at_dt)
            new_full = new_c.return_pct

            if benefit > threshold:
                old_ret = close_live(pos, at_dt, partial=True)
                switch_count += 1
                success = new_full > cf_hold
                if success:
                    hold_decisions["success_switch"] += 1
                else:
                    hold_decisions["wrong_switch"] += 1
                switches.append(SwitchEvent(
                    at_time=at_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    from_symbol=pos.trade.symbol,
                    to_symbol=new_c.symbol,
                    threshold=threshold,
                    benefit_est=round(benefit, 4),
                    current_rem_ev=round(cur_ev, 4),
                    new_entry_ev=round(new_ev, 4),
                    switch_cost=round(cost, 4),
                    old_ret_at_switch=round(old_ret, 4),
                    old_counterfactual=round(cf_hold, 4),
                    new_actual_ret=round(new_full, 4),
                    success=success,
                ))
                slots[si] = LivePos(trade=new_c, weight=0.5, entry_dt=at_dt, scan_origin=scan_kst)
                held.add(new_c.symbol)
            else:
                if new_full > cf_hold + threshold:
                    hold_decisions["late_hold"] += 1
                else:
                    hold_decisions["correct_hold"] += 1

    for scan_kst in scans:
        now = parse_kst(scan_kst)

        for si in range(SLOTS):
            pos = slots[si]
            if pos and pos.trade.exit_dt <= now:
                close_live(pos, pos.trade.exit_dt, partial=False)
                slots[si] = None

        held = {p.trade.symbol for p in slots if p}
        cands = [c for c in catalog[scan_kst] if c.symbol not in held]
        for c in cands:
            if all(s is not None for s in slots):
                break
            si = next(i for i in range(SLOTS) if slots[i] is None)
            slots[si] = LivePos(trade=c, weight=0.5, entry_dt=now, scan_origin=scan_kst)
            held.add(c.symbol)

        rotation_check(now, scan_kst)

        tick = now + timedelta(minutes=5)
        end = now + timedelta(hours=4)
        while tick < end:
            active = any(p and p.trade.exit_dt > tick for p in slots)
            if not active:
                break
            rotation_check(tick, scan_kst)
            tick += timedelta(minutes=5)

    for si in range(SLOTS):
        if slots[si]:
            close_live(slots[si], slots[si].trade.exit_dt, partial=False)

    daily_vals = list(daily_pnl.values())
    wins = [r for r in realized if r > 0]
    losses = [r for r in realized if r <= 0]
    gw, gl = sum(wins), abs(sum(losses))
    pf = gw / gl if gl > 0 else 99.0
    total_days = len({s[:10] for s in catalog.keys()}) or 1
    n_dec = sum(hold_decisions.values()) or 1

    stats = {
        "track": "B_meta_rotation",
        "threshold": threshold,
        "trade_count": trade_count,
        "switch_count": switch_count,
        "daily_return_mean": round(statistics.mean(daily_vals), 4) if daily_vals else 0,
        "total_daily_return": round(sum(daily_vals) / total_days, 4) if daily_vals else 0,
        "ev": round(statistics.mean(realized), 4) if realized else 0,
        "win_rate": round(len(wins) / len(realized) * 100, 2) if realized else 0,
        "profit_factor": round(pf, 4),
        "max_dd": round(mdd, 4),
        "avg_hold_min": round(statistics.mean(hold_times), 1) if hold_times else 0,
        "avg_trade_return": round(statistics.mean(realized), 4) if realized else 0,
        "switch_success_pct": round(hold_decisions["success_switch"] / max(1, switch_count) * 100, 2),
        "wrong_switch_pct": round(hold_decisions["wrong_switch"] / max(1, switch_count) * 100, 2),
        "correct_hold_pct": round(hold_decisions["correct_hold"] / n_dec * 100, 2),
        "late_hold_pct": round(hold_decisions["late_hold"] / n_dec * 100, 2),
    }
    return stats, switches, hold_decisions


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p19.CACHE_DIR = Path("logs") / "phase19_winner_dna" / "kline_cache"
    p16.CACHE_DIR = p19.CACHE_DIR

    safe_print("R012 building Top5 catalog...")
    catalog = build_top5_catalog(build_catalog())
    n_scans = len(catalog)
    safe_print(f"R012 {n_scans} scans | Track A...")

    track_a = simulate_portfolio(catalog, slots=SLOTS, replacement="keep", exposure=1.0)
    track_a["track"] = "A_hold_until_exit"
    track_a["switch_count"] = 0

    safe_print("R012 Track B — meta rotation thresholds...")
    track_b_rows: list[dict] = []
    all_switches: list[SwitchEvent] = []
    for thr in THRESHOLDS:
        st, sw, _ = simulate_track_b(catalog, thr)
        track_b_rows.append(st)
        all_switches.extend(sw)
        safe_print(f"  threshold={thr}% daily={st['total_daily_return']}% switches={st['switch_count']}")

    best_b = max(track_b_rows, key=lambda x: x["total_daily_return"])
    best_thr_switches = [s for s in all_switches if s.threshold == best_b["threshold"]]
    best_ex = sorted(best_thr_switches, key=lambda s: s.new_actual_ret - s.old_counterfactual, reverse=True)[:20]
    worst_ex = sorted(best_thr_switches, key=lambda s: s.new_actual_ret - s.old_counterfactual)[:20]

    engine = {
        "version": "r012_meta_rotation_v1",
        "search": "A6_frozen",
        "entry": "immediate_100pct_R010",
        "exit": "dynamic_efr_R009",
        "position": {"slots": SLOTS, "universe": "top5"},
        "track_a_daily_return": track_a["total_daily_return"],
        "recommended": {
            "mode": "meta_rotation" if best_b["total_daily_return"] > track_a["total_daily_return"] else "hold_until_exit",
            "threshold_pct": best_b["threshold"],
            "rotation_rule": "switch if NewEntryEV - CurrentRemainingEV - costs > threshold",
            "check_interval_min": 5,
            "fee_pct": FEE_PCT,
            "slippage_pct": SLIPPAGE_PCT,
        },
        "track_b_best": {k: best_b[k] for k in (
            "threshold", "total_daily_return", "switch_count", "switch_success_pct",
            "profit_factor", "max_dd", "win_rate",
        )},
        "track_a": {k: track_a[k] for k in (
            "total_daily_return", "profit_factor", "max_dd", "win_rate", "trade_count", "avg_hold_min",
        )},
        "compatible_with": ["R013_portfolio", "R014_reverse", "R015_review"],
    }
    (OUT_DIR / "recommended_meta_rotation_engine.json").write_text(json.dumps(engine, indent=2), encoding="utf-8")

    lines = [
        "############################################################",
        "SCOUT RESEARCH R012 — META ROTATION ENGINE",
        "############################################################",
        "",
        f"Fixed stack | {n_scans} scans | 2 slots | Top5 universe",
        "",
        "=" * 62,
        "REPORT 01 — Track A vs Track B (best threshold)",
        "=" * 62,
        f"  Track A: daily={track_a['total_daily_return']}% wr={track_a['win_rate']}% "
        f"pf={track_a['profit_factor']} mdd={track_a['max_dd']}% trades={track_a['trade_count']} "
        f"hold={track_a['avg_hold_min']}m switches=0",
        f"  Track B (thr={best_b['threshold']}%): daily={best_b['total_daily_return']}% "
        f"wr={best_b['win_rate']}% pf={best_b['profit_factor']} mdd={best_b['max_dd']}% "
        f"switches={best_b['switch_count']} hold={best_b['avg_hold_min']}m",
        "",
        "=" * 62,
        "REPORT 02 — Switch Analysis (best threshold)",
        "=" * 62,
        f"  Switch success: {best_b['switch_success_pct']}%",
        f"  Wrong switch: {best_b['wrong_switch_pct']}%",
        f"  Correct hold: {best_b['correct_hold_pct']}%",
        f"  Late hold: {best_b['late_hold_pct']}%",
        "",
        "=" * 62,
        "REPORT 03 — Threshold Analysis",
        "=" * 62,
    ]
    for r in track_b_rows:
        lines.append(
            f"  thr={r['threshold']}% daily={r['total_daily_return']}% ev={r['ev']}% "
            f"pf={r['profit_factor']} mdd={r['max_dd']}% switches={r['switch_count']}"
        )

    lines.extend(["", "=" * 62, "RECOMMENDED ENGINE", "=" * 62, json.dumps(engine, indent=2)])
    report = "\n".join(lines) + "\n\n" + "\n".join(mission_summary_lines())
    (OUT_DIR / "research_r012_report.txt").write_text(report, encoding="utf-8")

    write_csv(OUT_DIR / "trackA.csv", [track_a])
    write_csv(OUT_DIR / "trackB.csv", track_b_rows)
    write_csv(OUT_DIR / "threshold_analysis.csv", track_b_rows)
    write_csv(OUT_DIR / "switch_analysis.csv", [{
        "threshold": best_b["threshold"],
        "switch_success_pct": best_b["switch_success_pct"],
        "wrong_switch_pct": best_b["wrong_switch_pct"],
        "correct_hold_pct": best_b["correct_hold_pct"],
        "late_hold_pct": best_b["late_hold_pct"],
        "switch_count": best_b["switch_count"],
    }])
    write_csv(OUT_DIR / "best_switch_examples.csv", [
        {
            "at_time": s.at_time, "from": s.from_symbol, "to": s.to_symbol,
            "benefit_est": s.benefit_est, "old_cf": s.old_counterfactual,
            "new_ret": s.new_actual_ret, "success": s.success,
        }
        for s in best_ex
    ])
    write_csv(OUT_DIR / "worst_switch_examples.csv", [
        {
            "at_time": s.at_time, "from": s.from_symbol, "to": s.to_symbol,
            "benefit_est": s.benefit_est, "old_cf": s.old_counterfactual,
            "new_ret": s.new_actual_ret, "success": s.success,
        }
        for s in worst_ex
    ])

    safe_print(report)


if __name__ == "__main__":
    run()
