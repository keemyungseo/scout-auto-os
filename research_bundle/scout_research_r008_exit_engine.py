"""
Scout Research R008 — Exit Engine

A6 frozen search. Rule / State / Hybrid exit simulation. Blind forward-only.
No formula/reject/rerank changes.

Usage:
  python scout_research_r008_exit_engine.py
  python scout_research_r008_exit_engine.py --tier top5
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path

import scout_phase19_winner_ranking_dna as p19
from season2_p37_scout_decision_hierarchy import write_csv

import scout_phase16_human_blind_test as p16
from scout_research_r005_execution_statistics import load_b001_scan_row, loo_a6_scan_rows, parse_kst, safe_print
from scout_research_r006_pilot_execution_engine import Bar, load_forward_bars, mfe_mae, unique_top_picks
from season2_scout_mission import mission_summary_lines

OUT_DIR = Path("logs") / "research_r008_exit_engine"

TP_GRID = (5, 7, 10, 12, 15, 18, 20, 25)
SL_GRID = (3, 5, 7, 8, 10, 12)
HOLD_GRID = (15, 30, 45, 60, 90, 120, 180, 240)
TRAIL_GRID = (2, 3, 5)

STATE_EXIT_RULES = (
    "exit_on_trend_lost",
    "exit_on_accel_lost",
    "exit_on_volume_weak",
    "exit_on_momentum_weak",
    "exit_on_state_bad",
    "hold_while_trend_alive",
)


@dataclass
class StateSnap:
    trend_alive: bool
    acceleration: bool
    expansion: bool
    volume_weak: bool
    momentum_weak: bool
    ret_1h_proxy: float
    vol_ratio: float


@dataclass
class ExitDetail:
    return_pct: float
    exit_reason: str
    bars_held: int
    hold_minutes: int
    mfe_pct: float
    mae_pct: float
    peak_capture_pct: float
    opportunity_loss_pct: float
    recovery_bars: int
    exit_snap: StateSnap
    search_a6: float
    search_state_1h: str
    search_state_2h: str


def hold_bars(mins: int) -> int:
    return max(1, mins // 5)


def state_snapshot(bars: list[Bar], i: int, entry_i: int) -> StateSnap:
    if i >= len(bars) or i < entry_i:
        return StateSnap(False, False, False, True, True, 0.0, 0.0)
    c = bars[i].c
    i0 = max(entry_i, i - 23)
    i1h = max(entry_i, i - 11)
    i2h = max(entry_i, i - 23)
    ret_2h = (c - bars[i2h].o) / bars[i2h].o * 100 if bars[i2h].o else 0
    ret_1h = (c - bars[i1h].o) / bars[i1h].o * 100 if bars[i1h].o else 0
    prev_1h = (bars[i1h].c - bars[max(entry_i, i1h - 12)].o) / bars[max(entry_i, i1h - 12)].o * 100 if i1h - 12 >= entry_i else 0
    closes = [bars[j].c for j in range(max(entry_i, i - 19), i + 1)]
    ma20 = statistics.mean(closes) if closes else c
    trend_alive = ret_2h > 0 or c >= ma20
    acceleration = ret_1h > prev_1h and ret_1h > 0
    rng_now = (bars[i].h - bars[i].l) / bars[i].o * 100 if bars[i].o else 0
    rng_prev = statistics.mean(
        [(bars[j].h - bars[j].l) / bars[j].o * 100 for j in range(max(entry_i, i - 6), i)]
    ) if i > entry_i else rng_now
    expansion = rng_now > rng_prev and ret_1h > 0
    vols = [bars[j].h for j in range(max(entry_i, i - 19), i + 1)]
    vol_proxy = bars[i].h - bars[i].l
    vol_ma = statistics.mean([bars[j].h - bars[j].l for j in range(max(entry_i, i - 19), i)]) if i > entry_i else vol_proxy
    vol_ratio = vol_proxy / vol_ma if vol_ma else 1.0
    volume_weak = vol_ratio < 0.8
    mom6 = sum((bars[j].c - bars[j].o) / bars[j].o * 100 if bars[j].o else 0 for j in range(max(entry_i, i - 5), i + 1))
    momentum_weak = mom6 < 0
    return StateSnap(trend_alive, acceleration, expansion, volume_weak, momentum_weak, ret_1h, vol_ratio)


def recovery_bars_after_mae(bars: list[Bar], entry_i: int, entry_px: float, exit_i: int) -> int:
    worst_i = entry_i
    worst = entry_px
    for j in range(entry_i, exit_i + 1):
        if bars[j].l < worst:
            worst, worst_i = bars[j].l, j
    for j in range(worst_i, exit_i + 1):
        if bars[j].c >= entry_px:
            return j - worst_i
    return exit_i - worst_i


def simulate_rule(
    bars: list[Bar],
    meta: dict,
    *,
    tp_pct: float | None = None,
    sl_pct: float | None = None,
    hold_min: int | None = None,
    trail_gap_pct: float | None = None,
) -> ExitDetail | None:
    entry_i = 0
    if entry_i >= len(bars):
        return None
    entry_px = bars[entry_i].o
    if entry_px <= 0:
        return None
    mfe, mae = mfe_mae(bars, entry_i, entry_px)
    peak = entry_px
    end_i = len(bars) - 1
    if hold_min is not None:
        end_i = min(entry_i + hold_bars(hold_min) - 1, len(bars) - 1)

    for j in range(entry_i, end_i + 1):
        b = bars[j]
        peak = max(peak, b.h)
        tp_px = entry_px * (1 + tp_pct / 100) if tp_pct is not None else None
        sl_px = entry_px * (1 - sl_pct / 100) if sl_pct is not None else None
        trail_px = peak * (1 - trail_gap_pct / 100) if trail_gap_pct is not None else None

        if sl_px is not None and b.l <= sl_px:
            snap = state_snapshot(bars, j, entry_i)
            return _detail(-sl_pct, "sl", j, entry_i, entry_px, mfe, mae, snap, meta, bars)
        if tp_px is not None and b.h >= tp_px:
            snap = state_snapshot(bars, j, entry_i)
            return _detail(tp_pct, "tp", j, entry_i, entry_px, mfe, mae, snap, meta, bars)
        if trail_px is not None and b.l <= trail_px:
            ret = (trail_px - entry_px) / entry_px * 100
            snap = state_snapshot(bars, j, entry_i)
            return _detail(ret, "trail", j, entry_i, entry_px, mfe, mae, snap, meta, bars)

    close_px = bars[end_i].c
    ret = (close_px - entry_px) / entry_px * 100
    snap = state_snapshot(bars, end_i, entry_i)
    reason = "time" if hold_min is not None else "eod"
    return _detail(ret, reason, end_i, entry_i, entry_px, mfe, mae, snap, meta, bars)


def _detail(
    ret: float, reason: str, exit_i: int, entry_i: int, entry_px: float,
    mfe: float, mae: float, snap: StateSnap, meta: dict, bars: list[Bar],
) -> ExitDetail:
    bars_held = exit_i - entry_i + 1
    if mfe > 0 and ret > 0:
        pc = min(100.0, ret / mfe * 100)
    else:
        pc = 0.0
    opp = max(0.0, mfe - max(ret, 0))
    rec = recovery_bars_after_mae(bars, entry_i, entry_px, exit_i)
    return ExitDetail(
        return_pct=round(ret, 4),
        exit_reason=reason,
        bars_held=bars_held,
        hold_minutes=bars_held * 5,
        mfe_pct=round(mfe, 4),
        mae_pct=round(mae, 4),
        peak_capture_pct=round(pc, 4),
        opportunity_loss_pct=round(opp, 4),
        recovery_bars=rec,
        exit_snap=snap,
        search_a6=meta.get("a6_score", 0),
        search_state_1h=meta.get("state_1h", ""),
        search_state_2h=meta.get("state_2h", ""),
    )


def simulate_state(bars: list[Bar], meta: dict, rule_id: str, max_hold_min: int = 240) -> ExitDetail | None:
    entry_i = 0
    if not bars:
        return None
    entry_px = bars[entry_i].o
    if entry_px <= 0:
        return None
    mfe, mae = mfe_mae(bars, entry_i, entry_px)
    end_i = min(entry_i + hold_bars(max_hold_min) - 1, len(bars) - 1)
    entry_snap = state_snapshot(bars, entry_i, entry_i)

    for j in range(entry_i + 1, end_i + 1):
        snap = state_snapshot(bars, j, entry_i)
        fire = False
        if rule_id == "exit_on_trend_lost" and entry_snap.trend_alive and not snap.trend_alive:
            fire = True
        elif rule_id == "exit_on_accel_lost" and entry_snap.acceleration and not snap.acceleration:
            fire = True
        elif rule_id == "exit_on_volume_weak" and snap.volume_weak:
            fire = True
        elif rule_id == "exit_on_momentum_weak" and snap.momentum_weak:
            fire = True
        elif rule_id == "exit_on_state_bad" and (
            not snap.trend_alive or snap.volume_weak or snap.momentum_weak
        ):
            fire = True
        elif rule_id == "hold_while_trend_alive" and not snap.trend_alive:
            fire = True
        if fire:
            ret = (bars[j].c - entry_px) / entry_px * 100
            return _detail(ret, f"state_{rule_id}", j, entry_i, entry_px, mfe, mae, snap, meta, bars)

    ret = (bars[end_i].c - entry_px) / entry_px * 100
    snap = state_snapshot(bars, end_i, entry_i)
    return _detail(ret, "state_time", end_i, entry_i, entry_px, mfe, mae, snap, meta, bars)


def simulate_hybrid(
    bars: list[Bar],
    meta: dict,
    *,
    tp_pct: float | None,
    sl_pct: float | None,
    hold_min: int,
    trail_gap_pct: float | None = None,
    require_trend: bool = False,
    require_accel: bool = False,
    exit_on_state_bad: bool = True,
) -> ExitDetail | None:
    entry_i = 0
    if not bars:
        return None
    entry_px = bars[entry_i].o
    if entry_px <= 0:
        return None
    mfe, mae = mfe_mae(bars, entry_i, entry_px)
    peak = entry_px
    end_i = min(entry_i + hold_bars(hold_min) - 1, len(bars) - 1)

    for j in range(entry_i, end_i + 1):
        b = bars[j]
        peak = max(peak, b.h)
        snap = state_snapshot(bars, j, entry_i)
        state_bad = not snap.trend_alive or snap.volume_weak or snap.momentum_weak

        if exit_on_state_bad and state_bad:
            ret = (b.c - entry_px) / entry_px * 100
            return _detail(ret, "hybrid_state", j, entry_i, entry_px, mfe, mae, snap, meta, bars)

        sl_px = entry_px * (1 - sl_pct / 100) if sl_pct else None
        if sl_px is not None and b.l <= sl_px:
            return _detail(-sl_pct, "hybrid_sl", j, entry_i, entry_px, mfe, mae, snap, meta, bars)

        guard_ok = True
        if require_trend and not snap.trend_alive:
            guard_ok = False
        if require_accel and not snap.acceleration:
            guard_ok = False

        tp_px = entry_px * (1 + tp_pct / 100) if tp_pct else None
        if tp_px is not None and b.h >= tp_px and not guard_ok:
            return _detail(tp_pct, "hybrid_tp", j, entry_i, entry_px, mfe, mae, snap, meta, bars)
        if tp_px is not None and b.h >= tp_px and guard_ok:
            continue

        trail_px = peak * (1 - trail_gap_pct / 100) if trail_gap_pct else None
        if trail_px is not None and b.l <= trail_px:
            ret = (trail_px - entry_px) / entry_px * 100
            return _detail(ret, "hybrid_trail", j, entry_i, entry_px, mfe, mae, snap, meta, bars)

    ret = (bars[end_i].c - entry_px) / entry_px * 100
    snap = state_snapshot(bars, end_i, entry_i)
    return _detail(ret, "hybrid_time", end_i, entry_i, entry_px, mfe, mae, snap, meta, bars)


def longest_losing_streak(returns: list[float]) -> int:
    best = cur = 0
    for r in returns:
        if r <= 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def aggregate_stats(details: list[ExitDetail], rule_id: str, rule_type: str, params: dict) -> dict:
    rets = [d.return_pct for d in details]
    if not rets:
        return {}
    wins = [r for r in rets if r > 0]
    losses = [r for r in rets if r <= 0]
    gw, gl = sum(wins), abs(sum(losses))
    pf = gw / gl if gl > 0 else 99.0
    std = statistics.stdev(rets) if len(rets) > 1 else 0.0
    sharpe = statistics.mean(rets) / std if std > 0 else 0.0
    downs = [r for r in rets if r < 0]
    ds = statistics.stdev(downs) if len(downs) > 1 else (abs(downs[0]) if downs else 1.0)
    sortino = statistics.mean(rets) / ds if ds > 0 else 0.0
    eq = 100.0
    peak = 100.0
    mdd = 0.0
    for r in rets:
        eq *= 1 + r / 100
        peak = max(peak, eq)
        mdd = max(mdd, (peak - eq) / peak * 100)
    holds = [d.hold_minutes for d in details]
    return {
        "rule_id": rule_id,
        "rule_type": rule_type,
        **params,
        "n": len(rets),
        "win_rate": round(len(wins) / len(rets) * 100, 2),
        "loss_rate": round(len(losses) / len(rets) * 100, 2),
        "ev": round(statistics.mean(rets), 4),
        "profit_factor": round(pf, 4),
        "mean_return": round(statistics.mean(rets), 4),
        "median_return": round(statistics.median(rets), 4),
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "max_dd": round(mdd, 4),
        "avg_hold_min": round(statistics.mean(holds), 1),
        "median_hold_min": round(statistics.median(holds), 1),
        "peak_capture_pct": round(statistics.mean([d.peak_capture_pct for d in details]), 4),
        "opportunity_loss": round(statistics.mean([d.opportunity_loss_pct for d in details]), 4),
        "mae": round(statistics.mean([d.mae_pct for d in details]), 4),
        "mfe": round(statistics.mean([d.mfe_pct for d in details]), 4),
        "recovery_bars": round(statistics.mean([d.recovery_bars for d in details]), 2),
        "longest_losing_streak": longest_losing_streak(rets),
        "score": round(statistics.mean(rets) * min(pf, 5) - mdd * 0.05, 4),
    }


def load_trades(tier: str) -> list[tuple[dict, list[Bar]]]:
    if tier == "top1":
        scan_rows = loo_a6_scan_rows()
        b001 = load_b001_scan_row()
        if b001:
            scan_rows.append(b001)
        picks = []
        for sr in scan_rows:
            if sr.get("top2"):
                r = sr["top2"][0]
                picks.append({
                    "search_time": sr["scan_kst"],
                    "symbol": r["symbol"],
                    "a6_score": round(r["a6"], 4),
                    "state_1h": r["states"].get("1h", ""),
                    "state_2h": r["states"].get("2h", ""),
                })
    else:
        picks = unique_top_picks(tier)
    out: list[tuple[dict, list[Bar]]] = []
    for i, p in enumerate(picks, 1):
        bars = load_forward_bars(p["symbol"], p["search_time"])
        if bars:
            out.append((p, bars))
        if i % 200 == 0:
            safe_print(f"  paths {i}/{len(picks)}")
    return out


def run_rule_grid(trades: list[tuple[dict, list[Bar]]]) -> list[dict]:
    rows: list[dict] = []
    total = len(TP_GRID) * len(SL_GRID) * len(HOLD_GRID)
    done = 0
    for tp in TP_GRID:
        for sl in SL_GRID:
            for hold in HOLD_GRID:
                details: list[ExitDetail] = []
                for meta, bars in trades:
                    d = simulate_rule(bars, meta, tp_pct=tp, sl_pct=sl, hold_min=hold)
                    if d:
                        details.append(d)
                rid = f"rule_tp{tp}_sl{sl}_h{hold}"
                row = aggregate_stats(details, rid, "rule", {"tp": tp, "sl": sl, "hold_min": hold})
                if row:
                    rows.append(row)
                done += 1
                if done % 50 == 0:
                    safe_print(f"  rule grid {done}/{total}")
    for trail in TRAIL_GRID:
        for hold in (60, 90, 120):
            details = []
            for meta, bars in trades:
                d = simulate_rule(bars, meta, trail_gap_pct=trail, hold_min=hold)
                if d:
                    details.append(d)
            rid = f"rule_trail{trail}_h{hold}"
            rows.append(aggregate_stats(details, rid, "rule_trail", {"trail": trail, "hold_min": hold}))
    rows.sort(key=lambda x: (x["score"], x["ev"]), reverse=True)
    return rows


def run_state_grid(trades: list[tuple[dict, list[Bar]]]) -> list[dict]:
    rows: list[dict] = []
    for rule_id in STATE_EXIT_RULES:
        details = []
        for meta, bars in trades:
            d = simulate_state(bars, meta, rule_id)
            if d:
                details.append(d)
        rows.append(aggregate_stats(details, rule_id, "state", {"state_rule": rule_id}))
    rows.sort(key=lambda x: (x["score"], x["ev"]), reverse=True)
    return rows


def run_hybrid_grid(trades: list[tuple[dict, list[Bar]]], top_rules: list[dict]) -> list[dict]:
    rows: list[dict] = []
    bases = top_rules[:8]
    guards = (
        (False, False, True),
        (True, False, True),
        (True, True, True),
        (True, True, False),
    )
    for base in bases:
        tp, sl, hold = base["tp"], base["sl"], base["hold_min"]
        for trail in (None, 3, 5):
            for req_t, req_a, exit_bad in guards:
                details = []
                for meta, bars in trades:
                    d = simulate_hybrid(
                        bars, meta,
                        tp_pct=tp, sl_pct=sl, hold_min=hold,
                        trail_gap_pct=trail,
                        require_trend=req_t, require_accel=req_a,
                        exit_on_state_bad=exit_bad,
                    )
                    if d:
                        details.append(d)
                rid = f"hyb_tp{tp}_sl{sl}_h{hold}_tr{trail}_t{int(req_t)}_a{int(req_a)}"
                params = {
                    "tp": tp, "sl": sl, "hold_min": hold, "trail": trail,
                    "require_trend": req_t, "require_accel": req_a, "exit_on_state_bad": exit_bad,
                }
                rows.append(aggregate_stats(details, rid, "hybrid", params))
    rows.sort(key=lambda x: (x["score"], x["ev"]), reverse=True)
    return rows


def exit_analysis_csv(details: list[ExitDetail], good_threshold: float = 3.0) -> tuple[list[dict], list[dict]]:
    good = [d for d in details if d.return_pct >= good_threshold]
    bad = [d for d in details if d.return_pct < 0]

    def summarize(label: str, subset: list[ExitDetail]) -> dict:
        if not subset:
            return {"group": label, "n": 0}
        return {
            "group": label,
            "n": len(subset),
            "avg_return": round(statistics.mean([d.return_pct for d in subset]), 4),
            "avg_a6": round(statistics.mean([d.search_a6 for d in subset]), 4),
            "pct_trend_alive_exit": round(sum(1 for d in subset if d.exit_snap.trend_alive) / len(subset) * 100, 2),
            "pct_accel_exit": round(sum(1 for d in subset if d.exit_snap.acceleration) / len(subset) * 100, 2),
            "pct_volume_weak_exit": round(sum(1 for d in subset if d.exit_snap.volume_weak) / len(subset) * 100, 2),
            "pct_expansion_exit": round(sum(1 for d in subset if d.exit_snap.expansion) / len(subset) * 100, 2),
            "avg_hold_min": round(statistics.mean([d.hold_minutes for d in subset]), 1),
            "avg_peak_capture": round(statistics.mean([d.peak_capture_pct for d in subset]), 4),
        }

    return [summarize("good_exit", good), summarize("bad_exit", bad)], good + bad


def detail_to_row(d: ExitDetail, rule_id: str) -> dict:
    return {
        "rule_id": rule_id,
        "return_pct": d.return_pct,
        "exit_reason": d.exit_reason,
        "hold_minutes": d.hold_minutes,
        "mfe": d.mfe_pct,
        "mae": d.mae_pct,
        "peak_capture": d.peak_capture_pct,
        "opportunity_loss": d.opportunity_loss_pct,
        "search_a6": d.search_a6,
        "search_state_1h": d.search_state_1h,
        "search_state_2h": d.search_state_2h,
        "exit_trend_alive": d.exit_snap.trend_alive,
        "exit_acceleration": d.exit_snap.acceleration,
        "exit_expansion": d.exit_snap.expansion,
        "exit_volume_weak": d.exit_snap.volume_weak,
        "exit_momentum_weak": d.exit_snap.momentum_weak,
        "exit_vol_ratio": round(d.exit_snap.vol_ratio, 4),
    }


def build_pilot_json(best: dict) -> dict:
    rt = best.get("rule_type", "rule")
    return {
        "version": "r008_exit_v1",
        "search_formula": "A6_frozen",
        "entry": "immediate",
        "exit": "Hybrid" if rt == "hybrid" else ("State" if rt == "state" else "Rule"),
        "take_profit": best.get("tp"),
        "stop_loss": best.get("sl"),
        "max_hold": best.get("hold_min"),
        "trailing": best.get("trail"),
        "state_guard": {
            "trend_alive": best.get("require_trend"),
            "acceleration": best.get("require_accel"),
            "exit_on_state_bad": best.get("exit_on_state_bad"),
            "state_rule": best.get("state_rule"),
        },
        "expected_ev": best.get("ev"),
        "win_rate": best.get("win_rate"),
        "profit_factor": best.get("profit_factor"),
        "peak_capture": best.get("peak_capture_pct"),
        "max_dd": best.get("max_dd"),
        "robust_score": best.get("score"),
        "rule_id": best.get("rule_id"),
    }


def run(tier: str = "top5") -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p19.CACHE_DIR = Path("logs") / "phase19_winner_dna" / "kline_cache"
    p16.CACHE_DIR = p19.CACHE_DIR

    safe_print(f"R008 loading {tier} trades (blind forward-only)...")
    trades = load_trades(tier)
    n = len(trades)
    safe_print(f"R008 {n} trades")

    safe_print("R008 A — Rule Based Exit grid...")
    rule_rows = run_rule_grid(trades)

    safe_print("R008 B — State Based Exit...")
    state_rows = run_state_grid(trades)

    safe_print("R008 C — Hybrid Exit...")
    hybrid_rows = run_hybrid_grid(
        trades, [r for r in rule_rows if r.get("tp") and r.get("sl") and r.get("hold_min")],
    )

    all_ranked = sorted(rule_rows + state_rows + hybrid_rows, key=lambda x: (x["score"], x["ev"]), reverse=True)
    top10 = all_ranked[:10]
    best = top10[0]

    best_details: list[ExitDetail] = []
    for meta, bars in trades:
        if best.get("rule_type") == "hybrid":
            d = simulate_hybrid(
                bars, meta,
                tp_pct=best.get("tp"), sl_pct=best.get("sl"), hold_min=best.get("hold_min", 60),
                trail_gap_pct=best.get("trail"),
                require_trend=bool(best.get("require_trend")),
                require_accel=bool(best.get("require_accel")),
                exit_on_state_bad=bool(best.get("exit_on_state_bad", True)),
            )
        elif best.get("rule_type") == "state":
            d = simulate_state(bars, meta, best.get("state_rule", "exit_on_state_bad"))
        else:
            d = simulate_rule(
                bars, meta,
                tp_pct=best.get("tp"), sl_pct=best.get("sl"), hold_min=best.get("hold_min"),
                trail_gap_pct=best.get("trail"),
            )
        if d:
            best_details.append(d)

    analysis, _ = exit_analysis_csv(best_details)
    trade_log = [detail_to_row(d, best["rule_id"]) for d in best_details]

    pilot = build_pilot_json(best)
    (OUT_DIR / "pilot_exit_rule.json").write_text(json.dumps(pilot, indent=2), encoding="utf-8")

    lines = [
        "############################################################",
        "SCOUT RESEARCH R008 — EXIT ENGINE",
        "############################################################",
        "",
        f"A6 frozen | tier={tier} | trades={n} | entry=immediate | blind forward-only",
        "",
        "=" * 62,
        "TOP10 EXIT RULES (Rule + State + Hybrid)",
        "=" * 62,
    ]
    for i, r in enumerate(top10, 1):
        lines.append(
            f"  #{i} [{r['rule_type']}] {r['rule_id']} | ev={r['ev']}% wr={r['win_rate']}% "
            f"pf={r['profit_factor']} pc={r['peak_capture_pct']}% dd={r['max_dd']}% score={r['score']}"
        )

    lines.extend(["", "=" * 62, "EXIT ANALYSIS (best rule good vs bad)", "=" * 62])
    for a in analysis:
        lines.append(f"  {a}")

    lines.extend(["", "=" * 62, "PILOT EXIT RULE", "=" * 62, json.dumps(pilot, indent=2)])
    report = "\n".join(lines) + "\n\n" + "\n".join(mission_summary_lines())
    (OUT_DIR / "research_r008_report.txt").write_text(report, encoding="utf-8")

    write_csv(OUT_DIR / "rule_based_exits.csv", rule_rows)
    write_csv(OUT_DIR / "state_based_exits.csv", state_rows)
    write_csv(OUT_DIR / "hybrid_exits.csv", hybrid_rows)
    write_csv(OUT_DIR / "top10_exit_rules.csv", top10)
    write_csv(OUT_DIR / "exit_analysis.csv", analysis)
    write_csv(OUT_DIR / "best_rule_trade_log.csv", trade_log)

    safe_print(report)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="top5", choices=("top1", "top2", "top5", "top7"))
    args = ap.parse_args()
    run(tier=args.tier)


if __name__ == "__main__":
    main()
