"""Replay metrics — portfolio, entry/exit quality, lift calculations."""

from __future__ import annotations

import statistics

from scout_auto_os.engine.research.ranking_engine.metrics import equity_mdd, sharpe
from scout_auto_os.engine.research.system_replay.patterns import pattern_summary


def _profit_factor(returns: list[float]) -> float:
    wins = sum(r for r in returns if r > 0)
    losses = abs(sum(r for r in returns if r < 0))
    if losses <= 0:
        return round(wins, 2) if wins else 0.0
    return round(wins / losses, 2)


def _win_rate(returns: list[float], threshold: float = 3.0) -> float:
    if not returns:
        return 0.0
    return round(sum(1 for r in returns if r >= threshold) / len(returns) * 100, 2)


def aggregate_trades(trades: list[dict], system_id: str = "") -> dict:
    if not trades:
        return {"system_id": system_id, "trade_count": 0}
    rets = [float(t["actual_return"]) for t in trades]
    holds = [int(t["hold_minutes"]) for t in trades]
    entry_delays = [float(t.get("entry_delay_min", 0)) for t in trades]
    exit_delays = [float(t.get("exit_delay_min", 0)) for t in trades]
    cumulative = round(sum(rets), 4)
    patterns = pattern_summary(trades)

    false_exit = sum(
        1 for t in trades
        if "trail" in str(t.get("exit_reason", ""))
        and float(t.get("peak_roi", 0)) - float(t["actual_return"]) > 8
    )
    late_exit = sum(
        1 for t in trades
        if int(t["hold_minutes"]) >= 120 and float(t["actual_return"]) < 0
    )
    missed_exit = sum(
        1 for t in trades
        if float(t.get("peak_roi", 0)) - float(t["actual_return"]) > 5
        and str(t.get("exit_reason", "")) == "hold_2h"
    )

    return {
        "system_id": system_id,
        "trade_count": len(trades),
        "total_roi": cumulative,
        "avg_roi": round(statistics.mean(rets), 4),
        "profit_factor": _profit_factor(rets),
        "sharpe": sharpe(rets),
        "mdd": equity_mdd(rets),
        "win_rate": _win_rate(rets),
        "avg_hold_minutes": round(statistics.mean(holds), 1),
        "max_hold_minutes": max(holds),
        "avg_entry_delay_min": round(statistics.mean(entry_delays), 2) if entry_delays else 0,
        "avg_exit_delay_min": round(statistics.mean(exit_delays), 2) if exit_delays else 0,
        "false_exit_count": false_exit,
        "late_exit_count": late_exit,
        "missed_exit_count": missed_exit,
        **patterns,
    }


def entry_exit_breakdown(trades: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for t in trades:
        hold2h = float(t.get("hold_2h_counterfactual", t["actual_return"]))
        actual = float(t["actual_return"])
        entry_score = float(t.get("entry_score", 0))
        peak = float(t.get("peak_roi", 0))
        exit_delta = round(actual - hold2h, 4)
        entry_quality = round(hold2h, 4)
        exit_quality = round(exit_delta, 4)
        search_contribution = entry_quality
        exit_contribution = exit_quality
        rows.append({
            "trade_id": t.get("trade_id", ""),
            "system_id": t.get("system_id", ""),
            "symbol": t["symbol"],
            "direction": t["direction"],
            "entry_score": entry_score,
            "entry_quality": entry_quality,
            "exit_quality": exit_quality,
            "search_contribution": search_contribution,
            "exit_contribution": exit_contribution,
            "hold_2h_return": hold2h,
            "actual_return": actual,
            "exit_delta_vs_hold2h": exit_delta,
            "peak_roi": peak,
            "peak_capture_pct": round(actual / peak * 100, 2) if peak > 1e-6 else 0,
            "exit_helped": int(exit_delta > 0.5),
            "search_good_exit_bad": int(hold2h >= 3 and exit_delta < -2),
            "search_flat_exit_saved": int(hold2h < 1 and exit_delta > 1),
        })
    return rows


def compute_pe_lift(trades_c: list[dict], trades_a: list[dict]) -> float:
    """PE improvement: mean(C actual - A hold2h) on matched entries."""
    a_map = {
        (t["symbol"], t["direction"], t["entry_scan"]): float(t.get("hold_2h_counterfactual", t["actual_return"]))
        for t in trades_a
    }
    lifts: list[float] = []
    for t in trades_c:
        key = (t["symbol"], t["direction"], t["entry_scan"])
        if key in a_map:
            lifts.append(float(t["actual_return"]) - a_map[key])
    return round(statistics.mean(lifts), 4) if lifts else 0.0


def compute_expectation_lift(trades_d: list[dict], trades_c: list[dict]) -> float:
    c_map = {
        (t["symbol"], t["direction"], t["entry_scan"]): float(t["actual_return"])
        for t in trades_c
    }
    lifts: list[float] = []
    for t in trades_d:
        key = (t["symbol"], t["direction"], t["entry_scan"])
        if key in c_map:
            lifts.append(float(t["actual_return"]) - c_map[key])
    return round(statistics.mean(lifts), 4) if lifts else 0.0
