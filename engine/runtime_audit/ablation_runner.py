"""Ablation replay — incremental module stack vs baseline on forward klines."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from scout_auto_os.engine.portfolio.backtest import filter_2h_scans
from scout_auto_os.engine.research.directional.engines import rank_long, rank_short
from scout_auto_os.engine.research.directional.entry_filter.constants import (
    LONG_DIRECTION_CHAMPION,
    SHORT_DIRECTION_CHAMPION,
)
from scout_auto_os.engine.research.ranking_engine.metrics import equity_mdd, sharpe
from scout_auto_os.engine.research.safe import research_safe
from scout_auto_os.engine.research.zero_base.forward_eval import BAR_MINUTES, compute_forward_metrics
from scout_auto_os.engine.research.zero_base.runner import load_candidates_jsonl, load_forward_klines
from scout_auto_os.engine.runtime_audit.module_registry import ABLATION_SCENARIOS, MODULES


@dataclass
class TradeSim:
    scan_kst: str
    symbol: str
    direction: str
    return_pct: float
    hold_minutes: int
    peak_roi: float
    max_drawdown: float
    exit_reason: str
    scenario: str


def _g(features: dict, key: str, default: float = 0.0) -> float:
    return float(features.get(key, default))


def score_a6(row: dict) -> float:
    f = row.get("features") or {}
    h4 = _g(f, "h4_score")
    if h4 > 0:
        return h4
    return (
        _g(f, "5m_compression") * 2
        + _g(f, "15m_current_return_pct") * 1.5
        + max(0.0, _g(f, "1h_current_return_pct")) * 0.5
    )


def score_ranking(row: dict, peers: list[dict], direction: str) -> float:
    sym = row["symbol"]
    if direction == "long":
        ranked = rank_long(peers, LONG_DIRECTION_CHAMPION, top_k=len(peers))
    else:
        ranked = rank_short(peers, SHORT_DIRECTION_CHAMPION, top_k=len(peers))
    if sym not in ranked:
        return 0.0
    rank = ranked.index(sym)
    n = max(len(ranked), 1)
    base = score_a6(row)
    return base + (1.0 - rank / n) * 50.0


def _long_roi(klines: list, bar_i: int) -> float:
    entry = float(klines[0][1])
    if entry <= 0:
        return 0.0
    px = float(klines[min(bar_i, len(klines) - 1)][4])
    return round((px - entry) / entry * 100, 4)


def _short_roi(klines: list, bar_i: int) -> float:
    entry = float(klines[0][1])
    if entry <= 0:
        return 0.0
    px = float(klines[min(bar_i, len(klines) - 1)][4])
    return round((entry - px) / entry * 100, 4)


def _roi_at(klines: list, bar_i: int, direction: str) -> float:
    return _short_roi(klines, bar_i) if direction == "short" else _long_roi(klines, bar_i)


def _peak_and_mdd(klines: list, end_i: int, direction: str) -> tuple[float, float]:
    rois = [_roi_at(klines, i, direction) for i in range(end_i + 1)]
    if not rois:
        return 0.0, 0.0
    peak = max(rois)
    trough = min(rois)
    mdd = trough - peak if direction == "long" else peak - min(rois)
    return round(peak, 4), round(abs(mdd), 4)


def simulate_exit(
    klines: list,
    direction: str,
    mode: str,
) -> tuple[float, int, str, float, float]:
    """Return (return_pct, hold_minutes, exit_reason, peak_roi, max_drawdown)."""
    if not klines or len(klines) < 2:
        return 0.0, 0, "no_data", 0.0, 0.0

    max_bar = len(klines) - 1
    hold_2h_bar = min(max_bar, 120 // BAR_MINUTES - 1)
    hold_90m_bar = min(max_bar, 90 // BAR_MINUTES - 1)
    hold_1h_bar = min(max_bar, 60 // BAR_MINUTES - 1)
    min_hold_bar = min(max_bar, 30 // BAR_MINUTES - 1)

    if mode == "hold_2h":
        end = hold_2h_bar
        peak, mdd = _peak_and_mdd(klines, end, direction)
        return _roi_at(klines, end, direction), end * BAR_MINUTES, "hold_2h", peak, mdd

    peak_roi = 0.0
    exit_i = hold_2h_bar
    reason = "hold_2h"

    for i in range(min_hold_bar, max_bar + 1):
        roi = _roi_at(klines, i, direction)
        peak_roi = max(peak_roi, roi)
        trail = peak_roi - roi

        if mode == "expectation_proxy":
            if i >= hold_90m_bar:
                exit_i, reason = i, "expectation_horizon"
                break
            if i >= hold_1h_bar and roi < peak_roi * 0.4:
                exit_i, reason = i, "expectation_underperform"
                break

        elif mode == "pe_proxy":
            if trail >= 5.0 and i >= min_hold_bar:
                exit_i, reason = i, "pe_roi_trail5"
                break
            if i >= hold_2h_bar:
                exit_i, reason = i, "pe_max_hold"
                break

        elif mode == "state_exit":
            if i >= min_hold_bar and trail >= 8.0:
                exit_i, reason = i, "state_alive_collapse"
                break
            if direction == "long" and _roi_at(klines, i, direction) <= -10.0:
                exit_i, reason = i, "state_protective_sl"
                break
            if direction == "short" and _roi_at(klines, i, direction) <= -10.0:
                exit_i, reason = i, "state_protective_sl"
                break
            if i >= hold_2h_bar:
                exit_i, reason = i, "state_hold_target"
                break

        elif mode == "full_exit":
            if trail >= 5.0 and i >= min_hold_bar:
                exit_i, reason = i, "full_trail"
                break
            if i >= min_hold_bar and roi < peak_roi * 0.35:
                exit_i, reason = i, "full_expectation_fail"
                break
            if i >= hold_2h_bar:
                exit_i, reason = i, "full_hold_cap"
                break
        else:
            exit_i = hold_2h_bar

    peak, mdd = _peak_and_mdd(klines, exit_i, direction)
    return _roi_at(klines, exit_i, direction), exit_i * BAR_MINUTES, reason, peak, mdd


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


def _metrics(trades: list[TradeSim]) -> dict:
    if not trades:
        return {"trade_count": 0}
    rets = [t.return_pct for t in trades]
    holds = [t.hold_minutes for t in trades]
    return {
        "trade_count": len(trades),
        "avg_roi": round(statistics.mean(rets), 4),
        "win_rate": _win_rate(rets),
        "profit_factor": _profit_factor(rets),
        "mdd": equity_mdd(rets),
        "sharpe": sharpe(rets),
        "avg_hold_minutes": round(statistics.mean(holds), 1),
        "missed_exit_count": sum(1 for t in trades if t.exit_reason == "hold_2h" and t.peak_roi - t.return_pct > 5),
        "late_exit_count": sum(1 for t in trades if t.hold_minutes >= 120 and t.return_pct < 0),
        "false_exit_count": sum(1 for t in trades if "trail" in t.exit_reason and t.peak_roi - t.return_pct > 8),
        "reentry_opportunity_loss": round(
            statistics.mean(max(0.0, t.peak_roi - t.return_pct) for t in trades), 4,
        ),
    }


def _pick_symbols(
    rows: list[dict],
    scenario: dict,
    direction: str,
) -> list[str]:
    use_ranking = "ranking_engine" in scenario["modules"]
    scorer: Callable[[dict], float]
    if use_ranking:
        scorer = lambda r: score_ranking(r, rows, direction)
    else:
        scorer = score_a6

    ranked = sorted(rows, key=scorer, reverse=True)
    if direction == "long":
        k = scenario["slots_long"]
    else:
        k = scenario.get("slots_short", 0)
    if k <= 0:
        return []
    return [r["symbol"] for r in ranked[:k]]


class AblationRunner:
    def __init__(
        self,
        data_dir: Path,
        candidates_path: Path,
        forward_path: Path,
        lookback_scans: int | None = None,
    ) -> None:
        self.data_dir = data_dir
        self.out_dir = data_dir / "runtime_audit"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.candidates_path = candidates_path
        self.forward_path = forward_path
        self.lookback_scans = lookback_scans

    @research_safe("runtime_ablation")
    def run(self) -> dict:
        by_scan = load_candidates_jsonl(self.candidates_path)
        fwd = load_forward_klines(self.forward_path)
        scans = filter_2h_scans(sorted(by_scan.keys()))
        if self.lookback_scans:
            scans = scans[-self.lookback_scans :]

        comparison: list[dict] = []
        baseline_avg: float | None = None

        for sid, scenario in ABLATION_SCENARIOS.items():
            trades: list[TradeSim] = []
            for scan_kst in scans:
                rows = by_scan[scan_kst]
                for direction in ("long", "short"):
                    syms = _pick_symbols(rows, scenario, direction)
                    for sym in syms:
                        klines = fwd.get((scan_kst, sym))
                        if not klines:
                            continue
                        ret, hold, reason, peak, mdd = simulate_exit(
                            klines, direction, scenario["exit_mode"],
                        )
                        trades.append(TradeSim(
                            scan_kst=scan_kst,
                            symbol=sym,
                            direction=direction,
                            return_pct=ret,
                            hold_minutes=hold,
                            peak_roi=peak,
                            max_drawdown=mdd,
                            exit_reason=reason,
                            scenario=sid,
                        ))

            long_t = [t for t in trades if t.direction == "long"]
            short_t = [t for t in trades if t.direction == "short"]
            m_all = _metrics(trades)
            m_long = _metrics(long_t)
            m_short = _metrics(short_t)

            est_cpu = sum(MODULES[m].est_cpu_ms_per_tick for m in scenario["modules"] if m in MODULES)

            if sid == "baseline":
                baseline_avg = m_all.get("avg_roi", 0.0)

            lift = 0.0
            if baseline_avg and baseline_avg != 0:
                lift = round((m_all.get("avg_roi", 0) - baseline_avg) / abs(baseline_avg) * 100, 2)

            comparison.append({
                "scenario_id": sid,
                "label": scenario["label"],
                "modules": "|".join(scenario["modules"]),
                "exit_mode": scenario["exit_mode"],
                "trade_count": m_all.get("trade_count", 0),
                "long_trades": m_long.get("trade_count", 0),
                "short_trades": m_short.get("trade_count", 0),
                "avg_roi": m_all.get("avg_roi", 0),
                "long_avg_roi": m_long.get("avg_roi", 0),
                "short_avg_roi": m_short.get("avg_roi", 0),
                "win_rate": m_all.get("win_rate", 0),
                "long_win_rate": m_long.get("win_rate", 0),
                "short_win_rate": m_short.get("win_rate", 0),
                "profit_factor": m_all.get("profit_factor", 0),
                "mdd": m_all.get("mdd", 0),
                "sharpe": m_all.get("sharpe", 0),
                "avg_hold_minutes": m_all.get("avg_hold_minutes", 0),
                "missed_exit_count": m_all.get("missed_exit_count", 0),
                "late_exit_count": m_all.get("late_exit_count", 0),
                "false_exit_count": m_all.get("false_exit_count", 0),
                "reentry_opportunity_loss": m_all.get("reentry_opportunity_loss", 0),
                "roi_lift_vs_baseline_pct": lift,
                "est_cpu_ms_per_tick": est_cpu,
            })

        return {"comparison": comparison, "scan_count": len(scans)}
