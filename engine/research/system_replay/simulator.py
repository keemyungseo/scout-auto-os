"""Portfolio replay simulator — Long3/Short3, 15-day, systems A-D."""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta

from scout_auto_os.engine.portfolio.backtest import filter_2h_scans, _parse_scan
from scout_auto_os.engine.portfolio.engine import PortfolioEngine
from scout_auto_os.engine.research.system_replay.constants import BARS_PER_SCAN, REPLAY_SEED, SYSTEMS
from scout_auto_os.engine.research.system_replay.patterns import classify_patterns
from scout_auto_os.engine.research.zero_base.forward_eval import BAR_MINUTES
from scout_auto_os.engine.runtime_audit.ablation_runner import _peak_and_mdd, _roi_at, simulate_exit


@dataclass
class LivePosition:
    symbol: str
    direction: str
    entry_scan: str
    entry_scan_idx: int
    entry_score: float
    live_pattern: str
    last_bar: int = -1
    expected_return: float = 3.0
    expected_horizon_min: int = 120


def _hold2h_counterfactual(klines: list, direction: str) -> tuple[float, float, float]:
    ret, hold, _, peak, mdd = simulate_exit(klines, direction, "hold_2h")
    return ret, peak, mdd


def _scan_bar_index(entry_idx: int, current_idx: int) -> int:
    return min((current_idx - entry_idx) * BARS_PER_SCAN, 9999)


def _exit_at_bar(
    klines: list,
    direction: str,
    exit_mode: str,
    up_to_bar: int,
) -> tuple[float, int, str, float, float]:
    if not klines:
        return 0.0, 0, "no_data", 0.0, 0.0
    end = min(up_to_bar, len(klines) - 1)
    sub = klines[: end + 1]
    if len(sub) < 2:
        return 0.0, 0, "no_data", 0.0, 0.0

    if exit_mode == "hold_2h":
        hold_bar = min(len(sub) - 1, 120 // BAR_MINUTES - 1)
        if end < hold_bar:
            return _roi_at(sub, end, direction), end * BAR_MINUTES, "holding", *_peak_and_mdd(sub, end, direction)
        peak, mdd = _peak_and_mdd(sub, hold_bar, direction)
        return _roi_at(sub, hold_bar, direction), hold_bar * BAR_MINUTES, "hold_2h", peak, mdd

    ret, hold, reason, peak, mdd = simulate_exit(sub, direction, exit_mode)
    return ret, hold, reason, peak, mdd


def _entry_exit_scores(actual: float, peak: float, hold2h: float, entry_score: float) -> tuple[float, float]:
    entry_q = round(hold2h, 4)
    exit_q = round(actual - hold2h, 4)
    cap = round(actual / peak * 100, 2) if peak > 1e-6 else 0.0
    exit_score = round(cap * 0.5 + max(0, exit_q) * 10 + entry_score * 0.1, 2)
    entry_score_out = round(entry_q * 10 + entry_score * 0.2, 2)
    return entry_score_out, exit_score


def _close_position(
    pos: LivePosition,
    klines: list,
    bar_i: int,
    exit_mode: str,
    system_id: str,
    exit_scan: str,
    reason_override: str = "",
) -> dict:
    actual, hold, reason, peak, mdd = _exit_at_bar(klines, pos.direction, exit_mode, bar_i)
    if reason_override:
        reason = reason_override
    hold2h, peak2h, mdd2h = _hold2h_counterfactual(klines, pos.direction)
    if peak2h > peak:
        peak = peak2h
    entry_q, exit_q = _entry_exit_scores(actual, peak, hold2h, pos.entry_score)
    ideal_hold = 120
    exit_delay = max(0, hold - ideal_hold) if actual < hold2h else max(0, ideal_hold - hold)

    row = {
        "trade_id": f"t_{uuid.uuid4().hex[:10]}",
        "system_id": system_id,
        "symbol": pos.symbol,
        "direction": pos.direction,
        "entry_scan": pos.entry_scan,
        "exit_scan": exit_scan,
        "entry_score": pos.entry_score,
        "exit_score": exit_q,
        "entry_quality_score": entry_q,
        "exit_quality_score": exit_q,
        "expected_return": pos.expected_return,
        "expected_horizon_min": pos.expected_horizon_min,
        "actual_return": round(actual, 4),
        "hold_2h_counterfactual": round(hold2h, 4),
        "return_difference": round(actual - hold2h, 4),
        "hold_minutes": hold,
        "peak_roi": round(peak, 4),
        "max_drawdown": round(mdd, 4),
        "exit_reason": reason,
        "live_pattern": pos.live_pattern,
        "entry_delay_min": 0.0,
        "exit_delay_min": round(exit_delay, 2),
    }
    row.update(classify_patterns(row))
    return row


def simulate_system(
    system_id: str,
    scans: list[str],
    by_scan: dict[str, list[dict]],
    fwd: dict[tuple[str, str], list],
    engine: PortfolioEngine,
) -> list[dict]:
    random.seed(REPLAY_SEED)
    exit_mode = SYSTEMS[system_id]["exit_mode"]
    open_positions: list[LivePosition] = []
    closed_trades: list[dict] = []

    for scan_idx, scan_kst in enumerate(scans):
        hold_until = scans[scan_idx + 1] if scan_idx + 1 < len(scans) else scan_kst
        rows = [{"symbol": r["symbol"], "features": r["features"]} for r in by_scan[scan_kst]]

        still_open: list[LivePosition] = []
        for pos in open_positions:
            klines = fwd.get((pos.entry_scan, pos.symbol), [])
            if not klines:
                continue
            if scan_idx <= pos.entry_scan_idx:
                still_open.append(pos)
                continue
            max_bar = _scan_bar_index(pos.entry_scan_idx, scan_idx)
            if max_bar <= pos.last_bar:
                still_open.append(pos)
                continue

            exited = False
            if exit_mode == "hold_2h":
                if scan_idx - pos.entry_scan_idx >= 1:
                    closed_trades.append(_close_position(
                        pos, klines, max_bar, exit_mode, system_id, scan_kst,
                    ))
                    exited = True
            else:
                hold_cap = min(len(klines) - 1, 120 // BAR_MINUTES - 1)
                start_bar = max(pos.last_bar + 1, 0)
                for bar in range(start_bar, max_bar + 1):
                    _, _, reason, _, _ = _exit_at_bar(klines, pos.direction, exit_mode, bar)
                    min_bar = max(0, 30 // BAR_MINUTES - 1)
                    if reason not in ("holding",) and bar >= min_bar:
                        closed_trades.append(_close_position(
                            pos, klines, bar, exit_mode, system_id, scan_kst,
                        ))
                        exited = True
                        break
                    if bar >= hold_cap:
                        closed_trades.append(_close_position(
                            pos, klines, bar, exit_mode, system_id, scan_kst, "max_hold_cap",
                        ))
                        exited = True
                        break
                if not exited:
                    pos.last_bar = max_bar
                    still_open.append(pos)
                    exited = True

            if not exited:
                still_open.append(pos)

        open_positions = still_open

        result = engine.process_scan(rows, scan_kst, hold_until_scan=hold_until)

        out_keys = {(rep["out_symbol"], rep["direction"]) for rep in result["replacements"]}
        if out_keys:
            kept: list[LivePosition] = []
            for pos in open_positions:
                if (pos.symbol, pos.direction) in out_keys:
                    klines = fwd.get((pos.entry_scan, pos.symbol), [])
                    bar = _scan_bar_index(pos.entry_scan_idx, scan_idx)
                    closed_trades.append(_close_position(
                        pos, klines, bar, exit_mode, system_id, scan_kst, "slot_replacement",
                    ))
                else:
                    kept.append(pos)
            open_positions = kept

        for entry in result["new_entries"]:
            open_positions.append(LivePosition(
                symbol=entry["symbol"],
                direction=entry["direction"],
                entry_scan=scan_kst,
                entry_scan_idx=scan_idx,
                entry_score=float(entry.get("entry_score", 0)),
                live_pattern=str(entry.get("live_pattern", "")),
                expected_return=3.0,
                expected_horizon_min=90 if entry["direction"] == "short" else 120,
            ))

    for pos in open_positions:
        klines = fwd.get((pos.entry_scan, pos.symbol), [])
        bar = min(len(klines) - 1, 120 // BAR_MINUTES - 1) if klines else 0
        closed_trades.append(_close_position(
            pos, klines, bar, exit_mode, system_id, scans[-1], "replay_end",
        ))

    return closed_trades


def filter_replay_scans(all_scans: list[str], replay_days: int = 15) -> list[str]:
    scans = filter_2h_scans(all_scans)
    if not scans:
        return []
    last = _parse_scan(scans[-1])
    cut = last - timedelta(days=replay_days)
    return [s for s in scans if _parse_scan(s) >= cut]
