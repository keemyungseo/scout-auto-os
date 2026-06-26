"""Position evaluation metrics from thesis + live bars."""

from __future__ import annotations

from dataclasses import dataclass

from scout_research_r006_pilot_execution_engine import Bar

from scout_auto_os.engine.position_evaluation.side_rules import (
    mfe_mae_from_bars,
    roi_pct,
    side_state_signals,
)
from scout_auto_os.engine.position_evaluation.thesis import TradeThesis


@dataclass
class EvaluationMetrics:
    current_roi: float
    elapsed_minutes: int
    progress_vs_expected_return: float
    progress_vs_expected_time: float
    mfe: float
    mae: float
    peak_roi: float
    drawdown_from_peak: float
    momentum_alive: bool
    volume_alive: bool
    trend_alive: bool
    range_expansion: bool
    reversal_warning: bool
    thesis_validity_score: float
    exit_pressure_score: float
    hold_confidence: float
    current_price: float


class PositionEvaluator:
    def __init__(self) -> None:
        self._peak_roi: dict[str, float] = {}

    def reset_peak(self, position_id: str) -> None:
        self._peak_roi.pop(position_id, None)

    def evaluate(
        self,
        thesis: TradeThesis,
        position_id: str,
        side: str,
        entry_price: float,
        current_price: float,
        elapsed_minutes: int,
        bars: list[Bar],
    ) -> EvaluationMetrics:
        cur_roi = roi_pct(side, entry_price, current_price)
        peak = self._peak_roi.get(position_id, cur_roi)
        if cur_roi > peak:
            peak = cur_roi
            self._peak_roi[position_id] = peak

        mfe, mae = mfe_mae_from_bars(side, bars, entry_price)
        sig = side_state_signals(side, bars, 0)

        exp_ret = max(thesis.expected_return_pct, 0.01)
        exp_hor = max(thesis.expected_horizon_min, 1)
        prog_ret = round(cur_roi / exp_ret * 100, 2)
        prog_time = round(elapsed_minutes / exp_hor * 100, 2)
        dd_peak = round(peak - cur_roi, 4)

        validity = self._thesis_validity(thesis, cur_roi, elapsed_minutes, prog_ret, prog_time, sig)
        exit_pressure = self._exit_pressure(thesis, cur_roi, elapsed_minutes, prog_ret, prog_time, sig, dd_peak)
        hold_conf = round(max(0.0, min(100.0, validity - exit_pressure * 0.6)), 2)

        return EvaluationMetrics(
            current_roi=cur_roi,
            elapsed_minutes=elapsed_minutes,
            progress_vs_expected_return=prog_ret,
            progress_vs_expected_time=prog_time,
            mfe=mfe,
            mae=abs(mae),
            peak_roi=peak,
            drawdown_from_peak=dd_peak,
            momentum_alive=sig["momentum_alive"],
            volume_alive=sig["volume_alive"],
            trend_alive=sig["trend_alive"],
            range_expansion=sig["range_expansion"],
            reversal_warning=sig["reversal_warning"],
            thesis_validity_score=validity,
            exit_pressure_score=exit_pressure,
            hold_confidence=hold_conf,
            current_price=current_price,
        )

    def _thesis_validity(self, thesis, roi, elapsed, prog_ret, prog_time, sig) -> float:
        score = 70.0
        if prog_ret >= 80:
            score += 15
        elif prog_ret >= 50:
            score += 8
        elif prog_ret < 25 and prog_time >= 100:
            score -= 25
        if sig["momentum_alive"]:
            score += 8
        if sig["trend_alive"]:
            score += 5
        if sig["reversal_warning"]:
            score -= 15
        if roi <= -thesis.initial_stop_pct:
            score -= 30
        return round(max(0.0, min(100.0, score)), 2)

    def _exit_pressure(self, thesis, roi, elapsed, prog_ret, prog_time, sig, dd_peak) -> float:
        pressure = 0.0
        if elapsed >= thesis.max_hold_minutes:
            pressure += 50
        if prog_time >= 100 and prog_ret < 50:
            pressure += 30
        if prog_time >= 100 and prog_ret < 25:
            pressure += 20
        if sig["reversal_warning"]:
            pressure += 20
        if not sig["momentum_alive"] and elapsed >= thesis.expected_horizon_min:
            pressure += 15
        if dd_peak >= 5 and roi >= thesis.expected_return_pct:
            pressure += 10
        if roi <= -thesis.initial_stop_pct:
            pressure += 40
        if elapsed >= thesis.expected_horizon_min * 2 and roi < thesis.expected_return_pct * 0.5:
            pressure += 25
        return round(min(100.0, pressure), 2)
