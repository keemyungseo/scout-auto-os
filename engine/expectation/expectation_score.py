"""Expectation Score 0-100."""

from __future__ import annotations

from dataclasses import dataclass

from scout_auto_os.engine.expectation.progress_tracker import ProgressSnapshot


@dataclass
class ExpectationScoreResult:
    score: float
    label: str
    components: dict


def compute_expectation_score(
    progress: ProgressSnapshot,
    *,
    momentum_alive: bool,
    trend_alive: bool,
    volume_alive: bool,
    peak_roi: float,
    expected_peak_window: int,
    elapsed_min: int,
    exit_pressure: float,
    entry_success_prob: float,
) -> ExpectationScoreResult:
    ratio = progress.progress_ratio
    roi_component = _clamp(ratio, 0, 150) / 150 * 35
    time_component = _clamp(100 - abs(progress.time_progress_ratio - min(ratio, 120)), 0, 100) / 100 * 15

    prob_shift = (ratio - 100) / 200
    prob_component = _clamp(entry_success_prob + prob_shift, 0, 1) * 15

    alive = 0
    if momentum_alive:
        alive += 8
    if trend_alive:
        alive += 7
    if volume_alive:
        alive += 5

    peak_near = 1.0 if elapsed_min >= expected_peak_window * 0.7 else 0.5
    peak_component = _clamp(peak_roi / max(progress.expected_roi_now, 0.01), 0, 2) * 10 * peak_near

    pressure_penalty = _clamp(exit_pressure, 0, 100) / 100 * 20

    raw = roi_component + time_component + prob_component + alive + peak_component - pressure_penalty
    score = round(_clamp(raw, 0, 100), 2)

    if score >= 85:
        label = "well_above_expectation"
    elif score >= 65:
        label = "on_track"
    elif score >= 40:
        label = "below_expectation"
    else:
        label = "thesis_failure_zone"

    return ExpectationScoreResult(
        score=score,
        label=label,
        components={
            "roi_progress": round(roi_component, 2),
            "time_alignment": round(time_component, 2),
            "success_prob": round(prob_component, 2),
            "alive_signals": alive,
            "peak_approach": round(peak_component, 2),
            "exit_pressure_penalty": round(pressure_penalty, 2),
        },
    )


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))
