"""Thesis lifecycle state machine."""

from __future__ import annotations

from dataclasses import dataclass

from scout_auto_os.engine.expectation.expectation_score import ExpectationScoreResult
from scout_auto_os.engine.expectation.progress_tracker import ProgressSnapshot

STATES = (
    "EARLY",
    "ON_TRACK",
    "OUTPERFORM",
    "UNDERPERFORM",
    "THESIS_COMPLETE",
    "THESIS_FAILED",
    "EXIT_READY",
)


@dataclass
class ThesisStateResult:
    state: str
    transition_reason: str
    prior_state: str


def compute_thesis_state(
    progress: ProgressSnapshot,
    score: ExpectationScoreResult,
    *,
    prior_state: str = "EARLY",
    elapsed_min: int,
    expected_horizon: int,
    expected_return: float,
    current_roi: float,
    max_hold_min: int,
) -> ThesisStateResult:
    ratio = progress.progress_ratio
    reason = ""

    if prior_state == "EXIT_READY":
        return ThesisStateResult("EXIT_READY", "maintain_exit_ready", prior_state)

    if elapsed_min < min(30, expected_horizon // 4):
        return ThesisStateResult("EARLY", "within_first_quarter_horizon", prior_state)

    if current_roi >= expected_return and ratio >= 100:
        if score.score >= 70 and elapsed_min <= expected_horizon * 1.2:
            return ThesisStateResult("THESIS_COMPLETE", "target_return_met_on_schedule", prior_state)
        return ThesisStateResult("THESIS_COMPLETE", "target_return_met", prior_state)

    if ratio >= 130 and current_roi >= expected_return * 0.8:
        return ThesisStateResult("OUTPERFORM", "progress_ratio_above_130", prior_state)

    if ratio < 55 and elapsed_min >= expected_horizon * 0.8:
        if score.score < 40:
            return ThesisStateResult(
                "THESIS_FAILED",
                "slow_progress_low_expectation_score",
                prior_state,
            )
        return ThesisStateResult("UNDERPERFORM", "progress_ratio_below_55", prior_state)

    if ratio < 75 and elapsed_min >= expected_horizon:
        return ThesisStateResult("UNDERPERFORM", "horizon_passed_under_75pct", prior_state)

    if score.score < 25 or (elapsed_min >= max_hold_min and ratio < 80):
        return ThesisStateResult("EXIT_READY", "expectation_score_critical_or_max_hold", prior_state)

    if score.score < 40 and elapsed_min >= expected_horizon * 1.5:
        return ThesisStateResult("THESIS_FAILED", "met_overstay_pattern", prior_state)

    if 75 <= ratio <= 130:
        return ThesisStateResult("ON_TRACK", "progress_within_expected_band", prior_state)

    if ratio > 100:
        return ThesisStateResult("OUTPERFORM", "ahead_of_curve", prior_state)

    return ThesisStateResult("UNDERPERFORM", "behind_expected_curve", prior_state)
