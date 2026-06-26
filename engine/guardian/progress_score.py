"""Guardian score 0–100 from weighted progress components."""

from __future__ import annotations

from scout_auto_os.engine.guardian.progress_config import GuardianProgressWeights
from scout_auto_os.engine.guardian.progress_metrics import ProgressMetrics


def _clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


def roi_progress_component(m: ProgressMetrics) -> float:
    """0–100: how much of expected ROI is delivered."""
    if m.expected_roi <= 0:
        return _clamp(50.0 + m.current_roi * 2)
    ratio = m.progress_ratio
    if ratio >= 1.0:
        return _clamp(70.0 + min(ratio - 1.0, 0.5) * 60.0)
    return _clamp(ratio * 100.0)


def time_alignment_component(m: ProgressMetrics) -> float:
    """0–100: ROI pace vs time pace (aligned = high)."""
    if m.time_progress <= 0:
        return 50.0
    pace = m.progress_ratio / m.time_progress
    return _clamp(pace * 50.0)


def drawdown_health_component(m: ProgressMetrics) -> float:
    """0–100: lower drawdown pressure = healthier."""
    return _clamp(100.0 - m.drawdown_pressure * 100.0)


def value_score_component(contract: dict) -> float:
    return _clamp(float(contract.get("value_score", 50)))


def win_probability_component(contract: dict) -> float:
    return _clamp(float(contract.get("expected_win_prob", 0.5)) * 100.0)


def compute_guardian_score(
    metrics: ProgressMetrics,
    contract: dict,
    weights: GuardianProgressWeights | None = None,
) -> float:
    w = (weights or GuardianProgressWeights()).normalized()
    score = (
        roi_progress_component(metrics) * w.roi_progress
        + time_alignment_component(metrics) * w.time_alignment
        + drawdown_health_component(metrics) * w.drawdown_health
        + value_score_component(contract) * w.value_score
        + win_probability_component(contract) * w.win_probability
    )
    return round(_clamp(score), 2)
