"""Guardian state classification — explainable, rule-based."""

from __future__ import annotations

from scout_auto_os.engine.guardian.progress_metrics import ProgressMetrics

STATES = (
    "NOT_STARTED",
    "BUILDING",
    "ON_TRACK",
    "AHEAD",
    "LATE",
    "THESIS_WEAKENING",
    "THESIS_FAILED",
    "COMPLETED",
)

MET_HORIZON_MULT = 2.0
MET_PEAK_GIVEBACK_PCT = 5.0


def classify_guardian_state(
    metrics: ProgressMetrics,
    contract: dict,
) -> tuple[str, list[str]]:
    """Return (state, reason_fragments)."""
    lines: list[str] = []

    if metrics.elapsed_minutes <= 0:
        lines.append("elapsed=0 — position not yet evaluated")
        return "NOT_STARTED", lines

    peak_giveback = metrics.peak_roi - metrics.current_roi

    # MET-style extended hold failure
    met_like = (
        metrics.time_progress >= MET_HORIZON_MULT
        and metrics.progress_ratio < 0.5
        and peak_giveback >= MET_PEAK_GIVEBACK_PCT
    )
    if met_like:
        lines.append(
            f"time_progress {metrics.time_progress:.1f}x exceeds {MET_HORIZON_MULT}x horizon "
            f"with progress {metrics.progress_ratio:.2f} and peak giveback {peak_giveback:.2f}%"
        )
        return "THESIS_FAILED", lines

    # Hard thesis failure
    if metrics.time_progress > 1.0 and metrics.progress_ratio < 0.3:
        lines.append(
            f"past horizon (time_progress {metrics.time_progress:.2f}) "
            f"with very weak progress {metrics.progress_ratio:.2f}"
        )
        return "THESIS_FAILED", lines

    if metrics.drawdown_pressure >= 1.0:
        lines.append(
            f"drawdown_pressure {metrics.drawdown_pressure:.2f} >= 1.0 — "
            f"giveback exceeds contract tolerance"
        )
        return "THESIS_FAILED", lines

    # Completed target
    if metrics.progress_ratio >= 1.0 and metrics.drawdown_pressure < 0.5:
        lines.append(
            f"progress {metrics.progress_ratio:.2f} reached expected ROI "
            f"with low drawdown pressure {metrics.drawdown_pressure:.2f}"
        )
        return "COMPLETED", lines

    # Ahead of schedule
    if metrics.progress_ratio > 1.0 or metrics.peak_progress > 1.0:
        lines.append(
            f"ahead: progress {metrics.progress_ratio:.2f} or peak_progress "
            f"{metrics.peak_progress:.2f} exceeds contract target"
        )
        return "AHEAD", lines

    # Weakening before failure
    if metrics.drawdown_pressure >= 0.6:
        lines.append(
            f"drawdown_pressure {metrics.drawdown_pressure:.2f} >= 0.6 — "
            f"momentum weakening vs entry thesis"
        )
        return "THESIS_WEAKENING", lines

    if metrics.time_progress > 0.8 and metrics.progress_ratio < 0.5:
        lines.append(
            f"time {metrics.time_progress:.2f} advanced but ROI progress "
            f"{metrics.progress_ratio:.2f} still below half target"
        )
        return "THESIS_WEAKENING", lines

    # Late but not yet failed
    if metrics.time_progress > 1.0 and metrics.progress_ratio < 0.8:
        lines.append(
            f"horizon exceeded (time_progress {metrics.time_progress:.2f}) "
            f"without reaching 80% of expected ROI"
        )
        return "LATE", lines

    # Early build phase
    if metrics.progress_ratio < 0.3 and metrics.time_progress < 0.5:
        lines.append(
            f"early phase: progress {metrics.progress_ratio:.2f}, "
            f"time {metrics.time_progress:.2f} — thesis building"
        )
        return "BUILDING", lines

    lines.append(
        f"progress {metrics.progress_ratio:.2f} and time {metrics.time_progress:.2f} "
        f"within expected band for {contract.get('exit_profile', '')} profile"
    )
    return "ON_TRACK", lines
