"""Minimal rule-based recommendation from Guardian state + metrics."""

from __future__ import annotations

from scout_auto_os.engine.guardian.guardian_actions import GuardianAction
from scout_auto_os.engine.guardian.progress_metrics import ProgressMetrics

EMERGENCY_ROI_PCT = -15.0
EMERGENCY_DD_PRESSURE = 1.5


def recommend_action(
    state: str,
    metrics: ProgressMetrics,
    contract: dict,
    *,
    manual: bool = False,
) -> tuple[str, list[str]]:
    """Return (recommendation, reason_lines)."""
    lines: list[str] = [f"guardian_state={state}"]

    if manual or contract.get("gate_action") == "NO_ACTION":
        lines.append("manual or protected — NO_ACTION")
        return GuardianAction.NO_ACTION.value, lines

    if metrics.current_roi <= EMERGENCY_ROI_PCT:
        lines.append(f"current_roi {metrics.current_roi:.2f}% <= emergency floor {EMERGENCY_ROI_PCT}%")
        return GuardianAction.EMERGENCY_EXIT.value, lines

    if metrics.drawdown_pressure >= EMERGENCY_DD_PRESSURE:
        lines.append(f"drawdown_pressure {metrics.drawdown_pressure:.2f} >= {EMERGENCY_DD_PRESSURE}")
        return GuardianAction.EMERGENCY_EXIT.value, lines

    if state == "THESIS_FAILED":
        lines.append("thesis failed — exit to stop holding invalid entry logic")
        return GuardianAction.EXIT.value, lines

    if state == "THESIS_WEAKENING":
        if metrics.drawdown_pressure >= 1.0:
            lines.append("weakening with drawdown over contract limit — EXIT")
            return GuardianAction.EXIT.value, lines
        lines.append("thesis weakening — reduce exposure or prepare trail")
        return GuardianAction.REDUCE.value, lines

    if state == "LATE":
        if metrics.progress_ratio < 0.5:
            lines.append("late and under half target — EXIT")
            return GuardianAction.EXIT.value, lines
        lines.append("past horizon with partial progress — REDUCE")
        return GuardianAction.REDUCE.value, lines

    if state in ("AHEAD", "COMPLETED"):
        lines.append("target met or exceeded — TRAIL to protect gains")
        return GuardianAction.TRAIL.value, lines

    if state == "NOT_STARTED":
        lines.append("no elapsed time — observe")
        return GuardianAction.NO_ACTION.value, lines

    lines.append("thesis still valid — HOLD")
    return GuardianAction.HOLD.value, lines
