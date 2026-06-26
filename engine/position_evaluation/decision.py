"""Thesis-based action decision."""

from __future__ import annotations

from dataclasses import dataclass

from scout_auto_os.engine.position_evaluation.evaluator import EvaluationMetrics
from scout_auto_os.engine.position_evaluation.thesis import TradeThesis

try:
    from scout_auto_os.engine.expectation.runner import ExpectationReview
except ImportError:
    ExpectationReview = None  # type: ignore


ACTIONS = (
    "HOLD",
    "REDUCE",
    "EXIT",
    "TRAIL",
    "STOP_TIGHTEN",
    "TAKE_PROFIT",
    "THESIS_UPDATE",
    "NO_ACTION_MANUAL_POSITION",
)


@dataclass
class PositionDecision:
    action: str
    should_exit: bool
    reason_lines: list[str]
    thesis_update_reason: str = ""

    @property
    def reason(self) -> str:
        return " | ".join(self.reason_lines)

    @property
    def action_reason(self) -> str:
        return "\n".join(f"- {line}" for line in self.reason_lines)


def decide(
    metrics: EvaluationMetrics,
    thesis: TradeThesis,
    *,
    is_manual: bool = False,
    expectation: "ExpectationReview | None" = None,
) -> PositionDecision:
    if is_manual or thesis.source.upper() == "MANUAL" or not thesis.auto_manage:
        return PositionDecision(
            action="NO_ACTION_MANUAL_POSITION",
            should_exit=False,
            reason_lines=["manual position — observe only, no bot action"],
        )

    lines = []
    if expectation:
        p = expectation.progress
        lines.extend([
            f"expected_roi_now {p.expected_roi_now}% vs current {p.current_roi}%",
            f"progress_ratio {p.progress_ratio}% progress_delta {p.progress_delta}%",
            f"expectation_score {expectation.score.score} thesis_state {expectation.state.state}",
        ])
    lines.extend([
        f"expected_horizon {thesis.expected_horizon_min}m — elapsed {metrics.elapsed_minutes}m",
        f"expected_return {thesis.expected_return_pct}% — current ROI {metrics.current_roi}%",
        f"progress_return {metrics.progress_vs_expected_return}% progress_time {metrics.progress_vs_expected_time}%",
        f"momentum_alive {metrics.momentum_alive} trend_alive {metrics.trend_alive}",
        f"drawdown_from_peak {metrics.drawdown_from_peak}%",
        f"thesis_validity {metrics.thesis_validity_score} exit_pressure {metrics.exit_pressure_score}",
    ])

    if expectation:
        st = expectation.state.state
        if st == "EXIT_READY":
            return PositionDecision("EXIT", True, lines + ["action: EXIT — expectation EXIT_READY"])
        if st == "THESIS_FAILED":
            if metrics.exit_pressure_score >= 45 or expectation.score.score < 35:
                return PositionDecision("EXIT", True, lines + ["action: EXIT — THESIS_FAILED"])
            return PositionDecision("STOP_TIGHTEN", False, lines + ["action: STOP_TIGHTEN — THESIS_FAILED"])
        if st == "THESIS_COMPLETE" and expectation.extension:
            return PositionDecision(
                "TRAIL", False, lines + ["action: TRAIL — extension thesis trailing mode"],
                thesis_update_reason=expectation.extension.reason,
            )
        if st == "OUTPERFORM" and metrics.momentum_alive:
            return PositionDecision("TRAIL", False, lines + ["action: TRAIL — OUTPERFORM"])
        if st == "UNDERPERFORM" and metrics.elapsed_minutes >= thesis.expected_horizon_min:
            if expectation.progress.progress_ratio < 50:
                return PositionDecision("EXIT", True, lines + ["action: EXIT — underperform vs expected curve"])
            return PositionDecision("STOP_TIGHTEN", False, lines + ["action: STOP_TIGHTEN — underperform"])

    if metrics.current_roi <= -thesis.initial_stop_pct and metrics.elapsed_minutes >= 30:
        lines.append("protective stop breached")
        return PositionDecision("EXIT", True, lines + ["action: EXIT — stop loss"])

    if metrics.elapsed_minutes >= thesis.max_hold_minutes:
        lines.append("max_hold exceeded")
        return PositionDecision("EXIT", True, lines + ["action: EXIT — max_hold forced review"])

    if metrics.progress_vs_expected_time >= 100 and metrics.progress_vs_expected_return < 25:
        lines.append("horizon exceeded with weak return — thesis weakened")
        if metrics.exit_pressure_score >= 55:
            return PositionDecision("EXIT", True, lines + ["action: EXIT — thesis failure"])
        return PositionDecision("STOP_TIGHTEN", False, lines + ["action: STOP_TIGHTEN — thesis weak"])

    if metrics.current_roi >= thesis.initial_take_profit_pct:
        lines.append("take profit zone reached")
        return PositionDecision("TAKE_PROFIT", True, lines + ["action: TAKE_PROFIT"])

    if metrics.current_roi >= thesis.expected_return_pct and metrics.momentum_alive:
        lines.append("target exceeded with momentum — trail profits")
        return PositionDecision("TRAIL", False, lines + ["action: TRAIL — protect peak ROI"])

    if metrics.current_roi >= thesis.expected_return_pct * 1.5 and metrics.drawdown_from_peak >= 3:
        lines.append("peak giveback after extended profit")
        return PositionDecision("EXIT", True, lines + ["action: EXIT — peak trail breach"])

    if metrics.reversal_warning and metrics.current_roi >= thesis.expected_return_pct:
        lines.append("reversal warning after target")
        return PositionDecision("TRAIL", False, lines + ["action: TRAIL — reversal warning"])

    if metrics.progress_vs_expected_return >= 100 and not metrics.momentum_alive:
        lines.append("target met but momentum lost")
        return PositionDecision("THESIS_UPDATE", False, lines + ["action: THESIS_UPDATE — downgrade hold"])

    if metrics.exit_pressure_score >= 70:
        return PositionDecision("EXIT", True, lines + ["action: EXIT — high exit pressure"])

    return PositionDecision("HOLD", False, lines + ["action: HOLD"])
