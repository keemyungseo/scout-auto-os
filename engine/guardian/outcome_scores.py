"""Rule-based Guardian outcome scores and grades — explainable, no ML."""

from __future__ import annotations

from dataclasses import dataclass

from scout_auto_os.engine.guardian.outcome_metrics import TradeOutcomeFacts

GRADES = ("EXCELLENT", "GOOD", "NORMAL", "POOR", "FAILED")


@dataclass
class GuardianOutcomeEvaluation:
    facts: TradeOutcomeFacts
    exit_timing_score: float
    trail_timing_score: float
    hold_quality_score: float
    drawdown_control_score: float
    contract_adherence_score: float
    overall_guardian_score: float
    outcome_grade: str
    explanation: str

    def to_row(self) -> dict:
        f = self.facts
        return {
            "trade_id": f.trade_id,
            "symbol": f.symbol,
            "side": f.side,
            "entry_time": f.entry_time,
            "exit_time": f.exit_time,
            "hold_minutes": f.hold_minutes,
            "final_roi": f.final_roi,
            "peak_roi": f.peak_roi,
            "max_drawdown": f.max_drawdown,
            "hold_count": f.hold_count,
            "trail_start_minutes": f.trail_start_minutes if f.trail_start_minutes is not None else "",
            "reduce_count": f.reduce_count,
            "exit_minutes": f.exit_minutes if f.exit_minutes is not None else "",
            "emergency": int(f.emergency),
            "final_recommendation": f.final_recommendation,
            "final_state": f.final_state,
            "exit_timing_score": self.exit_timing_score,
            "trail_timing_score": self.trail_timing_score,
            "hold_quality_score": self.hold_quality_score,
            "drawdown_control_score": self.drawdown_control_score,
            "contract_adherence_score": self.contract_adherence_score,
            "overall_guardian_score": self.overall_guardian_score,
            "outcome_grade": self.outcome_grade,
            "explanation": self.explanation,
            "entry_reason": f.entry_reason,
            "predicted_dna": f.predicted_dna,
            "confidence": f.confidence,
            "expected_roi": f.expected_roi,
            "expected_horizon": f.expected_horizon,
        }


def _clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


def _grade(score: float) -> str:
    if score >= 85:
        return "EXCELLENT"
    if score >= 70:
        return "GOOD"
    if score >= 50:
        return "NORMAL"
    if score >= 30:
        return "POOR"
    return "FAILED"


def score_exit_timing(f: TradeOutcomeFacts) -> float:
    """Peak capture + timely exit recommendation."""
    if f.peak_roi > 0.5:
        peak_capture = f.final_roi / f.peak_roi
    elif f.peak_roi < -0.5:
        peak_capture = 1.0 if f.final_roi >= f.peak_roi else 0.3
    else:
        peak_capture = 0.5

    score = peak_capture * 70.0
    if f.exit_minutes is not None and f.exit_minutes < f.hold_minutes:
        score += 15.0
    if f.expected_roi > 0 and f.final_roi >= f.expected_roi * 0.8:
        score += 15.0
    elif f.expected_roi > 0 and f.final_roi < 0:
        score -= 20.0
    if f.emergency:
        score -= 10.0
    return round(_clamp(score), 2)


def score_trail_timing(f: TradeOutcomeFacts) -> float:
    """TRAIL when ahead of contract — earlier trail on strong peak."""
    if f.trail_start_minutes is None:
        if f.peak_roi > f.expected_peak_roi and f.final_roi < f.peak_roi * 0.7:
            return 35.0
        return 55.0

    if f.peak_roi <= 0:
        return 50.0

    trail_roi_ratio = f.trail_start_minutes / max(f.hold_minutes, 1)
    peak_before_trail = f.peak_roi >= f.expected_peak_roi * 0.8
    score = 50.0
    if peak_before_trail:
        score += 25.0
    if 0.2 <= trail_roi_ratio <= 0.75:
        score += 20.0
    if f.final_roi >= f.peak_roi * 0.75:
        score += 10.0
    return round(_clamp(score), 2)


def score_hold_quality(f: TradeOutcomeFacts, points: list[dict]) -> float:
    """Healthy states vs weakening/failed share."""
    if not points:
        return 50.0
    good = sum(
        1 for p in points
        if p.get("guardian_state") in ("BUILDING", "ON_TRACK", "COMPLETED", "AHEAD")
    )
    bad = sum(
        1 for p in points
        if p.get("guardian_state") in ("THESIS_WEAKENING", "THESIS_FAILED", "LATE")
    )
    total = good + bad
    if total == 0:
        return 50.0
    ratio = good / total
    hold_bonus = min(15.0, f.hold_count * 2.0) if ratio > 0.5 else 0.0
    return round(_clamp(ratio * 85.0 + hold_bonus), 2)


def score_drawdown_control(f: TradeOutcomeFacts) -> float:
    """Actual max drawdown vs contract tolerance."""
    expected = max(f.expected_drawdown, 1.0)
    ratio = f.max_drawdown / expected
    if ratio <= 0.5:
        return 95.0
    if ratio <= 1.0:
        return round(90.0 - (ratio - 0.5) * 40.0, 2)
    return round(_clamp(50.0 - (ratio - 1.0) * 30.0), 2)


def score_contract_adherence(f: TradeOutcomeFacts) -> float:
    """Final ROI vs contract expectation."""
    if f.expected_roi <= 0:
        if f.final_roi >= f.expected_roi:
            return 70.0
        return round(_clamp(50.0 + f.final_roi * 2.0), 2)

    progress = f.final_roi / f.expected_roi
    score = min(100.0, progress * 80.0)
    if f.final_state in ("COMPLETED", "AHEAD") and progress >= 0.8:
        score += 15.0
    if f.final_state == "THESIS_FAILED":
        score -= 25.0
    return round(_clamp(score), 2)


def build_explanation(
    f: TradeOutcomeFacts,
    scores: dict[str, float],
    grade: str,
) -> str:
    parts = [
        f"grade={grade} overall={scores['overall']:.1f}",
        f"trade {f.symbol} {f.side}: final_roi={f.final_roi:.2f}% peak={f.peak_roi:.2f}% max_dd={f.max_drawdown:.2f}%",
        f"hold={f.hold_minutes}m trail_start={f.trail_start_minutes} exit_rec={f.exit_minutes} emergency={f.emergency}",
        f"scores exit={scores['exit']:.0f} trail={scores['trail']:.0f} hold={scores['hold']:.0f} "
        f"dd={scores['dd']:.0f} contract={scores['contract']:.0f}",
    ]
    if grade == "EXCELLENT":
        parts.append(
            "Excellent: strong peak capture, drawdown within contract, states aligned with thesis."
        )
    elif grade == "FAILED":
        parts.append(
            "Failed: large giveback from peak, THESIS_FAILED dominance, or exit timing missed deterioration."
        )
    elif f.peak_roi > 0 and f.final_roi < f.peak_roi * 0.5:
        parts.append("Peak giveback exceeded half of peak ROI — trail/exit timing could improve.")
    if f.entry_reason:
        parts.append(f"entry_reason: {f.entry_reason[:120]}")
    return " | ".join(parts)


def evaluate_trade_outcome(
    facts: TradeOutcomeFacts,
    points: list[dict],
) -> GuardianOutcomeEvaluation:
    exit_s = score_exit_timing(facts)
    trail_s = score_trail_timing(facts)
    hold_s = score_hold_quality(facts, points)
    dd_s = score_drawdown_control(facts)
    contract_s = score_contract_adherence(facts)

    overall = round(
        exit_s * 0.25
        + trail_s * 0.20
        + hold_s * 0.20
        + dd_s * 0.20
        + contract_s * 0.15,
        2,
    )
    grade = _grade(overall)
    explanation = build_explanation(
        facts,
        {"exit": exit_s, "trail": trail_s, "hold": hold_s, "dd": dd_s, "contract": contract_s, "overall": overall},
        grade,
    )

    return GuardianOutcomeEvaluation(
        facts=facts,
        exit_timing_score=exit_s,
        trail_timing_score=trail_s,
        hold_quality_score=hold_s,
        drawdown_control_score=dd_s,
        contract_adherence_score=contract_s,
        overall_guardian_score=overall,
        outcome_grade=grade,
        explanation=explanation,
    )
