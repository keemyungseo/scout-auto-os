"""Portfolio value scoring — rule-based, no ML."""

from __future__ import annotations

from scout_auto_os.engine.portfolio.decision.models import PortfolioPosition, PredatorCandidate

DEFAULT_POSITION_WEIGHTS = {
    "guardian_score": 0.30,
    "expected_roi": 0.20,
    "current_roi": 0.20,
    "remaining_opportunity": 0.15,
    "confidence": 0.15,
}

DEFAULT_CANDIDATE_WEIGHTS = {
    "value_score": 0.35,
    "expected_roi": 0.25,
    "win_prob": 0.20,
    "confidence": 0.20,
}


def _weights(config: dict | None, key: str, default: dict) -> dict:
    if not config:
        return default
    section = config.get("portfolio_decision", {})
    return section.get(key, default)


def _clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


def normalize_roi(roi: float) -> float:
    """Map ROI % to 0–100 scale (empirical range -30..50)."""
    return _clamp((roi + 30.0) / 80.0 * 100.0)


def remaining_opportunity(expected_roi: float, current_roi: float) -> float:
    if expected_roi > 1.0:
        return _clamp(max(0.0, expected_roi - current_roi) / expected_roi * 100.0)
    if expected_roi < -1.0:
        return _clamp(50.0 + (current_roi - expected_roi))
    return _clamp(50.0 + current_roi)


def score_position(position: PortfolioPosition, config: dict | None = None) -> float:
    w = _weights(config, "position_weights", DEFAULT_POSITION_WEIGHTS)
    gs = _clamp(position.guardian_score)
    er = normalize_roi(position.expected_roi)
    cr = normalize_roi(position.current_roi)
    ro = remaining_opportunity(position.expected_roi, position.current_roi)
    conf = _clamp(position.confidence)

    score = (
        gs * w["guardian_score"]
        + er * w["expected_roi"]
        + cr * w["current_roi"]
        + ro * w["remaining_opportunity"]
        + conf * w["confidence"]
    )
    return round(_clamp(score), 2)


def score_candidate(candidate: PredatorCandidate, config: dict | None = None) -> float:
    w = _weights(config, "candidate_weights", DEFAULT_CANDIDATE_WEIGHTS)
    vs = _clamp(candidate.value_score)
    er = normalize_roi(candidate.expected_roi)
    wp = _clamp(candidate.expected_win_prob * 100.0)
    conf = _clamp(candidate.confidence)

    score = (
        vs * w["value_score"]
        + er * w["expected_roi"]
        + wp * w["win_prob"]
        + conf * w["confidence"]
    )
    return round(_clamp(score), 2)


def rescore_book_positions(positions: list[PortfolioPosition], config: dict | None = None) -> None:
    for p in positions:
        p.portfolio_value_score = score_position(p, config)
