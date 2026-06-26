"""Trade Value Score 0-100."""

from __future__ import annotations

import statistics


def _clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


def _norm_positive(v: float, p95: float) -> float:
    if p95 <= 0:
        return 50.0
    return _clamp(v / p95 * 100)


def _norm_inverse(v: float, p95: float) -> float:
    if p95 <= 0:
        return 50.0
    return _clamp(100 - v / p95 * 100)


def fit_norm_params(rows: list[dict], key: str) -> dict:
    vals = [abs(float(r["y"][key])) for r in rows]
    if not vals:
        return {"p95": 1.0, "median": 0.0}
    vals.sort()
    p95 = vals[int(len(vals) * 0.95)] if len(vals) > 1 else vals[0]
    return {"p95": max(p95, 0.01), "median": statistics.median(vals)}


def compute_value_score(
    pred: dict,
    norms: dict,
) -> float:
    """0-100 from predicted ROI, win prob, drawdown, hold efficiency, sharpe."""
    roi_n = _norm_positive(float(pred.get("expected_roi", 0)), norms["roi_p95"])
    win_n = _clamp(float(pred.get("expected_win_prob", 0)) * 100)
    dd_n = _norm_inverse(float(pred.get("expected_drawdown", 0)), norms["dd_p95"])
    peak = float(pred.get("expected_peak_roi", 0))
    roi = float(pred.get("expected_roi", 0))
    hold_eff = _clamp(roi / peak * 100 if abs(peak) > 0.01 else 0)
    sharpe_n = _norm_positive(float(pred.get("expected_sharpe_contrib", 0)), norms["sharpe_p95"])

    score = (
        roi_n * 0.30
        + win_n * 0.25
        + dd_n * 0.20
        + hold_eff * 0.15
        + sharpe_n * 0.10
    )
    return round(_clamp(score), 2)


def size_multiplier(score: float) -> float:
    if score >= 90:
        return 1.0
    if score >= 80:
        return 0.8
    if score >= 70:
        return 0.6
    if score >= 60:
        return 0.3
    return 0.0
