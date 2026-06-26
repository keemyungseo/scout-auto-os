"""Guardian progress ratio calculations."""

from __future__ import annotations

from dataclasses import dataclass

from scout_auto_os.engine.guardian.decision_rules import expected_horizon_minutes


def _safe_div(numerator: float, denominator: float, *, floor: float = 0.01) -> float:
    denom = denominator if abs(denominator) >= floor else floor
    return numerator / denom


@dataclass(frozen=True)
class ProgressMetrics:
    progress_ratio: float
    time_progress: float
    peak_progress: float
    drawdown_pressure: float
    current_roi: float
    peak_roi: float
    drawdown_from_peak: float
    elapsed_minutes: int
    expected_roi: float
    expected_peak_roi: float
    expected_drawdown: float
    expected_horizon: int


def normalize_contract(contract: dict) -> dict:
    """Ensure contract has expected_horizon and contract_id."""
    out = dict(contract)
    if not out.get("expected_horizon"):
        out["expected_horizon"] = expected_horizon_minutes(out)
    if not out.get("contract_id"):
        sym = out.get("symbol", "")
        side = out.get("side", "long")
        out["contract_id"] = f"{sym}|{side}"
    return out


def compute_progress_metrics(contract: dict, position: dict) -> ProgressMetrics:
    c = normalize_contract(contract)
    expected_roi = float(c.get("expected_roi", 0))
    expected_peak = float(c.get("expected_peak_roi", expected_roi))
    expected_dd = float(c.get("expected_drawdown", 1.0))
    horizon = int(c.get("expected_horizon") or position.get("expected_horizon") or 120)

    current_roi = float(position.get("current_roi", 0))
    peak_roi = float(position.get("peak_roi", current_roi))
    dd_from_peak = float(position.get("drawdown_from_peak", max(0.0, peak_roi - current_roi)))
    elapsed = int(position.get("elapsed_minutes", 0))

    return ProgressMetrics(
        progress_ratio=_safe_div(current_roi, expected_roi, floor=0.1),
        time_progress=_safe_div(elapsed, horizon, floor=1.0),
        peak_progress=_safe_div(peak_roi, expected_peak, floor=0.1),
        drawdown_pressure=_safe_div(dd_from_peak, expected_dd, floor=1.0),
        current_roi=current_roi,
        peak_roi=peak_roi,
        drawdown_from_peak=dd_from_peak,
        elapsed_minutes=elapsed,
        expected_roi=expected_roi,
        expected_peak_roi=expected_peak,
        expected_drawdown=expected_dd,
        expected_horizon=horizon,
    )
