"""Rule-based Guardian metrics and decision thresholds."""

from __future__ import annotations

from dataclasses import dataclass

from scout_auto_os.engine.guardian.guardian_actions import GuardianAction

# Progress bands
PROGRESS_EARLY_MAX = 0.5
PROGRESS_TARGET_MIN = 0.8
PROGRESS_TARGET_MAX = 1.0

# Pressure thresholds
DRAWDOWN_EXIT_PRESSURE = 1.0
DRAWDOWN_EMERGENCY_PRESSURE = 1.5
DRAWDOWN_WEAKENING_PRESSURE = 0.6

# Time
HORIZON_REVIEW_RATIO = 1.0
HORIZON_MET_RATIO = 2.0

# ROI floors
EMERGENCY_ROI_PCT = -15.0
MET_PEAK_GIVEBACK_PCT = 5.0


@dataclass(frozen=True)
class GuardianMetrics:
    progress_ratio: float
    time_progress: float
    drawdown_pressure: float
    overperformance: bool
    current_roi: float
    expected_roi: float
    expected_peak_roi: float
    expected_drawdown: float
    expected_horizon: int
    elapsed_minutes: int
    peak_roi: float
    drawdown_from_peak: float


def expected_horizon_minutes(contract: dict) -> int:
    """Horizon from exit profile + side — not predicted by contract."""
    side = str(contract.get("side", "long")).upper()
    profile = contract.get("exit_profile", "early_exit")
    if profile == "runner":
        return 120 if side == "LONG" else 90
    return 90 if side == "SHORT" else 120


def _safe_div(numerator: float, denominator: float, *, floor: float = 0.01) -> float:
    denom = denominator if abs(denominator) >= floor else floor
    return numerator / denom


def compute_metrics(contract: dict, position: dict) -> GuardianMetrics:
    expected_roi = float(contract.get("expected_roi", 0))
    expected_peak = float(contract.get("expected_peak_roi", expected_roi))
    expected_dd = float(contract.get("expected_drawdown", 1.0))
    horizon = int(position.get("expected_horizon") or expected_horizon_minutes(contract))

    current_roi = float(position.get("current_roi", 0))
    elapsed = int(position.get("elapsed_minutes", 0))
    peak_roi = float(position.get("peak_roi", current_roi))
    dd_from_peak = float(position.get("drawdown_from_peak", max(0.0, peak_roi - current_roi)))

    return GuardianMetrics(
        progress_ratio=_safe_div(current_roi, expected_roi, floor=0.1),
        time_progress=_safe_div(elapsed, horizon, floor=1.0),
        drawdown_pressure=_safe_div(dd_from_peak, expected_dd, floor=1.0),
        overperformance=current_roi > expected_peak,
        current_roi=current_roi,
        expected_roi=expected_roi,
        expected_peak_roi=expected_peak,
        expected_drawdown=expected_dd,
        expected_horizon=horizon,
        elapsed_minutes=elapsed,
        peak_roi=peak_roi,
        drawdown_from_peak=dd_from_peak,
    )


def _fmt_pct(v: float) -> str:
    return f"{v:.2f}%"


def evaluate_rules(
    metrics: GuardianMetrics,
    contract: dict,
    *,
    manual: bool = False,
) -> tuple[str, list[str]]:
    """Return (action, reason_lines) — highest-priority rule wins."""
    lines: list[str] = [
        f"contract expected_roi={_fmt_pct(metrics.expected_roi)} "
        f"expected_peak={_fmt_pct(metrics.expected_peak_roi)} "
        f"expected_drawdown={_fmt_pct(metrics.expected_drawdown)} "
        f"exit_profile={contract.get('exit_profile', '')} dna={contract.get('dna_type', '')}",
        f"position current_roi={_fmt_pct(metrics.current_roi)} elapsed={metrics.elapsed_minutes}m "
        f"peak={_fmt_pct(metrics.peak_roi)} dd_from_peak={_fmt_pct(metrics.drawdown_from_peak)}",
        f"progress_ratio={metrics.progress_ratio:.3f} "
        f"time_progress={metrics.time_progress:.3f} "
        f"drawdown_pressure={metrics.drawdown_pressure:.3f} "
        f"overperformance={metrics.overperformance}",
    ]

    if manual or contract.get("gate_action") == "NO_ACTION":
        lines.append("manual or gate NO_ACTION — observe only")
        return GuardianAction.NO_ACTION.value, lines

    # Emergency: catastrophic loss or extreme drawdown pressure
    if metrics.current_roi <= EMERGENCY_ROI_PCT:
        lines.append(
            f"EMERGENCY_EXIT: current_roi {_fmt_pct(metrics.current_roi)} "
            f"<= emergency floor {_fmt_pct(EMERGENCY_ROI_PCT)}"
        )
        return GuardianAction.EMERGENCY_EXIT.value, lines

    if metrics.drawdown_pressure >= DRAWDOWN_EMERGENCY_PRESSURE:
        lines.append(
            f"EMERGENCY_EXIT: drawdown_pressure {metrics.drawdown_pressure:.2f} "
            f">= {DRAWDOWN_EMERGENCY_PRESSURE} (peak giveback exceeds safe band)"
        )
        return GuardianAction.EMERGENCY_EXIT.value, lines

    # Drawdown exceeds contract tolerance → full exit
    if metrics.drawdown_pressure > DRAWDOWN_EXIT_PRESSURE:
        lines.append(
            f"EXIT: drawdown_pressure {metrics.drawdown_pressure:.2f} > {DRAWDOWN_EXIT_PRESSURE} "
            f"(giveback {_fmt_pct(metrics.drawdown_from_peak)} vs expected max "
            f"{_fmt_pct(metrics.expected_drawdown)})"
        )
        return GuardianAction.EXIT.value, lines

    weakening = metrics.drawdown_pressure >= DRAWDOWN_WEAKENING_PRESSURE

    # Horizon exceeded — re-evaluate (MET-style extended hold)
    if metrics.time_progress > HORIZON_REVIEW_RATIO:
        met_like = (
            metrics.time_progress >= HORIZON_MET_RATIO
            and metrics.progress_ratio < 0.5
            and (metrics.peak_roi - metrics.current_roi) >= MET_PEAK_GIVEBACK_PCT
        )
        if met_like:
            lines.append(
                f"EXIT: horizon exceeded {metrics.time_progress:.1f}x with weak progress "
                f"({metrics.progress_ratio:.2f}) and peak giveback "
                f"{_fmt_pct(metrics.peak_roi - metrics.current_roi)} — MET-style extended hold"
            )
            return GuardianAction.EXIT.value, lines

        if metrics.progress_ratio < 0.5:
            lines.append(
                f"EXIT: time_progress {metrics.time_progress:.2f} > {HORIZON_REVIEW_RATIO} "
                f"but progress_ratio {metrics.progress_ratio:.2f} < 0.5 — thesis underdelivered"
            )
            return GuardianAction.EXIT.value, lines

        if metrics.overperformance or metrics.progress_ratio > 1.0:
            lines.append(
                f"TRAIL: past horizon with strong progress ({metrics.progress_ratio:.2f})"
                + (f" or overperformance vs expected_peak {_fmt_pct(metrics.expected_peak_roi)}"
                   if metrics.overperformance else "")
            )
            return GuardianAction.TRAIL.value, lines

        if weakening:
            lines.append(
                f"REDUCE: past horizon, moderate progress {metrics.progress_ratio:.2f}, "
                f"drawdown_pressure {metrics.drawdown_pressure:.2f} — trim exposure"
            )
            return GuardianAction.REDUCE.value, lines

        lines.append(
            f"HOLD: past horizon but progress {metrics.progress_ratio:.2f} still acceptable "
            f"without weakening pressure"
        )
        return GuardianAction.HOLD.value, lines

    # Ahead of schedule — trail winners
    if metrics.overperformance:
        lines.append(
            f"TRAIL: current {_fmt_pct(metrics.current_roi)} exceeds expected_peak "
            f"{_fmt_pct(metrics.expected_peak_roi)} — protect outperformance"
        )
        return GuardianAction.TRAIL.value, lines
    if metrics.progress_ratio >= 1.0:
        lines.append(
            f"TRAIL: progress_ratio {metrics.progress_ratio:.2f} >= 1.0 "
            f"(target {_fmt_pct(metrics.expected_roi)} reached) — tighten stop / trail"
        )
        return GuardianAction.TRAIL.value, lines

    # Near target band
    if PROGRESS_TARGET_MIN <= metrics.progress_ratio <= PROGRESS_TARGET_MAX:
        if weakening:
            lines.append(
                f"TRAIL: progress {metrics.progress_ratio:.2f} in target band "
                f"but drawdown_pressure {metrics.drawdown_pressure:.2f} — momentum weakening"
            )
            return GuardianAction.TRAIL.value, lines
        lines.append(
            f"HOLD: progress {metrics.progress_ratio:.2f} in target band "
            f"({PROGRESS_TARGET_MIN}-{PROGRESS_TARGET_MAX}), state healthy"
        )
        return GuardianAction.HOLD.value, lines

    # Early phase
    if metrics.progress_ratio < PROGRESS_EARLY_MAX:
        lines.append(
            f"HOLD: progress {metrics.progress_ratio:.2f} < {PROGRESS_EARLY_MAX} — "
            f"still early vs expected {_fmt_pct(metrics.expected_roi)}"
        )
        return GuardianAction.HOLD.value, lines

    # Mid progress — default hold unless weakening with early_exit profile
    if (
        contract.get("exit_profile") == "early_exit"
        and contract.get("early_exit_allowed")
        and weakening
        and metrics.progress_ratio < PROGRESS_TARGET_MIN
    ):
        lines.append(
            f"REDUCE: early_exit profile, progress {metrics.progress_ratio:.2f}, "
            f"weakening before target — partial de-risk"
        )
        return GuardianAction.REDUCE.value, lines

    lines.append(
        f"HOLD: mid progress {metrics.progress_ratio:.2f}, no exit trigger fired"
    )
    return GuardianAction.HOLD.value, lines
