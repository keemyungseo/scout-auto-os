"""Rule-based online state estimation — no forward lookahead."""

from __future__ import annotations


def estimate_state(row: dict, history: list[dict]) -> str:
    """
    Causal state from current bar features only.
    Uses past rows in history for peak context, not future bars.
    """
    minutes = int(row.get("minutes_from_entry", 0))
    ret = float(row.get("return_pct", 0))
    mfe = float(row.get("mfe_pct", 0))
    mae = float(row.get("mae_pct", 0))
    velocity = float(row.get("velocity_pct_per_hour", 0))
    acceleration = float(row.get("acceleration_pct_per_hour2", 0))
    momentum = float(row.get("momentum_pct", 0))
    slope = float(row.get("slope_return", 0))
    dd_peak = float(row.get("drawdown_from_peak_pct", 0))
    peak_ret = float(row.get("peak_return_pct", 0))
    vol_ratio = float(row.get("volume_ratio", 1.0))
    range_ratio = float(row.get("range_ratio", 1.0))
    body_pct = float(row.get("body_pct", 0))

    if minutes == 0:
        return "EARLY_BREAKOUT"

    # Flat / no participation
    if minutes >= 30 and mfe < 0.8 and abs(ret) < 0.5:
        return "DEAD"

    # Adverse reversal
    if ret <= -1.5:
        return "REVERSAL"
    if peak_ret >= 2.0 and ret < 0 and minutes >= 15:
        return "REVERSAL"

    # Trap: strong excursion then collapse
    if mfe >= 3.0 and ret < max(0.5, mfe * 0.35) and dd_peak <= -1.5:
        return "FAKE_BREAKOUT"
    if mfe >= 4.0 and ret < 1.0 and velocity < -0.1:
        return "FAKE_BREAKOUT"

    # Late-stage fade near highs
    if peak_ret >= 2.0 and dd_peak <= -0.4 * max(peak_ret, 0.1) and velocity < 0:
        return "EXHAUSTION"

    # Re-acceleration phase
    if velocity > 0.15 and acceleration > 0.03 and ret > 0 and momentum > 0:
        return "ACCELERATION"

    # Retrace while still positive
    if peak_ret >= 1.5 and dd_peak <= -1.0 and ret > 0:
        return "PULLBACK"

    # Early expansion window
    if minutes <= 45 and (vol_ratio >= 1.2 or range_ratio >= 1.15 or body_pct >= 1.0):
        if ret >= 0 or mfe >= 0.5:
            return "EARLY_BREAKOUT"

    # Steady trend
    if ret >= 1.0 and velocity > 0.05 and dd_peak > -0.5 * max(peak_ret, 1.0):
        return "HEALTHY_TREND"

    # Positive but losing steam
    if ret > 0 and (velocity <= 0.05 or slope < 0):
        return "WEAK_TREND"

    if ret > 0:
        return "HEALTHY_TREND"

    if mae <= -1.0 and ret < 0:
        return "REVERSAL"

    if minutes >= 60 and mfe < 1.0:
        return "DEAD"

    return "WEAK_TREND"


def annotate_states(timeline: list[dict]) -> list[dict]:
    history: list[dict] = []
    out: list[dict] = []
    for row in timeline:
        state = estimate_state(row, history)
        annotated = {**row, "state": state}
        out.append(annotated)
        history.append(annotated)
    return out
