"""Build per-signal forward timelines from 15m kline bars."""

from __future__ import annotations

from scout_auto_os.engine.research.signal_lifecycle.constants import (
    BAR_MINUTES,
    MIN_TRACK_BARS,
    MOMENTUM_LOOKBACK_BARS,
    PREFERRED_TRACK_BARS,
)


def _directional_close_return(close: float, entry: float, direction: str) -> float:
    raw = (close - entry) / entry * 100
    return round(-raw if direction == "short" else raw, 4)


def _bar_favorable_pct(bar: list, entry: float, direction: str) -> float:
    if direction == "long":
        return (float(bar[2]) - entry) / entry * 100
    return (entry - float(bar[3])) / entry * 100


def _bar_adverse_pct(bar: list, entry: float, direction: str) -> float:
    if direction == "long":
        return (float(bar[3]) - entry) / entry * 100
    return (float(bar[2]) - entry) / entry * 100


def _bar_body_pct(bar: list) -> float:
    o, c = float(bar[1]), float(bar[4])
    mid = (o + c) / 2 or 1.0
    return round(abs(c - o) / mid * 100, 4)


def _bar_range_pct(bar: list) -> float:
    o = float(bar[1]) or 1.0
    return round((float(bar[2]) - float(bar[3])) / o * 100, 4)


def _bar_atr_proxy_pct(bar: list) -> float:
    c = float(bar[4]) or 1.0
    return round((float(bar[2]) - float(bar[3])) / c * 100, 4)


def track_bars_available(klines: list) -> int:
    if not klines:
        return 0
    return min(PREFERRED_TRACK_BARS, max(MIN_TRACK_BARS, len(klines) - 1))


def build_signal_timeline(
    klines: list,
    direction: str,
    scan_kst: str,
    symbol: str,
    signal_id: str,
) -> tuple[list[dict], dict]:
    """Return (timeline_rows, summary_features) for one champion signal."""
    if not klines or len(klines) < 2:
        return [], {}

    entry = float(klines[0][1])
    if entry <= 0:
        return [], {}

    n_bars = track_bars_available(klines)
    vol_base = float(klines[0][5]) or 1.0
    body_base = _bar_body_pct(klines[0]) or 0.01
    range_base = _bar_range_pct(klines[0]) or 0.01
    atr_base = _bar_atr_proxy_pct(klines[0]) or 0.01

    returns: list[float] = []
    mfe_series: list[float] = []
    mae_series: list[float] = []
    velocities: list[float] = []
    timeline: list[dict] = []

    running_mfe = float("-inf")
    running_mae = float("inf")

    for i in range(n_bars + 1):
        bar = klines[i]
        minutes = i * BAR_MINUTES
        close = float(bar[4])
        ret = _directional_close_return(close, entry, direction)
        fav = _bar_favorable_pct(bar, entry, direction)
        adv = _bar_adverse_pct(bar, entry, direction)
        running_mfe = max(running_mfe, fav)
        running_mae = min(running_mae, adv)

        returns.append(ret)
        mfe_series.append(round(running_mfe, 4))
        mae_series.append(round(running_mae, 4))

        if i == 0:
            velocity = 0.0
        else:
            dt_h = BAR_MINUTES / 60.0
            velocity = round((ret - returns[i - 1]) / dt_h, 4)
        velocities.append(velocity)

        if i >= MOMENTUM_LOOKBACK_BARS:
            momentum = round(ret - returns[i - MOMENTUM_LOOKBACK_BARS], 4)
        else:
            momentum = ret

        vol = float(bar[5])
        body = _bar_body_pct(bar)
        rng = _bar_range_pct(bar)
        atr = _bar_atr_proxy_pct(bar)

        if i == 0:
            acceleration = 0.0
        else:
            dt_h = BAR_MINUTES / 60.0
            acceleration = round((velocity - velocities[i - 1]) / dt_h, 4)

        phase = _phase_at(i, n_bars, velocity, acceleration, ret, returns)

        timeline.append(
            {
                "signal_id": signal_id,
                "direction": direction,
                "scan_time_kst": scan_kst,
                "symbol": symbol,
                "bar_index": i,
                "minutes_from_entry": minutes,
                "return_pct": ret,
                "mfe_pct": mfe_series[-1],
                "mae_pct": mae_series[-1],
                "velocity_pct_per_hour": velocity,
                "acceleration_pct_per_hour2": acceleration,
                "volume_ratio": round(vol / vol_base, 4),
                "atr_ratio": round(atr / atr_base, 4),
                "body_pct": body,
                "body_ratio": round(body / body_base, 4),
                "range_pct": rng,
                "range_ratio": round(rng / range_base, 4),
                "momentum_pct": momentum,
                "lifecycle_phase": phase,
            },
        )

    peak_i = max(range(len(returns)), key=lambda j: returns[j])
    peak_time = peak_i * BAR_MINUTES
    end_time = n_bars * BAR_MINUTES

    def _ret_at_minutes(m: int) -> float:
        idx = min(m // BAR_MINUTES, len(returns) - 1)
        return returns[idx]

    summary = {
        "signal_id": signal_id,
        "direction": direction,
        "scan_time_kst": scan_kst,
        "symbol": symbol,
        "entry_price": round(entry, 8),
        "track_bars": n_bars,
        "track_hours": round(end_time / 60, 2),
        "bar_minutes": BAR_MINUTES,
        "return_30m": _ret_at_minutes(30),
        "return_1h": _ret_at_minutes(60),
        "return_2h": _ret_at_minutes(120),
        "return_6h": _ret_at_minutes(360),
        "return_12h": _ret_at_minutes(720) if end_time >= 720 else _ret_at_minutes(end_time),
        "return_at_end": returns[-1],
        "mfe_full": mfe_series[-1],
        "mae_full": mae_series[-1],
        "peak_time_min": peak_time,
        "peak_return_pct": returns[peak_i],
        "end_time_min": end_time,
        "return_90m": _ret_at_minutes(90),
        "peak_fraction": round(peak_time / end_time, 4) if end_time else 0.0,
        "velocity_peak": max(velocities) if velocities else 0.0,
        "early_mae_1h": min(mae_series[: min(5, len(mae_series))]),
    }
    return timeline, summary


def _phase_at(
    bar_i: int,
    n_bars: int,
    velocity: float,
    acceleration: float,
    ret: float,
    returns: list[float],
) -> str:
    if bar_i == 0:
        return "birth"
    if bar_i >= n_bars:
        return "termination"

    if velocity > 0.2 and acceleration > 0.05:
        return "acceleration"
    if velocity > 0.05:
        return "growth"
    if bar_i >= 2 and ret < returns[bar_i - 1] and ret < returns[bar_i - 2]:
        return "deceleration"
    if velocity <= 0 and bar_i >= 3:
        return "deceleration"
    return "growth"
