"""Online feature computation at each post-entry timestep."""

from __future__ import annotations

from scout_auto_os.engine.research.online_state_machine.constants import BAR_MINUTES, SLOPE_LOOKBACK
from scout_auto_os.engine.research.signal_lifecycle.timeline import build_signal_timeline


def _slope(values: list[float], end_i: int, lookback: int) -> float:
    start = max(0, end_i - lookback + 1)
    window = values[start : end_i + 1]
    if len(window) < 2:
        return 0.0
    return round((window[-1] - window[0]) / max(len(window) - 1, 1), 4)


def enrich_timeline_online(timeline: list[dict]) -> list[dict]:
    """Add drawdown, slope, peak_return — all computable online (causal)."""
    if not timeline:
        return []
    returns = [float(r["return_pct"]) for r in timeline]
    peak_returns: list[float] = []
    running_peak = float("-inf")
    enriched: list[dict] = []
    for i, row in enumerate(timeline):
        running_peak = max(running_peak, returns[i])
        peak_returns.append(round(running_peak, 4))
        dd = round(returns[i] - running_peak, 4)
        enriched.append(
            {
                **row,
                "peak_return_pct": peak_returns[-1],
                "drawdown_from_peak_pct": dd,
                "slope_return": _slope(returns, i, SLOPE_LOOKBACK),
                "bar_minutes": BAR_MINUTES,
            },
        )
    return enriched
