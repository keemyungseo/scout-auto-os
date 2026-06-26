"""Observation-window features (first bar after scan — execution-time only)."""

from __future__ import annotations

from scout_auto_os.engine.research.execution_research.constants import BAR_MINUTES


def _dir_ret(raw_pct: float, direction: str) -> float:
    return -raw_pct if direction == "short" else raw_pct


def compute_observation_features(
    klines: list,
    direction: str,
    scan_features: dict,
) -> dict | None:
    """
    Features available after first observation bar closes (~15m in bundle; 5m target noted in report).
    No lookahead beyond bar 0.
    """
    if not klines:
        return None
    bar = klines[0]
    entry = float(bar[1])
    if entry <= 0:
        return None

    o, h, l, c, vol = float(bar[1]), float(bar[2]), float(bar[3]), float(bar[4]), float(bar[5])
    close_ret = _dir_ret((c - entry) / entry * 100, direction)
    high_ret = _dir_ret((h - entry) / entry * 100, direction)
    low_ret = _dir_ret((l - entry) / entry * 100, direction)

    vwap = (h + l + c) / 3.0
    vwap_dev = _dir_ret((c - vwap) / vwap * 100, direction)

    vol_base = float(scan_features.get("15m_current_volume_ratio", 1.0)) or 1.0
    volume_surge = round(vol / (vol_base * 1_000_000 + 1e-6), 4) if vol_base else round(vol, 4)
    volume_ratio = round(float(scan_features.get("15m_current_volume_ratio", 1.0)), 4)

    range_pct = (h - l) / entry * 100
    scan_range = float(scan_features.get("1h_current_range_pct", 1.0)) or 1.0
    atr_increase = round(range_pct / max(scan_range, 0.5), 4)

    scan_ret = float(scan_features.get("1h_current_return_pct", 0))
    new_high_break = 1.0 if high_ret > 0.5 else 0.0
    prior_high_break = 1.0 if _dir_ret(scan_ret, direction) > 0 and close_ret > 0 else 0.0

    false_breakout = 1.0 if high_ret >= 1.5 and close_ret <= 0.3 else 0.0
    momentum_persist = 1.0 if close_ret > 0 and _dir_ret(scan_ret, direction) > 0 else 0.0

    return {
        "obs_return_pct": round(close_ret, 4),
        "obs_high_pct": round(high_ret, 4),
        "obs_low_pct": round(low_ret, 4),
        "volume_surge": volume_surge,
        "volume_ratio_scan": volume_ratio,
        "vwap_deviation_pct": round(vwap_dev, 4),
        "atr_increase_ratio": atr_increase,
        "new_high_breakout": new_high_break,
        "prior_high_break": prior_high_break,
        "false_breakout_flag": false_breakout,
        "momentum_persist": momentum_persist,
        "observation_minutes": BAR_MINUTES,
    }


def execution_score(features: dict, weights: dict[str, float] | None = None) -> float:
    """Composite execution score — higher = preferred for entry."""
    w = weights or DEFAULT_WEIGHTS
    penalty = float(features.get("false_breakout_flag", 0))
    score = (
        w["obs_return"] * float(features.get("obs_return_pct", 0))
        + w["volume"] * min(float(features.get("volume_ratio_scan", 0)), 3.0)
        + w["vwap"] * float(features.get("vwap_deviation_pct", 0))
        + w["atr"] * float(features.get("atr_increase_ratio", 0))
        + w["breakout"] * float(features.get("new_high_breakout", 0)) * 2.0
        + w["prior_break"] * float(features.get("prior_high_break", 0)) * 1.5
        + w["momentum"] * float(features.get("momentum_persist", 0)) * 2.0
        + w["false_penalty"] * penalty * 5.0
    )
    return round(score, 4)


DEFAULT_WEIGHTS = {
    "obs_return": 0.35,
    "volume": 0.15,
    "vwap": 0.10,
    "atr": 0.10,
    "breakout": 0.15,
    "prior_break": 0.10,
    "momentum": 0.15,
    "false_penalty": -1.0,
}
