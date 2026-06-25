"""Direction pattern labeler — 6 zero-base patterns from scan features."""

from __future__ import annotations

LONG_PATTERNS = (
    "DOWN_BASE_UP",       # DOWN → BASE → UP
    "DOWN_UP",            # DOWN → UP (V reversal)
    "UP_CONTINUATION",    # UP → UP continuation
    "UP_ACCELERATION",    # UP → UP acceleration
)

SHORT_PATTERNS = (
    "UP_BASE_DOWN",       # UP → BASE → DOWN
    "UP_DOWN",            # UP → DOWN (V reversal)
    "DOWN_CONTINUATION",  # DOWN → DOWN continuation
    "DOWN_ACCELERATION",  # DOWN → DOWN acceleration
)

ALL_PATTERNS = LONG_PATTERNS + SHORT_PATTERNS + ("UNLABELED",)


def _g(f: dict, key: str, default: float = 0.0) -> float:
    return float(f.get(key, default))


def label_direction_pattern(features: dict) -> str:
    """Classify symbol at scan time into nearest directional pattern."""
    p15 = _g(features, "15m_previous_return_pct")
    c15 = _g(features, "15m_current_return_pct")
    p1h = _g(features, "1h_previous_return_pct")
    c1h = _g(features, "1h_current_return_pct")
    c2h = _g(features, "2h_current_return_pct")
    comp = _g(features, "5m_compression")
    mom = _g(features, "5m_momentum")
    seq = _g(features, "5m_seq_return_sum_6")

    # LONG: DOWN → BASE → UP
    if p1h < -0.2 and comp >= 0.5 and c15 > 0 and c1h >= -0.1:
        return "DOWN_BASE_UP"
    # LONG: DOWN → UP (V)
    if p15 < -0.4 and c15 > 0.5:
        return "DOWN_UP"
    # LONG: UP acceleration
    if c1h > 0.3 and c15 > 0 and (mom > 1.5 or seq > 1.0):
        return "UP_ACCELERATION"
    # LONG: UP continuation
    if c1h > 0.25 and c15 > 0 and c2h >= 0:
        return "UP_CONTINUATION"

    # SHORT: UP → BASE → DOWN
    if p1h > 0.2 and comp >= 0.5 and c15 < 0 and c1h <= 0.1:
        return "UP_BASE_DOWN"
    # SHORT: UP → DOWN (V)
    if p15 > 0.4 and c15 < -0.5:
        return "UP_DOWN"
    # SHORT: DOWN acceleration
    if c1h < -0.3 and c15 < 0 and (mom < -1.5 or seq < -1.0):
        return "DOWN_ACCELERATION"
    # SHORT: DOWN continuation
    if c1h < -0.25 and c15 < 0:
        return "DOWN_CONTINUATION"

    return "UNLABELED"


def pattern_side(pattern: str) -> str:
    if pattern in LONG_PATTERNS:
        return "long"
    if pattern in SHORT_PATTERNS:
        return "short"
    return "neutral"
