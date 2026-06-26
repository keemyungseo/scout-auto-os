"""Tracking horizons and classification thresholds (empirical, not operational)."""

from __future__ import annotations

BAR_MINUTES = 15
MIN_TRACK_HOURS = 6
PREFERRED_TRACK_HOURS = 12
MIN_TRACK_BARS = MIN_TRACK_HOURS * 60 // BAR_MINUTES  # 24
PREFERRED_TRACK_BARS = PREFERRED_TRACK_HOURS * 60 // BAR_MINUTES  # 48

SUCCESS_RETURN_PCT = 3.0
MOMENTUM_LOOKBACK_BARS = 4  # ~1h at 15m resolution

LIFECYCLE_LABELS = (
    "Immediate Explosion",
    "Slow Trend",
    "Delayed Breakout",
    "Fake Breakout",
    "V-Reversal",
    "Continuous Trend",
    "Late Runner",
    "Dead Signal",
    "Unclassified",
)

PHASE_LABELS = (
    "birth",
    "growth",
    "acceleration",
    "deceleration",
    "termination",
)
