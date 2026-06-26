"""Lifecycle classifier constants."""

from __future__ import annotations

TRAIN_RATIO = 0.7
MIN_CLASS_SAMPLES = 3

# Primary lifecycle labels (from Signal Lifecycle Engine V1)
PRIMARY_LABELS = (
    "Continuous Trend",
    "Slow Trend",
    "Late Runner",
    "Delayed Breakout",
    "Immediate Explosion",
    "Fake Breakout",
    "Dead Signal",
    "V-Reversal",
    "Unclassified",
)

LABEL_SHORT = {
    "Continuous Trend": "continuous",
    "Slow Trend": "slow",
    "Late Runner": "late_runner",
    "Delayed Breakout": "delayed",
    "Immediate Explosion": "explosion",
    "Fake Breakout": "fake",
    "Dead Signal": "dead",
    "V-Reversal": "v_reversal",
    "Unclassified": "unclassified",
}

# Display order for probability columns in CSV output
PROB_DISPLAY_ORDER = (
    "Continuous Trend",
    "Slow Trend",
    "Late Runner",
    "Delayed Breakout",
    "Immediate Explosion",
    "Fake Breakout",
    "Dead Signal",
)
