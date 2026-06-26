"""Portfolio engine defaults."""

from __future__ import annotations

MAX_LONG_SLOTS = 3
MAX_SHORT_SLOTS = 3
REPLACEMENT_MARGIN = 0.08
HOLD_HOURS = 2
SCAN_INTERVAL_HOURS = 2
MIN_REPLACE_SCORE_GAP = 0.08

SCORE_WEIGHTS = {
    "direction_confidence": 0.20,
    "pattern_confidence": 0.15,
    "rule_confidence": 0.15,
    "feature_match_ratio": 0.15,
    "rule_margin": 0.20,
    "recency": 0.075,
    "signal_freshness": 0.075,
}
