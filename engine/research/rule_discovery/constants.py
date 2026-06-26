"""Rule discovery search constants."""

from __future__ import annotations

TRAIN_RATIO = 0.7
PRECISION_TOLERANCE = 0.02
MIN_BLIND_PASS = 8
MIN_TRAIN_PASS = 12
TOP_ATOMS = 35
MAX_CANDIDATES = 400
TOP_OUTPUT = 20
SUCCESS_RETURN_PCT = 3.0

PREFERRED_FEATURES = (
    "1h_current_body_pct",
    "2h_current_body_pct",
    "1h_current_range_pct",
    "2h_current_range_pct",
    "1h_current_return_pct",
    "2h_current_return_pct",
    "15m_current_body_pct",
    "15m_current_range_pct",
    "15m_current_volume_ratio",
    "1h_current_volume_ratio",
    "5m_momentum",
    "15m_current_ma20_distance_pct",
    "1h_current_close_position",
    "2h_current_close_position",
)

WINDOW_PAIRS = (
    ("15m_current_body_pct", "15m_previous_body_pct"),
    ("15m_current_range_pct", "15m_previous_range_pct"),
    ("1h_current_body_pct", "1h_previous_body_pct"),
    ("1h_current_range_pct", "1h_previous_range_pct"),
    ("2h_current_body_pct", "2h_previous_body_pct"),
    ("1h_current_return_pct", "1h_previous_return_pct"),
)

RATIO_PAIRS = (
    ("1h_current_body_pct", "1h_current_range_pct"),
    ("2h_current_body_pct", "2h_current_range_pct"),
    ("1h_current_body_pct", "2h_current_body_pct"),
    ("1h_current_range_pct", "2h_current_range_pct"),
    ("15m_current_body_pct", "15m_current_range_pct"),
)

DIFF_PAIRS = (
    ("1h_current_body_pct", "2h_current_body_pct"),
    ("1h_current_return_pct", "2h_current_return_pct"),
    ("15m_current_body_pct", "1h_current_body_pct"),
    ("1h_current_range_pct", "2h_current_range_pct"),
)
