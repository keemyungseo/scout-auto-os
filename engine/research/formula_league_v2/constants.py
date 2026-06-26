"""Formula League V2 constants — frozen execution stack, search-only."""

from __future__ import annotations

TRAIN_RATIO = 0.7
TOP_K = 5
SUCCESS_RETURN_PCT = 3.0
HORIZONS = ("2h", "4h", "6h", "12h")
HORIZON_RETURN_KEY = {
    "2h": "return_2h",
    "4h": "return_4h",
    "6h": "return_6h",
    "12h": "return_12h",
}

MIN_TRAIN_PASS = 8
TOP_ATOMS = 80
MAX_FORMULAS = 6000
SURVIVOR_ROUND_TOP_PCT = 0.15
SURVIVOR_MIN_ROUNDS_WIN = 0.55
MIN_SURVIVOR_TRADES = 12
BASELINE_FORMULA_ID = "A6_frozen"

# Expanded search feature pool (maps to candidates.jsonl + derived)
SEARCH_FEATURE_ALIASES: dict[str, str] = {
    "body": "1h_current_body_pct",
    "upper_wick": "1h_current_close_position",
    "lower_wick": "1h_current_close_position",
    "atr": "1h_current_range_pct",
    "atr_expansion": "5m_release",
    "atr_compression": "5m_compression",
    "volume": "15m_current_volume_ratio",
    "volume_ratio": "15m_current_volume_ratio",
    "relative_volume": "5m_volume_ma_ratio",
    "vwap_distance": "1h_current_ma20_distance_pct",
    "ma_distance": "1h_current_ma20_distance_pct",
    "ma_slope": "2h_current_return_pct",
    "ema": "1h_current_ma20_distance_pct",
    "range": "1h_current_range_pct",
    "range_expansion": "1h_current_range_pct",
    "compression": "5m_compression",
    "breakout": "5m_release",
    "false_breakout": "5m_seq_return_sum_6",
    "high_break": "5m_seq_high_break_6",
    "low_break": "5m_seq_low_break_6",
    "relative_strength": "1h_current_return_pct",
    "obv": "5m_seq_volume_energy_6",
    "momentum": "5m_momentum",
    "gap": "15m_current_return_pct",
    "clv": "1h_current_close_position",
    "position_in_daily_range": "1h_current_close_position",
}

RATIO_PAIRS = (
    ("1h_current_body_pct", "1h_current_range_pct"),
    ("1h_current_return_pct", "15m_current_return_pct"),
    ("1h_current_return_pct", "2h_current_return_pct"),
    ("5m_momentum", "1h_current_return_pct"),
    ("15m_current_volume_ratio", "5m_volume_ma_ratio"),
    ("1h_current_range_pct", "5m_compression"),
    ("1h_current_ma20_distance_pct", "1h_current_range_pct"),
    ("5m_seq_return_sum_6", "1h_current_return_pct"),
)

DIFF_PAIRS = (
    ("1h_current_return_pct", "15m_current_return_pct"),
    ("1h_current_body_pct", "1h_previous_body_pct"),
    ("1h_current_range_pct", "1h_previous_range_pct"),
    ("2h_current_return_pct", "1h_current_return_pct"),
    ("1h_current_ma20_distance_pct", "2h_current_ma20_distance_pct"),
)

WINDOW_PAIRS = (
    ("1h_current_return_pct", "1h_previous_return_pct"),
    ("1h_current_range_pct", "1h_previous_range_pct"),
    ("15m_current_body_pct", "15m_previous_body_pct"),
)

RANK_FEATURES = (
    "1h_current_body_pct",
    "1h_current_range_pct",
    "1h_current_return_pct",
    "2h_current_return_pct",
    "15m_current_volume_ratio",
    "5m_momentum",
    "1h_current_ma20_distance_pct",
    "5m_compression",
    "5m_release",
    "5m_seq_return_sum_6",
)

LINEAR_WEIGHT_GRID = (-2.0, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 3.0)
