"""Execution rule discovery constants."""

from __future__ import annotations

TRAIN_RATIO = 0.7
TOP5_SIZE = 5
TOP2_SIZE = 2
MIN_TRAIN_GROUPS = 3
MIN_TRAIN_PASS = 4
TOP_ATOMS = 30
MAX_RULES = 350
TOP_OUTPUT = 20
SUCCESS_RETURN_PCT = 3.0

EXEC_NUMERIC_FEATURES = (
    "entry_score",
    "direction_confidence",
    "pattern_confidence",
    "rule_confidence",
    "feature_match_ratio",
    "rule_margin",
    "recency",
    "signal_freshness",
    "top5_rank",
    "top5_rank_pct",
    "gap_to_best_entry",
    "obs_return_pct",
    "obs_high_pct",
    "obs_low_pct",
    "obs_body_pct",
    "obs_range_pct",
    "body_range_ratio",
    "volume_ratio_scan",
    "vwap_deviation_pct",
    "atr_increase_ratio",
    "new_high_breakout",
    "prior_high_break",
    "false_breakout_flag",
    "momentum_persist",
    "rank_obs_return_top5",
    "execution_score",
)

EXEC_RATIO_PAIRS = (
    ("obs_body_pct", "obs_range_pct"),
    ("obs_return_pct", "entry_score"),
    ("obs_return_pct", "rule_margin"),
    ("vwap_deviation_pct", "atr_increase_ratio"),
    ("obs_high_pct", "obs_low_pct"),
)

EXEC_DIFF_PAIRS = (
    ("obs_return_pct", "gap_to_best_entry"),
    ("entry_score", "obs_return_pct"),
    ("obs_high_pct", "obs_return_pct"),
)
