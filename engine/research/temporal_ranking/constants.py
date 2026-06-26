"""Temporal Ranking Engine V1 constants."""

from __future__ import annotations

RANDOM_SEED = 42
TRAIN_RATIO = 0.7
SEQUENCE_LENGTHS = (3, 6, 9, 12, 24)
SCAN_INTERVAL_HOURS = 2
MODEL_NAMES = ("catboost_ranker", "lightgbm_ranker", "xgboost_ranker")
BASELINE_MODEL = "catboost_ranker"

TEMPORAL_BASE_KEYS = (
    "entry_score",
    "direction_confidence",
    "a6_formula_score",
    "phase20_base_score",
    "dna_1h_current_return_pct",
    "dna_1h_current_range_pct",
    "dna_15m_current_volume_ratio",
    "dna_5m_compression",
    "dna_5m_release",
    "dna_5m_momentum",
    "ctx_rank_5m_release",
    "ctx_rank_1h_current_return_pct",
    "search_rank_pct_return",
    "cluster_top_score",
    "meta_derived_range_expansion",
)

LEAK_FORBIDDEN_PREFIXES = (
    "return_",
    "label_",
    "outcome_",
    "max_up",
    "exec_obs_",
)
