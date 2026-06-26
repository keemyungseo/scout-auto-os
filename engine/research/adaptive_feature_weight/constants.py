"""Adaptive feature weight constants — analysis only, no model change."""

from __future__ import annotations

RANDOM_SEED = 42
TRAIN_RATIO = 0.7
MIN_CONDITION_SAMPLES = 40
MIN_CONDITION_SCANS = 3
FROZEN_RANKING_MODEL = "catboost_ranker"
WEIGHT_CLIP = (0.25, 4.0)
TOP_FEATURES = 20
INTERACTION_TOP_N = 15

CONDITION_DEFS: dict[str, dict] = {
    "high_volatility": {"field": "dna_1h_current_range_pct", "agg": "median", "op": "gte"},
    "low_volatility": {"field": "dna_1h_current_range_pct", "agg": "median", "op": "lte"},
    "strong_trend": {"field": "dna_1h_current_return_pct", "agg": "median", "op": "gte"},
    "weak_trend": {"field": "dna_1h_current_return_pct", "agg": "median", "op": "lte"},
    "volume_surge": {"field": "dna_15m_current_volume_ratio", "agg": "median", "op": "gte"},
    "volume_decline": {"field": "dna_15m_current_volume_ratio", "agg": "median", "op": "lte"},
    "breakout": {"field": "meta_derived_breakout_flag", "agg": "mean", "op": "gte", "threshold": 0.3},
    "compression": {"field": "dna_5m_compression", "agg": "median", "op": "gte"},
    "momentum": {"field": "dna_5m_momentum", "agg": "median", "op": "gte"},
    "range_expansion": {"field": "meta_derived_range_expansion", "agg": "median", "op": "gte"},
    "reversal": {"field": "dna_1h_current_return_pct", "agg": "median", "op": "lte", "extra": "reversal"},
    "bull_leader": {"field": "regime_bull", "agg": "mean", "op": "gte", "threshold": 0.5},
    "bear_leader": {"field": "regime_bear", "agg": "mean", "op": "gte", "threshold": 0.5},
    "sideway": {"field": "regime_sideway", "agg": "mean", "op": "gte", "threshold": 0.5},
}
