"""Rule portfolio constants — frozen baselines, no tuning."""

from __future__ import annotations

MIN_REGIME_TRADES = 2
MIN_CONFIDENCE_TRADES = 8
SUCCESS_RETURN_PCT = 3.0
BASELINE_RULE_ID = "execution_score_v1"
BASELINE_RULE_EXPR = "execution_score (manual weights)"

CLUSTER_LABELS = (
    "bull_trend",
    "strong_breakout",
    "high_atr",
    "low_atr",
    "compression",
    "recovery",
    "reversal",
    "sideways",
    "execution_aligned",
    "mixed",
)

FEATURE_CLUSTER_HINTS: dict[str, str] = {
    "direction_confidence": "bull_trend",
    "pattern_confidence": "bull_trend",
    "rank_obs_return_top5": "strong_breakout",
    "obs_return_pct": "strong_breakout",
    "obs_high_pct": "strong_breakout",
    "momentum_persist": "strong_breakout",
    "new_high_breakout": "strong_breakout",
    "prior_high_break": "strong_breakout",
    "atr_increase_ratio": "high_atr",
    "obs_range_pct": "low_atr",
    "body_range_ratio": "compression",
    "false_breakout_flag": "reversal",
    "obs_low_pct": "reversal",
    "feature_match_ratio": "sideways",
    "rule_margin": "sideways",
    "top5_rank_pct": "sideways",
    "execution_score": "execution_aligned",
    "gap_to_best_entry": "recovery",
    "vwap_deviation_pct": "recovery",
}
