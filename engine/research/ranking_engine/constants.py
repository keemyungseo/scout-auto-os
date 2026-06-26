"""Ranking Engine V1 constants."""

from __future__ import annotations

RANDOM_SEED = 42
TRAIN_RATIO = 0.7
TOP_K = 5
SUCCESS_RETURN_PCT = 3.0
HORIZONS = ("2h", "4h", "6h", "12h")

RELEVANCE_GRADES = {
    1: 4,
    2: 3,
    3: 2,
    4: 1,
    5: 1,
}

MODEL_NAMES = (
    "lightgbm_ranker",
    "xgboost_ranker",
    "catboost_ranker",
    "random_forest",
    "extra_trees",
    "logistic_top3",
    "ridge_return",
)

BASELINE_NAMES = (
    "current_search_a6",
    "entry_score_top5",
    "formula_league_v2",
    "execution_score_proxy",
)
