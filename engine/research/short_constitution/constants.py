"""Short Constitution V1 constants — independent from Long."""

from __future__ import annotations

RANDOM_SEED = 42
TRAIN_RATIO = 0.7
DIRECTION = "short"
MODEL = "catboost_ranker"
BASELINE_LABEL_ID = "baseline_max_down_2h"
SUCCESS_RETURN_PCT = 3.0
MIN_REGIME_SCANS = 8

RELEVANCE_GRADES = {1: 4, 2: 3, 3: 2, 4: 1, 5: 1}

LONG_CONSTITUTION = {
    "features": "ranking_engine_v1",
    "model": "catboost_ranker",
    "label": "return_minus_dd",
    "blind_avg_2h": 5.2608,
}
