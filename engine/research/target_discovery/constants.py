"""Target Discovery Engine V1 constants."""

from __future__ import annotations

RANDOM_SEED = 42
TRAIN_RATIO = 0.7
BASELINE_LABEL_ID = "baseline_max_up_4h"
BASELINE_MODEL = "catboost_ranker"
SUCCESS_RETURN_PCT = 3.0

RELEVANCE_GRADES = {
    1: 4,
    2: 3,
    3: 2,
    4: 1,
    5: 1,
}
