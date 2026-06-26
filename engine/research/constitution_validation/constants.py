"""Frozen SCOUT Constitution — no tuning allowed."""

from __future__ import annotations

RANDOM_SEED = 42
TRAIN_RATIO = 0.7
FROZEN_MODEL = "catboost_ranker"
FROZEN_LABEL_ID = "return_minus_dd"
TOP_K = 5
SUCCESS_RETURN_PCT = 3.0
MIN_REGIME_SAMPLES = 15
MIN_FOLD_SAMPLES = 20
TARGET_CALENDAR_DAYS = 90

CONSTITUTION = {
    "features": "ranking_engine_v1",
    "model": FROZEN_MODEL,
    "label": FROZEN_LABEL_ID,
    "tuning": "forbidden",
}
