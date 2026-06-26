"""Research Infrastructure V1 constants — frozen constitution reference."""

from __future__ import annotations

DATASET_VERSION = "scout_constitution_v1"
CONSTITUTION = {
    "features": "ranking_engine_v1",
    "model": "catboost_ranker",
    "label": "return_minus_dd",
    "tuning": "forbidden",
}

RANDOM_SEED = 42
TRAIN_RATIO = 0.7
TOP_K_ARCHIVE = 50
TOP_K_PICK = 5

CALENDAR_WINDOWS = (15, 30, 60, 90, 180)
VALIDATION_TARGET_DAYS = 90
LIVE_VALIDATION_TARGET_DAYS = 90
MIN_REGIME_SCANS = 10
MIN_SAMPLES_PER_VALIDATION = 500

REGIME_AXES = (
    "market_simple",
    "market_mapped",
    "volatility",
    "structure",
    "dynamics",
)

LABEL_COLUMNS = (
    "return_2h",
    "return_minus_dd",
    "max_drawdown_2h",
    "max_up_2h",
    "min_return_2h",
    "mfe_2h",
    "mae_2h",
    "intrabar_sharpe",
    "return_30m",
    "return_1h",
    "return_4h",
)
