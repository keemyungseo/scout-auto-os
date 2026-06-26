"""Short Execution Research V1 — frozen constitutions, execution layer only."""

from __future__ import annotations

RANDOM_SEED = 42
TRAIN_RATIO = 0.7
DIRECTION = "short"
BAR_MINUTES = 15
PICK_TOP = 3
LONG_PICK_TOP = 3

FROZEN_SHORT_LABEL = "risk_adjusted_short"
FROZEN_LONG_LABEL = "return_minus_dd"

CHECKPOINT_MINUTES = (5, 15, 30, 60, 120, 240)

# Forward simulation horizon — matches constitution blind window
MAX_FORWARD_MINUTES = 240

LIVE_LOG_PATHS = (
    "logs/auto_os/positions.csv",
    "logs/auto_os/daily_report_*.txt",
    "data/trades.db",
    "data/position_review.csv",
    "data/position_state_cache.json",
    "data/scout_auto_os.db",
)
