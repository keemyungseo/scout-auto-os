"""Shared utilities for A6 core export package."""

from __future__ import annotations

import math
import statistics
from datetime import datetime, timezone, timedelta

# Original: top10_gainer_learning_20260613.py
MIN_PRICE = 0.05
MAX_PRICE = 400.0

# Original: scout_phase19_winner_ranking_dna.py Pattern B (frozen)
PATTERN_B_MACD_MIN = -0.0016
PATTERN_B_RANGE_MIN = 1.4768

# Original: scout_phase16_human_blind_test.py RANK_WEIGHTS
RANK_WEIGHTS = {
    "young_birth": -1.4343174277392308,
    "birth_age_min": 1.5586524217896396,
    "ignition_age_min": -0.22688287801448231,
    "ma_slope_accel": -0.12327724295144488,
    "volume_ma_ratio": -0.06664243036600889,
}

KST = timezone(timedelta(hours=9))
WINNER_TOP_N = 3


def parse_kst(s: str) -> datetime:
    # Original: scout_phase16_human_blind_test.parse_kst
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=KST)


def ohlcv(kline: list) -> tuple[float, float, float, float, float]:
    # Original: season2_universe_blind_test.ohlcv
    return (
        float(kline[1]),
        float(kline[2]),
        float(kline[3]),
        float(kline[4]),
        float(kline[5]),
    )


def ema(values: list[float], period: int) -> float:
    # Original: season2_p40_scout_transition_triggers.ema
    if not values:
        return 0.0
    if len(values) < period:
        return statistics.mean(values)
    k = 2 / (period + 1)
    ema_val = values[0]
    for v in values[1:]:
        ema_val = v * k + ema_val * (1 - k)
    return ema_val


def g(f: dict, key: str, default: float = 0.0) -> float:
    # Original: scout_phase20_winner_state_ranking.g
    return float(f.get(key, default))


def percentile(vals: list[float], p: float) -> float:
    # Original: scout_phase20_winner_state_ranking.percentile
    if not vals:
        return 0.0
    s = sorted(vals)
    idx = (len(s) - 1) * p
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return s[lo]
    w = idx - lo
    return s[lo] * (1 - w) + s[hi] * w


def slice_klines_to_end(klines: list[list], end_ms: int) -> list[list]:
    """Keep candles with open_time <= end_ms."""
    return [k for k in klines if int(k[0]) <= end_ms]
