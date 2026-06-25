"""Long / Short directional candidate engines (research only)."""

from __future__ import annotations

import random
from typing import Callable

from scout_auto_os.engine.research.directional.patterns import label_direction_pattern

LONG_ENGINES = (
    "RANDOM_LONG",
    "A6_LONG",
    "LONG_BASE_REVERSAL",
    "LONG_V_REVERSAL",
    "LONG_CONTINUATION",
    "LONG_ACCELERATION",
)

SHORT_ENGINES = (
    "RANDOM_SHORT",
    "SHORT_TOP_REVERSAL",
    "SHORT_V_REVERSAL",
    "SHORT_CONTINUATION",
    "SHORT_ACCELERATION",
)


def _g(f: dict, key: str, default: float = 0.0) -> float:
    return float(f.get(key, default))


def _score_long_base(row: dict) -> float:
    f = row["features"]
    p1h = _g(f, "1h_previous_return_pct")
    comp = _g(f, "5m_compression")
    c15 = _g(f, "15m_current_return_pct")
    c1h = _g(f, "1h_current_return_pct")
    if p1h >= 0:
        return 0.0
    return comp * 2 + c15 * 1.5 + max(0, c1h) * 0.5 + abs(p1h) * 0.3


def _score_long_v(row: dict) -> float:
    f = row["features"]
    p15 = _g(f, "15m_previous_return_pct")
    c15 = _g(f, "15m_current_return_pct")
    if p15 >= 0:
        return c15 * 0.1
    return (c15 - p15) * 2 + _g(f, "15m_current_volume_ratio") * 0.5


def _score_long_cont(row: dict) -> float:
    f = row["features"]
    c1h = _g(f, "1h_current_return_pct")
    c15 = _g(f, "15m_current_return_pct")
    if c1h <= 0:
        return 0.0
    return c1h * 2 + c15 + _g(f, "2h_current_return_pct") * 0.3


def _score_long_accel(row: dict) -> float:
    f = row["features"]
    return (
        _g(f, "5m_momentum") * 2
        + _g(f, "5m_seq_return_sum_6") * 0.5
        + _g(f, "15m_current_return_pct")
        + _g(f, "5m_range_energy") * 0.1
    )


def _score_a6_long(row: dict) -> float:
    return _g(row["features"], "h4_score")


def _score_short_top(row: dict) -> float:
    f = row["features"]
    p1h = _g(f, "1h_previous_return_pct")
    comp = _g(f, "5m_compression")
    c15 = _g(f, "15m_current_return_pct")
    if p1h <= 0:
        return 0.0
    return comp * 2 + abs(c15) * 1.5 + p1h * 0.3 + _g(f, "1h_current_range_pct") * 0.05


def _score_short_v(row: dict) -> float:
    f = row["features"]
    p15 = _g(f, "15m_previous_return_pct")
    c15 = _g(f, "15m_current_return_pct")
    if p15 <= 0:
        return abs(c15) * 0.1
    return (p15 - c15) * 2 + _g(f, "15m_current_volume_ratio") * 0.5


def _score_short_cont(row: dict) -> float:
    f = row["features"]
    c1h = _g(f, "1h_current_return_pct")
    c15 = _g(f, "15m_current_return_pct")
    if c1h >= 0:
        return 0.0
    return abs(c1h) * 2 + abs(c15) + abs(_g(f, "2h_current_return_pct")) * 0.3


def _score_short_accel(row: dict) -> float:
    f = row["features"]
    return (
        abs(min(0, _g(f, "5m_momentum"))) * 2
        + abs(min(0, _g(f, "5m_seq_return_sum_6"))) * 0.5
        + abs(min(0, _g(f, "15m_current_return_pct")))
    )


LONG_SCORERS: dict[str, Callable[[dict], float]] = {
    "LONG_BASE_REVERSAL": _score_long_base,
    "LONG_V_REVERSAL": _score_long_v,
    "LONG_CONTINUATION": _score_long_cont,
    "LONG_ACCELERATION": _score_long_accel,
    "A6_LONG": _score_a6_long,
}

SHORT_SCORERS: dict[str, Callable[[dict], float]] = {
    "SHORT_TOP_REVERSAL": _score_short_top,
    "SHORT_V_REVERSAL": _score_short_v,
    "SHORT_CONTINUATION": _score_short_cont,
    "SHORT_ACCELERATION": _score_short_accel,
}


def rank_long(rows: list[dict], engine: str, top_k: int = 5) -> list[str]:
    if engine == "RANDOM_LONG":
        syms = [r["symbol"] for r in rows]
        return random.sample(syms, min(top_k, len(syms))) if len(syms) > top_k else syms[:top_k]
    scorer = LONG_SCORERS.get(engine)
    if not scorer:
        return []
    ranked = sorted(rows, key=scorer, reverse=True)
    return [r["symbol"] for r in ranked[:top_k]]


def rank_short(rows: list[dict], engine: str, top_k: int = 5) -> list[str]:
    if engine == "RANDOM_SHORT":
        syms = [r["symbol"] for r in rows]
        return random.sample(syms, min(top_k, len(syms))) if len(syms) > top_k else syms[:top_k]
    scorer = SHORT_SCORERS.get(engine)
    if not scorer:
        return []
    ranked = sorted(rows, key=scorer, reverse=True)
    return [r["symbol"] for r in ranked[:top_k]]
