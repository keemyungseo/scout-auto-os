"""Independent candidate engines — each produces its own TOP-N ranking."""

from __future__ import annotations

import random
from typing import Callable

CANDIDATE_ENGINES = (
    "RANDOM_BASELINE",
    "A6_CURRENT",
    "PURE_MOMENTUM_15M",
    "PURE_MOMENTUM_1H",
    "VOLUME_EXPLOSION",
    "QUOTE_VOLUME_LEADER",
    "BREAKOUT_15M_HIGH",
    "BREAKOUT_1H_HIGH",
    "COMPRESSION_RELEASE",
    "RANGE_EXPANSION",
    "VWAP_RECLAIM",
    "PULLBACK_CONTINUATION",
    "REVERSAL_AFTER_DUMP",
    "HIGH_VOLUME_GREEN_CANDLE",
    "MULTI_TIMEFRAME_ALIGNMENT",
    "BTC_RELATIVE_STRENGTH",
    "ALT_ROTATION_STRENGTH",
)

EVAL_INTERVALS = ("5m", "15m", "30m", "1h")
TRAIN_CUTOFF = "2026-06-01"
TOP_K_OPTIONS = (5, 10, 20)


def _g(f: dict, key: str, default: float = 0.0) -> float:
    return float(f.get(key, default))


def _score_a6(row: dict, ctx: dict) -> float:
    return _g(row["features"], "h4_score")


def _score_mom15(row: dict, ctx: dict) -> float:
    return _g(row["features"], "15m_current_return_pct")


def _score_mom1h(row: dict, ctx: dict) -> float:
    return _g(row["features"], "1h_current_return_pct")


def _score_vol_explosion(row: dict, ctx: dict) -> float:
    f = row["features"]
    return _g(f, "15m_current_volume_ratio") * 2 + _g(f, "5m_seq_volume_energy_6")


def _score_quote_vol(row: dict, ctx: dict) -> float:
    f = row["features"]
    return _g(f, "30m_current_volume_ratio") * _g(f, "1h_current_volume_ratio")


def _score_breakout_15m(row: dict, ctx: dict) -> float:
    f = row["features"]
    return _g(f, "15m_current_close_position") * _g(f, "15m_current_range_pct")


def _score_breakout_1h(row: dict, ctx: dict) -> float:
    f = row["features"]
    return _g(f, "1h_current_close_position") * _g(f, "1h_current_range_pct")


def _score_compression(row: dict, ctx: dict) -> float:
    f = row["features"]
    comp = max(0.1, _g(f, "5m_compression"))
    return _g(f, "5m_release") * 3 + _g(f, "5m_range_energy") / comp


def _score_range_exp(row: dict, ctx: dict) -> float:
    f = row["features"]
    return _g(f, "1h_current_range_pct") + _g(f, "30m_current_range_pct") * 0.5


def _score_vwap(row: dict, ctx: dict) -> float:
    f = row["features"]
    return _g(f, "15m_current_close_position") * 2 + max(0, _g(f, "15m_current_ma20_distance_pct"))


def _score_pullback(row: dict, ctx: dict) -> float:
    f = row["features"]
    prev = _g(f, "15m_previous_return_pct")
    cur = _g(f, "15m_current_return_pct")
    if prev < 0 and cur > 0:
        return cur - prev
    return cur * 0.1


def _score_reversal(row: dict, ctx: dict) -> float:
    f = row["features"]
    dump = _g(f, "2h_previous_return_pct")
    bounce = _g(f, "15m_current_return_pct")
    if dump < -1 and bounce > 0:
        return bounce - dump * 0.5
    return bounce * 0.05


def _score_hv_green(row: dict, ctx: dict) -> float:
    f = row["features"]
    ret = max(0, _g(f, "15m_current_return_pct"))
    return _g(f, "15m_current_volume_ratio") * ret


def _score_mtf(row: dict, ctx: dict) -> float:
    f = row["features"]
    total = 0.0
    for k in ("15m_current_return_pct", "30m_current_return_pct", "1h_current_return_pct", "2h_current_return_pct"):
        v = _g(f, k)
        if v > 0:
            total += v
    return total


def _score_btc_rel(row: dict, ctx: dict) -> float:
    return _g(row["features"], "1h_current_return_pct") - ctx.get("median_1h_ret", 0)


def _score_alt_rot(row: dict, ctx: dict) -> float:
    rel = _g(row["features"], "1h_current_return_pct") - ctx.get("median_1h_ret", 0)
    return rel * 1.5 if ctx.get("btc_4h_weak", False) else rel * 0.5


SCORERS: dict[str, Callable[[dict, dict], float]] = {
    "A6_CURRENT": _score_a6,
    "PURE_MOMENTUM_15M": _score_mom15,
    "PURE_MOMENTUM_1H": _score_mom1h,
    "VOLUME_EXPLOSION": _score_vol_explosion,
    "QUOTE_VOLUME_LEADER": _score_quote_vol,
    "BREAKOUT_15M_HIGH": _score_breakout_15m,
    "BREAKOUT_1H_HIGH": _score_breakout_1h,
    "COMPRESSION_RELEASE": _score_compression,
    "RANGE_EXPANSION": _score_range_exp,
    "VWAP_RECLAIM": _score_vwap,
    "PULLBACK_CONTINUATION": _score_pullback,
    "REVERSAL_AFTER_DUMP": _score_reversal,
    "HIGH_VOLUME_GREEN_CANDLE": _score_hv_green,
    "MULTI_TIMEFRAME_ALIGNMENT": _score_mtf,
    "BTC_RELATIVE_STRENGTH": _score_btc_rel,
    "ALT_ROTATION_STRENGTH": _score_alt_rot,
}


def scan_context(rows: list[dict]) -> dict:
    rets = [_g(r["features"], "1h_current_return_pct") for r in rows]
    rets_sorted = sorted(rets)
    med = rets_sorted[len(rets_sorted) // 2] if rets_sorted else 0.0
    return {"median_1h_ret": med, "btc_4h_weak": med < 0}


def rank_engine(
    rows: list[dict],
    engine: str,
    top_k: int = 5,
    ctx: dict | None = None,
) -> list[str]:
    if engine == "RANDOM_BASELINE":
        syms = [r["symbol"] for r in rows]
        if len(syms) <= top_k:
            return syms
        return random.sample(syms, top_k)
    scorer = SCORERS.get(engine)
    if not scorer:
        return []
    ctx = ctx or scan_context(rows)
    ranked = sorted(rows, key=lambda r: scorer(r, ctx), reverse=True)
    return [r["symbol"] for r in ranked[:top_k]]


def rank_all_engines(
    rows: list[dict],
    top_k: int = 5,
) -> dict[str, list[str]]:
    ctx = scan_context(rows)
    return {eng: rank_engine(rows, eng, top_k, ctx) for eng in CANDIDATE_ENGINES if eng != "RANDOM_BASELINE"}
