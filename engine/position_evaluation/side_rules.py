"""Side-aware ROI, MFE/MAE, protective stop, and market state signals."""

from __future__ import annotations

import statistics

from scout_research_r006_pilot_execution_engine import Bar


def normalize_side(side: str) -> str:
    s = (side or "LONG").upper()
    return "SHORT" if s == "SHORT" else "LONG"


def roi_pct(side: str, entry: float, current: float) -> float:
    if entry <= 0 or current <= 0:
        return 0.0
    raw = (current - entry) / entry * 100
    return round(-raw if normalize_side(side) == "SHORT" else raw, 4)


def mfe_mae_from_bars(side: str, bars: list[Bar], entry: float) -> tuple[float, float]:
    if not bars or entry <= 0:
        return 0.0, 0.0
    side = normalize_side(side)
    highs = [b.h for b in bars]
    lows = [b.l for b in bars]
    if side == "LONG":
        mfe = (max(highs) - entry) / entry * 100
        mae = (min(lows) - entry) / entry * 100
    else:
        mfe = (entry - min(lows)) / entry * 100
        mae = (entry - max(highs)) / entry * 100
    return round(mfe, 4), round(mae, 4)


def protective_stop_hit(side: str, bars: list[Bar], entry: float, stop_pct: float) -> bool:
    if not bars or entry <= 0 or stop_pct <= 0:
        return False
    side = normalize_side(side)
    last = bars[-1]
    if side == "LONG":
        sl_px = entry * (1 - stop_pct / 100)
        return last.l <= sl_px
    sl_px = entry * (1 + stop_pct / 100)
    return last.h >= sl_px


def side_state_signals(side: str, bars: list[Bar], entry_i: int = 0) -> dict:
    """Side-aware momentum/trend/volume signals (no forward leak — bars only up to now)."""
    side = normalize_side(side)
    if not bars or len(bars) <= entry_i:
        return {
            "momentum_alive": False,
            "volume_alive": False,
            "trend_alive": False,
            "range_expansion": False,
            "reversal_warning": False,
        }

    i = len(bars) - 1
    c = bars[i].c
    i1h = max(entry_i, i - 11)
    i2h = max(entry_i, i - 23)
    ret_1h = (c - bars[i1h].o) / bars[i1h].o * 100 if bars[i1h].o else 0
    ret_2h = (c - bars[i2h].o) / bars[i2h].o * 100 if bars[i2h].o else 0
    closes = [bars[j].c for j in range(max(entry_i, i - 19), i + 1)]
    ma20 = statistics.mean(closes) if closes else c

    if side == "LONG":
        trend_alive = ret_2h > 0 or c >= ma20
        momentum_alive = ret_1h > 0
        reversal_warning = ret_1h < -0.5 and ret_2h > 0
    else:
        trend_alive = ret_2h < 0 or c <= ma20
        momentum_alive = ret_1h < 0
        reversal_warning = ret_1h > 0.5 and ret_2h < 0

    rng_now = (bars[i].h - bars[i].l) / bars[i].o * 100 if bars[i].o else 0
    rng_prev = statistics.mean(
        [(bars[j].h - bars[j].l) / bars[j].o * 100 for j in range(max(entry_i, i - 6), i)]
    ) if i > entry_i else rng_now
    range_expansion = rng_now > rng_prev * 1.05

    vol_proxy = bars[i].h - bars[i].l
    vol_ma = statistics.mean([bars[j].h - bars[j].l for j in range(max(entry_i, i - 19), i)]) if i > entry_i else vol_proxy
    vol_ratio = vol_proxy / vol_ma if vol_ma else 1.0
    volume_alive = vol_ratio >= 0.75

    if side == "SHORT" and ret_1h > 1.0 and vol_ratio > 1.2:
        reversal_warning = True

    return {
        "momentum_alive": momentum_alive,
        "volume_alive": volume_alive,
        "trend_alive": trend_alive,
        "range_expansion": range_expansion,
        "reversal_warning": reversal_warning,
        "ret_1h_proxy": round(ret_1h, 4),
        "vol_ratio": round(vol_ratio, 4),
    }


def compute_side_alive_score(side: str, bars: list[Bar], entry_i: int = 0) -> float:
    sig = side_state_signals(side, bars, entry_i)
    score = 0.0
    if sig["trend_alive"]:
        score += 25
    if sig["momentum_alive"]:
        score += 25
    if sig["volume_alive"]:
        score += 25
    if sig["range_expansion"]:
        score += 15
    if sig["reversal_warning"]:
        score -= 20
    return round(max(0.0, min(100.0, score)), 2)
