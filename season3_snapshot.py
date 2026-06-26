"""
Scout Season3 - Snapshot Feature Extraction

Captures market state as numeric snapshot. No rule labels. No process variables.
"""

from __future__ import annotations

import math
import statistics
from datetime import datetime, timedelta

import top10_gainer_learning_20260613 as t10

FEATURE_KEYS = (
    "open", "high", "low", "close", "volume",
    "ha_open", "ha_close", "ha_body_ratio",
    "volume_ma6", "volume_ma12", "volume_ma24", "volume_ratio_ma24",
    "dollar_volume", "dollar_volume_ma24", "dollar_volume_ratio",
    "atr_pct", "atr_ratio", "range_pct", "body_ratio",
    "upper_wick_ratio", "lower_wick_ratio", "range_compression",
    "ma5_dist_pct", "ma10_dist_pct", "ma20_dist_pct", "ma60_dist_pct",
    "ma5_slope_pct", "ma20_slope_pct", "ma60_slope_pct",
    "vwap_dist_pct",
    "return_15m_pct", "return_30m_pct", "return_1h_pct", "return_2h_pct",
    "return_4h_pct", "return_24h_pct",
    "prev1_body_ratio", "prev3_mean_body", "prev6_mean_range",
    "btc_return_24h", "btc_return_2h", "btc_atr_pct", "btc_stable_flag",
)


def ohlcv(k: list) -> tuple[float, float, float, float, float]:
    return float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])


def heikin_ashi_series(klines: list[list]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    ha_open = float(klines[0][1])
    for k in klines:
        o, h, l, c, _ = ohlcv(k)
        ha_close = (o + h + l + c) / 4.0
        ha_open = (ha_open + ha_close) / 2.0 if out else o
        out.append((ha_open, ha_close))
        ha_open = (ha_open + ha_close) / 2.0
    return out


def ma(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def slope_pct(klines: list[list], period: int, shift: int = 3) -> float:
    if len(klines) < period + shift + 1:
        return 0.0
    closes = [float(k[4]) for k in klines]
    recent = ma(closes[-period - 1 : -1])
    prior = ma(closes[-period - shift - 1 : -shift - 1])
    if prior == 0:
        return 0.0
    return (recent - prior) / prior * 100


def return_over(klines: list[list], n_back: int) -> float:
    if len(klines) <= n_back:
        return 0.0
    c0 = float(klines[-1][4])
    c1 = float(klines[-1 - n_back][4])
    if c1 == 0:
        return 0.0
    return (c0 - c1) / c1 * 100


def compression_ratio(klines: list[list], window: int = 6) -> float:
    if len(klines) < window * 2 + 1:
        return 1.0
    recent = klines[-window - 1 : -1]
    prior = klines[-window * 2 - 1 : -window - 1]
    r_recent = statistics.mean(t10.candle_range_metric(k) for k in recent)
    r_prior = statistics.mean(t10.candle_range_metric(k) for k in prior)
    if r_prior <= 0:
        return 1.0
    return r_recent / r_prior


def vwap_distance(klines: list[list], window: int = 24) -> float:
    if len(klines) < window + 1:
        return 0.0
    chunk = klines[-window - 1 : -1]
    num = sum(((ohlcv(k)[1] + ohlcv(k)[2] + ohlcv(k)[3]) / 3.0) * ohlcv(k)[4] for k in chunk)
    den = sum(ohlcv(k)[4] for k in chunk)
    if den <= 0:
        return 0.0
    vwap = num / den
    close = float(klines[-1][4])
    return (close - vwap) / vwap * 100 if vwap else 0.0


def build_snapshot_features(
    klines_2h: list[list],
    klines_1h: list[list] | None = None,
    klines_15m: list[list] | None = None,
    btc_metrics: dict | None = None,
) -> dict[str, float] | None:
    if len(klines_2h) < t10.RANKING_KLINES_2H:
        return None

    signal = klines_2h[-1]
    o, h, l, c, vol = ohlcv(signal)
    if not (t10.MIN_PRICE <= c <= t10.MAX_PRICE):
        return None

    prev_24 = klines_2h[-(t10.CANDLES_24H_2H + 1) : -1]
    vols = [ohlcv(k)[4] for k in klines_2h[-25:-1]]
    vol_ma6 = ma(vols[-6:])
    vol_ma12 = ma(vols[-12:])
    vol_ma24 = ma(vols[-24:]) if len(vols) >= 24 else ma(vols)
    dollar = c * vol
    dollar_ma24 = ma([ohlcv(k)[4] * ohlcv(k)[3] for k in klines_2h[-25:-1]])

    body = abs(c - o) / o * 100 if o else 0
    rng = (h - l) / o * 100 if o else 0
    upper = (h - max(o, c)) / o * 100 if o else 0
    lower = (min(o, c) - l) / o * 100 if o else 0

    ha = heikin_ashi_series(klines_2h[-10:])
    ha_o, ha_c = ha[-1]
    ha_body = abs(ha_c - ha_o) / c * 100 if c else 0

    atr = t10.average_true_range_percent(prev_24, c)
    cur_tr = t10.true_range(signal, float(klines_2h[-2][4]))
    atr_now = cur_tr / c * 100 if c else 0

    ret_15m = ret_30m = 0.0
    if klines_15m and len(klines_15m) >= 3:
        ret_15m = return_over(klines_15m, 1)
        ret_30m = return_over(klines_15m, 2)
    elif klines_1h and len(klines_1h) >= 2:
        ret_30m = return_over(klines_1h, 1)

    ret_1h = return_over(klines_1h, 1) if klines_1h and len(klines_1h) >= 2 else return_over(klines_2h, 1)
    ret_2h = return_over(klines_2h, 1)
    ret_4h = return_over(klines_2h, 2)
    ret_24h = return_over(klines_2h, t10.CANDLES_24H_2H)

    btc_r24 = (btc_metrics or {}).get("btc_return_24h", 0.0)
    btc_r2 = (btc_metrics or {}).get("btc_return_2h", 0.0)
    btc_atr = (btc_metrics or {}).get("btc_atr_pct", 0.0)
    btc_stable = 1.0 if abs(btc_r2) < 0.8 and abs(btc_r24) < 3.0 else 0.0

    snap: dict[str, float] = {
        "open": o, "high": h, "low": l, "close": c, "volume": vol,
        "ha_open": ha_o, "ha_close": ha_c, "ha_body_ratio": ha_body,
        "volume_ma6": vol_ma6, "volume_ma12": vol_ma12, "volume_ma24": vol_ma24,
        "volume_ratio_ma24": vol / vol_ma24 if vol_ma24 > 0 else 0,
        "dollar_volume": dollar,
        "dollar_volume_ma24": dollar_ma24,
        "dollar_volume_ratio": dollar / dollar_ma24 if dollar_ma24 > 0 else 0,
        "atr_pct": atr_now, "atr_ratio": atr_now / atr if atr > 0 else 0,
        "range_pct": rng, "body_ratio": body / rng if rng > 0 else 0,
        "upper_wick_ratio": upper / rng if rng > 0 else 0,
        "lower_wick_ratio": lower / rng if rng > 0 else 0,
        "range_compression": compression_ratio(klines_2h),
        "ma5_dist_pct": t10.distance_from_ma_percent(c, ma([float(k[4]) for k in klines_2h[-6:-1]])),
        "ma10_dist_pct": t10.distance_from_ma_percent(c, ma([float(k[4]) for k in klines_2h[-11:-1]])),
        "ma20_dist_pct": t10.distance_from_ma_percent(c, ma([float(k[4]) for k in klines_2h[-21:-1]])),
        "ma60_dist_pct": t10.distance_from_ma_percent(c, ma([float(k[4]) for k in klines_2h[-61:-1]])) if len(klines_2h) >= 62 else 0,
        "ma5_slope_pct": slope_pct(klines_2h, 5),
        "ma20_slope_pct": slope_pct(klines_2h, 20),
        "ma60_slope_pct": slope_pct(klines_2h, min(60, len(klines_2h) - 2)),
        "vwap_dist_pct": vwap_distance(klines_2h),
        "return_15m_pct": ret_15m, "return_30m_pct": ret_30m,
        "return_1h_pct": ret_1h, "return_2h_pct": ret_2h,
        "return_4h_pct": ret_4h, "return_24h_pct": ret_24h,
        "prev1_body_ratio": t10.candle_body_metric(klines_2h[-2]) if len(klines_2h) >= 2 else 0,
        "prev3_mean_body": ma([t10.candle_body_metric(k) for k in klines_2h[-4:-1]]),
        "prev6_mean_range": ma([t10.candle_range_metric(k) for k in klines_2h[-7:-1]]),
        "btc_return_24h": btc_r24, "btc_return_2h": btc_r2,
        "btc_atr_pct": btc_atr, "btc_stable_flag": btc_stable,
    }
    return snap


def forward_outcomes(
    entry: float,
    forward_klines: list[list],
    scan_dt: datetime,
    interval_hours: float = 2.0,
) -> dict[str, float]:
    """Max excursion / drawdown at 30m, 1h, 2h, 4h horizons (2h candle grid approximated)."""
    if entry <= 0:
        return {}

    horizons = {
        "30m": timedelta(minutes=30),
        "1h": timedelta(hours=1),
        "2h": timedelta(hours=2),
        "4h": timedelta(hours=4),
    }
    out: dict[str, float] = {}
    max_high = entry
    min_low = entry

    for label, delta in horizons.items():
        target = scan_dt + delta
        hi, lo = entry, entry
        for candle in forward_klines:
            if t10.kline_close_dt(candle) > target:
                break
            _, h, l, _, _ = ohlcv(candle)
            hi = max(hi, h)
            lo = min(lo, l)
        out[f"max_excursion_{label}"] = (hi - entry) / entry * 100
        out[f"max_drawdown_{label}"] = (entry - lo) / entry * 100

    # Best excursion in 4h window
    for candle in forward_klines:
        if t10.kline_close_dt(candle) > scan_dt + timedelta(hours=4):
            break
        _, h, l, _, _ = ohlcv(candle)
        max_high = max(max_high, h)
        min_low = min(min_low, l)

    best = (max_high - entry) / entry * 100
    out["max_excursion_best"] = best
    out["max_drawdown_best"] = (entry - min_low) / entry * 100
    out["hit_5pct_plus"] = 1.0 if best >= 5.0 else 0.0
    out["hit_10pct_plus"] = 1.0 if best >= 10.0 else 0.0
    return out
