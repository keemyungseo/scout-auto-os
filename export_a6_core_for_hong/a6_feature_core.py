"""
A6 feature extraction core.
Original sources:
  - scout_phase19_winner_ranking_dna.py (extract_dna_features, tf_pair, Pattern B)
  - scout_phase13_5m_sequence_ignition.py (window_seq, compression_length, compute_at_anchor)
  - scout_phase16_human_blind_test.py (macd_sig, lifecycle_15m, h4_score)
"""

from __future__ import annotations

import statistics
from typing import TYPE_CHECKING

from export_a6_core_for_hong.a6_common import (
    MAX_PRICE,
    MIN_PRICE,
    PATTERN_B_MACD_MIN,
    PATTERN_B_RANGE_MIN,
    RANK_WEIGHTS,
    ema,
    g,
    ohlcv,
    slice_klines_to_end,
)

if TYPE_CHECKING:
    import pandas as pd


def df_to_klines(df: "pd.DataFrame") -> list[list]:
    """
    Convert OHLCV dataframe to Binance kline list format.
    Required columns: open_time, open, high, low, close, volume
    open_time: milliseconds (int) or datetime convertible to ms.
    """
    import pandas as pd

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    out: list[list] = []
    for row in df.itertuples(index=False):
        d = row._asdict() if hasattr(row, "_asdict") else dict(zip(df.columns, row))
        ts = d.get("open_time", d.get("timestamp"))
        if hasattr(ts, "timestamp"):
            t_ms = int(ts.timestamp() * 1000)
        else:
            t_ms = int(ts)
        out.append([
            t_ms,
            float(d["open"]),
            float(d["high"]),
            float(d["low"]),
            float(d["close"]),
            float(d["volume"]),
        ])
    out.sort(key=lambda x: x[0])
    return out


def pattern_b_pass(macd_signal: float, range_pct_15m: float, price: float) -> bool:
    # Original: scout_phase19_winner_ranking_dna.extract_dna_features filter block
    if not (MIN_PRICE <= price <= MAX_PRICE):
        return False
    if macd_signal < PATTERN_B_MACD_MIN:
        return False
    if range_pct_15m < PATTERN_B_RANGE_MIN:
        return False
    return True


def ma_slope(closes: list[float]) -> float:
    # Original: scout_phase13_5m_sequence_ignition.ma_slope
    if len(closes) < 12:
        return 0.0
    recent = statistics.mean(closes[-6:-1])
    prior = statistics.mean(closes[-12:-7]) if len(closes) >= 12 else recent
    return (recent - prior) / prior * 100 if prior else 0.0


def compression_length(klines: list[list], end_i: int, max_rng: float = 1.5) -> int:
    # Original: scout_phase13_5m_sequence_ignition.compression_length
    n = 0
    for i in range(end_i - 1, max(0, end_i - 48), -1):
        o, h, l, _, _ = ohlcv(klines[i])
        if o <= 0 or (h - l) / o * 100 > max_rng:
            break
        n += 1
    return n


def candle_row(k: list, vol_ma: float) -> dict:
    # Original: scout_phase13_5m_sequence_ignition.candle_row
    o, h, l, c, vol = ohlcv(k)
    ret = (c - o) / o * 100 if o else 0.0
    rng = (h - l) / o * 100 if o else 0.0
    body = abs(c - o) / o * 100 if o else 0.0
    br = body / rng if rng > 0 else 0.0
    vr = vol / vol_ma if vol_ma > 0 else 0.0
    cp = (c - l) / (h - l) if h > l else 0.5
    return {
        "return_pct": ret,
        "range_pct": rng,
        "body_ratio": br,
        "volume_ma_ratio": vr,
        "close_position": cp,
        "positive": ret > 0,
        "pos_ret": max(ret, 0),
    }


def window_seq(klines: list[list], end_i: int, n: int) -> dict:
    # Original: scout_phase13_5m_sequence_ignition.window_seq
    if end_i < n or end_i >= len(klines):
        return {}
    vols = [ohlcv(klines[j])[4] for j in range(max(0, end_i - 24), end_i)]
    vol_ma = statistics.mean(vols[-20:]) if vols else 1.0

    rows = []
    for i in range(end_i - n + 1, end_i + 1):
        rows.append(candle_row(klines[i], vol_ma))

    return_sum = sum(r["return_pct"] for r in rows)
    pos_count = sum(1 for r in rows if r["positive"])
    vol_persist = sum(1 for r in rows if r["volume_ma_ratio"] >= 1.0) / n
    vol_energy = sum(r["volume_ma_ratio"] * r["pos_ret"] for r in rows)
    body_energy = sum(r["body_ratio"] * r["pos_ret"] for r in rows)
    range_energy = sum(r["range_pct"] * r["pos_ret"] for r in rows)
    close_strength = sum(1 for r in rows if r["close_position"] >= 0.6) / n

    closes = [float(klines[i][4]) for i in range(max(0, end_i - 24), end_i + 1)]
    slope_now = ma_slope(closes)
    slope_prior = ma_slope(closes[:-n]) if len(closes) > n + 6 else slope_now
    slope_delta = slope_now - slope_prior
    slope_accel = slope_delta / max(abs(slope_prior), 0.01)

    comp_len = compression_length(klines, end_i - n + 1)
    pre_start = max(0, end_i - n - 12)
    pre = [candle_row(klines[i], vol_ma) for i in range(pre_start, end_i - n + 1)]
    comp_release = 0
    if pre and rows:
        pre_rng = statistics.mean([x["range_pct"] for x in pre[-12:]])
        pre_vol = statistics.mean([x["volume_ma_ratio"] for x in pre[-12:]])
        rec_rng = statistics.mean([x["range_pct"] for x in rows])
        rec_vol = statistics.mean([x["volume_ma_ratio"] for x in rows])
        if comp_len >= 4 and rec_rng > pre_rng * 1.2 and rec_vol > pre_vol * 1.1:
            comp_release = 1

    first_abnormal = 0
    for i, r in enumerate(rows):
        if i == 0:
            continue
        prev = rows[i - 1]
        if (
            r["range_pct"] > prev["range_pct"]
            and r["body_ratio"] > prev["body_ratio"]
            and r["volume_ma_ratio"] > prev["volume_ma_ratio"]
        ):
            first_abnormal = 1
            break

    return {
        f"seq_return_sum_{n}": round(return_sum, 4),
        f"seq_positive_count_{n}": pos_count,
        f"seq_volume_persistence_{n}": round(vol_persist, 4),
        f"seq_volume_energy_{n}": round(vol_energy, 4),
        f"seq_body_energy_{n}": round(body_energy, 4),
        f"seq_range_energy_{n}": round(range_energy, 4),
        f"seq_slope_delta_{n}": round(slope_delta, 4),
        f"seq_slope_accel_{n}": round(slope_accel, 4),
        f"seq_close_strength_{n}": round(close_strength, 4),
        f"seq_compression_release_{n}": float(comp_release),
        f"first_abnormal_candle_{n}": float(first_abnormal),
    }


def compute_at_anchor(klines: list[list], anchor_i: int) -> dict:
    # Original: scout_phase13_5m_sequence_ignition.compute_at_anchor
    feats: dict = {}
    o, h, l, c, vol = ohlcv(klines[anchor_i])
    vols = [ohlcv(klines[j])[4] for j in range(max(0, anchor_i - 24), anchor_i)]
    vol_ma = statistics.mean(vols[-20:]) if vols else 1.0
    feats["volume_ma_ratio"] = round(vol / vol_ma if vol_ma else 0, 4)
    feats["close_position"] = round((c - l) / (h - l) if h > l else 0.5, 4)
    feats["compression_length"] = float(compression_length(klines, anchor_i))
    for n in (3, 6, 9, 12):
        feats.update(window_seq(klines, anchor_i, n))
    return feats


def macd_sig(closes: list[float]) -> float:
    # Original: scout_phase16_human_blind_test.macd_sig
    if len(closes) < 26:
        return 0.0
    hist = [ema(closes[:i], 12) - ema(closes[:i], 26) for i in range(26, len(closes) + 1)]
    return ema(hist, 9) if hist else 0.0


def ma_slope_pct(klines: list[list]) -> float:
    # Original: scout_phase16_human_blind_test.ma_slope_pct
    if len(klines) < 14:
        return 0.0
    closes = [float(k[4]) for k in klines]
    recent = statistics.mean(closes[-6:-1])
    prior = statistics.mean(closes[-12:-7]) if len(closes) >= 12 else recent
    return (recent - prior) / prior * 100 if prior else 0.0


def lifecycle_15m(klines: list[list]) -> dict:
    # Original: scout_phase16_human_blind_test.lifecycle_15m
    anchor = len(klines) - 1
    window = klines[max(0, anchor - 48): anchor + 1]
    base = min(ohlcv(k)[2] for k in window) if window else ohlcv(klines[-1])[2]
    birth_i = ign_i = anchor
    for i in range(anchor, max(0, anchor - 32), -1):
        if ohlcv(klines[i])[4] >= base * 1.03:
            birth_i = i
            break
    for i in range(anchor, max(0, anchor - 24), -1):
        vols = [ohlcv(klines[j])[4] for j in range(max(0, i - 8), i)]
        if vols and statistics.mean(vols) > 0 and ohlcv(klines[i])[4] / statistics.mean(vols) >= 1.2:
            ign_i = i
            break
    slope = ma_slope_pct(klines)
    slope_p = ma_slope_pct(klines[:-4]) if len(klines) > 20 else slope
    return {
        "birth_age_min": float((anchor - birth_i) * 15),
        "ignition_age_min": float((anchor - ign_i) * 15),
        "young_birth": 1.0 if (anchor - birth_i) * 15 <= 45 else 0.0,
        "ma_slope_accel": slope - slope_p,
        "ma_slope": slope,
    }


def h4_score(lc: dict, vol_ratio: float) -> float:
    # Original: scout_phase16_human_blind_test.h4_score
    ff = {
        "young_birth": lc.get("young_birth", 0),
        "birth_age_min": lc.get("birth_age_min", 0),
        "ignition_age_min": lc.get("ignition_age_min", 0),
        "ma_slope_accel": lc.get("ma_slope_accel", 0),
        "volume_ma_ratio": vol_ratio,
    }
    return sum(RANK_WEIGHTS[k] * ff.get(k, 0) for k in RANK_WEIGHTS)


def tf_pair(klines: list[list]) -> dict | None:
    # Original: scout_phase19_winner_ranking_dna.tf_pair
    if len(klines) < 25:
        return None
    anchor = len(klines) - 1

    def candle(i: int) -> dict:
        o, h, l, c, vol = ohlcv(klines[i])
        vols = [ohlcv(klines[j])[4] for j in range(max(0, i - 24), i)]
        vol_ma = statistics.mean(vols[-20:]) if vols else vol
        closes = [float(klines[j][4]) for j in range(max(0, i - 21), i + 1)]
        ma20 = statistics.mean(closes[-20:]) if len(closes) >= 20 else c
        ret = (c - o) / o * 100 if o else 0
        rng = (h - l) / o * 100 if o else 0
        body = abs(c - o) / o * 100 if o else 0
        comp = 0
        for j in range(i - 1, max(0, i - 20), -1):
            jo, jh, jl, _, _ = ohlcv(klines[j])
            if jo > 0 and (jh - jl) / jo * 100 <= 2.0:
                comp += 1
            else:
                break
        return {
            "volume_ratio": round(vol / vol_ma if vol_ma else 0, 4),
            "return_pct": round(ret, 4),
            "body_pct": round(body, 4),
            "close_position": round((c - l) / (h - l) if h > l else 0.5, 4),
            "range_pct": round(rng, 4),
            "compression_length": float(comp),
            "ma20_distance_pct": round((c - ma20) / ma20 * 100 if ma20 else 0, 4),
        }

    return {"current": candle(anchor), "previous": candle(anchor - 1)}


def extract_dna_features_from_klines(
    k5: list[list],
    k15: list[list],
    end_ms: int,
    *,
    k30: list[list] | None = None,
    k1h: list[list] | None = None,
    k2h: list[list] | None = None,
) -> dict | None:
    """
    Core feature extraction (no network).
    Original: scout_phase19_winner_ranking_dna.extract_dna_features
    """
    k5 = slice_klines_to_end(k5, end_ms)
    k15 = slice_klines_to_end(k15, end_ms)
    if len(k5) < 40 or len(k15) < 30:
        return None

    o, h, l, c, vol = ohlcv(k15[-1])
    rng = (h - l) / o * 100 if o else 0
    closes15 = [float(k[4]) for k in k15]
    ms = macd_sig(closes15)
    if not pattern_b_pass(ms, rng, c):
        return None

    anchor = len(k5) - 1
    seq6 = window_seq(k5, anchor, 6)
    seq_feats = compute_at_anchor(k5, anchor)
    closes5 = [float(k5[i][4]) for i in range(max(0, anchor - 24), anchor + 1)]
    momentum5 = ma_slope(closes5)

    feats: dict[str, float] = {
        "5m_volume_ma_ratio": seq_feats.get("volume_ma_ratio", 0),
        "5m_seq_volume_energy_6": seq6.get("seq_volume_energy_6", 0),
        "5m_seq_return_sum_6": seq6.get("seq_return_sum_6", 0),
        "5m_seq_body_energy_6": seq6.get("seq_body_energy_6", 0),
        "5m_seq_positive_count_6": float(seq6.get("seq_positive_count_6", 0)),
        "5m_first_abnormal_candle_6": seq6.get("first_abnormal_candle_6", 0),
        "5m_compression": float(compression_length(k5, anchor)),
        "5m_release": seq6.get("seq_compression_release_6", 0),
        "5m_body_position": seq_feats.get("close_position", 0),
        "5m_range_energy": seq6.get("seq_range_energy_6", 0),
        "5m_momentum": momentum5,
    }

    tf_map = {
        "15m": k15,
        "30m": k30 or k15,
        "1h": k1h or k15,
        "2h": k2h or k15,
    }
    for label, kl in tf_map.items():
        pair = tf_pair(kl)
        if not pair:
            continue
        for side in ("current", "previous"):
            p = pair[side]
            prefix = f"{label}_{side}"
            feats[f"{prefix}_volume_ratio"] = p["volume_ratio"]
            feats[f"{prefix}_return_pct"] = p["return_pct"]
            feats[f"{prefix}_body_pct"] = p["body_pct"]
            feats[f"{prefix}_close_position"] = p["close_position"]
            feats[f"{prefix}_range_pct"] = p["range_pct"]
            feats[f"{prefix}_compression"] = p["compression_length"]
            feats[f"{prefix}_ma20_distance_pct"] = p["ma20_distance_pct"]

    vols = [ohlcv(k)[4] for k in k15[-25:-1]]
    vol_ma = statistics.mean(vols[-24:]) if vols else 0.0
    lc = lifecycle_15m(k15)
    feats["h4_score"] = h4_score(lc, vol / vol_ma if vol_ma else 0)
    feats["price"] = c
    return feats


def extract_dna_features_from_dataframes(
    dfs: dict[str, "pd.DataFrame"],
    end_ms: int,
) -> dict | None:
    """DataFrame wrapper for external projects."""
    klines = {tf: df_to_klines(df) for tf, df in dfs.items()}
    return extract_dna_features_from_klines(
        klines["5m"],
        klines["15m"],
        end_ms,
        k30=klines.get("30m"),
        k1h=klines.get("1h"),
        k2h=klines.get("2h"),
    )


# Alias matching original project name
extract_dna_features = extract_dna_features_from_klines
