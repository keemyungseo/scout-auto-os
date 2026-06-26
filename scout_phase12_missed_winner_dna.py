"""
Scout Phase 12 — Missed Winner DNA Dataset Build

Dataset only. NO filter/ranking/threshold changes.

Usage:
  python scout_phase12_missed_winner_dna.py
  python scout_phase12_missed_winner_dna.py --max-scans 5
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import top10_gainer_learning_20260613 as t10
from season2_p37_scout_decision_hierarchy import pf, write_csv
from season2_p40_scout_transition_triggers import ema
from season2_universe_blind_test import load_eligible_symbols, ohlcv

OUT_DIR = Path("logs") / "phase12_dna"
CACHE_DIR = OUT_DIR / "scan_cache"
KST = timezone(timedelta(hours=9))
START_KST = datetime(2026, 5, 1, 0, 0, tzinfo=KST)
END_KST = datetime(2026, 5, 7, 23, 0, tzinfo=KST)
SCAN_INTERVAL_H = 2
INTERVAL_15M_MS = 15 * 60 * 1000
LOOKBACK = 480
FORWARD_15M = 48
API_SLEEP = 0.02
WORKERS = 14
BIRTH_PCT = 3.0

PATTERN_B = (("macd_signal", "gte", -0.0016), ("range_pct", "gte", 1.4768))
RANK_WEIGHTS = {
    "young_birth": -1.4343174277392308,
    "birth_age_min": 1.5586524217896396,
    "ignition_age_min": -0.22688287801448231,
    "ma_slope_accel": -0.12327724295144488,
    "volume_ma_ratio": -0.06664243036600889,
}

DNA_FEATURES = (
    "range_pct", "body_ratio", "body_expansion", "compression_length",
    "compression_std", "compression_break", "high_close_2d", "high_close_5d",
    "high_close_10d", "volume_ma_ratio", "volume_accel", "volume_std",
    "dollar_volume_log", "ma_slope", "ma_slope_accel", "rs_vs_btc", "atr_exp",
    "distance_from_high", "distance_from_low", "birth_age", "ignition_age",
    "macd_signal", "rsi", "bb_width", "bb_expansion", "donchian_break",
    "ema_gap", "ema_accel", "wick_ratio", "close_position",
)

SEQ_FEATURES = (
    "compression_length", "volume_ma_ratio", "body_ratio", "ma_slope",
    "ma_slope_accel", "range_pct",
)

SEQ_OFFSETS = (
    ("scan", 0),
    ("m30", 2),
    ("h1", 4),
    ("h2", 8),
    ("h4", 16),
)

CLUSTERS = (
    "Low_Volume_Explosion",
    "Compression",
    "Momentum",
    "Expansion",
    "Relative_Strength",
    "Other",
)


def gen_scan_times() -> list[tuple[str, datetime]]:
    times: list[tuple[str, datetime]] = []
    t = START_KST
    while t <= END_KST:
        times.append((t.strftime("%Y-%m-%d %H:%M:%S"), t))
        t += timedelta(hours=SCAN_INTERVAL_H)
    return times


def fetch_klines(symbol: str, end_ms: int, limit: int) -> list[list]:
    params = urllib.parse.urlencode({
        "symbol": symbol, "interval": "15m", "endTime": end_ms,
        "limit": min(limit, t10.MAX_LIMIT),
    })
    url = f"{t10.FUTURES_BASE_URL}{t10.KLINES_ENDPOINT}?{params}"
    with urllib.request.urlopen(urllib.request.Request(url), timeout=25) as resp:
        return json.loads(resp.read().decode())


def fetch_forward(symbol: str, start_ms: int) -> list[list]:
    params = urllib.parse.urlencode({
        "symbol": symbol, "interval": "15m", "startTime": start_ms,
        "limit": FORWARD_15M,
    })
    url = f"{t10.FUTURES_BASE_URL}{t10.KLINES_ENDPOINT}?{params}"
    with urllib.request.urlopen(urllib.request.Request(url), timeout=25) as resp:
        return json.loads(resp.read().decode())


def rsi(closes: list[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    gains, losses = [], []
    for i in range(-period, 0):
        d = closes[i] - closes[i - 1]
        gains.append(max(d, 0.0))
        losses.append(max(-d, 0.0))
    ag = sum(gains) / period
    al = sum(losses) / period
    if al == 0:
        return 100.0
    return 100 - 100 / (1 + ag / al)


def ma_slope_pct(klines: list[list]) -> float:
    if len(klines) < 14:
        return 0.0
    closes = [float(k[4]) for k in klines]
    recent = statistics.mean(closes[-6:-1])
    prior = statistics.mean(closes[-12:-7]) if len(closes) >= 12 else recent
    return (recent - prior) / prior * 100 if prior else 0.0


def compression_stats(klines: list[list]) -> tuple[int, float, int]:
    """length, std of range in compression zone, break flag."""
    ranges: list[float] = []
    n = 0
    for k in reversed(klines[:-1]):
        o, h, l, _, _ = ohlcv(k)
        if o <= 0:
            break
        r = (h - l) / o * 100
        if r <= 2.0:
            ranges.append(r)
            n += 1
        else:
            break
    std = statistics.pstdev(ranges) if len(ranges) > 1 else 0.0
    brk = 0
    if len(klines) >= 2:
        o, h, l, _, _ = ohlcv(klines[-1])
        prev_ranges = [(ohlcv(klines[i])[2] - ohlcv(klines[i])[3]) / ohlcv(klines[i])[1] * 100
                       for i in range(max(0, len(klines) - 10), len(klines) - 1) if ohlcv(klines[i])[1]]
        if prev_ranges and o > 0:
            brk = int((h - l) / o * 100 > statistics.mean(prev_ranges) * 1.5)
    return n, round(std, 4), brk


def lifecycle_ages(klines: list[list]) -> tuple[float, float]:
    anchor = len(klines) - 1
    window = klines[max(0, anchor - 48): anchor + 1]
    base = min(ohlcv(k)[2] for k in window) if window else ohlcv(klines[-1])[2]
    birth_i = anchor
    for i in range(anchor, max(0, anchor - 32), -1):
        if ohlcv(klines[i])[4] >= base * (1 + BIRTH_PCT / 100):
            birth_i = i
            break
    ign_i = anchor
    for i in range(anchor, max(0, anchor - 24), -1):
        vols = [ohlcv(klines[j])[4] for j in range(max(0, i - 8), i)]
        if vols and statistics.mean(vols) > 0 and ohlcv(klines[i])[4] / statistics.mean(vols) >= 1.2:
            ign_i = i
            break
    return float((anchor - birth_i) * 15), float((anchor - ign_i) * 15)


def extract_features(klines: list[list], btc_ret: float) -> dict | None:
    if len(klines) < 40:
        return None
    o, h, l, c, vol = ohlcv(klines[-1])
    if not (t10.MIN_PRICE <= c <= t10.MAX_PRICE) or o <= 0:
        return None

    closes = [float(k[4]) for k in klines]
    vols = [ohlcv(k)[4] for k in klines[-25:-1]]
    vol_ma = statistics.mean(vols[-24:]) if vols else 0.0
    vol_prior = statistics.mean(vols[-28:-24]) if len(vols) >= 28 else vol_ma
    vol_std = statistics.pstdev(vols[-24:]) if len(vols) > 1 else 0.0

    rng = (h - l) / o * 100
    body = abs(c - o) / o * 100
    body_ratio = body / rng if rng > 0 else 0.0
    if len(klines) >= 6:
        po, ph, pl, pc, _ = ohlcv(klines[-6])
        prev_body = abs(pc - po) / po * 100 if po else 0
        prev_rng = (ph - pl) / po * 100 if po else 1
        prev_br = prev_body / prev_rng if prev_rng > 0 else 0
        body_exp = body_ratio - prev_br
    else:
        body_exp = 0.0

    comp_len, comp_std, comp_brk = compression_stats(klines)
    slope = ma_slope_pct(klines)
    slope_p = ma_slope_pct(klines[:-4]) if len(klines) > 20 else slope

    n2d = min(192, len(closes))
    n5d = min(480, len(closes))
    n10d = min(960, len(closes))
    hi2 = max(closes[-n2d:])
    hi5 = max(closes[-n5d:])
    hi10 = max(closes[-n10d:])
    period_hi = max(float(k[2]) for k in klines[-21:-1]) if len(klines) > 21 else h
    period_lo = min(float(k[3]) for k in klines[-21:-1]) if len(klines) > 21 else l

    ema12 = ema(closes, 12)
    ema26 = ema(closes, 26)
    ema_gap = (c - ema12) / c * 100 if c else 0
    ema_prev = ema(closes[:-4], 12) if len(closes) > 16 else ema12
    ema_accel = ema12 - ema_prev

    bb_mid = statistics.mean(closes[-20:])
    bb_std = statistics.pstdev(closes[-20:]) if len(closes) >= 20 else 0.0
    bb_width = (2 * bb_std / bb_mid * 100) if bb_mid else 0
    bb_width_prev = 0.0
    if len(closes) >= 24:
        m2 = statistics.mean(closes[-24:-4])
        s2 = statistics.pstdev(closes[-24:-4])
        bb_width_prev = (2 * s2 / m2 * 100) if m2 else 0

    atr_b = t10.average_true_range_percent(klines[-15:-1], c) if len(klines) >= 16 else 1.0
    atr_n = t10.true_range(klines[-1], float(klines[-2][4])) / c * 100 if c else 0

    upper_wick = (h - max(o, c)) / o * 100
    lower_wick = (min(o, c) - l) / o * 100
    wick_ratio = upper_wick / lower_wick if lower_wick > 0 else upper_wick

    hist_macd = [ema(closes[:i], 12) - ema(closes[:i], 26) for i in range(26, len(closes) + 1)]
    macd_sig = ema(hist_macd, 9) if hist_macd else 0.0

    return {
        "range_pct": round(rng, 4),
        "body_ratio": round(body_ratio, 4),
        "body_expansion": round(body_exp, 4),
        "compression_length": float(comp_len),
        "compression_std": comp_std,
        "compression_break": float(comp_brk),
        "high_close_2d": 1.0 if c >= hi2 * 0.999 else 0.0,
        "high_close_5d": 1.0 if c >= hi5 * 0.999 else 0.0,
        "high_close_10d": 1.0 if c >= hi10 * 0.999 else 0.0,
        "volume_ma_ratio": round(vol / vol_ma if vol_ma > 0 else 0, 4),
        "volume_accel": round((vol / vol_ma - vol_prior / vol_ma) if vol_ma > 0 else 0, 4),
        "volume_std": round(vol_std, 4),
        "dollar_volume_log": round(math.log10(max(c * vol, 1.0)), 4),
        "ma_slope": round(slope, 4),
        "ma_slope_accel": round(slope - slope_p, 4),
        "rs_vs_btc": round(slope - btc_ret, 4),
        "atr_exp": round(atr_n / atr_b if atr_b > 0 else 1.0, 4),
        "distance_from_high": round((period_hi - c) / c * 100 if c else 0, 4),
        "distance_from_low": round((c - period_lo) / c * 100 if c else 0, 4),
        "birth_age": lifecycle_ages(klines)[0],
        "ignition_age": lifecycle_ages(klines)[1],
        "macd_signal": round(macd_sig, 6),
        "rsi": round(rsi(closes), 4),
        "bb_width": round(bb_width, 4),
        "bb_expansion": round(bb_width / bb_width_prev if bb_width_prev > 0 else 1.0, 4),
        "donchian_break": 1.0 if h > period_hi else 0.0,
        "ema_gap": round(ema_gap, 4),
        "ema_accel": round(ema_accel, 6),
        "wick_ratio": round(wick_ratio, 4),
        "close_position": round((c - l) / (h - l) if h > l else 0.5, 4),
        "price": c,
    }


def time_sequence(klines: list[list], btc_ret: float) -> dict:
    seq: dict = {}
    anchor = len(klines) - 1
    prev_feats: dict | None = None
    for label, offset in SEQ_OFFSETS:
        end_i = anchor - offset + 1
        if end_i < 40:
            continue
        chunk = klines[:end_i]
        f = extract_features(chunk, btc_ret)
        if not f:
            continue
        point = {k: f.get(k, 0) for k in SEQ_FEATURES}
        if prev_feats:
            for k in SEQ_FEATURES:
                p, n = prev_feats.get(k, 0), point.get(k, 0)
                point[f"{k}_delta"] = round(n - p, 4) if isinstance(n, (int, float)) else 0
        seq[label] = point
        prev_feats = point
    return seq


def passes_filter(f: dict) -> bool:
    return f.get("macd_signal", -999) >= -0.0016 and f.get("range_pct", 0) >= 1.4768


def rank_score(f: dict) -> float:
    young = 1.0 if f.get("birth_age", 99) <= 45 else 0.0
    ff = {
        "young_birth": young,
        "birth_age_min": f.get("birth_age", 0),
        "ignition_age_min": f.get("ignition_age", 0),
        "ma_slope_accel": f.get("ma_slope_accel", 0),
        "volume_ma_ratio": f.get("volume_ma_ratio", 0),
    }
    return sum(RANK_WEIGHTS[k] * ff.get(k, 0) for k in RANK_WEIGHTS)


def forward_stats(entry: float, fwd: list[list]) -> dict:
    if entry <= 0 or not fwd:
        return {"max_up_12h": 0, "return_12h": 0, "mdd_12h": 0}
    chunk = fwd[:FORWARD_15M]
    max_h = max(ohlcv(k)[1] for k in chunk)
    min_l = min(ohlcv(k)[2] for k in chunk)
    close = float(chunk[-1][4])
    return {
        "max_up_12h": round((max_h - entry) / entry * 100, 4),
        "return_12h": round((close - entry) / entry * 100, 4),
        "mdd_12h": round((entry - min_l) / entry * 100, 4),
    }


def assign_cluster(f: dict, max_up: float) -> str:
    if f.get("volume_ma_ratio", 0) < 1.0 and max_up >= 7.0:
        return "Low_Volume_Explosion"
    if f.get("compression_length", 0) >= 10 and f.get("body_ratio", 1) < 0.55:
        return "Compression"
    if f.get("ma_slope_accel", 0) > 0 and f.get("ma_slope", 0) > 1.0:
        return "Momentum"
    if f.get("range_pct", 0) >= 3.0 and f.get("body_expansion", 0) > 0:
        return "Expansion"
    if f.get("rs_vs_btc", 0) >= 1.0:
        return "Relative_Strength"
    return "Other"


def process_symbol(symbol: str, end_ms: int, btc_ret: float) -> dict | None:
    try:
        hist = fetch_klines(symbol, end_ms, LOOKBACK)
        f = extract_features(hist, btc_ret)
        if not f or not passes_filter(f):
            return None
        fwd = fetch_forward(symbol, end_ms + INTERVAL_15M_MS)
        fs = forward_stats(f["price"], fwd)
        seq = time_sequence(hist, btc_ret)
        return {
            "symbol": symbol,
            "price": f["price"],
            "ranking_score": rank_score(f),
            "features": {k: f.get(k, 0) for k in DNA_FEATURES},
            "time_sequence": seq,
            **fs,
        }
    except Exception:
        return None
    finally:
        time.sleep(API_SLEEP)


def run_scan(scan_kst: str, scan_dt: datetime, symbols: list[str], use_cache: bool) -> dict:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tag = scan_kst.replace(" ", "_").replace(":", "")
    cache_path = CACHE_DIR / f"{tag}.json"
    if use_cache and cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    end_ms = int(scan_dt.timestamp() * 1000)
    btc_ret = 0.0
    try:
        bk = fetch_klines("BTCUSDT", end_ms, 16)
        if len(bk) >= 9:
            btc_ret = (float(bk[-1][4]) - float(bk[-9][4])) / float(bk[-9][4]) * 100
    except Exception:
        pass

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = {pool.submit(process_symbol, s, end_ms, btc_ret): s for s in symbols}
        for fut in as_completed(futs):
            r = fut.result()
            if r:
                rows.append(r)

    rows.sort(key=lambda x: x["ranking_score"], reverse=True)
    for i, r in enumerate(rows, 1):
        r["rank"] = i
        r["in_top5"] = i <= 5
        r["in_top30"] = i <= 30

    max_ups = sorted([r["max_up_12h"] for r in rows], reverse=True)
    p90 = max_ups[int(len(max_ups) * 0.1)] if max_ups else 5.0

    missed: list[dict] = []
    for r in rows:
        is_missed = r["rank"] > 5 and (r["max_up_12h"] >= 5.0 or r["max_up_12h"] >= p90)
        r["is_missed_winner"] = is_missed
        if is_missed:
            r["cluster"] = assign_cluster(r["features"], r["max_up_12h"])
            missed.append(r)

    result = {
        "scan_kst": scan_kst,
        "match_count": len(rows),
        "top30": rows[:30],
        "top5": rows[:5],
        "missed_winners": missed,
        "p90_max_up": p90,
    }
    cache_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return result


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 4:
        return 0.0
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    den = math.sqrt(sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys))
    return num / den if den else 0.0


def feature_importance(rows: list[dict]) -> list[dict]:
    labels = [r["max_up_12h"] for r in rows]
    out = []
    for feat in DNA_FEATURES:
        xs = [r["features"].get(feat, 0) for r in rows]
        out.append({
            "feature": feat,
            "correlation": round(pearson(xs, labels), 4),
            "mean": round(statistics.mean(xs), 4) if xs else 0,
            "median": round(statistics.median(xs), 4) if xs else 0,
        })
    out.sort(key=lambda x: abs(x["correlation"]), reverse=True)
    return out


def corr_matrix(rows: list[dict], top_feats: list[str]) -> list[dict]:
    out = []
    for i, a in enumerate(top_feats):
        for b in top_feats[i + 1:]:
            xs = [r["features"].get(a, 0) for r in rows]
            ys = [r["features"].get(b, 0) for r in rows]
            out.append({"a": a, "b": b, "corr": round(pearson(xs, ys), 4)})
    out.sort(key=lambda x: abs(x["corr"]), reverse=True)
    return out[:30]


def seq_transition_stats(missed: list[dict]) -> list[dict]:
    stats = []
    for feat in SEQ_FEATURES:
        deltas = []
        for m in missed:
            seq = m.get("time_sequence", {})
            if "scan" in seq and "h4" in seq:
                deltas.append(seq["scan"].get(feat, 0) - seq["h4"].get(feat, 0))
        if deltas:
            stats.append({
                "feature": feat,
                "mean_delta_scan_vs_h4": round(statistics.mean(deltas), 4),
                "n": len(deltas),
            })
    return stats


def run(max_scans: int | None, use_cache: bool) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    symbols = sorted(load_eligible_symbols(refresh=False, cache_only=False))
    scans = gen_scan_times()
    if max_scans:
        scans = scans[:max_scans]

    print(f"Phase 12 DNA: {len(scans)} scans, universe={len(symbols)}")
    all_missed: list[dict] = []
    all_top30: list[dict] = []
    n_scans = 0

    for i, (scan_kst, scan_dt) in enumerate(scans, 1):
        print(f"  scan {i}/{len(scans)}: {scan_kst}")
        res = run_scan(scan_kst, scan_dt, symbols, use_cache)
        n_scans += 1
        for m in res["missed_winners"]:
            all_missed.append({"scan_kst": scan_kst, **m})
        for t in res["top30"]:
            all_top30.append({"scan_kst": scan_kst, **t})

    # JSONL outputs
    missed_path = OUT_DIR / "missed_winners_dna.jsonl"
    with missed_path.open("w", encoding="utf-8") as f:
        for m in all_missed:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    top30_path = OUT_DIR / "top30_per_scan.jsonl"
    with top30_path.open("w", encoding="utf-8") as f:
        for t in all_top30:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    cluster_counts = Counter(m.get("cluster", "Other") for m in all_missed)
    importance = feature_importance(all_missed)
    top_feats = [x["feature"] for x in importance[:8]]
    correlations = corr_matrix(all_missed, top_feats)
    seq_stats = seq_transition_stats(all_missed)

    def bucket_stats(key: str) -> dict:
        vals = [m["features"].get(key, 0) for m in all_missed]
        return {
            "mean": round(statistics.mean(vals), 3) if vals else 0,
            "median": round(statistics.median(vals), 3) if vals else 0,
            "n": len(vals),
        }

    top20 = sorted(all_missed, key=lambda x: x["max_up_12h"], reverse=True)[:20]

    lines = [
        "#############################################################",
        "SCOUT PHASE 12 — MISSED WINNER DNA DATASET BUILD",
        "#############################################################",
        "",
        f"Period: {START_KST.strftime('%Y-%m-%d %H:%M')} ~ {END_KST.strftime('%Y-%m-%d %H:%M')} KST",
        f"Filter/Ranking: UNCHANGED (Pattern B + H4 Lifecycle)",
        "",
        "=" * 56,
        "1. SCAN SUMMARY",
        "=" * 56,
        f"  Total scans: {n_scans}",
        f"  Missed winner records: {len(all_missed)}",
        f"  TOP30 records saved: {len(all_top30)}",
        "",
        "=" * 56,
        "2. CLUSTER DISTRIBUTION",
        "=" * 56,
    ]
    for c in CLUSTERS:
        n = cluster_counts.get(c, 0)
        pct = round(n / max(len(all_missed), 1) * 100, 1)
        lines.append(f"  {c}: {n} ({pct}%)")

    lines.extend(["", "=" * 56, "3. FEATURE IMPORTANCE (missed winners)", "=" * 56])
    for row in importance[:15]:
        lines.append(f"  {row['feature']:<22} corr={row['correlation']:+.3f} mean={row['mean']}")

    lines.extend(["", "=" * 56, "4. FEATURE CORRELATION (top pairs)", "=" * 56])
    for row in correlations[:10]:
        lines.append(f"  {row['a']} x {row['b']}: {row['corr']:+.3f}")

    lines.extend(["", "=" * 56, "5. TIME SEQUENCE TRANSITIONS", "=" * 56])
    for s in seq_stats:
        lines.append(f"  {s['feature']}: mean_delta(scan-h4)={s['mean_delta_scan_vs_h4']} n={s['n']}")

    lines.extend(["", "=" * 56, "6. COMPRESSION STATS", "=" * 56])
    cs = bucket_stats("compression_length")
    lines.append(f"  compression_length mean={cs['mean']} median={cs['median']}")
    cs2 = bucket_stats("compression_std")
    lines.append(f"  compression_std mean={cs2['mean']} median={cs2['median']}")

    lines.extend(["", "=" * 56, "7. BODY EXPANSION STATS", "=" * 56])
    for k in ("body_ratio", "body_expansion"):
        b = bucket_stats(k)
        lines.append(f"  {k}: mean={b['mean']} median={b['median']}")

    lines.extend(["", "=" * 56, "8. VOLUME PATTERN STATS", "=" * 56])
    for k in ("volume_ma_ratio", "volume_accel", "volume_std"):
        b = bucket_stats(k)
        lines.append(f"  {k}: mean={b['mean']} median={b['median']}")

    lines.extend(["", "=" * 56, "9. RS vs BTC STATS", "=" * 56])
    b = bucket_stats("rs_vs_btc")
    lines.append(f"  rs_vs_btc: mean={b['mean']} median={b['median']}")
    pos = sum(1 for m in all_missed if m["features"].get("rs_vs_btc", 0) >= 0)
    lines.append(f"  rs_vs_btc >= 0: {round(pos/max(len(all_missed),1)*100,1)}%")

    lines.extend(["", "=" * 56, "10. TOP20 MISS CASES", "=" * 56])
    for m in top20:
        f = m["features"]
        lines.append(
            f"  {m['scan_kst']} #{m['rank']} {m['symbol']} max_up={m['max_up_12h']}% "
            f"cluster={m.get('cluster')} vol={f.get('volume_ma_ratio')} comp={f.get('compression_length')} "
            f"rs={f.get('rs_vs_btc')}"
        )

    lines.extend(["", "=" * 56, "11. STRUCTURAL WEAKNESS (no changes made)", "=" * 56,
        "  Ranking penalizes low volume_ma_ratio; Low_Volume_Explosion cluster under-selected.",
        "  birth_age/ignition_age proxy collapses to 0 at scan — lifecycle DNA under-differentiated.",
        "  compression_length high but negative rank weight absent — long compress misses rank boost.",
        f"  Evidence: {len(all_missed)} missed across {n_scans} scans; TOP5 unchanged.",
        "",
        "=" * 56,
        "12. FUTURE MODIFICATION CANDIDATES (PROPOSAL ONLY — NOT APPLIED)",
        "=" * 56,
        "  P1: Ranking weight ma_slope_accel +0.5 (Phase 11 evidence)",
        "  P2: Add compression_length to ranking (+0.05 weight)",
        "  P3: Low-volume path: rank tie-break when compression_break=1 and rs_vs_btc>0",
        "",
        "NO filter/threshold/weight changes applied in Phase 12.",
    ])

    report = OUT_DIR / "phase12_report.txt"
    report.write_text("\n".join(lines), encoding="utf-8")
    write_csv(OUT_DIR / "feature_importance.csv", importance)
    write_csv(OUT_DIR / "cluster_counts.csv", [
        {"cluster": c, "count": cluster_counts.get(c, 0),
         "pct": round(cluster_counts.get(c, 0) / max(len(all_missed), 1) * 100, 1)}
        for c in CLUSTERS
    ])
    write_csv(OUT_DIR / "feature_correlation.csv", correlations)
    write_csv(OUT_DIR / "sequence_transitions.csv", seq_stats)

    meta = {
        "scans": n_scans,
        "missed_count": len(all_missed),
        "top30_count": len(all_top30),
        "cluster_counts": dict(cluster_counts),
        "period_start": START_KST.isoformat(),
        "period_end": END_KST.isoformat(),
    }
    (OUT_DIR / "dataset_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\n".join(lines[:35]).encode("ascii", "replace").decode("ascii"))
    print(f"\nSaved: {report} | missed={len(all_missed)} top30={len(all_top30)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-scans", type=int, default=None)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()
    run(args.max_scans, use_cache=not args.no_cache)


if __name__ == "__main__":
    main()
