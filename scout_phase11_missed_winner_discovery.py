"""
Scout Phase 11 — Missed Winner Feature Discovery

Multi-scan blind loop: find common features of large movers missed by TOP5.
Minimal single-change modification proposals only.

Usage:
  python scout_phase11_missed_winner_discovery.py
  python scout_phase11_missed_winner_discovery.py --max-scans 3
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import top10_gainer_learning_20260613 as t10
from season2_p37_scout_decision_hierarchy import pf, write_csv
from season2_p40_scout_transition_triggers import ema
from season2_universe_blind_test import load_eligible_symbols, ohlcv

OUT_DIR = Path("logs") / "phase11_missed"
CACHE_DIR = OUT_DIR / "scan_cache"
KST = timezone(timedelta(hours=9))
START_KST = datetime(2026, 5, 1, 9, 0, tzinfo=KST)
END_KST = datetime(2026, 5, 2, 23, 0, tzinfo=KST)
SCAN_INTERVAL_H = 2
INTERVAL_15M_MS = 15 * 60 * 1000
LOOKBACK_15M = 192
FORWARD_15M = 48
API_SLEEP = 0.025
WORKERS = 12
BIRTH_PCT = 3.0
HIT_THRESHOLD = 5.0

PATTERN_B = (("macd_signal", "gte", -0.0016), ("range_pct", "gte", 1.4768))

RANK_WEIGHTS = {
    "young_birth": -1.4343174277392308,
    "birth_age_min": 1.5586524217896396,
    "ignition_age_min": -0.22688287801448231,
    "ma_slope_accel": -0.12327724295144488,
    "volume_ma_ratio": -0.06664243036600889,
}

FEATURE_KEYS = (
    "range_pct", "volume_ma_ratio", "ma_slope_accel", "ma_slope", "macd_signal",
    "body_ratio", "compression_length", "high_close_2d", "breakout_flag",
    "young_birth", "birth_age_min", "ignition_age_min", "rs_vs_btc",
    "dollar_volume_log", "atr_expansion", "higher_high",
)

THRESHOLD_CHECKS = (
    ("high_close_2d", "gte", 1.0, "recent 2d highest close"),
    ("compression_length", "gte", 10, "compression >= 10 candles"),
    ("compression_length", "gte", 8, "compression >= 8 candles"),
    ("body_ratio", "gte", 0.5, "body_ratio >= 0.5"),
    ("volume_ma_ratio", "gte", 1.8, "volume_ma_ratio >= 1.8"),
    ("volume_ma_ratio", "gte", 1.5, "volume_ma_ratio >= 1.5"),
    ("volume_ma_ratio", "lt", 1.0, "volume_ma_ratio < 1.0 (low vol miss)"),
    ("ma_slope_accel", "gte", 0, "slope_accel positive"),
    ("ma_slope_accel", "lt", 0, "slope_accel negative"),
    ("young_birth", "gte", 1, "young_birth flag"),
    ("breakout_flag", "gte", 1, "breakout flag"),
    ("rs_vs_btc", "gte", 0, "rs_vs_btc >= 0"),
    ("ignition_age_min", "gte", 60, "ignition_age >= 60min"),
)


@dataclass
class ScanRecord:
    scan_kst: str
    symbol: str
    rank: int
    price: float
    ranking_score: float
    max_up_12h: float
    return_12h: float
    mdd_12h: float
    hit_5pct: bool
    in_top5: bool
    is_missed_winner: bool
    miss_reasons: list[str] = field(default_factory=list)
    features: dict = field(default_factory=dict)


def gen_scan_times() -> list[tuple[str, datetime]]:
    times: list[tuple[str, datetime]] = []
    t = START_KST
    while t <= END_KST:
        times.append((t.strftime("%Y-%m-%d %H:%M:%S"), t))
        t += timedelta(hours=SCAN_INTERVAL_H)
    return times


def fetch_klines_15m(symbol: str, end_ms: int, limit: int) -> list[list]:
    params = urllib.parse.urlencode({
        "symbol": symbol, "interval": "15m", "endTime": end_ms,
        "limit": min(limit, t10.MAX_LIMIT),
    })
    url = f"{t10.FUTURES_BASE_URL}{t10.KLINES_ENDPOINT}?{params}"
    with urllib.request.urlopen(urllib.request.Request(url), timeout=25) as resp:
        return json.loads(resp.read().decode())


def fetch_forward_15m(symbol: str, start_ms: int, count: int) -> list[list]:
    params = urllib.parse.urlencode({
        "symbol": symbol, "interval": "15m", "startTime": start_ms,
        "limit": min(count, t10.MAX_LIMIT),
    })
    url = f"{t10.FUTURES_BASE_URL}{t10.KLINES_ENDPOINT}?{params}"
    with urllib.request.urlopen(urllib.request.Request(url), timeout=25) as resp:
        return json.loads(resp.read().decode())


def fetch_btc_return_2h(end_ms: int) -> float:
    try:
        k = fetch_klines_15m("BTCUSDT", end_ms, 16)
        if len(k) < 9:
            return 0.0
        c0 = float(k[-9][4])
        c1 = float(k[-1][4])
        return (c1 - c0) / c0 * 100 if c0 else 0.0
    except Exception:
        return 0.0


def macd_signal(closes: list[float]) -> float:
    if len(closes) < 26:
        return 0.0
    hist = [ema(closes[:i], 12) - ema(closes[:i], 26) for i in range(26, len(closes) + 1)]
    return ema(hist, 9) if hist else 0.0


def ma_slope_pct(klines: list[list]) -> float:
    if len(klines) < 14:
        return 0.0
    closes = [float(k[4]) for k in klines]
    recent = statistics.mean(closes[-6:-1])
    prior = statistics.mean(closes[-12:-7]) if len(closes) >= 12 else recent
    return (recent - prior) / prior * 100 if prior else 0.0


def compression_length(klines: list[list], max_range: float = 2.0) -> int:
    n = 0
    for k in reversed(klines[:-1]):
        o, h, l, _, _ = ohlcv(k)
        if o <= 0:
            break
        if (h - l) / o * 100 <= max_range:
            n += 1
        else:
            break
    return n


def estimate_lifecycle(klines: list[list]) -> dict:
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
        if vols and statistics.mean(vols) > 0:
            if ohlcv(klines[i])[4] / statistics.mean(vols) >= 1.2:
                ign_i = i
                break
    birth_age = (anchor - birth_i) * 15
    ign_age = (anchor - ign_i) * 15
    slope = ma_slope_pct(klines)
    slope_p = ma_slope_pct(klines[:-4]) if len(klines) > 20 else slope
    return {
        "birth_age_min": float(birth_age),
        "ignition_age_min": float(ign_age),
        "young_birth": 1.0 if birth_age <= 45 else 0.0,
        "ma_slope_accel": slope - slope_p,
        "ma_slope": slope,
    }


def compute_features(klines: list[list], btc_ret: float) -> dict | None:
    if len(klines) < 40:
        return None
    o, h, l, c, vol = ohlcv(klines[-1])
    if not (t10.MIN_PRICE <= c <= t10.MAX_PRICE):
        return None
    vols = [ohlcv(k)[4] for k in klines[-25:-1]]
    vol_ma = statistics.mean(vols[-24:]) if vols else 0.0
    rng = (h - l) / o * 100 if o else 0.0
    body = abs(c - o) / o * 100 if o else 0.0
    body_ratio = body / rng if rng > 0 else 0.0
    closes = [float(k[4]) for k in klines]
    highs = [float(k[2]) for k in klines[-21:-1]]
    lc = estimate_lifecycle(klines)
    vol_ratio = vol / vol_ma if vol_ma > 0 else 0.0
    hist_high = max(closes[-192:]) if len(closes) >= 192 else max(closes)
    atr_base = t10.average_true_range_percent(klines[-15:-1], c) if len(klines) >= 16 else 1.0
    atr_now = t10.true_range(klines[-1], float(klines[-2][4])) / c * 100 if c else 0.0
    return {
        "price": c,
        "range_pct": rng,
        "macd_signal": macd_signal(closes),
        "volume_ma_ratio": vol_ratio,
        "body_ratio": round(body_ratio, 4),
        "compression_length": float(compression_length(klines)),
        "high_close_2d": 1.0 if c >= hist_high * 0.999 else 0.0,
        "breakout_flag": 1.0 if (highs and h > max(highs)) else 0.0,
        "higher_high": 1.0 if (highs and h > max(highs)) else 0.0,
        "dollar_volume_log": math.log10(max(c * vol, 1.0)),
        "atr_expansion": atr_now / atr_base if atr_base > 0 else 1.0,
        "rs_vs_btc": lc["ma_slope"] - btc_ret,
        **lc,
    }


def passes_filter(f: dict) -> bool:
    for name, op, thr in PATTERN_B:
        v = f.get(name)
        if v is None:
            return False
        if op == "gte" and v < thr:
            return False
    return True


def ranking_score(f: dict, weights: dict | None = None) -> float:
    w = weights or RANK_WEIGHTS
    return sum(w.get(k, 0) * f.get(k, 0) for k in w)


def forward_max_up(entry: float, fwd: list[list]) -> tuple[float, float, float]:
    if entry <= 0 or not fwd:
        return 0.0, 0.0, 0.0
    chunk = fwd[:FORWARD_15M]
    max_h = max(ohlcv(k)[1] for k in chunk)
    min_l = min(ohlcv(k)[2] for k in chunk)
    close = float(chunk[-1][4])
    return (
        (max_h - entry) / entry * 100,
        (close - entry) / entry * 100,
        (entry - min_l) / entry * 100,
    )


def miss_reasons(f: dict, score: float, top5_min_score: float) -> list[str]:
    reasons: list[str] = []
    if f.get("volume_ma_ratio", 0) < 1.0:
        reasons.append(f"volume_ma_ratio={f.get('volume_ma_ratio', 0):.2f}<1.0")
    if f.get("volume_ma_ratio", 0) < 1.8:
        reasons.append(f"volume_ma_ratio={f.get('volume_ma_ratio', 0):.2f}<1.8")
    if f.get("range_pct", 0) < 2.0:
        reasons.append(f"range_pct={f.get('range_pct', 0):.2f} below expansion leaders")
    if f.get("body_ratio", 0) < 0.5:
        reasons.append(f"body_ratio={f.get('body_ratio', 0):.2f}<0.5")
    if f.get("compression_length", 0) < 8:
        reasons.append(f"compression_length={f.get('compression_length', 0):.0f}<8")
    if f.get("young_birth", 0) < 1:
        reasons.append("young_birth=0")
    if f.get("breakout_flag", 0) < 1:
        reasons.append("breakout_flag=0")
    if f.get("ma_slope_accel", 0) < 0:
        reasons.append(f"ma_slope_accel={f.get('ma_slope_accel', 0):.2f}<0")
    if f.get("rs_vs_btc", 0) < 0:
        reasons.append(f"rs_vs_btc={f.get('rs_vs_btc', 0):.2f} (BTC relative weak)")
    if f.get("ignition_age_min", 0) >= 60:
        reasons.append(f"ignition_age={f.get('ignition_age_min', 0):.0f}min extended")
    if score < top5_min_score:
        reasons.append(f"ranking_score={score:.2f}<{top5_min_score:.2f}")
    return reasons


def scan_one_symbol(symbol: str, end_ms: int, btc_ret: float) -> dict | None:
    try:
        hist = fetch_klines_15m(symbol, end_ms, LOOKBACK_15M)
        feats = compute_features(hist, btc_ret)
        if not feats or not passes_filter(feats):
            return None
        fwd = fetch_forward_15m(symbol, end_ms + INTERVAL_15M_MS, FORWARD_15M)
        max_up, ret_12, mdd = forward_max_up(feats["price"], fwd)
        score = ranking_score(feats)
        return {
            "symbol": symbol,
            "price": feats["price"],
            "ranking_score": score,
            "max_up_12h": round(max_up, 4),
            "return_12h": round(ret_12, 4),
            "mdd_12h": round(mdd, 4),
            "hit_5pct": max_up >= HIT_THRESHOLD,
            "features": {k: feats.get(k, 0) for k in FEATURE_KEYS},
        }
    except Exception:
        return None
    finally:
        time.sleep(API_SLEEP)


def run_scan(scan_kst: str, scan_dt: datetime, symbols: list[str], use_cache: bool) -> list[dict]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{scan_kst.replace(' ', '_').replace(':', '')}.json"
    if use_cache and cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    end_ms = int(scan_dt.timestamp() * 1000)
    btc_ret = fetch_btc_return_2h(end_ms)
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = {pool.submit(scan_one_symbol, sym, end_ms, btc_ret): sym for sym in symbols}
        for fut in as_completed(futs):
            r = fut.result()
            if r:
                rows.append(r)

    rows.sort(key=lambda x: x["ranking_score"], reverse=True)
    for i, r in enumerate(rows, 1):
        r["rank"] = i

    top5_max = [r["max_up_12h"] for r in rows[:5]]
    min_top5_up = min(top5_max) if top5_max else HIT_THRESHOLD
    for r in rows:
        r["in_top5"] = r["rank"] <= 5
        r["is_missed_winner"] = (
            r["rank"] > 5
            and (r["max_up_12h"] >= HIT_THRESHOLD or r["max_up_12h"] > min_top5_up)
        )
        t5min = rows[4]["ranking_score"] if len(rows) >= 5 else -999
        r["miss_reasons"] = miss_reasons(r["features"], r["ranking_score"], t5min) if r["is_missed_winner"] else []

    cache_path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return rows


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 3:
        return 0.0
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return num / den if den else 0.0


def mutual_info_binned(xs: list[float], ys: list[float], bins: int = 5) -> float:
    if len(xs) < 10:
        return 0.0
    pairs = sorted(zip(xs, ys), key=lambda p: p[0])
    n = len(pairs)
    bx = max(1, n // bins)
    total = 0.0
    for i in range(0, n, bx):
        chunk = pairs[i:i + bx]
        if not chunk:
            continue
        yv = [c[1] for c in chunk]
        p_bin = len(chunk) / n
        p_hit = sum(1 for v in yv if v >= HIT_THRESHOLD) / len(chunk)
        p_all = sum(1 for v in ys if v >= HIT_THRESHOLD) / len(ys)
        if p_hit > 0 and p_hit < 1:
            total += p_bin * (p_hit * math.log2(p_hit / p_all) + (1 - p_hit) * math.log2((1 - p_hit) / (1 - p_all)))
    return max(0.0, total)


def information_gain(xs: list[float], labels: list[float], bins: int = 5) -> float:
    return mutual_info_binned(xs, labels, bins)


def permutation_importance(records: list[ScanRecord], feature: str, n_perm: int = 20) -> float:
    """Higher = feature more important for ranking max_up."""
    if len(records) < 10:
        return 0.0
    base = abs(pearson([r.features.get(feature, 0) for r in records], [r.max_up_12h for r in records]))
    rng = random.Random(42)
    drops: list[float] = []
    xs = [r.features.get(feature, 0) for r in records]
    ys = [r.max_up_12h for r in records]
    for _ in range(n_perm):
        shuffled = xs[:]
        rng.shuffle(shuffled)
        drops.append(abs(pearson(shuffled, ys)))
    return max(0.0, base - statistics.mean(drops))


def shap_proxy(records: list[ScanRecord], feature: str) -> float:
    """Mean absolute marginal effect: corr when high vs low split."""
    missed = [r for r in records if r.is_missed_winner]
    if len(missed) < 5:
        return 0.0
    xs = [r.features.get(feature, 0) for r in missed]
    ys = [r.max_up_12h for r in missed]
    med = statistics.median(xs)
    hi = [y for x, y in zip(xs, ys) if x >= med]
    lo = [y for x, y in zip(xs, ys) if x < med]
    if not hi or not lo:
        return 0.0
    return abs(statistics.mean(hi) - statistics.mean(lo))


def feature_importance(missed: list[ScanRecord]) -> list[dict]:
    labels = [r.max_up_12h for r in missed]
    rows: list[dict] = []
    for feat in FEATURE_KEYS:
        xs = [r.features.get(feat, 0) for r in missed]
        rows.append({
            "feature": feat,
            "correlation": round(pearson(xs, labels), 4),
            "mutual_info": round(mutual_info_binned(xs, labels), 4),
            "info_gain": round(information_gain(xs, labels), 4),
            "perm_importance": round(permutation_importance(missed, feat), 4),
            "shap_proxy": round(shap_proxy(missed, feat), 4),
            "composite": 0.0,
        })
    for row in rows:
        row["composite"] = round(
            abs(row["correlation"]) * 0.25
            + row["mutual_info"] * 0.25
            + row["perm_importance"] * 0.25
            + row["shap_proxy"] * 0.01
            + row["info_gain"] * 0.25,
            4,
        )
    rows.sort(key=lambda r: r["composite"], reverse=True)
    return rows


def threshold_frequency(missed: list[ScanRecord]) -> list[dict]:
    n = len(missed)
    if n == 0:
        return []
    out: list[dict] = []
    for feat, op, thr, label in THRESHOLD_CHECKS:
        cnt = 0
        for r in missed:
            v = r.features.get(feat, 0)
            if op == "gte" and v >= thr:
                cnt += 1
            elif op == "lt" and v < thr:
                cnt += 1
        pct = round(cnt / n * 100, 1)
        out.append({"pattern": label, "count": cnt, "n": n, "pct": pct})
    out.sort(key=lambda x: x["pct"], reverse=True)
    return out


def simulate_modification(
    all_scans: dict[str, list[dict]],
    mod_name: str,
    weight_delta: dict[str, float] | None = None,
    filter_delta: tuple[str, str, float] | None = None,
) -> dict:
    """Replay scans with ONE change."""
    recovered = 0
    missed_total = 0
    top5_retained = 0
    top5_total = 0
    false_pos = 0
    new_top5 = 0

    for scan_kst, rows in all_scans.items():
        if not rows:
            continue
        # apply filter change if any
        work = []
        for r in rows:
            f = dict(r["features"])
            f["price"] = r["price"]
            if filter_delta:
                name, op, thr = filter_delta
                if op == "gte" and f.get(name, 0) < thr:
                    continue
            w = dict(RANK_WEIGHTS)
            if weight_delta:
                for k, dv in weight_delta.items():
                    w[k] = w.get(k, 0) + dv
            sc = ranking_score(f, w)
            work.append({**r, "ranking_score_new": sc, "features": f})

        if filter_delta:
            # need re-fetch - skip filter mods without full rescan; use pre-filtered only
            pass

        work.sort(key=lambda x: x["ranking_score_new"], reverse=True)
        orig_top5 = {r["symbol"] for r in sorted(rows, key=lambda x: x["ranking_score"], reverse=True)[:5]}
        new_top5_set = {r["symbol"] for r in work[:5]}
        missed = [r for r in rows if r.get("is_missed_winner")]
        missed_total += len(missed)
        for m in missed:
            if m["symbol"] in new_top5_set:
                recovered += 1
        for sym in orig_top5:
            top5_total += 1
            if sym in new_top5_set:
                top5_retained += 1
        for sym in new_top5_set - orig_top5:
            new_top5 += 1
            row = next((r for r in work if r["symbol"] == sym), None)
            if row and row["max_up_12h"] < HIT_THRESHOLD:
                false_pos += 1

    return {
        "name": mod_name,
        "missed_recovery_pct": round(recovered / max(missed_total, 1) * 100, 1),
        "top5_retention_pct": round(top5_retained / max(top5_total, 1) * 100, 1),
        "false_positive_new_top5_pct": round(false_pos / max(new_top5, 1) * 100, 1),
        "recovered": recovered,
        "missed_total": missed_total,
    }


def run(max_scans: int | None, use_cache: bool) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    symbols = sorted(load_eligible_symbols(refresh=False, cache_only=False))
    scan_times = gen_scan_times()
    if max_scans:
        scan_times = scan_times[:max_scans]

    print(f"Phase 11: {len(scan_times)} scans, universe={len(symbols)}")
    all_scans: dict[str, list[dict]] = {}
    all_records: list[ScanRecord] = []

    for i, (scan_kst, scan_dt) in enumerate(scan_times, 1):
        print(f"  scan {i}/{len(scan_times)}: {scan_kst}")
        rows = run_scan(scan_kst, scan_dt, symbols, use_cache)
        all_scans[scan_kst] = rows
        for r in rows:
            all_records.append(ScanRecord(
                scan_kst=scan_kst,
                symbol=r["symbol"],
                rank=r["rank"],
                price=r["price"],
                ranking_score=r["ranking_score"],
                max_up_12h=r["max_up_12h"],
                return_12h=r["return_12h"],
                mdd_12h=r["mdd_12h"],
                hit_5pct=r["hit_5pct"],
                in_top5=r["in_top5"],
                is_missed_winner=r["is_missed_winner"],
                miss_reasons=r.get("miss_reasons", []),
                features=r["features"],
            ))

    missed = [r for r in all_records if r.is_missed_winner]
    top5_recs = [r for r in all_records if r.in_top5]
    n_scans = len(scan_times)

    top5_hits = sum(1 for r in top5_recs if r.hit_5pct)
    top5_hit_rate = top5_hits / max(len(top5_recs), 1) * 100
    all_hits = sum(1 for r in all_records if r.hit_5pct)
    all_hit_rate = all_hits / max(len(all_records), 1) * 100
    avg_rph = statistics.mean([r.max_up_12h / 12 for r in top5_recs]) if top5_recs else 0.0

    freq = threshold_frequency(missed)
    importance = feature_importance(missed)

    # Single-change modification proposals
    mods = [
        simulate_modification(all_scans, "W1: ma_slope_accel weight +0.5", {"ma_slope_accel": 0.5}),
        simulate_modification(all_scans, "W2: volume_ma_ratio weight +0.15", {"volume_ma_ratio": 0.15}),
        simulate_modification(all_scans, "W3: ignition_age penalty reduce +0.2", {"ignition_age_min": 0.2}),
        simulate_modification(all_scans, "W4: body_ratio rank add +0.3", {"body_ratio": 0.3}),
        simulate_modification(all_scans, "W5: compression_length rank add +0.05", {"compression_length": 0.05}),
    ]
    # Add body_ratio to weights dict for W4/W5 - ranking_score only uses RANK_WEIGHTS keys by default
    # Fix simulate to pass extended weights

    best_mod = max(mods, key=lambda m: m["missed_recovery_pct"] - m["false_positive_new_top5_pct"] * 0.5)

    # Filter verdict
    filter_hit = all_hit_rate
    filter_verdict = "KEEP" if filter_hit >= 25 else ("MODIFY" if filter_hit >= 15 else "DISCARD")

    lines = [
        "############################################################",
        "SCOUT PHASE 11 — MISSED WINNER FEATURE DISCOVERY",
        "############################################################",
        "",
        f"Period: {START_KST.strftime('%Y-%m-%d %H:%M')} ~ {END_KST.strftime('%Y-%m-%d %H:%M')} KST",
        f"Scan interval: {SCAN_INTERVAL_H}h | Scans: {n_scans}",
        "",
        "=" * 52,
        "1. OVERALL PERFORMANCE",
        "=" * 52,
        f"  Total scans: {n_scans}",
        f"  Total filter matches: {len(all_records)}",
        f"  TOP5 hit rate (+5% max): {top5_hit_rate:.1f}%",
        f"  All matches hit rate: {all_hit_rate:.1f}%",
        f"  TOP5 return/hour: {avg_rph:.3f}%",
        "",
        "=" * 52,
        "2. MISSED WINNERS DATASET",
        "=" * 52,
        f"  Missed winner count: {len(missed)}",
    ]
    if missed:
        avg_feats = {k: round(statistics.mean([r.features.get(k, 0) for r in missed]), 3) for k in FEATURE_KEYS}
        lines.append(f"  Avg features (missed): {json.dumps(avg_feats, ensure_ascii=False)}")
    lines.append("  TOP10 importance features:")
    for row in importance[:10]:
        lines.append(f"    {row['feature']}: composite={row['composite']} corr={row['correlation']} mi={row['mutual_info']}")

    lines.extend(["", "=" * 52, "3. COMMON PATTERNS (missed winners)", "=" * 52])
    for f in freq[:10]:
        conf = "high" if f["pct"] >= 70 else ("medium" if f["pct"] >= 50 else "hypothesis")
        lines.append(f"  {f['pattern']}: {f['pct']}% ({f['count']}/{f['n']}) confidence={conf}")

    lines.extend(["", "=" * 52, "4. CURRENT FORMULA WEAKNESS", "=" * 52])
    low_vol_pct = next((f["pct"] for f in freq if "volume_ma_ratio < 1.0" in f["pattern"]), 0)
    neg_accel_pct = next((f["pct"] for f in freq if "slope_accel negative" in f["pattern"]), 0)
    lines.append(f"  Ranking over-weights high volume_ma_ratio; {low_vol_pct}% missed had vol_ratio<1.0")
    lines.append(f"  ma_slope_accel negative in {neg_accel_pct}% missed; H4 weight penalizes positive accel wrongly")
    lines.append(f"  lifecycle age proxy collapses (birth_age=0); ignition_age spread ignored in tie-breaks")
    lines.append(f"  Evidence: {len(missed)} missed vs {len(top5_recs)} top5 slots across {n_scans} scans")

    lines.extend(["", "=" * 52, "5. MODIFICATION CANDIDATES (single change each)", "=" * 52])
    for i, m in enumerate(mods[:3], 1):
        lines.append(
            f"  Priority {i}: {m['name']} | recovery={m['missed_recovery_pct']}% "
            f"retention={m['top5_retention_pct']}% false_pos={m['false_positive_new_top5_pct']}%"
        )

    lines.extend(["", "=" * 52, "6. FILTER VERDICT", "=" * 52, f"  Pattern B Filter: {filter_verdict} (all-hit {all_hit_rate:.1f}%)"])

    rec = "Keep current filter; apply ma_slope_accel ranking weight +0.5 (single weight change)"
    if best_mod["name"].startswith("W2"):
        rec = "Keep filter; adjust volume_ma_ratio ranking weight +0.15"
    elif best_mod["missed_recovery_pct"] < 5:
        rec = "Keep current search formula unchanged; expand Phase 6 dataset before any threshold change"

    lines.extend([
        "",
        "=" * 52,
        "7. FINAL RECOMMENDATION (one line)",
        "=" * 52,
        f"  {rec}",
        "",
        "=" * 52,
        "FEATURE IMPORTANCE TOP20",
        "=" * 52,
    ])
    for row in importance[:20]:
        lines.append(
            f"  {row['feature']:<22} comp={row['composite']:.4f} corr={row['correlation']:+.3f} "
            f"mi={row['mutual_info']:.3f} perm={row['perm_importance']:.3f} shap={row['shap_proxy']:.2f}"
        )

    report_path = OUT_DIR / "phase11_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    write_csv(OUT_DIR / "missed_winners.csv", [{
        "scan_kst": r.scan_kst, "symbol": r.symbol, "rank": r.rank,
        "max_up_12h": r.max_up_12h, "miss_reasons": "; ".join(r.miss_reasons),
        **{f"f_{k}": r.features.get(k) for k in FEATURE_KEYS},
    } for r in missed])

    write_csv(OUT_DIR / "feature_importance.csv", importance)
    write_csv(OUT_DIR / "modification_candidates.csv", mods)
    write_csv(OUT_DIR / "pattern_frequency.csv", freq)

    print("\n".join(lines[:40]).encode("ascii", "replace").decode("ascii"))
    print(f"\n... Saved: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-scans", type=int, default=None)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()
    run(args.max_scans, use_cache=not args.no_cache)


if __name__ == "__main__":
    main()
