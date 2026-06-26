"""
Scout Phase 14 — Out-of-Sample Blind Replay Validation

Holdout: 2026-05-08 ~ 2026-05-14. NO filter/threshold/weight changes.

Usage:
  python scout_phase14_blind_replay.py
  python scout_phase14_blind_replay.py --max-scans 5
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import top10_gainer_learning_20260613 as t10
from season2_p37_scout_decision_hierarchy import pf, write_csv
from season2_p40_scout_transition_triggers import ema
from season2_universe_blind_test import load_eligible_symbols, ohlcv

from scout_phase13_5m_sequence_ignition import (
    compute_at_anchor,
    fetch_klines_5m,
    candle_row,
)

OUT_DIR = Path("logs") / "phase14_holdout"
CACHE_DIR = OUT_DIR / "scan_cache"
PHASE13_REPORT = Path("logs") / "phase13_sequence" / "missed_sequence_avg.csv"
KST = timezone(timedelta(hours=9))
VAL_START = datetime(2026, 5, 8, 0, 0, tzinfo=KST)
VAL_END = datetime(2026, 5, 14, 23, 0, tzinfo=KST)
SCAN_INTERVAL_H = 2
INTERVAL_15M_MS = 15 * 60 * 1000
LOOKBACK_15M = 96
FORWARD_2H_15M = 8
API_SLEEP = 0.08
WORKERS = 12
MISS_THRESHOLD = 7.0
HIT_THRESHOLD = 5.0

PATTERN_B = (("macd_signal", "gte", -0.0016), ("range_pct", "gte", 1.4768))
RANK_WEIGHTS = {
    "young_birth": -1.4343174277392308,
    "birth_age_min": 1.5586524217896396,
    "ignition_age_min": -0.22688287801448231,
    "ma_slope_accel": -0.12327724295144488,
    "volume_ma_ratio": -0.06664243036600889,
}

MODELS = ("A", "B", "C", "D", "E")

REPORT_FEATURES = (
    "volume_ma_ratio", "seq_volume_energy_6", "seq_return_sum_6", "seq_body_energy_6",
    "first_abnormal_candle_6", "compression_length", "compression_std", "ma_slope",
    "ma_slope_accel", "range_pct", "body_ratio", "bb_expansion", "distance_from_low",
    "close_position", "high_close_2d", "rs_vs_btc",
)

TRAIN_IMPORTANCE = {
    "seq_volume_energy_6": 2.9708,
    "seq_return_sum_6": 0.4318,
    "first_abnormal_candle_6": 0.7742,
    "volume_ma_ratio": 1.0121,
    "seq_body_energy_6": 1.5043,
    "compression_length": 5.4516,
    "ma_slope_accel": 0.25,
}


def gen_scan_times() -> list[tuple[str, datetime]]:
    times: list[tuple[str, datetime]] = []
    t = VAL_START
    while t <= VAL_END:
        times.append((t.strftime("%Y-%m-%d %H:%M:%S"), t))
        t += timedelta(hours=SCAN_INTERVAL_H)
    return times


def fetch_15m(symbol: str, end_ms: int, limit: int) -> list[list]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"15m_{symbol}_{end_ms}.json"
    p = CACHE_DIR / tag
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    params = urllib.parse.urlencode({
        "symbol": symbol, "interval": "15m", "endTime": end_ms,
        "limit": min(limit, t10.MAX_LIMIT),
    })
    url = f"{t10.FUTURES_BASE_URL}{t10.KLINES_ENDPOINT}?{params}"
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url), timeout=25) as resp:
                data = json.loads(resp.read().decode())
            p.write_text(json.dumps(data), encoding="utf-8")
            return data
        except urllib.error.HTTPError as e:
            if e.code in (418, 429) and attempt < 3:
                time.sleep(2 * (attempt + 1))
                continue
            raise
    return []


def fetch_fwd_15m(symbol: str, start_ms: int) -> list[list]:
    tag = f"15mfwd_{symbol}_{start_ms}.json"
    p = CACHE_DIR / tag
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    params = urllib.parse.urlencode({
        "symbol": symbol, "interval": "15m", "startTime": start_ms,
        "limit": FORWARD_2H_15M,
    })
    url = f"{t10.FUTURES_BASE_URL}{t10.KLINES_ENDPOINT}?{params}"
    with urllib.request.urlopen(urllib.request.Request(url), timeout=25) as resp:
        data = json.loads(resp.read().decode())
    p.write_text(json.dumps(data), encoding="utf-8")
    return data


def macd_sig(closes: list[float]) -> float:
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


def lifecycle_15m(klines: list[list]) -> dict:
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


def filter_features_15m(klines: list[list]) -> dict | None:
    if len(klines) < 30:
        return None
    o, h, l, c, vol = ohlcv(klines[-1])
    if not (t10.MIN_PRICE <= c <= t10.MAX_PRICE):
        return None
    rng = (h - l) / o * 100 if o else 0
    closes = [float(k[4]) for k in klines]
    ms = macd_sig(closes)
    if ms < -0.0016 or rng < 1.4768:
        return None
    vols = [ohlcv(k)[4] for k in klines[-25:-1]]
    vol_ma = statistics.mean(vols[-24:]) if vols else 0.0
    lc = lifecycle_15m(klines)
    return {
        "price": c,
        "range_pct": rng,
        "macd_signal": ms,
        "volume_ma_ratio": vol / vol_ma if vol_ma else 0,
        **lc,
    }


def h4_score(f15: dict) -> float:
    ff = {
        "young_birth": f15.get("young_birth", 0),
        "birth_age_min": f15.get("birth_age_min", 0),
        "ignition_age_min": f15.get("ignition_age_min", 0),
        "ma_slope_accel": f15.get("ma_slope_accel", 0),
        "volume_ma_ratio": f15.get("volume_ma_ratio", 0),
    }
    return sum(RANK_WEIGHTS[k] * ff.get(k, 0) for k in RANK_WEIGHTS)


def model_score(model: str, h4: float, seq: dict) -> float:
    if model == "A":
        return h4
    if model == "B":
        return h4 + seq.get("seq_volume_energy_6", 0)
    if model == "C":
        return h4 + seq.get("seq_return_sum_6", 0)
    if model == "D":
        return h4 + seq.get("first_abnormal_candle_6", 0)
    aux = (
        0.5 * seq.get("seq_volume_energy_6", 0)
        + 0.3 * seq.get("seq_return_sum_6", 0)
        + 0.2 * seq.get("first_abnormal_candle_6", 0)
    )
    return h4 + aux


def seq_6_candles(klines_5m: list[list]) -> list[dict]:
    if len(klines_5m) < 6:
        return []
    vols = [ohlcv(k)[4] for k in klines_5m[-26:-1]]
    vol_ma = statistics.mean(vols[-20:]) if vols else 1.0
    out = []
    for k in klines_5m[-6:]:
        r = candle_row(k, vol_ma)
        out.append({
            "vol_ratio": round(r["volume_ma_ratio"], 3),
            "return_pct": round(r["return_pct"], 3),
            "body_ratio": round(r["body_ratio"], 3),
            "range_pct": round(r["range_pct"], 3),
            "sign": "+" if r["positive"] else "-",
        })
    return out


def assign_cluster(f: dict, max_up: float) -> str:
    if f.get("volume_ma_ratio", 0) < 1.0 and max_up >= 7:
        return "Low_Volume_Explosion"
    if f.get("compression_length", 0) >= 10 and f.get("seq_compression_release_6", 0) >= 1:
        return "Compression_Release"
    if f.get("ma_slope_accel", 0) > 0 and f.get("ma_slope", 0) > 1:
        return "Momentum"
    if f.get("range_pct", 0) >= 3:
        return "Expansion"
    return "Other"


def zest_like(f: dict) -> bool:
    return (
        f.get("volume_ma_ratio", 0) < 1.2
        and f.get("seq_positive_count_6", 0) >= 3
        and f.get("seq_volume_energy_6", 0) >= 1.5
        and f.get("seq_return_sum_6", 0) >= 1.0
        and f.get("first_abnormal_candle_6", 0) >= 1
    )


def forward_2h_max(entry: float, fwd: list[list]) -> float:
    if entry <= 0 or not fwd:
        return 0.0
    max_h = max(ohlcv(k)[1] for k in fwd[:FORWARD_2H_15M])
    return (max_h - entry) / entry * 100


def process_symbol(symbol: str, end_ms: int) -> dict | None:
    try:
        k15 = fetch_15m(symbol, end_ms, LOOKBACK_15M)
        f15 = filter_features_15m(k15)
        if not f15:
            return None
        k5 = fetch_klines_5m(symbol, end_ms, 120)
        seq = compute_at_anchor(k5, len(k5) - 1) if len(k5) >= 40 else {}
        merged = {**f15, **seq}
        merged["volume_ma_ratio"] = seq.get("volume_ma_ratio", f15.get("volume_ma_ratio", 0))
        h4 = h4_score(f15)
        fwd = fetch_fwd_15m(symbol, end_ms + INTERVAL_15M_MS)
        max2h = forward_2h_max(f15["price"], fwd)
        scores = {m: model_score(m, h4, seq) for m in MODELS}
        return {
            "symbol": symbol,
            "price": f15["price"],
            "h4_score": h4,
            "scores": scores,
            "features": merged,
            "seq_6": seq_6_candles(k5),
            "max_up_2h": round(max2h, 4),
            "hit_5pct": max2h >= HIT_THRESHOLD,
            "hit_7pct": max2h >= MISS_THRESHOLD,
            "zest_like": zest_like(merged),
        }
    except Exception:
        return None
    finally:
        time.sleep(API_SLEEP)


def rank_models(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for m in MODELS:
        ranked = sorted(rows, key=lambda r: r["scores"][m], reverse=True)
        for i, r in enumerate(ranked, 1):
            r[f"rank_{m}"] = i
        out[m] = ranked
    return out


def miss_reason(f: dict, rank_a: int, score_a: float, top5_min: float) -> str:
    parts = []
    if f.get("volume_ma_ratio", 0) < 1.0:
        parts.append(f"volume_ma_ratio={f.get('volume_ma_ratio',0):.2f}<1.0")
    if f.get("seq_volume_energy_6", 0) >= 1.5:
        parts.append(f"seq_volume_energy_6={f.get('seq_volume_energy_6',0):.2f} strong but H4 score low")
    if f.get("ma_slope_accel", 0) < 0:
        parts.append(f"ma_slope_accel={f.get('ma_slope_accel',0):.2f}<0")
    if score_a < top5_min:
        parts.append(f"h4_score={score_a:.2f}<top5_threshold={top5_min:.2f}")
    parts.append(f"rank_A={rank_a}")
    return "; ".join(parts)


def model_metrics(ranked: list[dict], model: str) -> dict:
    top5 = ranked[:5]
    top10 = ranked[:10]
    t5_hits = sum(1 for r in top5 if r["hit_5pct"])
    t10_hits = sum(1 for r in top10 if r["hit_5pct"])
    t5_fp = sum(1 for r in top5 if not r["hit_5pct"])
    avg_max = statistics.mean([r["max_up_2h"] for r in top5]) if top5 else 0
    all_winners = [r for r in ranked if r["hit_7pct"]]
    t5_recovered = sum(1 for r in top5 if r["hit_7pct"])
    recall = t5_recovered / max(len(all_winners), 1) * 100
    precision = t5_hits / max(len(top5), 1) * 100
    return {
        "model": model,
        "top5_hit_pct": round(t5_hits / max(len(top5), 1) * 100, 1),
        "top10_hit_pct": round(t10_hits / max(len(top10), 1) * 100, 1),
        "avg_max_up_2h": round(avg_max, 2),
        "false_positive_top5": t5_fp,
        "precision_pct": round(precision, 1),
        "recall_7pct_pct": round(recall, 1),
        "top5_n": len(top5),
    }


def run(max_scans: int | None = None) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    symbols = sorted(load_eligible_symbols(refresh=False, cache_only=False))
    scans = gen_scan_times()
    if max_scans:
        scans = scans[:max_scans]

    print(f"Phase 14 holdout: {len(scans)} scans, universe={len(symbols)}")

    all_missed: list[dict] = []
    all_rows: list[dict] = []
    zest_rows: list[dict] = []
    model_metrics_all: dict[str, list] = {m: [] for m in MODELS}
    scan_summaries: list[dict] = []

    for i, (scan_kst, scan_dt) in enumerate(scans, 1):
        print(f"  scan {i}/{len(scans)}: {scan_kst}")
        end_ms = int(scan_dt.timestamp() * 1000)
        rows: list[dict] = []
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futs = {pool.submit(process_symbol, sym, end_ms): sym for sym in symbols}
            for fut in as_completed(futs):
                r = fut.result()
                if r:
                    r["scan_kst"] = scan_kst
                    rows.append(r)

        if not rows:
            scan_summaries.append({"scan_kst": scan_kst, "matches": 0})
            continue

        ranked_by_model = rank_models(rows)
        ranked_a = ranked_by_model["A"]

        for m in MODELS:
            model_metrics_all[m].append(model_metrics(ranked_by_model[m], m))

        winners = [r for r in rows if r["hit_7pct"]]
        top5_a = ranked_a[:5]
        top5_min = top5_a[-1]["scores"]["A"] if top5_a else -999

        for r in rows:
            if r["rank_A"] > 5 and r["hit_7pct"]:
                r["cluster"] = assign_cluster(r["features"], r["max_up_2h"])
                r["miss_reason"] = miss_reason(r["features"], r["rank_A"], r["scores"]["A"], top5_min)
                r["ranks"] = {m: r[f"rank_{m}"] for m in MODELS}
                all_missed.append(r)
            if r.get("zest_like"):
                zest_rows.append(r)
            all_rows.append(r)

        scan_summaries.append({
            "scan_kst": scan_kst,
            "matches": len(rows),
            "winners_7pct": len(winners),
            "missed_a": sum(1 for r in rows if r["rank_A"] > 5 and r["hit_7pct"]),
        })

        cache_out = {
            "scan_kst": scan_kst,
            "models": {
                m: [{"symbol": r["symbol"], "rank": r[f"rank_{m}"], "max_up_2h": r["max_up_2h"],
                     "score": r["scores"][m]} for r in ranked_by_model[m][:30]]
                for m in MODELS
            },
        }
        tag = scan_kst.replace(" ", "_").replace(":", "")
        (CACHE_DIR / f"scan_{tag}.json").write_text(json.dumps(cache_out, ensure_ascii=False), encoding="utf-8")

    # Aggregate metrics
    agg: dict[str, dict] = {}
    for m in MODELS:
        ms = model_metrics_all[m]
        if not ms:
            continue
        agg[m] = {
            "model": m,
            "top5_hit_pct": round(statistics.mean([x["top5_hit_pct"] for x in ms]), 1),
            "top10_hit_pct": round(statistics.mean([x["top10_hit_pct"] for x in ms]), 1),
            "avg_max_up_2h": round(statistics.mean([x["avg_max_up_2h"] for x in ms]), 2),
            "false_positive_top5": sum(x["false_positive_top5"] for x in ms),
            "precision_pct": round(statistics.mean([x["precision_pct"] for x in ms]), 1),
            "recall_7pct_pct": round(statistics.mean([x["recall_7pct_pct"] for x in ms]), 1),
            "scans": len(ms),
        }

    # Holdout feature importance (missed winners)
    val_imp: dict[str, float] = {}
    if all_missed:
        labels = [r["max_up_2h"] for r in all_missed]
        for feat in REPORT_FEATURES:
            xs = [r["features"].get(feat, 0) for r in all_missed]
            if len(xs) >= 4:
                mx, my = statistics.mean(xs), statistics.mean(labels)
                num = sum((a - mx) * (b - my) for a, b in zip(xs, labels))
                den = math.sqrt(sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in labels))
                val_imp[feat] = round(num / den if den else 0, 4)

    feat_verdict: list[dict] = []
    for feat in ("seq_volume_energy_6", "seq_return_sum_6", "first_abnormal_candle_6",
                 "volume_ma_ratio", "seq_body_energy_6"):
        tr = TRAIN_IMPORTANCE.get(feat, 0)
        va = val_imp.get(feat, 0)
        if abs(va) >= abs(tr) * 0.5 and va > 0:
            verdict = "KEEP"
        elif va > 0:
            verdict = "MODIFY"
        else:
            verdict = "DISCARD"
        feat_verdict.append({"feature": feat, "train_corr_proxy": tr, "val_corr": va, "verdict": verdict})

    low_vol_holdout = sum(1 for r in all_missed if r.get("cluster") == "Low_Volume_Explosion")
    low_vol_pct = low_vol_holdout / max(len(all_missed), 1) * 100

    # Model E vs A comparison for Q5
    missed_recovered_e = sum(1 for r in all_missed if r["ranks"]["E"] <= 5 and r["ranks"]["A"] > 5)
    missed_recovered_b = sum(1 for r in all_missed if r["ranks"]["B"] <= 5 and r["ranks"]["A"] > 5)
    fp_e_extra = agg.get("E", {}).get("false_positive_top5", 0) - agg.get("A", {}).get("false_positive_top5", 0)

    # Final verdict
    a_hit = agg.get("A", {}).get("top5_hit_pct", 0)
    e_hit = agg.get("E", {}).get("top5_hit_pct", 0)
    e_recall = agg.get("E", {}).get("recall_7pct_pct", 0)
    a_recall = agg.get("A", {}).get("recall_7pct_pct", 0)
    if e_hit > a_hit + 3 and e_recall > a_recall and fp_e_extra <= 5:
        final = "KEEP"
    elif e_hit >= a_hit or e_recall > a_recall + 5:
        final = "MODIFY"
    else:
        final = "DISCARD"

    lines = [
        "###############################################################",
        "SCOUT PHASE 14 — OUT-OF-SAMPLE BLIND REPLAY VALIDATION",
        "###############################################################",
        "",
        f"Validation: {VAL_START.strftime('%Y-%m-%d')} ~ {VAL_END.strftime('%Y-%m-%d')} KST",
        "Training used for reference ONLY — no re-fit.",
        "",
        "=" * 58,
        "1. SCAN SUMMARY",
        "=" * 58,
        f"  Total scans: {len(scans)}",
        f"  Total 2h winners (>=7%): {sum(s.get('winners_7pct',0) for s in scan_summaries)}",
        f"  Total missed (Model A TOP5外 >=7%): {len(all_missed)}",
        "",
        "=" * 58,
        "2. MODEL COMPARISON (avg across scans)",
        "=" * 58,
    ]
    for m in MODELS:
        a = agg.get(m, {})
        lines.append(
            f"  Model {m}: TOP5_hit={a.get('top5_hit_pct')}% TOP10_hit={a.get('top10_hit_pct')}% "
            f"avg_max_2h={a.get('avg_max_up_2h')}% FP_top5={a.get('false_positive_top5')} "
            f"precision={a.get('precision_pct')}% recall_7={a.get('recall_7pct_pct')}%"
        )

    lines.extend(["", "=" * 58, "3. ZEST-LIKE DETECTION", "=" * 58])
    if zest_rows:
        z_top5 = sum(1 for r in zest_rows if r.get("rank_A", 99) <= 5)
        z_top10 = sum(1 for r in zest_rows if r.get("rank_A", 99) <= 10)
        lines.append(f"  Detected: {len(zest_rows)}")
        lines.append(f"  Avg max_up_2h: {statistics.mean([r['max_up_2h'] for r in zest_rows]):.2f}%")
        lines.append(f"  TOP5 rate (Model A): {round(z_top5/len(zest_rows)*100,1)}%")
        lines.append(f"  TOP10 rate (Model A): {round(z_top10/len(zest_rows)*100,1)}%")
    else:
        lines.append("  Detected: 0")

    lines.extend(["", "=" * 58, "4. HOLDOUT FEATURE IMPORTANCE vs TRAINING", "=" * 58])
    for fv in feat_verdict:
        delta = "increase" if fv["val_corr"] > fv["train_corr_proxy"] * 0.1 else (
            "decrease" if fv["val_corr"] < fv["train_corr_proxy"] * 0.5 else "stable")
        lines.append(f"  {fv['feature']}: val={fv['val_corr']:+.3f} train_proxy={fv['train_corr_proxy']} -> {fv['verdict']} ({delta})")

    lines.extend(["", "=" * 58, "5. KEY QUESTIONS", "=" * 58])
    lines.append(f"  Q1 Low Volume Explosion reproduced? {low_vol_pct:.1f}% of missed ({low_vol_holdout}/{len(all_missed)})")
    lines.append(f"  Q2 seq_volume_energy helps TOP5? B hit={agg.get('B',{}).get('top5_hit_pct')}% vs A={a_hit}%")
    lines.append(f"  Q3 seq_return_sum value? C hit={agg.get('C',{}).get('top5_hit_pct')}% recall={agg.get('C',{}).get('recall_7pct_pct')}%")
    lines.append(f"  Q4 first_abnormal_candle? D hit={agg.get('D',{}).get('top5_hit_pct')}%")
    lines.append(f"  Q5 aux_score: E recovered {missed_recovered_e} missed vs B={missed_recovered_b}, FP delta={fp_e_extra}")

    lines.extend(["", "=" * 58, "6. TOP MISSED CASES (Model A)", "=" * 58])
    for r in sorted(all_missed, key=lambda x: x["max_up_2h"], reverse=True)[:10]:
        f = r["features"]
        lines.append(
            f"  {r['scan_kst']} #{r['rank_A']} {r['symbol']} max2h={r['max_up_2h']}% cluster={r.get('cluster')}"
        )
        lines.append(f"    ranks A={r['ranks']['A']} B={r['ranks']['B']} C={r['ranks']['C']} D={r['ranks']['D']} E={r['ranks']['E']}")
        lines.append(f"    {r.get('miss_reason','')}")

    lines.extend(["", "=" * 58, "7. FINAL VERDICT", "=" * 58,
        f"  Decision: {final}",
        f"  Evidence: A TOP5_hit={a_hit}% E TOP5_hit={e_hit}% | A recall={a_recall}% E recall={e_recall}%",
        f"  Missed recovered by E: {missed_recovered_e}/{len(all_missed)} | FP increase: {fp_e_extra}",
        "",
        f"ONE LINE: {final} — aux_score {'improves' if e_hit > a_hit else 'does not improve'} holdout TOP5 hit "
        f"({a_hit}% -> {e_hit}%) with recall ({a_recall}% -> {e_recall}%).",
    ])

    report = OUT_DIR / "phase14_report.txt"
    report.write_text("\n".join(lines), encoding="utf-8")

    with (OUT_DIR / "missed_holdout.jsonl").open("w", encoding="utf-8") as f:
        for r in all_missed:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    write_csv(OUT_DIR / "model_comparison.csv", [agg[m] for m in MODELS if m in agg])
    write_csv(OUT_DIR / "feature_verdict.csv", feat_verdict)
    write_csv(OUT_DIR / "scan_summary.csv", scan_summaries)

    print("\n".join(lines[-20:]).encode("ascii", "replace").decode("ascii"))
    print(f"\nSaved: {report}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-scans", type=int, default=None)
    args = parser.parse_args()
    run(args.max_scans)


if __name__ == "__main__":
    main()
