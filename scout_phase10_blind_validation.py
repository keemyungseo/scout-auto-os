"""
Scout Phase 10 — Blind Timestamp Validation

Fixed Filter (Pattern B) + H4 Lifecycle Composite Ranking.
NO future data during search. Forward data ONLY in AFTER REPORT.

Usage:
  python scout_phase10_blind_validation.py
  python scout_phase10_blind_validation.py --scan-kst "2026-06-10 11:00:00"
"""

from __future__ import annotations

import argparse
import json
import math
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

OUT_DIR = Path("logs") / "phase10_blind"
KST = timezone(timedelta(hours=9))
INTERVAL_15M_MS = 15 * 60 * 1000
LOOKBACK_15M = 96
FORWARD_15M = 48
API_SLEEP = 0.03
WORKERS = 10
BIRTH_PCT = 3.0

PATTERN_B = (
    ("macd_signal", "gte", -0.0016),
    ("range_pct", "gte", 1.4768),
)

# Phase 9 H4 Lifecycle Composite — frozen TRAIN weights (no macd)
RANK_WEIGHTS = {
    "young_birth": -1.4343174277392308,
    "birth_age_min": 1.5586524217896396,
    "ignition_age_min": -0.22688287801448231,
    "ma_slope_accel": -0.12327724295144488,
    "volume_ma_ratio": -0.06664243036600889,
}

COHORT = {
    "expected_max_12h": 7.70,
    "expected_return": 7.70,
    "return_per_hour": 0.64,
    "expected_mdd": 5.90,
    "expected_hold_min": 720,
    "base_confidence": 71.0,
}


@dataclass
class Candidate:
    symbol: str
    price: float
    phase_estimate: str
    birth_age: float
    ignition_age: float
    range_pct: float
    volume_ma_ratio: float
    ma_slope_accel: float
    young_birth: float
    macd_signal: float
    confidence: float
    ranking_score: float
    rank: int = 0
    predictions: dict = field(default_factory=dict)
    forward: dict = field(default_factory=dict)
    actual_phase: str = ""
    phase_transitions: list = field(default_factory=list)
    features: dict = field(default_factory=dict)


def parse_kst(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=KST)


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


def macd_values(closes: list[float]) -> tuple[float, float]:
    if len(closes) < 26:
        return 0.0, 0.0
    line = ema(closes, 12) - ema(closes, 26)
    hist = [ema(closes[:i], 12) - ema(closes[:i], 26) for i in range(26, len(closes) + 1)]
    signal = ema(hist, 9) if hist else line
    return line, signal


def ma_slope_pct(klines: list[list]) -> float:
    if len(klines) < 14:
        return 0.0
    closes = [float(k[4]) for k in klines]
    recent = statistics.mean(closes[-6:-1])
    prior = statistics.mean(closes[-12:-7]) if len(closes) >= 12 else recent
    return (recent - prior) / prior * 100 if prior else 0.0


def estimate_lifecycle(klines: list[list]) -> dict:
    """Search-time only — lookback klines, no forward."""
    anchor = len(klines) - 1
    window = klines[max(0, anchor - 48): anchor + 1]
    base_price = min(ohlcv(k)[2] for k in window) if window else ohlcv(klines[-1])[2]

    birth_i = anchor
    for i in range(anchor, max(0, anchor - 32), -1):
        if ohlcv(klines[i])[4] >= base_price * (1.0 + BIRTH_PCT / 100.0):
            birth_i = i
            break

    ignition_i = anchor
    for i in range(anchor, max(0, anchor - 24), -1):
        vols = [ohlcv(klines[j])[4] for j in range(max(0, i - 8), i)]
        if vols and statistics.mean(vols) > 0:
            if ohlcv(klines[i])[4] / statistics.mean(vols) >= 1.2:
                ignition_i = i
                break

    birth_age = (anchor - birth_i) * 15
    ignition_age = (anchor - ignition_i) * 15
    slope_now = ma_slope_pct(klines)
    slope_prior = ma_slope_pct(klines[:-4]) if len(klines) > 20 else slope_now

    return {
        "birth_age_min": float(birth_age),
        "ignition_age_min": float(ignition_age),
        "young_birth": 1.0 if birth_age <= 45 else 0.0,
        "ma_slope_accel": slope_now - slope_prior,
        "ma_slope": slope_now,
    }


def compute_features(klines: list[list]) -> dict | None:
    if len(klines) < 30:
        return None
    o, h, l, c, vol = ohlcv(klines[-1])
    if not (t10.MIN_PRICE <= c <= t10.MAX_PRICE):
        return None
    vols = [ohlcv(k)[4] for k in klines[-25:-1]]
    vol_ma = statistics.mean(vols[-24:]) if vols else 0.0
    rng = (h - l) / o * 100 if o else 0.0
    closes = [float(k[4]) for k in klines]
    _, macd_sig = macd_values(closes)
    lc = estimate_lifecycle(klines)
    vol_ratio = vol / vol_ma if vol_ma > 0 else 0.0
    return {
        "price": c,
        "range_pct": rng,
        "macd_signal": macd_sig,
        "volume_ma_ratio": vol_ratio,
        **lc,
    }


def passes_pattern(f: dict) -> bool:
    for name, op, thr in PATTERN_B:
        v = f.get(name)
        if v is None:
            return False
        if op == "gte" and v < thr:
            return False
        if op == "lte" and v > thr:
            return False
    return True


def phase_estimate(f: dict) -> str:
    if f["young_birth"] >= 1.0 and f["volume_ma_ratio"] >= 1.0:
        return "Birth"
    if f["volume_ma_ratio"] >= 1.2 and f["range_pct"] < 2.5:
        return "Ignition"
    if f["ma_slope"] >= 0.3 and f["range_pct"] >= 1.4768:
        return "Birth"
    if abs(f["ma_slope"]) < 1.0:
        return "Accumulation"
    return "Birth"


def ranking_score(f: dict) -> float:
    return sum(RANK_WEIGHTS.get(k, 0) * f.get(k, 0) for k in RANK_WEIGHTS)


def confidence_score(f: dict, score: float) -> float:
    pct = min(100, max(0, 50 + score * 2))
    return round(min(100, COHORT["base_confidence"] + pct * 0.15), 1)


def blind_predictions(f: dict, score: float) -> dict:
    adj = min(10, max(0, score - statistics.mean(list(RANK_WEIGHTS.values())) * 30))
    exp_max = COHORT["expected_max_12h"] + adj * 0.2
    exp_mdd = COHORT["expected_mdd"] + max(0, -f.get("ma_slope_accel", 0)) * 0.5
    hold = 720 if f.get("young_birth", 0) >= 1 else 480
    return {
        "expected_phase": phase_estimate(f),
        "expected_max_up": round(exp_max, 2),
        "expected_mdd": round(exp_mdd, 2),
        "expected_return": round(exp_max, 2),
        "return_per_hour": round(exp_max / 12.0, 3),
        "confidence": confidence_score(f, score),
        "holding_time_min": hold,
    }


def scan_symbol(symbol: str, end_ms: int) -> Candidate | None:
    try:
        klines = fetch_klines_15m(symbol, end_ms, LOOKBACK_15M)
        feats = compute_features(klines)
        if not feats or not passes_pattern(feats):
            return None
        score = ranking_score(feats)
        preds = blind_predictions(feats, score)
        return Candidate(
            symbol=symbol,
            price=round(feats["price"], 8),
            phase_estimate=phase_estimate(feats),
            birth_age=round(feats["birth_age_min"], 0),
            ignition_age=round(feats["ignition_age_min"], 0),
            range_pct=round(feats["range_pct"], 4),
            volume_ma_ratio=round(feats["volume_ma_ratio"], 4),
            ma_slope_accel=round(feats["ma_slope_accel"], 4),
            young_birth=feats["young_birth"],
            macd_signal=round(feats["macd_signal"], 6),
            confidence=preds["confidence"],
            ranking_score=round(score, 4),
            predictions=preds,
            features=feats,
        )
    except Exception:
        return None
    finally:
        time.sleep(API_SLEEP)


def horizon_stats(entry: float, forward: list[list], candles: int) -> dict:
    if entry <= 0 or not forward:
        return {}
    chunk = forward[:candles]
    if not chunk:
        return {}
    close = float(chunk[-1][4])
    max_h = max(ohlcv(k)[1] for k in chunk)
    min_l = min(ohlcv(k)[2] for k in chunk)
    return {
        "return_pct": round((close - entry) / entry * 100, 4),
        "max_up_pct": round((max_h - entry) / entry * 100, 4),
        "mdd_pct": round((entry - min_l) / entry * 100, 4),
    }


def infer_actual_phase(entry: float, forward: list[list]) -> tuple[str, list]:
    if not forward:
        return "Unknown", []
    transitions: list[str] = []
    phase = "Accumulation"
    peak = entry
    birth_done = False
    for i, k in enumerate(forward):
        c, h = float(k[4]), float(k[1])
        ret = (c - entry) / entry * 100
        if h > peak:
            peak = h
        mins = (i + 1) * 15
        new_phase = phase
        if not birth_done and ret >= 3.0:
            new_phase, birth_done = "Birth", True
        elif birth_done and ret >= 5.0 and phase in ("Birth", "Ignition", "Accumulation"):
            new_phase = "Expansion"
        elif ret >= 7.0 and phase in ("Expansion", "Birth"):
            new_phase = "Continuation"
        elif peak > entry * 1.05 and c < peak * 0.97 and phase in ("Continuation", "Expansion"):
            new_phase = "Exhaustion"
        elif ret < 0 and phase == "Exhaustion":
            new_phase = "Distribution"
        if i == len(forward) - 1:
            new_phase = "End"
        if new_phase != phase:
            transitions.append(f"{mins}min:{phase}->{new_phase}")
            phase = new_phase
    return phase, transitions


def evaluate_forward(c: Candidate, end_ms: int) -> None:
    try:
        fwd = fetch_forward_15m(c.symbol, end_ms + INTERVAL_15M_MS, FORWARD_15M)
        entry = c.price
        horizons = {"15m": 1, "30m": 2, "1h": 4, "2h": 8, "4h": 16, "8h": 32, "12h": 48}
        c.forward = {h: horizon_stats(entry, fwd, n) for h, n in horizons.items()}
        if fwd:
            max_h = max(ohlcv(k)[1] for k in fwd)
            min_l = min(ohlcv(k)[2] for k in fwd)
            close_12 = float(fwd[min(len(fwd), 48) - 1][4])
            s = c.forward.get("12h", {})
            c.forward["summary"] = {
                "actual_return_12h": round((close_12 - entry) / entry * 100, 4),
                "actual_max_up_12h": round((max_h - entry) / entry * 100, 4),
                "actual_mdd_12h": round((entry - min_l) / entry * 100, 4),
                "hit_3pct_30m": c.forward.get("30m", {}).get("max_up_pct", 0) >= 3.0,
                "hit_5pct_1h": c.forward.get("1h", {}).get("max_up_pct", 0) >= 5.0,
                "hit_7pct_2h": c.forward.get("2h", {}).get("max_up_pct", 0) >= 7.0,
                "hit_10pct_4h": c.forward.get("4h", {}).get("max_up_pct", 0) >= 10.0,
            }
        c.actual_phase, c.phase_transitions = infer_actual_phase(entry, fwd)
    except Exception:
        c.forward, c.actual_phase = {}, "Unknown"


def spearman(pairs: list[tuple[float, float]]) -> float:
    n = len(pairs)
    if n < 3:
        return 0.0
    xs, ys = [p[0] for p in pairs], [p[1] for p in pairs]

    def ranks(vals: list[float]) -> list[float]:
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        r = [0.0] * len(vals)
        i = 0
        while i < len(vals):
            j = i
            while j + 1 < len(vals) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    d2 = sum((a - b) ** 2 for a, b in zip(rx, ry))
    return 1 - 6 * d2 / (n * (n * n - 1))


def cohort_hit_rates(group: list[Candidate]) -> dict:
    if not group:
        return {}
    s = [c.forward.get("summary", {}) for c in group if c.forward.get("summary")]
    n = len(s)
    return {
        "n": n,
        "hit_3pct": round(sum(1 for x in s if x.get("hit_3pct_30m")) / max(n, 1) * 100, 1),
        "hit_5pct": round(sum(1 for x in s if x.get("hit_5pct_1h")) / max(n, 1) * 100, 1),
        "hit_7pct": round(sum(1 for x in s if x.get("hit_7pct_2h")) / max(n, 1) * 100, 1),
        "hit_10pct": round(sum(1 for x in s if x.get("hit_10pct_4h")) / max(n, 1) * 100, 1),
        "avg_max_up": round(statistics.mean([x.get("actual_max_up_12h", 0) for x in s]), 2) if s else 0,
        "return_per_hour": round(statistics.mean([x.get("actual_max_up_12h", 0) for x in s]) / 12, 3) if s else 0,
    }


def top_k_accuracy(candidates: list[Candidate], k: int) -> bool:
    if not candidates:
        return False
    best = max(candidates, key=lambda c: c.forward.get("summary", {}).get("actual_max_up_12h", 0))
    topk = candidates[:k]
    return any(c.symbol == best.symbol for c in topk)


def miss_analysis(c: Candidate, top5_syms: set[str]) -> str:
    if c.symbol in top5_syms:
        return ""
    s = c.forward.get("summary", {})
    parts = []
    if c.birth_age > 45:
        parts.append(f"birth_age={c.birth_age:.0f}min>45 (young_birth=0)")
    if c.volume_ma_ratio < 1.0:
        parts.append(f"volume_ma_ratio={c.volume_ma_ratio:.2f}<1.0")
    if c.ma_slope_accel < 0:
        parts.append(f"ma_slope_accel={c.ma_slope_accel:.2f}<0")
    if c.ignition_age < 30:
        parts.append(f"ignition_age={c.ignition_age:.0f}min short")
    if c.ranking_score < statistics.mean([0, 0]):
        pass
    parts.append(f"ranking_score={c.ranking_score:.2f} below TOP5 threshold")
    return "; ".join(parts) if parts else "lower composite lifecycle score"


def failure_analysis(c: Candidate) -> str:
    s = c.forward.get("summary", {})
    max_up = s.get("actual_max_up_12h", 0)
    parts = []
    if max_up < 5:
        parts.append(f"max_up_12h={max_up:.2f}%<5%")
    if c.volume_ma_ratio < 0.8:
        parts.append(f"volume_ma_ratio={c.volume_ma_ratio:.2f} insufficient")
    if c.ma_slope_accel < 0:
        parts.append(f"ma_slope_accel={c.ma_slope_accel:.2f} deceleration")
    if c.birth_age > 60:
        parts.append(f"birth_age={c.birth_age:.0f}min late-cycle")
    if c.range_pct > 5:
        parts.append(f"range_pct={c.range_pct:.2f} overextended at entry")
    if c.macd_signal < -0.001:
        parts.append(f"macd_signal={c.macd_signal:.6f} weak filter margin")
    return "; ".join(parts) or "no follow-through after scan"


def top5_reason(c: Candidate, rank6: Candidate | None) -> str:
    reasons = []
    if c.young_birth >= 1:
        reasons.append(f"young_birth=1 (age={c.birth_age:.0f}min)")
    if c.birth_age > (rank6.birth_age if rank6 else 0):
        reasons.append(f"birth_age={c.birth_age:.0f} > rank6")
    if c.volume_ma_ratio > (rank6.volume_ma_ratio if rank6 else 0):
        reasons.append(f"vol_ratio={c.volume_ma_ratio:.2f}")
    reasons.append(f"score={c.ranking_score:.2f}")
    return "; ".join(reasons)


def final_scores(candidates: list[Candidate], top5: list[Candidate], top10: list[Candidate]) -> dict:
    all_h = cohort_hit_rates(candidates)
    t5 = cohort_hit_rates(top5)
    t10 = cohort_hit_rates(top10)
    summaries = [c.forward.get("summary", {}) for c in top5 if c.forward.get("summary")]

    filter_hit = sum(1 for c in candidates if c.forward.get("summary", {}).get("actual_max_up_12h", 0) >= 5.0)
    filter_quality = min(100, filter_hit / max(len(candidates), 1) * 100 * 1.5)

    exp_rph = statistics.mean([c.predictions.get("return_per_hour", 0) for c in top5]) if top5 else 0
    act_rph = t5.get("return_per_hour", 0)
    ranking_quality = min(100, act_rph / max(exp_rph, 0.01) * 50 + t5.get("hit_5pct", 0) * 0.5)

    phase_ok = sum(
        1 for c in top5
        if c.phase_estimate in ("Birth", "Ignition") and c.actual_phase in ("Birth", "Expansion", "Continuation", "End")
    )
    lifecycle_acc = phase_ok / max(len(top5), 1) * 100

    exp_ret_acc = min(100, 100 - abs(statistics.mean([c.predictions.get("expected_return", 0) for c in top5]) - t5.get("avg_max_up", 0))) if top5 and summaries else 0
    rph_acc = min(100, act_rph / max(exp_rph, 0.01) * 70) if exp_rph else 0
    top5_hit = t5.get("hit_5pct", 0)

    outside = [c for c in candidates if c.rank > 5]
    best_out = max(outside, key=lambda c: c.forward.get("summary", {}).get("actual_max_up_12h", 0), default=None)
    miss_penalty = 0
    if best_out and best_out.forward.get("summary", {}).get("actual_max_up_12h", 0) > t5.get("avg_max_up", 0):
        miss_penalty = 20

    conf_err = statistics.mean([
        abs(c.confidence - min(100, 50 + s.get("actual_max_up_12h", 0) * 5))
        for c, s in zip(top5, summaries)
    ]) if summaries else 50
    conf_cal = max(0, 100 - conf_err)

    scores = {
        "filter_quality": round(filter_quality, 1),
        "ranking_quality": round(ranking_quality, 1),
        "lifecycle_accuracy": round(lifecycle_acc, 1),
        "expected_return_accuracy": round(exp_ret_acc, 1),
        "return_per_hour_accuracy": round(rph_acc, 1),
        "top5_hit_rate": round(top5_hit, 1),
        "miss_opportunity": round(max(0, 100 - miss_penalty - (10 if best_out and best_out.rank <= 10 else 30)), 1),
        "confidence_calibration": round(conf_cal, 1),
    }
    scores["total"] = round(statistics.mean(list(scores.values())), 1)
    return scores


def decide_verdict(scores: dict, t5_hit: float, filter_hit_pct: float) -> str:
    if scores["total"] >= 60 and t5_hit >= 60 and filter_hit_pct >= 40:
        return "KEEP"
    if scores["total"] >= 40 or filter_hit_pct >= 35:
        return "MODIFY"
    return "DISCARD"


def run(scan_kst: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scan_dt = parse_kst(scan_kst)
    end_ms = int(scan_dt.timestamp() * 1000)
    symbols = sorted(load_eligible_symbols(refresh=False, cache_only=False))
    print(f"Phase 10 blind: {scan_kst} KST | universe={len(symbols)}")

    candidates: list[Candidate] = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = {pool.submit(scan_symbol, sym, end_ms): sym for sym in symbols}
        done = 0
        for fut in as_completed(futs):
            done += 1
            c = fut.result()
            if c:
                candidates.append(c)
            if done % 80 == 0:
                print(f"  scanned {done}/{len(symbols)}, matches={len(candidates)}")

    candidates.sort(key=lambda x: x.ranking_score, reverse=True)
    for i, c in enumerate(candidates, 1):
        c.rank = i

    print(f"Matches: {len(candidates)}. Forward evaluation...")
    for c in candidates:
        evaluate_forward(c, end_ms)
        time.sleep(API_SLEEP)

    top10 = candidates[:10]
    top5 = candidates[:5]
    rank6 = candidates[5] if len(candidates) > 5 else None

    all_stats = cohort_hit_rates(candidates)
    t10_stats = cohort_hit_rates(top10)
    t5_stats = cohort_hit_rates(top5)
    pairs = [(c.ranking_score, c.forward.get("summary", {}).get("actual_max_up_12h", 0)) for c in candidates if c.forward.get("summary")]
    rho = spearman(pairs)

    outside_top5 = [c for c in candidates if c.rank > 5]
    best_miss = max(outside_top5, key=lambda c: c.forward.get("summary", {}).get("actual_max_up_12h", 0)) if outside_top5 else None
    top5_syms = {c.symbol for c in top5}

    scores = final_scores(candidates, top5, top10)
    filter_hit_pct = round(sum(1 for c in candidates if c.forward.get("summary", {}).get("actual_max_up_12h", 0) >= 5.0) / max(len(candidates), 1) * 100, 1)
    verdict = decide_verdict(scores, t5_stats.get("hit_5pct", 0), filter_hit_pct)

    # Conservative auto-trading assessment
    auto_filter = "YES" if filter_hit_pct >= 45 and len(candidates) >= 10 else "NO"
    auto_rank = "YES" if t5_stats.get("hit_5pct", 0) >= 60 and scores["ranking_quality"] >= 55 else "NO"
    weakness = "holdout sample thin; lifecycle age from lookback proxy; TOP5 miss best non-ranked symbols"
    next_fix = "Ranking" if filter_hit_pct >= 40 else "Filter"
    project_completion = min(45, round(scores["total"] * 0.45, 0))

    lines = [
        "##############################################################",
        "SCOUT PHASE 10 — BLIND TIMESTAMP VALIDATION",
        "##############################################################",
        "",
        f"Search Time (KST): {scan_kst}",
        "FILTER: macd_signal >= -0.0016 AND range_pct >= 1.4768",
        "RANK: H4 Lifecycle Composite (macd excluded from ranking)",
        f"Universe: {len(symbols)} | Matches: {len(candidates)}",
        "",
        "=" * 60,
        "NO FUTURE SECTION",
        "=" * 60,
        "",
        f"{'Rank':<5}{'Symbol':<14}{'Price':<12}{'Phase':<12}{'BirthAge':<9}{'IgnAge':<8}"
        f"{'Range%':<8}{'VolRatio':<9}{'SlopeAcc':<9}{'Conf%':<7}{'Score'}",
    ]
    for c in top10:
        lines.append(
            f"{c.rank:<5}{c.symbol:<14}{c.price:<12}{c.phase_estimate:<12}"
            f"{c.birth_age:<9.0f}{c.ignition_age:<8.0f}{c.range_pct:<8}"
            f"{c.volume_ma_ratio:<9}{c.ma_slope_accel:<9}{c.confidence:<7}{c.ranking_score}"
        )

    lines.extend(["", "=" * 60, "BLIND PREDICTIONS (search-time only)", "=" * 60, ""])
    for c in top10:
        p = c.predictions
        lines.append(
            f"{c.symbol}: phase={p['expected_phase']} E[max]={p['expected_max_up']}% "
            f"E[MDD]={p['expected_mdd']}% E[ret]={p['expected_return']}% "
            f"r/h={p['return_per_hour']}% conf={p['confidence']}% hold={p['holding_time_min']}min"
        )

    lines.extend(["", "=" * 60, "TOP5 RECOMMENDATION (Return/Hour ranking)", "=" * 60, ""])
    for c in top5:
        p = c.predictions
        lines.append(
            f"#{c.rank} {c.symbol} r/h={p['return_per_hour']}% conf={p['confidence']}% | {top5_reason(c, rank6)}"
        )
    if rank6:
        lines.append(f"  vs #6 {rank6.symbol}: score {top5[-1].ranking_score:.2f} > {rank6.ranking_score:.2f}; "
                     f"birth_age {top5[-1].birth_age:.0f} vs {rank6.birth_age:.0f}")

    lines.extend(["", "=" * 60, "AFTER REPORT — REAL RESULTS (TOP5)", "=" * 60, ""])
    for c in top5:
        lines.append(f"\n--- {c.symbol} ---")
        lines.append(f"{'Hz':<6}{'Ret%':<10}{'MaxUp%':<10}{'MDD%'}")
        for h in ("15m", "30m", "1h", "2h", "4h", "8h", "12h"):
            d = c.forward.get(h, {})
            if d:
                lines.append(f"{h:<6}{d.get('return_pct',''):<10}{d.get('max_up_pct',''):<10}{d.get('mdd_pct','')}")

    lines.extend(["", "=" * 60, "LIFECYCLE EVALUATION", "=" * 60, ""])
    for c in top5:
        ok = c.phase_estimate in ("Birth", "Ignition") and c.actual_phase in ("Birth", "Expansion", "Continuation", "End")
        lines.append(
            f"{c.symbol}: expected={c.phase_estimate} actual={c.actual_phase} "
            f"{'OK' if ok else 'MISS'} | transitions={'; '.join(c.phase_transitions[:4])}"
        )

    lines.extend(["", "=" * 60, "RANKING EVALUATION", "=" * 60, ""])
    for label, st in [("ALL", all_stats), ("TOP10", t10_stats), ("TOP5", t5_stats)]:
        lines.append(
            f"  {label} n={st.get('n',0)} +3%={st.get('hit_3pct')}% +5%={st.get('hit_5pct')}% "
            f"+7%={st.get('hit_7pct')}% +10%={st.get('hit_10pct')}% r/h={st.get('return_per_hour')}"
        )
    lines.append(f"  Spearman(score vs max_up): {round(rho, 3)}")
    lines.append(f"  Top1 accuracy: {top_k_accuracy(candidates, 1)}")
    lines.append(f"  Top3 accuracy: {top_k_accuracy(candidates, 3)}")
    lines.append(f"  Top5 accuracy: {top_k_accuracy(candidates, 5)}")

    lines.extend(["", "=" * 60, "MISS OPPORTUNITY", "=" * 60, ""])
    if best_miss:
        s = best_miss.forward.get("summary", {})
        lines.append(
            f"  Best outside TOP5: #{best_miss.rank} {best_miss.symbol} "
            f"max_up={s.get('actual_max_up_12h')}% | {miss_analysis(best_miss, top5_syms)}"
        )

    lines.extend(["", "=" * 60, "FAILURE ANALYSIS (TOP5 failures)", "=" * 60, ""])
    for c in top5:
        s = c.forward.get("summary", {})
        ok = s.get("actual_max_up_12h", 0) >= 5.0
        lines.append(f"  {c.symbol}: {'PASS' if ok else 'FAIL'} — {failure_analysis(c)}")

    lines.extend(["", "=" * 60, "FINAL SCORE (100 each)", "=" * 60, ""])
    for k, v in scores.items():
        lines.append(f"  {k}: {v}")

    lines.extend([
        "",
        "=" * 60,
        "FINAL CONCLUSION",
        "=" * 60,
        f"Verdict: {verdict}",
        "",
        "Q1. Auto-trading ready (filter)? " + auto_filter,
        f"    Evidence: filter hit +5% = {filter_hit_pct}% ({len(candidates)} matches)",
        "Q2. Ranking trustworthy? " + auto_rank,
        f"    Evidence: TOP5 hit +5% = {t5_stats.get('hit_5pct')}% ranking_quality={scores['ranking_quality']}",
        f"Q3. Biggest weakness: {weakness}",
        f"Q4. Next fix: {next_fix} only (filter hit {filter_hit_pct}%)",
        f"Q5. Project completion (conservative, auto-trading): {project_completion}/100",
    ])

    tag = scan_kst.replace(" ", "_").replace(":", "")
    path = OUT_DIR / f"blind_{tag}.txt"
    path.write_text("\n".join(lines), encoding="utf-8")

    csv_rows = [{
        "rank": c.rank, "symbol": c.symbol, "price": c.price,
        "phase": c.phase_estimate, "birth_age": c.birth_age, "ignition_age": c.ignition_age,
        "range_pct": c.range_pct, "volume_ma_ratio": c.volume_ma_ratio,
        "ma_slope_accel": c.ma_slope_accel, "confidence": c.confidence,
        "ranking_score": c.ranking_score,
        **{f"pred_{k}": v for k, v in c.predictions.items()},
        **{f"actual_{k}": v for k, v in c.forward.get("summary", {}).items()},
        "actual_phase": c.actual_phase,
    } for c in candidates]
    write_csv(OUT_DIR / f"blind_{tag}.csv", csv_rows)

    print("\n".join(lines[-35:]).encode("ascii", "replace").decode("ascii"))
    print(f"\nSaved: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-kst", default="2026-06-10 11:00:00")
    args = parser.parse_args()
    run(args.scan_kst)


if __name__ == "__main__":
    main()
