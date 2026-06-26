"""
Scout Phase 8 — Blind Timestamp Simulation

Fixed Pattern B search at a historical timestamp.
NO future data during search. Forward data used ONLY in AFTER REPORT section.

Usage:
  python scout_phase8_blind_simulation.py
  python scout_phase8_blind_simulation.py --scan-kst "2026-06-03 11:00:00"
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

OUT_DIR = Path("logs") / "phase8_blind"
KST = timezone(timedelta(hours=9))
INTERVAL_15M = "15m"
INTERVAL_15M_MS = 15 * 60 * 1000
LOOKBACK_15M = 96
FORWARD_15M = 48
API_SLEEP = 0.03
WORKERS = 10

# Pattern B — FIXED (Phase 7 KEEP). Do not modify.
PATTERN_B = (
    ("macd_signal", "gte", -0.0016),
    ("range_pct", "gte", 1.4768),
)

# Pattern B holdout cohort priors (Phase 7). Fixed blind predictions.
COHORT = {
    "prob_30m_3pct": 35.0,
    "prob_1h_5pct": 42.0,
    "prob_2h_7pct": 40.0,
    "prob_4h_10pct": 20.0,
    "expected_max_12h": 7.70,
    "expected_return": 7.70,
    "return_per_hour": 0.64,
    "expected_mdd": 5.90,
    "expected_hold_min": 720,
    "base_confidence": 71.0,
    "holdout_hit_5": 50.0,
    "holdout_n": 10,
}


@dataclass
class Candidate:
    symbol: str
    price: float
    phase_estimate: str
    macd_signal: float
    range_pct: float
    volume_ma_ratio: float
    ma_slope: float
    confidence: float
    score: float
    rank: int = 0
    predictions: dict = field(default_factory=dict)
    forward: dict = field(default_factory=dict)
    actual_phase: str = ""
    phase_transitions: list = field(default_factory=list)


def parse_kst(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=KST)


def fetch_klines_15m(symbol: str, end_ms: int, limit: int) -> list[list]:
    params = urllib.parse.urlencode({
        "symbol": symbol,
        "interval": INTERVAL_15M,
        "endTime": end_ms,
        "limit": min(limit, t10.MAX_LIMIT),
    })
    url = f"{t10.FUTURES_BASE_URL}{t10.KLINES_ENDPOINT}?{params}"
    with urllib.request.urlopen(urllib.request.Request(url), timeout=25) as resp:
        return json.loads(resp.read().decode())


def fetch_forward_15m(symbol: str, start_ms: int, count: int) -> list[list]:
    params = urllib.parse.urlencode({
        "symbol": symbol,
        "interval": INTERVAL_15M,
        "startTime": start_ms,
        "limit": min(count, t10.MAX_LIMIT),
    })
    url = f"{t10.FUTURES_BASE_URL}{t10.KLINES_ENDPOINT}?{params}"
    with urllib.request.urlopen(urllib.request.Request(url), timeout=25) as resp:
        return json.loads(resp.read().decode())


def macd_values(closes: list[float]) -> tuple[float, float]:
    if len(closes) < 26:
        return 0.0, 0.0
    line = ema(closes, 12) - ema(closes, 26)
    hist: list[float] = []
    for i in range(26, len(closes) + 1):
        chunk = closes[:i]
        hist.append(ema(chunk, 12) - ema(chunk, 26))
    signal = ema(hist, 9) if hist else line
    return line, signal


def ma_slope_pct(klines: list[list]) -> float:
    if len(klines) < 14:
        return 0.0
    closes = [float(k[4]) for k in klines]
    recent = statistics.mean(closes[-6:-1])
    prior = statistics.mean(closes[-12:-7]) if len(closes) >= 12 else recent
    if prior == 0:
        return 0.0
    return (recent - prior) / prior * 100


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
    return {
        "price": c,
        "range_pct": rng,
        "macd_signal": macd_sig,
        "volume_ma_ratio": vol / vol_ma if vol_ma > 0 else 0.0,
        "ma_slope": ma_slope_pct(klines),
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
    if f["volume_ma_ratio"] >= 1.2 and f["range_pct"] < 2.0:
        return "Ignition"
    if f["ma_slope"] >= 0.3 and f["range_pct"] >= 1.4768:
        return "Birth"
    if abs(f["ma_slope"]) < 1.0:
        return "Accumulation"
    return "Birth"


def recommendation_score(f: dict) -> float:
    margin_macd = max(0.0, f["macd_signal"] - (-0.0016))
    margin_rng = max(0.0, f["range_pct"] - 1.4768)
    return (
        COHORT["return_per_hour"] * 10
        + margin_macd * 8000
        + margin_rng * 0.5
        + max(0.0, f["ma_slope"]) * 0.2
        + max(0.0, f["volume_ma_ratio"] - 1.0) * 0.3
    )


def confidence_score(f: dict) -> float:
    base = COHORT["base_confidence"]
    boost = min(15, recommendation_score(f) - COHORT["return_per_hour"] * 10)
    return round(min(100, base + boost), 1)


def blind_predictions(f: dict) -> dict:
    adj = min(8, recommendation_score(f) - COHORT["return_per_hour"] * 10)
    return {
        "prob_30m_3pct": round(COHORT["prob_30m_3pct"] + adj * 0.3, 1),
        "prob_1h_5pct": round(COHORT["prob_1h_5pct"] + adj * 0.2, 1),
        "prob_2h_7pct": round(COHORT["prob_2h_7pct"] + adj * 0.15, 1),
        "prob_4h_10pct": round(COHORT["prob_4h_10pct"] + adj * 0.1, 1),
        "expected_max_12h": COHORT["expected_max_12h"],
        "expected_return": COHORT["expected_return"],
        "return_per_hour": COHORT["return_per_hour"],
        "expected_mdd": COHORT["expected_mdd"],
        "expected_hold_min": COHORT["expected_hold_min"],
        "recommendation_score": round(recommendation_score(f), 3),
    }


def scan_symbol(symbol: str, scan_dt: datetime, end_ms: int) -> Candidate | None:
    try:
        klines = fetch_klines_15m(symbol, end_ms, LOOKBACK_15M)
        feats = compute_features(klines)
        if not feats or not passes_pattern(feats):
            return None
        preds = blind_predictions(feats)
        return Candidate(
            symbol=symbol,
            price=round(feats["price"], 8),
            phase_estimate=phase_estimate(feats),
            macd_signal=round(feats["macd_signal"], 6),
            range_pct=round(feats["range_pct"], 4),
            volume_ma_ratio=round(feats["volume_ma_ratio"], 4),
            ma_slope=round(feats["ma_slope"], 4),
            confidence=confidence_score(feats),
            score=preds["recommendation_score"],
            predictions=preds,
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
    ret = (close - entry) / entry * 100
    max_up = (max_h - entry) / entry * 100
    max_dn = (entry - min_l) / entry * 100
    return {
        "price": round(close, 8),
        "return_pct": round(ret, 4),
        "max_up_pct": round(max_up, 4),
        "max_down_pct": round(max_dn, 4),
        "mdd_pct": round(max_dn, 4),
    }


def infer_actual_phase(entry: float, forward: list[list]) -> tuple[str, list]:
    if not forward:
        return "Unknown", []
    transitions: list[str] = []
    phase = "Accumulation"
    peak = entry
    birth_done = False
    for i, k in enumerate(forward):
        c = float(k[4])
        h = float(k[1])
        ret = (c - entry) / entry * 100
        if h > peak:
            peak = h
        mins = (i + 1) * 15
        new_phase = phase
        if not birth_done and ret >= 3.0:
            new_phase = "Birth"
            birth_done = True
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


def evaluate_forward(c: Candidate, scan_dt: datetime, end_ms: int) -> None:
    start_ms = end_ms + INTERVAL_15M_MS
    try:
        fwd = fetch_forward_15m(c.symbol, start_ms, FORWARD_15M)
        entry = c.price
        horizons = {
            "15m": 1, "30m": 2, "1h": 4, "2h": 8, "4h": 16, "8h": 32, "12h": 48,
        }
        c.forward = {label: horizon_stats(entry, fwd, n) for label, n in horizons.items()}
        if fwd:
            max_h = max(ohlcv(k)[1] for k in fwd)
            min_l = min(ohlcv(k)[2] for k in fwd)
            close_12 = float(fwd[min(len(fwd), 48) - 1][4])
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
        c.forward = {}
        c.actual_phase = "Unknown"


def failure_reason(c: Candidate, success: bool) -> str:
    s = c.forward.get("summary", {})
    if success:
        parts = []
        if c.volume_ma_ratio >= 1.0:
            parts.append(f"volume_ma_ratio={c.volume_ma_ratio:.2f} supported move")
        if c.ma_slope >= 0.3:
            parts.append(f"ma_slope={c.ma_slope:.2f} persisted")
        if c.range_pct >= 1.8:
            parts.append(f"range_pct={c.range_pct:.2f} expansion confirmed")
        return "; ".join(parts) or "pattern features aligned with cohort"
    parts = []
    if c.volume_ma_ratio < 0.8:
        parts.append(f"volume_ma_ratio={c.volume_ma_ratio:.2f} below sustaining level")
    if c.ma_slope < 0:
        parts.append(f"ma_slope={c.ma_slope:.2f} negative slope after scan")
    if s.get("actual_max_up_12h", 0) < 5:
        parts.append(f"range_pct={c.range_pct:.2f} failed to convert to follow-through")
    if c.macd_signal < -0.001:
        parts.append(f"macd_signal={c.macd_signal:.6f} weak momentum")
    return "; ".join(parts) or "max_up_12h below cohort expectation"


def final_scores(candidates: list[Candidate], top5: list[Candidate]) -> dict:
    if not top5:
        return {k: 0 for k in ("hit_rate", "expected_return", "actual_return", "return_per_hour", "phase_accuracy", "confidence_calibration", "recommendation_quality", "total")}

    summaries = [c.forward.get("summary", {}) for c in top5 if c.forward.get("summary")]
    hits = sum(1 for s in summaries if s.get("actual_max_up_12h", 0) >= 5.0)
    hit_rate = hits / max(len(summaries), 1) * 100
    exp_ret = statistics.mean([c.predictions.get("expected_return", 0) for c in top5])
    act_ret = statistics.mean([s.get("actual_max_up_12h", 0) for s in summaries]) if summaries else 0
    exp_rph = statistics.mean([c.predictions.get("return_per_hour", 0) for c in top5])
    act_rph = statistics.mean([s.get("actual_max_up_12h", 0) / 12 for s in summaries]) if summaries else 0
    phase_ok = sum(1 for c in top5 if c.phase_estimate in ("Birth", "Ignition", "Expansion") and c.actual_phase in ("Birth", "Expansion", "Continuation", "End"))
    phase_acc = phase_ok / max(len(top5), 1) * 100
    conf_err = statistics.mean([abs(c.confidence - min(100, 50 + s.get("actual_max_up_12h", 0) * 5)) for c, s in zip(top5, summaries)]) if summaries else 50
    conf_cal = max(0, 100 - conf_err)
    rec_qual = hit_rate * 0.4 + min(100, act_rph / max(exp_rph, 0.01) * 50) * 0.6 if exp_rph else hit_rate

    scores = {
        "hit_rate": round(hit_rate, 1),
        "expected_return": round(min(100, exp_ret / 10 * 100), 1),
        "actual_return": round(min(100, act_ret / 10 * 100), 1),
        "return_per_hour": round(min(100, act_rph / max(exp_rph, 0.01) * 70), 1),
        "phase_accuracy": round(phase_acc, 1),
        "confidence_calibration": round(conf_cal, 1),
        "recommendation_quality": round(rec_qual, 1),
    }
    scores["total"] = round(statistics.mean(list(scores.values())), 1)
    return scores


def run(scan_kst: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scan_dt = parse_kst(scan_kst)
    end_ms = int(scan_dt.timestamp() * 1000)
    eligible = load_eligible_symbols(refresh=False, cache_only=False)
    symbols = sorted(eligible)
    print(f"Phase 8 blind scan: {scan_kst} KST | universe={len(symbols)}")

    candidates: list[Candidate] = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = {pool.submit(scan_symbol, sym, scan_dt, end_ms): sym for sym in symbols}
        done = 0
        for fut in as_completed(futs):
            done += 1
            c = fut.result()
            if c:
                candidates.append(c)
            if done % 80 == 0:
                print(f"  scanned {done}/{len(symbols)}, matches={len(candidates)}")

    candidates.sort(key=lambda x: x.score, reverse=True)
    for i, c in enumerate(candidates, 1):
        c.rank = i

    print(f"Matches: {len(candidates)}. Fetching forward data for AFTER REPORT...")
    for c in candidates:
        evaluate_forward(c, scan_dt, end_ms)
        time.sleep(API_SLEEP)

    top5 = candidates[:5]
    scores = final_scores(candidates, top5)

    # Decision
    hit_top5 = sum(1 for c in top5 if c.forward.get("summary", {}).get("actual_max_up_12h", 0) >= 5.0)
    if scores["total"] >= 55 and hit_top5 >= 2:
        decision = "KEEP"
    elif scores["total"] >= 35 or hit_top5 >= 1:
        decision = "MODIFY"
    else:
        decision = "DISCARD"

    lines = [
        "############################################################",
        "SCOUT PHASE 8 — BLIND TIMESTAMP SIMULATION",
        "############################################################",
        "",
        f"Search Time (KST): {scan_kst}",
        "Pattern B (FIXED): macd_signal >= -0.0016 AND range_pct >= 1.4768",
        f"Universe scanned: {len(symbols)} symbols",
        f"Matches: {len(candidates)}",
        "",
        "=" * 60,
        "SEARCH RESULTS (NO FUTURE DATA USED)",
        "=" * 60,
        "",
    ]

    for c in candidates:
        lines.append(
            f"#{c.rank} {c.symbol} price={c.price} phase={c.phase_estimate} "
            f"macd={c.macd_signal} range={c.range_pct}% vol_ratio={c.volume_ma_ratio} "
            f"ma_slope={c.ma_slope} conf={c.confidence}% score={c.score}"
        )

    lines.extend(["", "=" * 60, "BLIND PREDICTIONS (cohort priors + search-time margin)", "=" * 60, ""])
    for c in candidates[:15]:
        p = c.predictions
        lines.append(
            f"{c.symbol}: 30m+3%={p['prob_30m_3pct']}% 1h+5%={p['prob_1h_5pct']}% "
            f"2h+7%={p['prob_2h_7pct']}% 4h+10%={p['prob_4h_10pct']}% "
            f"E[max12h]={p['expected_max_12h']}% E[r/h]={p['return_per_hour']}% "
            f"E[MDD]={p['expected_mdd']}% hold={p['expected_hold_min']}min score={p['recommendation_score']}"
        )

    lines.extend(["", "=" * 60, "TOP 5 RECOMMENDATION (Return/Hour ranking)", "=" * 60, ""])
    lines.append(f"{'Rank':<5}{'Symbol':<14}{'ExpRet%':<10}{'Ret/h%':<10}{'Conf%':<8}{'HoldMin':<10}{'Rec'}")
    for c in top5:
        p = c.predictions
        lines.append(
            f"{c.rank:<5}{c.symbol:<14}{p['expected_return']:<10}{p['return_per_hour']:<10}"
            f"{c.confidence:<8}{p['expected_hold_min']:<10}BUY_CANDIDATE"
        )

    lines.extend(["", "=" * 60, "AFTER REPORT — REAL RESULTS", "=" * 60, ""])

    for c in top5:
        lines.append(f"\n--- {c.symbol} ---")
        lines.append(f"{'Horizon':<8}{'Price':<14}{'Return%':<10}{'MaxUp%':<10}{'MaxDn%':<10}{'MDD%'}")
        for h in ("15m", "30m", "1h", "2h", "4h", "8h", "12h"):
            d = c.forward.get(h, {})
            if d:
                lines.append(
                    f"{h:<8}{d.get('price',''):<14}{d.get('return_pct',''):<10}"
                    f"{d.get('max_up_pct',''):<10}{d.get('max_down_pct',''):<10}{d.get('mdd_pct','')}"
                )
        s = c.forward.get("summary", {})
        lines.append(f"12h summary: return={s.get('actual_return_12h')}% max_up={s.get('actual_max_up_12h')}% mdd={s.get('actual_mdd_12h')}%")

    lines.extend(["", "=" * 60, "LIFECYCLE ANALYSIS (post-hoc)", "=" * 60, ""])
    for c in top5:
        lines.append(f"{c.symbol}: expected={c.phase_estimate} actual={c.actual_phase} transitions={'; '.join(c.phase_transitions[:5])}")

    lines.extend(["", "=" * 60, "PREDICTION vs ACTUAL", "=" * 60, ""])
    lines.append(f"{'Symbol':<12}{'ExpRet':<10}{'ActMax':<10}{'ExpMDD':<10}{'ActMDD':<10}{'ExpPhase':<12}{'ActPhase'}")
    for c in top5:
        s = c.forward.get("summary", {})
        lines.append(
            f"{c.symbol:<12}{c.predictions.get('expected_return',''):<10}{s.get('actual_max_up_12h',''):<10}"
            f"{c.predictions.get('expected_mdd',''):<10}{s.get('actual_mdd_12h',''):<10}"
            f"{c.phase_estimate:<12}{c.actual_phase}"
        )

    lines.extend(["", "=" * 60, "FAILURE ANALYSIS", "=" * 60, ""])
    for c in top5:
        s = c.forward.get("summary", {})
        success = s.get("actual_max_up_12h", 0) >= 5.0
        lines.append(f"{c.symbol}: {'SUCCESS' if success else 'FAIL'} — {failure_reason(c, success)}")

    lines.extend(["", "=" * 60, "FINAL SCORE (100 max each dimension)", "=" * 60, ""])
    for k, v in scores.items():
        lines.append(f"  {k}: {v}")
    lines.extend([
        "",
        "=" * 60,
        "FINAL DECISION",
        "=" * 60,
        "",
        f"Decision: {decision}",
        f"Evidence: TOP5 hit_5pct={hit_top5}/5, total_score={scores['total']}, "
        f"universe_matches={len(candidates)}, holdout_prior_hit={COHORT['holdout_hit_5']}% (n={COHORT['holdout_n']})",
        "",
        f"Use in production? {decision} — numeric basis above.",
    ])

    report_path = OUT_DIR / f"blind_{scan_kst.replace(' ', '_').replace(':', '')}.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    csv_rows = []
    for c in candidates:
        row = {
            "rank": c.rank,
            "symbol": c.symbol,
            "price": c.price,
            "phase_estimate": c.phase_estimate,
            "macd_signal": c.macd_signal,
            "range_pct": c.range_pct,
            "volume_ma_ratio": c.volume_ma_ratio,
            "ma_slope": c.ma_slope,
            "confidence": c.confidence,
            "score": c.score,
            **{f"pred_{k}": v for k, v in c.predictions.items()},
            **{f"actual_{k}": v for k, v in c.forward.get("summary", {}).items()},
            "actual_phase": c.actual_phase,
        }
        csv_rows.append(row)
    write_csv(OUT_DIR / f"blind_{scan_kst.replace(' ', '_').replace(':', '')}.csv", csv_rows)

    print("\n".join(lines[-25:]).encode("ascii", "replace").decode("ascii"))
    print(f"\nSaved: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-kst", default="2026-06-03 11:00:00")
    args = parser.parse_args()
    run(args.scan_kst)


if __name__ == "__main__":
    main()
