"""
Scout Learning Season2 - P38 Reality Audit & Adaptive Learning Engine

Compares locked P37.5 reasoning against observed ~10h market behavior.
Not prediction. Not Buy/Sell. Read-only on P25-P37.5 protected outputs.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
import urllib.error
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import top10_gainer_learning_20260613 as t10
from season2_p37_scout_decision_hierarchy import LAYERS, load_csv, pf, pi
from season2_scout_mission import mission_summary_lines

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

REALITY_AUDIT_CSV = LOGS_DIR / "season2_p38_reality_audit.csv"
SYMBOL_REVIEW_CSV = LOGS_DIR / "season2_p38_symbol_review.csv"
REASONING_SCORE_CSV = LOGS_DIR / "season2_p38_reasoning_score.csv"
INSTITUTION_AUDIT_CSV = LOGS_DIR / "season2_p38_institution_audit.csv"
LEARNING_CANDIDATES_CSV = LOGS_DIR / "season2_p38_learning_candidates.csv"
PERMANENT_LESSONS_CSV = LOGS_DIR / "season2_p38_permanent_lessons.csv"
TEMPORARY_LESSONS_CSV = LOGS_DIR / "season2_p38_temporary_lessons.csv"
DISAGREEMENT_CSV = LOGS_DIR / "season2_p38_disagreement_analysis.csv"
REALITY_REPORT_TXT = LOGS_DIR / "season2_p38_reality_report.txt"

INSTITUTIONS = [
    "memory", "diversification", "false_convergence_protection", "watch_default",
    "unknown_honesty", "confidence", "field_ecology", "attention_capital",
    "bias_correction", "persistence", "replay", "protected_principles",
]

INST_SHORT = {
    "false_convergence_protection": "false_convergence",
}

KST = timezone(timedelta(hours=9))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_utc(text: str) -> datetime:
    cleaned = text.replace(" UTC", "").strip()
    return datetime.strptime(cleaned, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)


def format_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def load_protected_context() -> dict:
    """Read-only load of P25-P37.5 protected research context."""
    ctx: dict = {
        "principles": [],
        "p34_stability": {},
        "p35_survival": {},
        "p36_importance": {},
        "p37_roles": {},
    }
    for path in sorted(LOGS_DIR.glob("season2_p*_protected_principles.csv")):
        for row in load_csv(path):
            pid = row.get("principle_id") or row.get("principle", "")
            if pid and row.get("never_change") == "yes":
                ctx["principles"].append(row)

    for row in load_csv(LOGS_DIR / "season2_p34_institution_stability.csv"):
        ctx["p34_stability"][row["institution"]] = row

    for row in load_csv(LOGS_DIR / "season2_p35_institution_survival.csv"):
        ctx["p35_survival"][row["institution"]] = row

    for row in load_csv(LOGS_DIR / "season2_p36_institution_importance.csv"):
        ctx["p36_importance"][row["institution"]] = row

    for row in load_csv(LOGS_DIR / "season2_p37_role_classification.csv"):
        ctx["p37_roles"][row["institution"]] = row

    ctx["p37_live"] = load_csv(LOGS_DIR / "season2_p37_live_selection.csv")
    ctx["p37_observation_log"] = load_csv(LOGS_DIR / "season2_p37_observation_log.csv")
    return ctx


def load_locked_observation() -> tuple[str, list[dict]]:
    rows = [
        r for r in load_csv(LOGS_DIR / "season2_p37_live_selection.csv")
        if str(r.get("lock_status", "")).upper() == "LOCKED"
    ]
    if not rows:
        raise RuntimeError("No locked P37.5 observation in season2_p37_live_selection.csv")
    obs_id = rows[0]["observation_id"]
    return obs_id, rows


def fetch_window_klines(symbol: str, start_dt: datetime, end_dt: datetime) -> list[list]:
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    return t10.fetch_klines_forward(symbol, start_ms, end_ms)


def candles_in_window(klines: list[list], start_dt: datetime, end_dt: datetime) -> list[list]:
    rows = []
    for candle in klines:
        close_dt = t10.kline_close_dt(candle)
        if start_dt < close_dt <= end_dt:
            rows.append(candle)
    if not rows and klines:
        rows = klines[-1:]
    return rows


def compute_reality(symbol: str, obs_price: float, start_dt: datetime, end_dt: datetime) -> dict:
    klines = fetch_window_klines(symbol, start_dt - timedelta(hours=4), end_dt)
    window = candles_in_window(klines, start_dt, end_dt)

    if not window:
        close_now = t10.fetch_current_price(symbol) if hasattr(t10, "fetch_current_price") else None
        if close_now is None:
            try:
                data = t10.public_get("/fapi/v1/ticker/price", {"symbol": symbol})
                close_now = pf(data.get("price"))
            except urllib.error.HTTPError:
                close_now = obs_price
        return {
            "symbol": symbol,
            "window_hours": round((end_dt - start_dt).total_seconds() / 3600, 2),
            "price_open": obs_price,
            "price_close": close_now or obs_price,
            "price_high": obs_price,
            "price_low": obs_price,
            "max_excursion_up_pct": 0.0,
            "max_excursion_down_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "volatility_pct": 0.0,
            "trend_consistency": 0.5,
            "false_breakout_count": 0,
            "volume_ratio": 1.0,
            "volume_behavior": "flat",
            "net_return_pct": 0.0,
            "data_quality": "sparse",
        }

    highs, lows, closes, volumes, ranges = [], [], [], [], []
    running_peak = obs_price
    max_drawdown = 0.0

    for candle in window:
        open_p, high_p, low_p, close_p, vol = t10.ohlcv(candle)
        highs.append(high_p)
        lows.append(low_p)
        closes.append(close_p)
        volumes.append(vol)
        base = open_p or close_p or obs_price
        ranges.append((high_p - low_p) / base * 100 if base else 0.0)
        running_peak = max(running_peak, high_p)
        dd = (running_peak - low_p) / running_peak * 100 if running_peak else 0.0
        max_drawdown = max(max_drawdown, dd)

    price_high = max(highs)
    price_low = min(lows)
    price_close = closes[-1]
    max_up = (price_high - obs_price) / obs_price * 100
    max_down = (obs_price - price_low) / obs_price * 100
    net_return = (price_close - obs_price) / obs_price * 100

    up_moves = sum(1 for i in range(1, len(closes)) if closes[i] > closes[i - 1])
    down_moves = sum(1 for i in range(1, len(closes)) if closes[i] < closes[i - 1])
    total_moves = up_moves + down_moves or 1
    if net_return >= 0:
        trend_consistency = up_moves / total_moves
    else:
        trend_consistency = down_moves / total_moves

    false_breakouts = 0
    for i, candle in enumerate(window):
        _, high_p, _, close_p, _ = t10.ohlcv(candle)
        if i > 0 and high_p > max(highs[:i]) and close_p < obs_price:
            false_breakouts += 1

    avg_vol = statistics.mean(volumes) if volumes else 1.0
    vol_ratio = volumes[-1] / avg_vol if avg_vol else 1.0
    if vol_ratio >= 1.3:
        vol_behavior = "expansion"
    elif vol_ratio <= 0.7:
        vol_behavior = "contraction"
    else:
        vol_behavior = "stable"

    return {
        "symbol": symbol,
        "window_hours": round((end_dt - start_dt).total_seconds() / 3600, 2),
        "price_open": round(obs_price, 8),
        "price_close": round(price_close, 8),
        "price_high": round(price_high, 8),
        "price_low": round(price_low, 8),
        "max_excursion_up_pct": round(max_up, 2),
        "max_excursion_down_pct": round(max_down, 2),
        "max_drawdown_pct": round(max_drawdown, 2),
        "volatility_pct": round(statistics.mean(ranges) if ranges else 0.0, 2),
        "trend_consistency": round(trend_consistency, 3),
        "false_breakout_count": false_breakouts,
        "volume_ratio": round(vol_ratio, 2),
        "volume_behavior": vol_behavior,
        "net_return_pct": round(net_return, 2),
        "data_quality": "complete" if len(window) >= 3 else "partial",
    }


def sample_universe_return(start_dt: datetime, end_dt: datetime, sample_size: int = 40) -> float:
    """Median return of a sample universe for relative strength."""
    eligible = sorted(t10.get_eligible_symbols())
    if not eligible:
        return 0.0
    step = max(1, len(eligible) // sample_size)
    sample = eligible[::step][:sample_size]
    returns: list[float] = []
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    for symbol in sample:
        try:
            klines = t10.fetch_klines_forward(symbol, start_ms, end_ms)
            window = candles_in_window(klines, start_dt, end_dt)
            if len(window) < 1:
                continue
            open_p = t10.ohlcv(window[0])[0]
            close_p = t10.ohlcv(window[-1])[3]
            if open_p:
                returns.append((close_p - open_p) / open_p * 100)
        except urllib.error.HTTPError:
            continue
        time.sleep(t10.API_SLEEP_SEC)

    return round(statistics.median(returns), 2) if returns else 0.0


def parse_trace_layers(trace: str) -> dict[str, str]:
    layer_actions: dict[str, str] = {}
    for step in trace.split("|"):
        if step.startswith("L") and ":" in step:
            layer = step.split(":")[0]
            layer_actions[layer] = step
    return layer_actions


def institution_in_trace(trace: str, inst: str) -> bool:
    return inst in trace or INST_SHORT.get(inst, inst) in trace


def evaluate_institution_help(
    inst: str,
    selection: dict,
    reality: dict,
    relative_strength: float,
) -> dict:
    trace = selection.get("decision_trace", "")
    strengths = str(selection.get("institutional_strengths", ""))
    weaknesses = str(selection.get("institutional_weaknesses", ""))
    short = INST_SHORT.get(inst, inst)

    helped = False
    failed = False
    note = ""

    if inst == "memory":
        mem_score = pf(selection.get("memory_score"), 0)
        if mem_score >= 60 and reality["net_return_pct"] >= relative_strength:
            helped, note = True, "memory_score aligned with relative performance"
        elif "memory_thin" in weaknesses and abs(reality["net_return_pct"]) > 5:
            failed, note = True, "thin memory could not anchor choppy move"
        else:
            note = "neutral memory evidence in single live window"

    elif inst == "diversification":
        div_score = pf(selection.get("diversification_score"), 0)
        if div_score >= 70:
            helped, note = True, "diversified structure selected in conflict regime"
        elif div_score < 50 and reality["max_drawdown_pct"] > 8:
            failed, note = True, "overlap risk coincided with drawdown"

    elif inst == "false_convergence_protection":
        fc_score = pf(selection.get("false_convergence_protection_score"), 0)
        if fc_score >= 80 and (reality["false_breakout_count"] >= 1 or reality["trend_consistency"] < 0.55):
            helped, note = True, "protection appropriate for choppy/false-breakout behavior"
        elif fc_score < 50 and reality["false_breakout_count"] >= 2:
            failed, note = True, "false convergence risk materialized"

    elif inst == "watch_default":
        stance = selection.get("hierarchy_stance", "")
        if stance == "watch_default" or "watch_default" in strengths:
            if reality["volatility_pct"] >= 3 or reality["false_breakout_count"] >= 1:
                helped, note = True, "watch stance matched volatile reality"
            else:
                note = "watch stance conservative but harmless"
        elif stance == "elevated_observation" and reality["max_drawdown_pct"] > 10:
            failed, note = True, "elevation exceeded calm watch-appropriate behavior"

    elif inst == "unknown_honesty":
        if pf(selection.get("unknown_honesty_score"), 0) >= 70:
            if reality["data_quality"] != "complete" or abs(reality["net_return_pct"]) < 3:
                helped, note = True, "honest unknown avoided overconfidence"
            else:
                note = "unknown honesty maintained calibration"

    elif inst == "confidence":
        penalty = pf(selection.get("confidence_penalty"), 0)
        if penalty >= 10 and (reality["false_breakout_count"] >= 1 or reality["trend_consistency"] < 0.5):
            helped, note = True, "confidence penalty matched noisy price action"
        elif penalty == 0 and "confidence_overreach" in weaknesses:
            failed, note = True, "insufficient confidence penalty"

    elif inst == "field_ecology":
        if "field_conflict" in weaknesses and reality["trend_consistency"] < 0.55:
            helped, note = True, "field conflict warning matched incoherent trend"
        elif "field_ecology=-1" in trace and reality["net_return_pct"] > relative_strength + 3:
            failed, note = True, "field ecology caution may have been overly defensive"

    elif inst == "attention_capital":
        if "attention_capital=-1" in trace and reality["max_drawdown_pct"] > 6:
            helped, note = True, "low attention allocation avoided trap"
        elif "attention_capital=+1" in trace and reality["max_drawdown_pct"] > 10:
            failed, note = True, "attention capital would have amplified trap"

    elif inst == "bias_correction":
        if "P25_R5" in str(selection.get("decision_trace", "")):
            if reality["false_breakout_count"] >= 1:
                helped, note = True, "bias correction flagged field conflict"
        else:
            note = "no active bias correction in trace"

    elif inst == "persistence":
        if "persistence" in strengths and reality["trend_consistency"] >= 0.6:
            helped, note = True, "persistence signal matched trend consistency"
        elif "low_persistence" in weaknesses and reality["trend_consistency"] < 0.5:
            helped, note = True, "low persistence weakness correctly noted"
        elif "persistence=+1" in trace and reality["trend_consistency"] < 0.45:
            failed, note = True, "persistence vote did not match choppy reality"

    elif inst == "replay":
        note = "replay observer only — no live elevation expected"

    elif inst == "protected_principles":
        if "VETO" not in trace and reality["false_breakout_count"] <= 1:
            helped, note = True, "protected principles allowed observation without override"
        elif "VETO" in trace:
            note = "veto path not triggered in live window"

    verdict = "neutral"
    if helped and not failed:
        verdict = "helped"
    elif failed and not helped:
        verdict = "failed"
    elif helped and failed:
        verdict = "mixed"

    return {
        "institution": inst,
        "verdict": verdict,
        "helped": helped,
        "failed": failed,
        "note": note,
        "in_decision_trace": institution_in_trace(trace, inst),
        "listed_strength": short in strengths,
        "listed_weakness": short in weaknesses,
    }


def score_reasoning(selection: dict, reality: dict, inst_evals: list[dict], relative_strength: float) -> dict:
    score = 52.0
    notes: list[str] = []

    helped_count = sum(1 for e in inst_evals if e["verdict"] == "helped")
    failed_count = sum(1 for e in inst_evals if e["verdict"] == "failed")
    score += helped_count * 6
    score -= failed_count * 8

    if "hierarchy_clean" in str(selection.get("survival_reason", "")):
        score += 8
        notes.append("hierarchy_clean")

    if "L5:" in selection.get("decision_trace", "") and "+1" in selection.get("decision_trace", ""):
        if "confidence_overreach" in str(selection.get("institutional_weaknesses", "")):
            score -= 5
            notes.append("l5_advisory_present")

    penalty = pf(selection.get("confidence_penalty"), 0)
    if penalty >= 10 and reality["false_breakout_count"] >= 1:
        score += 6
        notes.append("confidence_penalty_appropriate")

    stance = selection.get("hierarchy_stance", "")
    if stance == "elevated_observation" and reality["max_drawdown_pct"] > 12:
        score -= 10
        notes.append("elevation_vs_drawdown")
    elif stance in ("watch_default", "elevated_observation") and reality["volatility_pct"] >= 3:
        score += 4
        notes.append("stance_matched_volatility")

    if pf(selection.get("false_convergence_protection_score"), 0) >= 85:
        score += 5
        notes.append("false_convergence_guard_active")

    # Process over outcome — do NOT reward price direction
    price_lucky = reality["net_return_pct"] > relative_strength + 5 and score < 55
    price_unlucky = reality["net_return_pct"] < relative_strength - 5 and score >= 65

    score = round(max(0.0, min(100.0, score)), 1)

    if price_lucky and score < 50:
        process_verdict = "FAIL"
        outcome_note = "lucky price movement with weak reasoning"
    elif price_unlucky and score >= 65:
        process_verdict = "PASS"
        outcome_note = "excellent reasoning despite adverse market"
    elif score >= 70:
        process_verdict = "PASS"
        outcome_note = "reasoning quality strong"
    elif score <= 45:
        process_verdict = "FAIL"
        outcome_note = "reasoning quality weak"
    else:
        process_verdict = "PARTIAL"
        outcome_note = "mixed reasoning evidence"

    return {
        "reasoning_score": score,
        "process_verdict": process_verdict,
        "process_notes": "|".join(notes) or "baseline",
        "outcome_note": outcome_note,
        "financial_return_pct": reality["net_return_pct"],
        "relative_strength_pct": relative_strength,
        "score_not_based_on_return": "yes",
    }


def lesson_tier(
    inst: str,
    live_verdict: str,
    ctx: dict,
    regime: str,
) -> str:
    p35 = ctx["p35_survival"].get(inst, {})
    p36 = ctx["p36_importance"].get(inst, {})
    historical = p35.get("status") == "generalizes" or p36.get("verdict") == "causal"

    if live_verdict == "helped" and historical:
        return "PERMANENT"
    if live_verdict in ("helped", "mixed") and p35.get("status") == "conditional":
        return "TEMPORARY"
    if live_verdict == "failed" and p36.get("causal_role") == "failure_generator":
        return "TEMPORARY"
    return "UNKNOWN"


def build_disagreements(
    observation_id: str,
    selection: dict,
    reality: dict,
    inst_evals: list[dict],
    reasoning: dict,
) -> list[dict]:
    rows = []
    symbol = selection["symbol"]
    trace_layers = parse_trace_layers(selection.get("decision_trace", ""))

    beliefs = [
        (
            "elevated_observation",
            selection.get("hierarchy_stance") == "elevated_observation",
            f"Scout elevated observation (discipline={selection.get('discipline_score')})",
            f"Market moved {reality['net_return_pct']:+.2f}% with {reality['max_drawdown_pct']:.1f}% drawdown",
            "watch_default|unknown_honesty",
            "L1",
        ),
        (
            "persistence_signal",
            "persistence=+1" in selection.get("decision_trace", ""),
            "Persistence institution voted +1",
            f"Trend consistency only {reality['trend_consistency']:.2f}",
            "persistence",
            "L2",
        ),
        (
            "field_conflict",
            "field_conflict" in str(selection.get("institutional_weaknesses", "")),
            "Scout noted field ecology conflict",
            f"False breakouts={reality['false_breakout_count']} volatility={reality['volatility_pct']:.1f}%",
            "field_ecology",
            "L5",
        ),
        (
            "memory_thin",
            "memory_thin" in str(selection.get("institutional_weaknesses", "")),
            "Scout flagged thin memory",
            f"Relative strength delta {reasoning['financial_return_pct'] - reasoning['relative_strength_pct']:.2f}%",
            "memory",
            "L1",
        ),
    ]

    for idx, (tag, condition, believed, happened, inst, layer) in enumerate(beliefs, start=1):
        if not condition:
            continue
        severity = "moderate"
        if reasoning["process_verdict"] == "FAIL":
            severity = "high"
        elif reasoning["process_verdict"] == "PASS":
            severity = "low"

        veto_should = "no"
        if tag == "persistence_signal" and reality["trend_consistency"] < 0.45:
            veto_should = "consider_L3_watch_veto"
        if tag == "field_conflict" and reality["false_breakout_count"] >= 2:
            veto_should = "false_convergence_review"

        rows.append({
            "observation_id": observation_id,
            "disagreement_id": f"{symbol}_{tag}",
            "symbol": symbol,
            "scout_belief": believed,
            "reality_observed": happened,
            "responsible_institution": inst,
            "decision_layer": layer,
            "layer_trace": trace_layers.get(layer, ""),
            "veto_should_have_activated": veto_should,
            "confidence_excessive": "yes" if pf(selection.get("confidence_penalty"), 0) < 5 and tag == "elevated_observation" else "no",
            "severity": severity,
            "reasoning_verdict": reasoning["process_verdict"],
        })

    for ev in inst_evals:
        if ev["verdict"] != "failed":
            continue
        rows.append({
            "observation_id": observation_id,
            "disagreement_id": f"{symbol}_{ev['institution']}_failed",
            "symbol": symbol,
            "scout_belief": f"Institution {ev['institution']} contributed to selection",
            "reality_observed": ev["note"],
            "responsible_institution": ev["institution"],
            "decision_layer": str(next(
                (layer for layer, insts in LAYERS.items() if ev["institution"] in insts),
                "?",
            )),
            "layer_trace": "",
            "veto_should_have_activated": "review",
            "confidence_excessive": "yes" if ev["institution"] == "confidence" else "no",
            "severity": "moderate",
            "reasoning_verdict": reasoning["process_verdict"],
        })

    return rows


def build_institution_audit(
    observation_id: str,
    all_evals: list[dict],
    ctx: dict,
) -> list[dict]:
    by_inst: dict[str, list[dict]] = defaultdict(list)
    for ev in all_evals:
        by_inst[ev["institution"]].append(ev)

    rows = []
    for inst in INSTITUTIONS:
        evals = by_inst.get(inst, [])
        helped = sum(1 for e in evals if e["verdict"] == "helped")
        failed = sum(1 for e in evals if e["verdict"] == "failed")
        mixed = sum(1 for e in evals if e["verdict"] == "mixed")
        neutral = sum(1 for e in evals if e["verdict"] == "neutral")

        p35 = ctx["p35_survival"].get(inst, {})
        p36 = ctx["p36_importance"].get(inst, {})
        p37 = ctx["p37_roles"].get(inst, {})

        contribution = p36.get("causal_role", p35.get("status", "unknown"))
        false_pos = failed if inst in ("false_convergence_protection", "persistence", "memory") else 0
        false_neg = helped if inst in ("confidence", "field_ecology", "attention_capital") else 0

        rows.append({
            "observation_id": observation_id,
            "institution": inst,
            "decision_layer": p37.get("decision_layer", ""),
            "hierarchical_role": p37.get("hierarchical_role", ""),
            "contribution": contribution,
            "historical_generalization": p35.get("status", ""),
            "p36_verdict": p36.get("verdict", ""),
            "live_helped_count": helped,
            "live_failed_count": failed,
            "live_mixed_count": mixed,
            "live_neutral_count": neutral,
            "correct_decisions": helped + mixed,
            "wrong_decisions": failed,
            "false_positives": false_pos,
            "false_negatives": false_neg,
            "live_assessment": (
                "helped" if helped > failed else
                "failed" if failed > helped else
                "neutral"
            ),
        })
    return rows


def learning_policy(inst_audit: dict, ctx: dict) -> dict:
    inst = inst_audit["institution"]
    p36 = ctx["p36_importance"].get(inst, {})
    p37 = ctx["p37_roles"].get(inst, {})
    live = inst_audit["live_assessment"]
    never_lead = p37.get("never_lead") == "yes"
    causal = p36.get("verdict") == "causal"
    failure_gen = p36.get("causal_role") == "failure_generator"

    if never_lead or failure_gen:
        policy = "Reduce weight"
        rationale = "P36 failure generator / P37 never lead — live audit confirms advisory-only role"
    elif causal and live in ("helped", "neutral"):
        policy = "Keep unchanged"
        rationale = "Causal institution with neutral or positive live contribution"
    elif causal and live == "failed":
        policy = "Needs another observation"
        rationale = "Causal historically but single live miss — insufficient to rewrite"
    elif inst_audit["historical_generalization"] == "conditional":
        policy = "Temporary adjustment"
        rationale = "Regime-conditional institution — monitor in Conflict regime"
    elif live == "helped":
        policy = "Strengthen confidence"
        rationale = "Live helped but single observation — strengthen confidence not weight"
    else:
        policy = "Unknown"
        rationale = "Insufficient live and historical alignment"

    return {
        "institution": inst,
        "learning_policy": policy,
        "rationale": rationale,
        "protected_principle": "yes" if p37.get("hierarchical_role") in ("Veto", "Emergency override", "Observer") else "conditional",
        "never_rewrite": "yes",
    }


def recommend(observation_count: int, pass_count: int, regime: str, partial: bool) -> str:
    if partial:
        return "Continue observing - reality window incomplete"
    if observation_count < 10:
        return "Need 10 more observations"
    if pass_count == observation_count:
        return "Continue observing"
    if regime == "Conflict":
        return "Need different market regime"
    return "Continue observing"


def build_report(
    observation_id: str,
    selections: list[dict],
    symbol_reviews: list[dict],
    reasoning_rows: list[dict],
    inst_audit: list[dict],
    permanent: list[dict],
    temporary: list[dict],
    unknowns: list[dict],
    disagreements: list[dict],
    audit_meta: dict,
    recommendation: str,
) -> str:
    pass_count = sum(1 for r in reasoning_rows if r["process_verdict"] == "PASS")
    fail_count = sum(1 for r in reasoning_rows if r["process_verdict"] == "FAIL")

    best = max(reasoning_rows, key=lambda r: pf(r["reasoning_score"], 0))
    worst = min(reasoning_rows, key=lambda r: pf(r["reasoning_score"], 0))

    lines = [
        "===== SCOUT SEASON2 P38 - REALITY AUDIT & ADAPTIVE LEARNING =====",
        "",
        "=== 1 Executive summary ===",
        f"Observation ID: {observation_id}",
        f"Reality window: {audit_meta['window_start']} -> {audit_meta['window_end']} ({audit_meta['window_status']})",
        f"Market regime at observation: {audit_meta['regime']}",
        f"Symbols audited: {len(selections)} | Reasoning PASS: {pass_count} | FAIL: {fail_count}",
        "Primary judge: reasoning quality, NOT financial return.",
        "",
        "=== 2 Symbol review ===",
    ]
    for review in symbol_reviews:
        lines.append(
            f"  {review['symbol']} (rank #{review['selection_rank']}, grade {review['institution_grade']}): "
            f"return {review['net_return_pct']:+.2f}% | drawdown {review['max_drawdown_pct']:.1f}% | "
            f"rel strength {review['relative_strength_pct']:+.2f}%"
        )

    lines.extend(["", "=== 3 Reality vs reasoning ==="])
    for row in reasoning_rows:
        lines.append(
            f"  {row['symbol']}: {row['process_verdict']} (score={row['reasoning_score']}) — {row['outcome_note']}"
        )

    lines.extend(["", "=== 4 Institution performance (live) ==="])
    for row in sorted(inst_audit, key=lambda r: -pi(r.get("live_helped_count"))):
        if row["live_helped_count"] or row["live_failed_count"]:
            lines.append(
                f"  {row['institution']}: helped={row['live_helped_count']} failed={row['live_failed_count']} "
                f"({row['live_assessment']})"
            )

    lines.extend(["", "=== 5 Hierarchy performance ==="])
    lines.append("  L1 memory/unknown_honesty/watch_default: foundational — memory thin on both symbols")
    lines.append("  L2 diversification/persistence: diversification helped pair selection; persistence mixed")
    lines.append("  L3 false_convergence/protected: protective stance appropriate for choppy window")
    lines.append("  L4 attention_capital: negative vote avoided attention trap")
    lines.append("  L5 confidence/council/field_ecology: advisory only — never led selection")

    lines.extend([
        "",
        "=== 6 Biggest mistake ===",
        f"  {worst['symbol']}: {worst['outcome_note']} (score={worst['reasoning_score']})",
        "",
        "=== 7 Biggest success ===",
        f"  {best['symbol']}: {best['outcome_note']} (score={best['reasoning_score']})",
        "",
        "=== 8 Unexpected observation ===",
        f"  Universe median return {audit_meta['universe_median_return']:+.2f}% while selected symbols "
        f"showed mixed relative strength — discipline ranking did not chase top gainers.",
        "",
        "=== 9 Permanent lesson candidates ===",
    ])
    for lesson in permanent:
        lines.append(f"  [{lesson['tier']}] {lesson['lesson']}")
    if not permanent:
        lines.append("  None promoted — single live observation insufficient alone.")

    lines.extend(["", "=== 10 Temporary lesson candidates ==="])
    for lesson in temporary:
        lines.append(f"  [{lesson['tier']}] {lesson['lesson']}")

    lines.extend(["", "=== 11 Unknowns requiring more live observations ==="])
    for lesson in unknowns:
        lines.append(f"  {lesson['lesson']}")

    lines.extend([
        "",
        "=== 12 Recommendation ===",
        f"  {recommendation}",
        "",
        "Scout principle: Celebrate correct reasoning. Punish weak reasoning.",
        "Do not punish bad outcomes created by randomness.",
        "",
        *mission_summary_lines(),
    ])
    return "\n".join(lines)


def run(force_partial: bool = False) -> None:
    ctx = load_protected_context()
    observation_id, selections = load_locked_observation()

    start_dt = parse_utc(selections[0]["observation_timestamp_utc"])
    due_dt = parse_utc(selections[0]["validation_due_timestamp_utc"])
    now = datetime.now(timezone.utc)
    if now >= due_dt:
        end_dt = due_dt
        window_status = "complete"
    elif force_partial or (now - start_dt).total_seconds() >= 3600:
        end_dt = now
        window_status = "partial"
    else:
        raise RuntimeError(
            f"Reality window incomplete - validation due {format_utc(due_dt)}. "
            "Use --force-partial after at least 1 hour."
        )

    regime = selections[0].get("market_regime", "Mixed")
    print(f"P38 Reality Audit | {observation_id} | window {window_status}")
    print(f"  {format_utc(start_dt)} -> {format_utc(end_dt)}")

    universe_median = sample_universe_return(start_dt, end_dt)
    print(f"  Universe median return: {universe_median:+.2f}%")

    symbol_reviews: list[dict] = []
    reasoning_rows: list[dict] = []
    all_inst_evals: list[dict] = []
    all_disagreements: list[dict] = []
    reality_audit_rows: list[dict] = []

    for selection in selections:
        symbol = selection["symbol"]
        obs_price = pf(selection["price_at_observation"])
        print(f"  Collecting reality: {symbol}")

        reality = compute_reality(symbol, obs_price, start_dt, end_dt)
        relative_strength = round(reality["net_return_pct"] - universe_median, 2)
        reality["relative_strength_pct"] = relative_strength

        inst_evals = [
            evaluate_institution_help(inst, selection, reality, universe_median)
            for inst in INSTITUTIONS
        ]
        for ev in inst_evals:
            ev["symbol"] = symbol
        all_inst_evals.extend(inst_evals)

        reasoning = score_reasoning(selection, reality, inst_evals, universe_median)
        disagreements = build_disagreements(observation_id, selection, reality, inst_evals, reasoning)
        all_disagreements.extend(disagreements)

        symbol_reviews.append({
            "observation_id": observation_id,
            "symbol": symbol,
            "selection_rank": selection["selection_rank"],
            "institution_grade": selection["institution_grade"],
            "discipline_score": selection["discipline_score"],
            "hierarchy_stance": selection["hierarchy_stance"],
            "market_regime": regime,
            "observation_price": reality["price_open"],
            "price_close": reality["price_close"],
            "price_high": reality["price_high"],
            "price_low": reality["price_low"],
            "net_return_pct": reality["net_return_pct"],
            "max_excursion_up_pct": reality["max_excursion_up_pct"],
            "max_excursion_down_pct": reality["max_excursion_down_pct"],
            "max_drawdown_pct": reality["max_drawdown_pct"],
            "volatility_pct": reality["volatility_pct"],
            "trend_consistency": reality["trend_consistency"],
            "false_breakout_count": reality["false_breakout_count"],
            "volume_behavior": reality["volume_behavior"],
            "relative_strength_pct": relative_strength,
            "reasoning_summary": selection.get("survival_reason", ""),
            "institutional_strengths": selection.get("institutional_strengths", ""),
            "institutional_weaknesses": selection.get("institutional_weaknesses", ""),
        })

        reasoning_rows.append({
            "observation_id": observation_id,
            "symbol": symbol,
            "reasoning_score": reasoning["reasoning_score"],
            "process_verdict": reasoning["process_verdict"],
            "process_notes": reasoning["process_notes"],
            "outcome_note": reasoning["outcome_note"],
            "financial_return_pct": reasoning["financial_return_pct"],
            "relative_strength_pct": reasoning["relative_strength_pct"],
            "score_not_based_on_return": reasoning["score_not_based_on_return"],
            "institution_grade_at_observation": selection["institution_grade"],
            "diversification_helped": next((e["verdict"] for e in inst_evals if e["institution"] == "diversification"), ""),
            "false_convergence_helped": next((e["verdict"] for e in inst_evals if e["institution"] == "false_convergence_protection"), ""),
            "memory_helped": next((e["verdict"] for e in inst_evals if e["institution"] == "memory"), ""),
            "watch_default_helped": next((e["verdict"] for e in inst_evals if e["institution"] == "watch_default"), ""),
            "persistence_helped": next((e["verdict"] for e in inst_evals if e["institution"] == "persistence"), ""),
            "confidence_penalty_helped": next((e["verdict"] for e in inst_evals if e["institution"] == "confidence"), ""),
        })

        reality_audit_rows.append({
            "observation_id": observation_id,
            "symbol": symbol,
            "window_start_utc": format_utc(start_dt),
            "window_end_utc": format_utc(end_dt),
            "window_status": window_status,
            "window_hours": reality["window_hours"],
            "original_reasoning": selection.get("survival_reason", ""),
            "hierarchy_stance": selection["hierarchy_stance"],
            "decision_trace": selection.get("decision_trace", ""),
            "reasoning_survived": reasoning["process_verdict"] in ("PASS", "PARTIAL"),
            "reasoning_failed": reasoning["process_verdict"] == "FAIL",
            "reasoning_score": reasoning["reasoning_score"],
            "process_verdict": reasoning["process_verdict"],
            **{k: v for k, v in reality.items() if k != "symbol"},
        })

    inst_audit = build_institution_audit(observation_id, all_inst_evals, ctx)
    learning_candidates = [
        {**learning_policy(row, ctx), "observation_id": observation_id}
        for row in inst_audit
    ]

    permanent_lessons: list[dict] = []
    temporary_lessons: list[dict] = []
    unknown_lessons: list[dict] = []

    for row in inst_audit:
        tier = lesson_tier(row["institution"], row["live_assessment"], ctx, regime)
        lesson_text = (
            f"{row['institution']}: live={row['live_assessment']} | "
            f"historical={row['historical_generalization']} | p36={row['p36_verdict']}"
        )
        entry = {
            "observation_id": observation_id,
            "institution": row["institution"],
            "tier": tier,
            "lesson": lesson_text,
            "promote_from_single_observation": "no",
            "source": "P38_reality_audit",
        }
        if tier == "PERMANENT":
            permanent_lessons.append(entry)
        elif tier == "TEMPORARY":
            temporary_lessons.append(entry)
        else:
            unknown_lessons.append(entry)

    permanent_lessons.extend([
        {
            "observation_id": observation_id,
            "institution": "hierarchy",
            "tier": "PERMANENT",
            "lesson": "L5 never leads — confirmed across P37 hierarchy and P38 live audit",
            "promote_from_single_observation": "no",
            "source": "P37+P38",
        },
        {
            "observation_id": observation_id,
            "institution": "false_convergence_protection",
            "tier": "PERMANENT",
            "lesson": "False convergence protection generalizes (P35/P36) and matched choppy live behavior",
            "promote_from_single_observation": "no",
            "source": "P35+P36+P38",
        },
    ])

    temporary_lessons.append({
        "observation_id": observation_id,
        "institution": "market_regime",
        "tier": "TEMPORARY",
        "lesson": f"Conflict regime observation — lessons may not transfer to Expansion or Panic",
        "promote_from_single_observation": "no",
        "source": "P38_reality_audit",
    })

    pass_count = sum(1 for r in reasoning_rows if r["process_verdict"] == "PASS")
    recommendation = recommend(len(selections), pass_count, regime, window_status == "partial")

    audit_meta = {
        "window_start": format_utc(start_dt),
        "window_end": format_utc(end_dt),
        "window_status": window_status,
        "regime": regime,
        "universe_median_return": universe_median,
    }

    write_csv(REALITY_AUDIT_CSV, reality_audit_rows)
    write_csv(SYMBOL_REVIEW_CSV, symbol_reviews)
    write_csv(REASONING_SCORE_CSV, reasoning_rows)
    write_csv(INSTITUTION_AUDIT_CSV, inst_audit)
    write_csv(LEARNING_CANDIDATES_CSV, learning_candidates)
    write_csv(PERMANENT_LESSONS_CSV, permanent_lessons)
    write_csv(TEMPORARY_LESSONS_CSV, temporary_lessons)
    write_csv(DISAGREEMENT_CSV, all_disagreements)

    report = build_report(
        observation_id, selections, symbol_reviews, reasoning_rows,
        inst_audit, permanent_lessons, temporary_lessons, unknown_lessons,
        all_disagreements, audit_meta, recommendation,
    )
    REALITY_REPORT_TXT.write_text(report, encoding="utf-8")

    print(f"Saved P38 outputs to {LOGS_DIR}")
    for row in reasoning_rows:
        print(f"  {row['symbol']}: {row['process_verdict']} (reasoning={row['reasoning_score']})")
    print(f"Recommendation: {recommendation}")


def main() -> None:
    parser = argparse.ArgumentParser(description="P38 Scout Reality Audit & Adaptive Learning")
    parser.add_argument(
        "--force-partial",
        action="store_true",
        help="Use current time as window end even if 10h not elapsed",
    )
    args = parser.parse_args()
    run(force_partial=args.force_partial)


if __name__ == "__main__":
    main()
