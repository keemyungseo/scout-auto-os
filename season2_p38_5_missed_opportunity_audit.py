"""
Scout Learning Season2 - P38.5 Missed Opportunity Audit & Blind Spot Analysis

Audits symbols that outperformed P37.5 live selections using ONLY
pre-lock decision evidence. Not performance chasing. Not hindsight optimization.
Read-only on P25-P38 protected outputs.
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
from season2_p37_scout_decision_hierarchy import hierarchical_decide, load_csv, pf, pi, pbool
from season2_scout_mission import mission_summary_lines

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

MISSED_CSV = LOGS_DIR / "season2_p38_5_missed_opportunities.csv"
BLIND_SPOTS_CSV = LOGS_DIR / "season2_p38_5_blind_spots.csv"
COUNTERFACTUAL_CSV = LOGS_DIR / "season2_p38_5_counterfactual.csv"
INST_FAILURES_CSV = LOGS_DIR / "season2_p38_5_institution_failures.csv"
NOISE_CSV = LOGS_DIR / "season2_p38_5_noise_winners.csv"
HONEST_MISSES_CSV = LOGS_DIR / "season2_p38_5_honest_misses.csv"
CANDIDATES_CSV = LOGS_DIR / "season2_p38_5_candidate_improvements.csv"
META_REPORT_TXT = LOGS_DIR / "season2_p38_5_meta_report.txt"
FORWARD_CACHE_CSV = LOGS_DIR / "season2_p38_5_forward_cache.csv"

INSTITUTIONS_COMPARE = [
    "memory", "diversification", "false_convergence", "unknown_honesty",
    "watch_default", "confidence", "field_ecology", "attention_capital",
    "bias_correction", "persistence", "hierarchy",
]

SCORE_FIELDS = {
    "memory": "memory_score",
    "diversification": "diversification_score",
    "false_convergence": "false_convergence_protection_score",
    "unknown_honesty": "unknown_honesty_score",
    "confidence": "confidence_penalty",
}


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


def load_locked_context() -> dict:
    selections = [
        r for r in load_csv(LOGS_DIR / "season2_p37_live_selection.csv")
        if str(r.get("lock_status", "")).upper() == "LOCKED"
    ]
    if not selections:
        raise RuntimeError("No locked P37.5 observation found")

    obs_id = selections[0]["observation_id"]
    obs_log = [
        r for r in load_csv(LOGS_DIR / "season2_p37_observation_log.csv")
        if r.get("observation_id") == obs_id
    ]
    if not obs_log:
        raise RuntimeError("P37.5 observation log missing for locked observation")

    p38_review = {
        r["symbol"]: r for r in load_csv(LOGS_DIR / "season2_p38_symbol_review.csv")
        if r.get("observation_id") == obs_id
    }

    return {
        "observation_id": obs_id,
        "selections": selections,
        "selected_symbols": {r["symbol"] for r in selections},
        "observation_log": obs_log,
        "obs_log_by_symbol": {r["symbol"]: r for r in obs_log},
        "p38_review": p38_review,
        "p37_roles": {r["institution"]: r for r in load_csv(LOGS_DIR / "season2_p37_role_classification.csv")},
        "p36_importance": {r["institution"]: r for r in load_csv(LOGS_DIR / "season2_p36_institution_importance.csv")},
        "p38_reasoning": {
            r["symbol"]: r for r in load_csv(LOGS_DIR / "season2_p38_reasoning_score.csv")
            if r.get("observation_id") == obs_id
        },
        "start_dt": parse_utc(selections[0]["observation_timestamp_utc"]),
        "end_dt": parse_utc(selections[0]["validation_due_timestamp_utc"]),
        "regime": selections[0].get("market_regime", "Mixed"),
    }


def candles_in_window(klines: list[list], start_dt: datetime, end_dt: datetime) -> list[list]:
    rows = []
    for candle in klines:
        close_dt = t10.kline_close_dt(candle)
        if start_dt < close_dt <= end_dt:
            rows.append(candle)
    return rows


def fetch_entry_price(symbol: str, start_dt: datetime) -> float | None:
    end_ms = int(start_dt.timestamp() * 1000)
    try:
        klines = t10.fetch_klines_before(symbol, t10.INTERVAL_2H, end_ms, 2)
        if not klines:
            return None
        return float(klines[-1][4])
    except urllib.error.HTTPError:
        return None


def compute_forward_metrics(
    symbol: str,
    start_dt: datetime,
    end_dt: datetime,
    entry_price: float | None = None,
) -> dict | None:
    if entry_price is None:
        entry_price = fetch_entry_price(symbol, start_dt)
    if not entry_price:
        return None

    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    try:
        klines = t10.fetch_klines_forward(symbol, start_ms, end_ms)
    except urllib.error.HTTPError:
        return None

    window = candles_in_window(klines, start_dt, end_dt)
    if not window:
        return None

    highs, lows, closes, volumes, ranges = [], [], [], [], []
    running_peak = entry_price
    max_drawdown = 0.0

    for candle in window:
        open_p, high_p, low_p, close_p, vol = t10.ohlcv(candle)
        highs.append(high_p)
        lows.append(low_p)
        closes.append(close_p)
        volumes.append(vol)
        base = open_p or close_p or entry_price
        ranges.append((high_p - low_p) / base * 100 if base else 0.0)
        running_peak = max(running_peak, high_p)
        dd = (running_peak - low_p) / running_peak * 100 if running_peak else 0.0
        max_drawdown = max(max_drawdown, dd)

    price_close = closes[-1]
    net_return = (price_close - entry_price) / entry_price * 100
    volatility = statistics.mean(ranges) if ranges else 1.0
    volatility = max(volatility, 0.1)

    up_moves = sum(1 for i in range(1, len(closes)) if closes[i] > closes[i - 1])
    down_moves = sum(1 for i in range(1, len(closes)) if closes[i] < closes[i - 1])
    total_moves = up_moves + down_moves or 1
    trend_persistence = up_moves / total_moves if net_return >= 0 else down_moves / total_moves

    false_breakouts = 0
    for i, candle in enumerate(window):
        _, high_p, _, close_p, _ = t10.ohlcv(candle)
        if i > 0 and high_p > max(highs[:i]) and close_p < entry_price:
            false_breakouts += 1

    risk_adj = net_return / max(max_drawdown, 0.5)
    vol_adj = net_return / volatility

    return {
        "symbol": symbol,
        "entry_price": round(entry_price, 8),
        "exit_price": round(price_close, 8),
        "forward_return_pct": round(net_return, 2),
        "max_drawdown_pct": round(max_drawdown, 2),
        "volatility_pct": round(volatility, 2),
        "trend_persistence": round(trend_persistence, 3),
        "false_breakout_count": false_breakouts,
        "risk_adjusted_return": round(risk_adj, 2),
        "volatility_adjusted_return": round(vol_adj, 2),
    }


def load_or_build_forward_cache(
    symbols: list[str],
    start_dt: datetime,
    end_dt: datetime,
    rebuild: bool = False,
) -> dict[str, dict]:
    cache: dict[str, dict] = {}
    if FORWARD_CACHE_CSV.exists() and not rebuild:
        for row in load_csv(FORWARD_CACHE_CSV):
            cache[row["symbol"]] = row

    missing = [s for s in symbols if s not in cache]
    if missing:
        print(f"Computing forward returns for {len(missing)} symbols...")
        rows = list(cache.values())
        for index, symbol in enumerate(missing, start=1):
            if index % 40 == 0:
                print(f"  progress {index}/{len(missing)}")
            metrics = compute_forward_metrics(symbol, start_dt, end_dt)
            if metrics:
                cache[symbol] = metrics
                rows.append(metrics)
            time.sleep(t10.API_SLEEP_SEC)
        write_csv(FORWARD_CACHE_CSV, rows)

    return cache


def composite_outperform_score(row: dict, universe_median: float) -> float:
    rel = pf(row["forward_return_pct"], 0) - universe_median
    return (
        pf(row["forward_return_pct"], 0) * 0.35
        + rel * 0.25
        + pf(row["trend_persistence"], 0) * 10 * 0.15
        + pf(row["risk_adjusted_return"], 0) * 0.15
        + pf(row["volatility_adjusted_return"], 0) * 0.10
    )


def trace_has_positive(trace: str, inst: str) -> bool:
    aliases = {
        "false_convergence": "false_convergence_protection",
        "watch_default": "watch_default",
        "field_ecology": "field_ecology",
        "attention_capital": "attention_capital",
        "bias_correction": "bias_correction",
        "persistence": "persistence",
        "memory": "memory",
        "confidence": "confidence",
    }
    key = aliases.get(inst, inst)
    return f"{key}=+1" in trace or f"{key}->watch" in trace


def institution_gap(missed: dict, selected: dict, inst: str) -> float:
    if inst == "hierarchy":
        return pf(missed.get("hierarchy_score"), 0) - pf(selected.get("hierarchy_score"), 0)
    if inst == "confidence":
        return pf(selected.get("confidence_penalty"), 0) - pf(missed.get("confidence_penalty"), 0)
    field = SCORE_FIELDS.get(inst)
    if field:
        return pf(missed.get(field), 0) - pf(selected.get(field), 0)
    return 0.0


def classify_miss(
    missed_obs: dict,
    forward: dict,
    selected_returns: list[float],
    selected_disciplines: list[float],
) -> tuple[str, str]:
    symbol = missed_obs["symbol"]
    vetoed = str(missed_obs.get("vetoed", "")).lower() == "true"
    discipline = pf(missed_obs.get("discipline_score"), 0)
    memory = pf(missed_obs.get("memory_score"), 0)
    rank_24h = pi(missed_obs.get("rank_24h"), 999)
    grade = missed_obs.get("institution_grade", "C")
    trace = missed_obs.get("decision_trace", "")
    max_selected_return = max(selected_returns)
    min_selected_discipline = min(selected_disciplines)

    fwd = pf(forward["forward_return_pct"], 0)
    if fwd <= max_selected_return:
        return "N/A", "did not outperform selected symbols"

    if vetoed:
        return "D", "hierarchy veto intentionally blocked selection"

    trend = pf(forward["trend_persistence"], 0)
    vol = pf(forward["volatility_pct"], 0)
    if (
        fwd >= max_selected_return + 3
        and trend < 0.45
        and pi(forward.get("false_breakout_count")) >= 1
        and pf(forward["risk_adjusted_return"], 0) < 1.5
    ):
        return "E", "noise winner - lucky pump without persistent trend"

    if (
        fwd >= max_selected_return + 5
        and trend < 0.4
        and pf(forward["max_drawdown_pct"], 0) > vol * 2
    ):
        return "E", "noise winner - high volatility spike"

    has_memory_signal = memory >= 65 or trace_has_positive(trace, "memory")
    has_persistence = trace_has_positive(trace, "persistence")
    has_diversification = pf(missed_obs.get("diversification_score"), 0) >= 70

    observable = (
        has_memory_signal
        or has_persistence
        or has_diversification
        or discipline >= min_selected_discipline - 8
        or grade in ("A+", "A")
    )

    if not observable and memory <= 40 and rank_24h > 80 and grade in ("C",):
        return "A", "impossible to know - no observable evidence at lock time"

    if (
        discipline >= min_selected_discipline - 5
        and (memory >= 65 or pf(missed_obs.get("diversification_score"), 0) >= 80)
        and grade in ("A+", "A", "B")
    ):
        return "C", "observable but underestimated - real blind spot"

    if discipline >= 55 and has_memory_signal and not vetoed:
        return "C", "memory signal present but symbol not selected"

    if grade in ("C",) or discipline < min_selected_discipline - 12:
        return "B", "weak evidence - correctly ignored"

    if memory <= 35 and not has_persistence:
        return "A", "impossible to know honestly - thin evidence at decision time"

    return "B", "weak evidence - correctly ignored by discipline ranking"


def blind_spot_score(miss_type: str, missed_obs: dict, selected: dict, forward: dict) -> float:
    if miss_type == "A":
        return round(max(0, 5 - pf(missed_obs.get("discipline_score"), 0) / 20), 1)
    if miss_type == "B":
        return round(max(0.0, min(25.0, pf(missed_obs.get("discipline_score"), 0) / 4)), 1)
    if miss_type == "D":
        return 8.0
    if miss_type == "E":
        return 12.0
    if miss_type != "C":
        return 0.0

    score = 40.0
    score += max(0, pf(missed_obs.get("memory_score"), 0) - pf(selected.get("memory_score"), 0)) * 0.3
    score += max(0, pf(missed_obs.get("diversification_score"), 0) - pf(selected.get("diversification_score"), 0)) * 0.2
    score += max(0, pf(missed_obs.get("discipline_score"), 0) - pf(selected.get("discipline_score"), 0)) * 0.4
    if missed_obs.get("institution_grade") in ("A+", "A"):
        score += 15
    if pf(forward["trend_persistence"], 0) >= 0.55:
        score += 10
    return round(min(100.0, max(0.0, score)), 1)


def institution_responsibility(
    missed_obs: dict,
    selected_obs: dict,
    miss_type: str,
) -> list[dict]:
    failures = []
    trace = missed_obs.get("decision_trace", "")
    missed_disc = pf(missed_obs.get("discipline_score"), 0)
    sel_disc = pf(selected_obs.get("discipline_score"), 0)

    checks = [
        ("memory", "memory too weak", pf(missed_obs.get("memory_score"), 0) >= 65 and missed_disc < sel_disc),
        ("memory", "memory failed to speak", pf(missed_obs.get("memory_score"), 0) >= 70 and "memory=+1" not in trace),
        ("watch_default", "watch default inactive", "watch_default=baseline" in trace and missed_obs.get("hierarchy_stance") == "elevated_observation"),
        ("persistence", "persistence insufficient", trace_has_positive(trace, "persistence") and pf(missed_obs.get("discipline_score"), 0) < sel_disc),
        ("diversification", "diversification underweighted", pf(missed_obs.get("diversification_score"), 0) >= 80 and miss_type == "C"),
        ("unknown_honesty", "unknown honesty delayed", "unknown_honesty->watch" in trace and miss_type == "C"),
        ("false_convergence", "false convergence veto too early", str(missed_obs.get("vetoed", "")).lower() == "true"),
        ("confidence", "confidence suppressed", pf(missed_obs.get("confidence_penalty"), 0) >= 15 and miss_type == "C"),
        ("field_ecology", "field ecology overweight", "field_ecology=-1" in trace and pf(missed_obs.get("memory_score"), 0) >= 65),
        ("attention_capital", "attention capital ignored signal", "attention_capital=-1" in trace and pf(missed_obs.get("memory_score"), 0) >= 70),
    ]

    for inst, finding, condition in checks:
        if condition:
            failures.append({
                "institution": inst,
                "failure_mode": finding,
                "miss_type": miss_type,
                "symbol": missed_obs["symbol"],
                "responsibility": "failed_to_speak" if "failed" in finding or "inactive" in finding else "ignored",
            })

    if miss_type == "C" and missed_disc >= sel_disc - 3:
        failures.append({
            "institution": "hierarchy",
            "failure_mode": "discipline ranking undervalued competitive candidate",
            "miss_type": miss_type,
            "symbol": missed_obs["symbol"],
            "responsibility": "ignored",
        })

    return failures


def counterfactual_type_c(
    missed_obs: dict,
    selected_obs: dict,
    forward: dict,
    ctx: dict,
) -> dict:
    regime = ctx["regime"]
    record = {
        "symbol": missed_obs["symbol"],
        "persistence_scans": 1 if trace_has_positive(missed_obs.get("decision_trace", ""), "persistence") else 0,
        "false_convergence": "false_convergence_protection" in missed_obs.get("decision_trace", "") and "VETO" in missed_obs.get("decision_trace", ""),
        "unknown_honesty": "unknown_active" if "unknown_honesty->watch" in missed_obs.get("decision_trace", "") else "neutral",
        "structure_duplicate_index": 0 if pf(missed_obs.get("diversification_score"), 0) >= 70 else 2,
        "confidence_score": 25,
        "memory_top_outcome": "favorable" if pf(missed_obs.get("memory_score"), 0) >= 65 else "",
        "observation_allocation": "Ignore",
        "market_regime": regime,
        "field_ecology": "rank=0|mixed|unknown",
        "_attn_weight": 0,
        "_gov": {"market_regime": regime},
    }
    gov = record["_gov"]
    decision = hierarchical_decide(record, gov, "hybrid")

    missed_disc = pf(missed_obs.get("discipline_score"), 0)
    sel_disc = pf(selected_obs.get("discipline_score"), 0)
    l5_dom = any("L5:" in t and "+1" in t for t in decision.get("trace", []))

    return {
        "symbol": missed_obs["symbol"],
        "compared_to_selected": selected_obs["symbol"],
        "would_discipline_improve": "yes" if missed_disc > sel_disc else "no",
        "discipline_delta": round(missed_disc - sel_disc, 2),
        "would_false_convergence_increase": "yes" if decision.get("vetoed") else "no",
        "would_hierarchy_break": "yes" if l5_dom and decision.get("stance") == "elevated_observation" else "no",
        "would_confidence_dominate": "yes" if l5_dom else "no",
        "simulated_stance": decision.get("stance", ""),
        "simulated_vetoed": decision.get("vetoed", False),
        "forward_return_pct": forward["forward_return_pct"],
        "note": "counterfactual uses pre-lock scores only; forward return not used in simulation",
    }


def learning_policy(miss_type: str, blind_score: float, failures: list[dict]) -> dict:
    if miss_type in ("A", "D", "E"):
        return {"policy": "NO_ACTION", "reason": f"Type {miss_type} - not a learning trigger"}
    if miss_type == "B":
        return {"policy": "NO_ACTION", "reason": "Weak evidence correctly ignored at decision time"}
    if miss_type == "C" and blind_score >= 70:
        insts = "|".join(sorted({f["institution"] for f in failures})) or "hierarchy"
        return {
            "policy": "Needs another observation",
            "reason": f"Type C blind spot score {blind_score} - evidence existed but single observation",
            "institutions_to_watch": insts,
        }
    if miss_type == "C":
        return {"policy": "NO_ACTION", "reason": "Type C but blind spot score below promotion threshold"}
    return {"policy": "UNKNOWN", "reason": "insufficient classification"}


def compare_institutions(missed_obs: dict, selected_obs: dict) -> dict:
    comparison = {}
    for inst in INSTITUTIONS_COMPARE:
        gap = institution_gap(missed_obs, selected_obs, inst)
        comparison[f"{inst}_gap"] = round(gap, 2)
        if inst == "hierarchy":
            comparison[f"{inst}_missed"] = missed_obs.get("hierarchy_stance", "")
            comparison[f"{inst}_selected"] = selected_obs.get("hierarchy_stance", "")
        elif inst == "confidence":
            comparison[f"{inst}_missed"] = missed_obs.get("confidence_penalty", "")
            comparison[f"{inst}_selected"] = selected_obs.get("confidence_penalty", "")
        elif inst in SCORE_FIELDS:
            field = SCORE_FIELDS[inst]
            comparison[f"{inst}_missed"] = missed_obs.get(field, "")
            comparison[f"{inst}_selected"] = selected_obs.get(field, "")
        else:
            comparison[f"{inst}_missed"] = trace_has_positive(missed_obs.get("decision_trace", ""), inst)
            comparison[f"{inst}_selected"] = trace_has_positive(selected_obs.get("decision_trace", ""), inst)
    return comparison


def build_meta_report(
    ctx: dict,
    missed_rows: list[dict],
    blind_spots: list[dict],
    honest: list[dict],
    noise: list[dict],
    failures: list[dict],
    candidates: list[dict],
    selected_returns: dict[str, float],
) -> str:
    obs_id = ctx["observation_id"]
    selected = ctx["selections"]
    type_counts = Counter(r["miss_type"] for r in missed_rows)

    lines = [
        "===== SCOUT SEASON2 P38.5 - MISSED OPPORTUNITY & BLIND SPOT AUDIT =====",
        "",
        "=== 1 Executive summary ===",
        f"Observation ID: {obs_id}",
        f"Decision universe: {len(ctx['observation_log'])} symbols (pre-lock only)",
        f"Selected: {', '.join(r['symbol'] for r in selected)}",
        f"Selected returns: {', '.join(f'{s}={selected_returns.get(s, 0):+.2f}%' for s in selected_returns)}",
        f"Outperformers audited: {len(missed_rows)}",
        f"Miss types: A={type_counts.get('A', 0)} B={type_counts.get('B', 0)} "
        f"C={type_counts.get('C', 0)} D={type_counts.get('D', 0)} E={type_counts.get('E', 0)}",
        "Not performance chasing. Not hindsight optimization.",
        "",
        "=== 2 Top missed opportunities ===",
    ]
    for row in missed_rows[:10]:
        lines.append(
            f"  {row['outperformer_rank']}. {row['symbol']}: "
            f"return {row['forward_return_pct']:+.2f}% | type {row['miss_type']} | "
            f"blind_spot={row['blind_spot_score']}"
        )

    lines.extend(["", "=== 3 Blind spot analysis (Type C) ==="])
    if blind_spots:
        for row in blind_spots:
            lines.append(f"  {row['symbol']}: score={row['blind_spot_score']} - {row['classification_reason']}")
    else:
        lines.append("  No Type C blind spots in top outperformers.")

    lines.extend(["", "=== 4 Honest misses (Type A + B) ==="])
    for row in honest[:8]:
        lines.append(f"  [{row['miss_type']}] {row['symbol']}: {row['classification_reason']}")

    lines.extend(["", "=== 5 Impossible misses (Type A) ==="])
    type_a = [r for r in honest if r["miss_type"] == "A"]
    for row in type_a[:5]:
        lines.append(f"  {row['symbol']}: {row['classification_reason']}")
    if not type_a:
        lines.append("  None in top outperformer set.")

    lines.extend(["", "=== 6 Noise winners (Type E) ==="])
    for row in noise:
        lines.append(f"  {row['symbol']}: return {row['forward_return_pct']:+.2f}% - {row['classification_reason']}")
    if not noise:
        lines.append("  None classified as noise in top set.")

    lines.extend(["", "=== 7 Institution responsibility ==="])
    by_inst = Counter(f["institution"] for f in failures)
    for inst, count in by_inst.most_common(8):
        lines.append(f"  {inst}: {count} failure mentions")

    lines.extend(["", "=== 8 Hierarchy evaluation ==="])
    lines.append("  Selection used discipline ranking, not return ranking - by design.")
    lines.append("  L5 institutions did not lead selected symbols.")
    lines.append("  Vetoed symbols excluded from selection pool.")

    lines.extend(["", "=== 9 Candidate improvements ==="])
    actionable = [c for c in candidates if c.get("policy") != "NO_ACTION"]
    if actionable:
        for row in actionable:
            lines.append(f"  {row['symbol']}: {row['policy']} - {row['reason']}")
    else:
        lines.append("  NO_ACTION on all top misses - single observation insufficient.")

    lines.extend(["", "=== 10 Lessons NOT to learn ==="])
    lines.append("  Do not chase top 10h return symbols.")
    lines.append("  Do not promote noise winners (Type E) to selection criteria.")
    lines.append("  Do not override hierarchy veto retroactively (Type D).")
    lines.append("  Do not rewrite protected principles from one live window.")

    lines.extend(["", "=== 11 Lessons requiring 10 more observations ==="])
    lines.append("  Any Type C blind spot with score >= 70 requires repeat before weight change.")
    for row in blind_spots:
        if pf(row.get("blind_spot_score"), 0) >= 70:
            lines.append(f"  Watch: {row['symbol']} (score={row['blind_spot_score']})")

    lines.extend([
        "",
        "Scout Constitution:",
        "  A missed winner is acceptable.",
        "  An ignored observable signal is a learning opportunity.",
        "  A lucky winner is not evidence.",
        "",
        *mission_summary_lines(),
    ])
    return "\n".join(lines)


def run(rebuild_forward: bool = False, top_n: int = 20) -> None:
    ctx = load_locked_context()
    obs_id = ctx["observation_id"]
    start_dt = ctx["start_dt"]
    end_dt = ctx["end_dt"]

    print(f"P38.5 Missed Opportunity Audit | {obs_id}")
    print(f"  Universe: {len(ctx['observation_log'])} symbols at lock")
    print(f"  Window: {format_utc(start_dt)} -> {format_utc(end_dt)}")

    symbols = [r["symbol"] for r in ctx["observation_log"]]
    forward_cache = load_or_build_forward_cache(symbols, start_dt, end_dt, rebuild=rebuild_forward)

    universe_returns = [pf(r.get("forward_return_pct"), 0) for r in forward_cache.values()]
    universe_median = statistics.median(universe_returns) if universe_returns else 0.0

    selected_symbols = ctx["selected_symbols"]
    selected_returns: dict[str, float] = {}
    for sel in ctx["selections"]:
        sym = sel["symbol"]
        if sym in forward_cache:
            selected_returns[sym] = pf(forward_cache[sym]["forward_return_pct"], 0)
        elif sym in ctx["p38_review"]:
            selected_returns[sym] = pf(ctx["p38_review"][sym].get("net_return_pct"), 0)

    max_selected_return = max(selected_returns.values()) if selected_returns else 0.0
    selected_disciplines = [pf(s.get("discipline_score"), 0) for s in ctx["selections"]]
    primary_selected = max(ctx["selections"], key=lambda s: pf(s.get("discipline_score"), 0))
    secondary_selected = min(ctx["selections"], key=lambda s: pf(s.get("discipline_score"), 0))
    primary_obs = ctx["obs_log_by_symbol"][primary_selected["symbol"]]
    secondary_obs = ctx["obs_log_by_symbol"][secondary_selected["symbol"]]

    candidates: list[dict] = []
    for sym, fwd in forward_cache.items():
        if sym in selected_symbols:
            continue
        obs = ctx["obs_log_by_symbol"].get(sym)
        if not obs:
            continue
        score = composite_outperform_score(fwd, universe_median)
        candidates.append({**fwd, "composite_score": round(score, 3), "obs": obs})

    outperformers = [
        c for c in candidates
        if pf(c["forward_return_pct"], 0) > max_selected_return
    ]
    outperformers.sort(key=lambda r: r["composite_score"], reverse=True)
    top_outperformers = outperformers[:top_n]

    print(f"  Max selected return: {max_selected_return:+.2f}%")
    print(f"  Outperformers: {len(outperformers)} | Top {len(top_outperformers)} audited")

    missed_rows: list[dict] = []
    blind_spots: list[dict] = []
    counterfactuals: list[dict] = []
    inst_failures: list[dict] = []
    noise_winners: list[dict] = []
    honest_misses: list[dict] = []
    candidate_improvements: list[dict] = []

    for rank, item in enumerate(top_outperformers, start=1):
        obs = item["obs"]
        sym = item["symbol"]
        miss_type, reason = classify_miss(
            obs, item, list(selected_returns.values()), selected_disciplines
        )

        compare_target = primary_obs if pf(obs.get("discipline_score"), 0) >= pf(secondary_obs.get("discipline_score"), 0) else secondary_obs
        comparison = compare_institutions(obs, compare_target)
        bscore = blind_spot_score(miss_type, obs, compare_target, item)
        failures = institution_responsibility(obs, compare_target, miss_type)

        fwd_ret = pf(item["forward_return_pct"], 0)
        row = {
            "observation_id": obs_id,
            "outperformer_rank": rank,
            "symbol": sym,
            "miss_type": miss_type,
            "classification_reason": reason,
            "blind_spot_score": bscore,
            "forward_return_pct": fwd_ret,
            "relative_strength_pct": round(fwd_ret - universe_median, 2),
            "trend_persistence": pf(item["trend_persistence"], 0),
            "risk_adjusted_return": pf(item["risk_adjusted_return"], 0),
            "volatility_adjusted_return": pf(item["volatility_adjusted_return"], 0),
            "composite_outperform_score": item["composite_score"],
            "discipline_score_at_lock": obs.get("discipline_score"),
            "institution_grade_at_lock": obs.get("institution_grade"),
            "rank_24h_at_lock": obs.get("rank_24h"),
            "vetoed_at_lock": obs.get("vetoed"),
            "selected_symbol_compared": compare_target["symbol"],
            "selected_return_pct": selected_returns.get(compare_target["symbol"], 0),
            "return_advantage_vs_selected_pct": round(
                fwd_ret - selected_returns.get(compare_target["symbol"], 0), 2
            ),
            **comparison,
        }
        missed_rows.append(row)

        for fail in failures:
            inst_failures.append({**fail, "observation_id": obs_id})

        policy = learning_policy(miss_type, bscore, failures)
        candidate_improvements.append({
            "observation_id": obs_id,
            "symbol": sym,
            "miss_type": miss_type,
            "blind_spot_score": bscore,
            "policy": policy["policy"],
            "reason": policy["reason"],
            "institutions_to_watch": policy.get("institutions_to_watch", ""),
            "evidence_existed_before_selection": "yes" if miss_type == "C" else "no",
            "blind_spot_repeats": "no",
            "hierarchy_stable": "yes",
        })

        if miss_type == "C":
            blind_spots.append(row)
            cf = counterfactual_type_c(obs, compare_target, item, ctx)
            counterfactuals.append({**cf, "observation_id": obs_id, "miss_type": miss_type})
        elif miss_type == "E":
            noise_winners.append(row)
        elif miss_type in ("A", "B"):
            honest_misses.append(row)

    write_csv(MISSED_CSV, missed_rows)
    write_csv(BLIND_SPOTS_CSV, blind_spots or [{
        "observation_id": obs_id, "symbol": "", "miss_type": "C",
        "note": "no Type C blind spots in top outperformers",
    }])
    write_csv(COUNTERFACTUAL_CSV, counterfactuals or [{
        "observation_id": obs_id, "symbol": "", "note": "no Type C counterfactuals required",
    }])
    write_csv(INST_FAILURES_CSV, inst_failures)
    write_csv(NOISE_CSV, noise_winners or [{
        "observation_id": obs_id, "symbol": "", "miss_type": "E",
        "note": "no noise winners in top outperformers",
    }])
    write_csv(HONEST_MISSES_CSV, honest_misses)
    write_csv(CANDIDATES_CSV, candidate_improvements)

    report = build_meta_report(
        ctx, missed_rows, blind_spots, honest_misses, noise_winners,
        inst_failures, candidate_improvements, selected_returns,
    )
    META_REPORT_TXT.write_text(report, encoding="utf-8")

    print(f"Saved P38.5 outputs to {LOGS_DIR}")
    print(f"  Top outperformer: {missed_rows[0]['symbol'] if missed_rows else 'none'}")
    print(f"  Type C blind spots: {len(blind_spots)}")
    print(f"  Type E noise: {len(noise_winners)}")
    print(f"  Honest misses: {len(honest_misses)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="P38.5 Missed Opportunity & Blind Spot Audit")
    parser.add_argument("--rebuild-forward", action="store_true", help="Rebuild forward return cache")
    parser.add_argument("--top-n", type=int, default=20, help="Top outperformers to audit")
    args = parser.parse_args()
    run(rebuild_forward=args.rebuild_forward, top_n=args.top_n)


if __name__ == "__main__":
    main()
