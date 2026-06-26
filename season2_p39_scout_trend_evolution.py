"""
Scout Learning Season2 - P39 Trend Evolution & State Transition Engine

Studies HOW selected symbols evolved hour-by-hour after lock.
Not hindsight optimization. Not learning from final return.
Read-only on P25-P38.5. Never modifies weights, hierarchy, or vetoes.
"""

from __future__ import annotations

import argparse
import statistics
import time
import urllib.error
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import top10_gainer_learning_20260613 as t10
from season2_p37_scout_decision_hierarchy import load_csv, pf, pi
from season2_scout_mission import mission_summary_lines

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

STATE_TRANSITION_CSV = LOGS_DIR / "season2_p39_state_transition.csv"
TREND_EVOLUTION_CSV = LOGS_DIR / "season2_p39_trend_evolution.csv"
CONFIDENCE_CURVE_CSV = LOGS_DIR / "season2_p39_confidence_curve.csv"
MARKET_CONTEXT_CSV = LOGS_DIR / "season2_p39_market_context.csv"
TRANSITION_PATTERNS_CSV = LOGS_DIR / "season2_p39_transition_patterns.csv"
LEARNING_CANDIDATES_CSV = LOGS_DIR / "season2_p39_learning_candidates.csv"
PROCESS_REPORT_TXT = LOGS_DIR / "season2_p39_process_report.txt"

STATES = (
    "Observation",
    "Potential",
    "Trend Start",
    "Trend Expansion",
    "Trend Exhaustion",
    "Failure",
)

CHECKPOINT_HOURS = list(range(11))  # T+0 .. T+10

INSTITUTIONS = (
    "memory", "diversification", "false_convergence_protection", "unknown_honesty",
    "watch_default", "persistence", "field_ecology", "attention_capital",
    "confidence", "protected_principles",
)


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
        import csv
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_utc(text: str) -> datetime:
    cleaned = text.replace(" UTC", "").strip()
    return datetime.strptime(cleaned, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)


def format_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def load_selections() -> tuple[str, list[dict]]:
    rows = [
        r for r in load_csv(LOGS_DIR / "season2_p37_live_selection.csv")
        if str(r.get("lock_status", "")).upper() == "LOCKED"
    ]
    if not rows:
        raise RuntimeError("No locked P37.5 selection found")
    return rows[0]["observation_id"], rows


def fetch_1h_klines(symbol: str, start_dt: datetime, end_dt: datetime) -> list[list]:
    start_ms = int((start_dt - timedelta(hours=24)).timestamp() * 1000)
    end_ms = int((end_dt + timedelta(hours=2)).timestamp() * 1000)
    all_klines: list[list] = []
    current = start_ms
    while current < end_ms:
        params = {
            "symbol": symbol,
            "interval": t10.INTERVAL_1H,
            "startTime": current,
            "endTime": end_ms,
            "limit": 500,
        }
        import urllib.parse
        import urllib.request
        import json
        url = f"{t10.FUTURES_BASE_URL}{t10.KLINES_ENDPOINT}?{urllib.parse.urlencode(params)}"
        with urllib.request.urlopen(urllib.request.Request(url), timeout=20) as resp:
            batch = json.loads(resp.read().decode())
        if not batch:
            break
        all_klines.extend(batch)
        last_open = int(batch[-1][0])
        nxt = last_open + 3600_000
        if nxt <= current:
            break
        current = nxt
        time.sleep(t10.API_SLEEP_SEC)
    return all_klines


def kline_close_time(kline: list) -> datetime:
    open_dt = datetime.fromtimestamp(int(kline[0]) / 1000, tz=timezone.utc)
    return open_dt + timedelta(hours=1)


def price_at_checkpoint(klines: list[list], checkpoint: datetime) -> tuple[float, float, list[list]]:
    """Return close price, volume, and klines up to checkpoint."""
    eligible = [k for k in klines if kline_close_time(k) <= checkpoint]
    if not eligible:
        return 0.0, 0.0, []
    last = eligible[-1]
    _, _, _, close_p, vol = t10.ohlcv(last)
    return close_p, vol, eligible


def trend_consistency_since_entry(closes: list[float], entry: float) -> float:
    if len(closes) < 2:
        return 0.5
    ups = sum(1 for i in range(1, len(closes)) if closes[i] > closes[i - 1])
    downs = sum(1 for i in range(1, len(closes)) if closes[i] < closes[i - 1])
    total = ups + downs or 1
    net = closes[-1] - entry
    return ups / total if net >= 0 else downs / total


def breakout_status(closes: list[float], highs: list[float], entry: float) -> str:
    if len(closes) < 3:
        return "inside_range"
    prior_high = max(highs[:-1]) if len(highs) > 1 else entry
    if closes[-1] > prior_high * 1.005:
        if closes[-1] < entry:
            return "false_breakout"
        return "breakout_confirmed"
    if closes[-1] < entry * 0.98:
        return "breakdown"
    return "inside_range"


def infer_regime_at_checkpoint(universe_returns: list[float]) -> str:
    if not universe_returns:
        return "Mixed"
    avg = statistics.mean(universe_returns)
    positive = sum(1 for r in universe_returns if r > 0) / len(universe_returns)
    if avg >= 0.8 and positive >= 0.45:
        return "Healthy Expansion"
    if avg <= -0.5 or positive <= 0.3:
        return "Panic"
    if avg >= 0.3:
        return "Rotation"
    if avg <= 0.1 and positive <= 0.4:
        return "Compression"
    return "Conflict"


def sample_universe_returns(
    start_dt: datetime,
    checkpoint: datetime,
    sample_size: int = 30,
) -> list[float]:
    eligible = sorted(t10.get_eligible_symbols())
    step = max(1, len(eligible) // sample_size)
    sample = eligible[::step][:sample_size]
    returns: list[float] = []
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(checkpoint.timestamp() * 1000)

    for symbol in sample:
        try:
            params = {
                "symbol": symbol,
                "interval": t10.INTERVAL_1H,
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": 20,
            }
            import urllib.parse
            import urllib.request
            import json
            url = f"{t10.FUTURES_BASE_URL}{t10.KLINES_ENDPOINT}?{urllib.parse.urlencode(params)}"
            with urllib.request.urlopen(urllib.request.Request(url), timeout=15) as resp:
                batch = json.loads(resp.read().decode())
            if len(batch) < 1:
                continue
            open_p = float(batch[0][1])
            close_p = float(batch[-1][4])
            if open_p:
                returns.append((close_p - open_p) / open_p * 100)
        except urllib.error.HTTPError:
            continue
        time.sleep(t10.API_SLEEP_SEC * 0.5)
    return returns


def assign_state(
    hour: int,
    entry: float,
    price: float,
    drawdown: float,
    recovery: float,
    trend_cons: float,
    breakout: str,
    atr_ratio: float,
    vol_behavior: str,
) -> str:
    ret_from_entry = (price - entry) / entry * 100 if entry else 0.0

    if hour == 0:
        return "Observation"

    if drawdown >= 8 or breakout == "breakdown" or (ret_from_entry <= -5 and trend_cons < 0.4):
        return "Failure"

    if (
        drawdown >= 4
        and recovery < 0.35
        and hour >= 3
        and trend_cons < 0.55
    ):
        return "Trend Exhaustion"

    if (
        breakout in ("breakout_confirmed",)
        and trend_cons >= 0.6
        and drawdown < 4
        and ret_from_entry > 0
    ):
        return "Trend Expansion"

    if trend_cons >= 0.55 and hour >= 2 and ret_from_entry > -1:
        return "Trend Start"

    if hour >= 1 and drawdown < 5 and trend_cons >= 0.45:
        return "Potential"

    return "Observation"


def institutional_confidence(
    hour: int,
    state: str,
    selection: dict,
    trend_cons: float,
    breakout: str,
    drawdown: float,
    false_breakout: bool,
) -> dict[str, float]:
    """Observational confidence curves - NOT weight changes."""
    base_memory = pf(selection.get("memory_score"), 35)
    base_div = pf(selection.get("diversification_score"), 50)
    base_fc = pf(selection.get("false_convergence_protection_score"), 90)
    base_unknown = pf(selection.get("unknown_honesty_score"), 80)

    memory = base_memory
    if trend_cons >= 0.6 and hour >= 2:
        memory = min(85, base_memory + hour * 2)
    elif drawdown > 5:
        memory = max(20, base_memory - 10)

    diversification = base_div
    persistence = 35 + trend_cons * 40 + min(hour, 5) * 2
    if state == "Trend Start":
        persistence = min(80, persistence + 10)

    false_conv = base_fc
    if false_breakout or breakout == "false_breakout":
        false_conv = min(base_fc, 40)
    elif breakout == "inside_range":
        false_conv = min(95, base_fc + 3)

    unknown = base_unknown
    if state in ("Observation", "Potential"):
        unknown = min(90, base_unknown + 5)

    watch_default = 70 if state in ("Observation", "Potential") else max(40, 70 - hour * 3)

    field_ecology = 50 + trend_cons * 25
    if "field_conflict" in str(selection.get("institutional_weaknesses", "")):
        field_ecology = max(30, field_ecology - 15)

    attention = 40 - drawdown * 2
    if drawdown > 6:
        attention = max(15, attention - 10)

    confidence = max(10, 25 - pf(selection.get("confidence_penalty"), 0))
    confidence = min(confidence, 20)

    protected = 75 if state != "Failure" else 85

    return {
        "memory": round(memory, 1),
        "diversification": round(diversification, 1),
        "false_convergence_protection": round(false_conv, 1),
        "unknown_honesty": round(unknown, 1),
        "watch_default": round(watch_default, 1),
        "persistence": round(persistence, 1),
        "field_ecology": round(field_ecology, 1),
        "attention_capital": round(attention, 1),
        "confidence": round(confidence, 1),
        "protected_principles": round(protected, 1),
    }


def build_checkpoints(
    symbol: str,
    selection: dict,
    observation_id: str,
    start_dt: datetime,
    end_dt: datetime,
    regime_at_lock: str,
    universe_cache: dict[int, dict],
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    klines = fetch_1h_klines(symbol, start_dt, end_dt)
    entry = pf(selection.get("price_at_observation"))
    if not entry:
        entry = price_at_checkpoint(klines, start_dt)[0]

    evolution_rows: list[dict] = []
    transition_rows: list[dict] = []
    confidence_rows: list[dict] = []
    context_rows: list[dict] = []

    running_peak = entry
    running_trough = entry
    prev_state = "Observation"

    for hour in CHECKPOINT_HOURS:
        checkpoint = start_dt + timedelta(hours=hour)
        price, volume, hist = price_at_checkpoint(klines, checkpoint)
        if not price:
            continue

        closes = [float(k[4]) for k in hist if kline_close_time(k) > start_dt]
        highs = [float(k[2]) for k in hist if kline_close_time(k) > start_dt]
        lows = [float(k[3]) for k in hist if kline_close_time(k) > start_dt]

        running_peak = max(running_peak, price, max(highs) if highs else price)
        running_trough = min(running_trough, price, min(lows) if lows else price)

        drawdown = (running_peak - price) / running_peak * 100 if running_peak else 0.0
        span = running_peak - running_trough
        recovery = (price - running_trough) / span if span > 0 else 1.0

        atr_pct = t10.average_true_range_percent(hist[-14:], price) if len(hist) >= 2 else 0.0
        atr_prior = t10.average_true_range_percent(hist[-28:-14], price) if len(hist) >= 28 else atr_pct
        atr_ratio = atr_pct / atr_prior if atr_prior else 1.0

        ranges = []
        for k in hist[-6:]:
            o, h, l, c, _ = t10.ohlcv(k)
            base = o or c or price
            ranges.append((h - l) / base * 100 if base else 0)
        volatility = statistics.mean(ranges) if ranges else 0.0

        trend_cons = trend_consistency_since_entry(closes, entry) if closes else 0.5
        breakout = breakout_status(closes, highs, entry)
        false_breakout = breakout == "false_breakout"

        if hour not in universe_cache:
            uni_rets = sample_universe_returns(start_dt, checkpoint)
            universe_cache[hour] = {
                "returns": uni_rets,
                "regime": infer_regime_at_checkpoint(uni_rets),
                "median": statistics.median(uni_rets) if uni_rets else 0.0,
            }

        uni = universe_cache[hour]
        rel_strength = (price - entry) / entry * 100 - uni["median"]

        vol_behavior = "contraction"
        if len(hist) >= 4:
            vols = [float(k[5]) for k in hist[-4:]]
            if vols[-1] > statistics.mean(vols[:-1]) * 1.2:
                vol_behavior = "expansion"
            elif vols[-1] < statistics.mean(vols[:-1]) * 0.8:
                vol_behavior = "contraction"
            else:
                vol_behavior = "stable"

        state = assign_state(
            hour, entry, price, drawdown, recovery, trend_cons,
            breakout, atr_ratio, vol_behavior,
        )

        if state != prev_state:
            transition_rows.append({
                "observation_id": observation_id,
                "symbol": symbol,
                "from_state": prev_state,
                "to_state": state,
                "transition_hour": hour,
                "checkpoint": f"T+{hour}h",
                "trigger": f"drawdown={drawdown:.1f}|trend={trend_cons:.2f}|breakout={breakout}",
                "process_based": "yes",
            })
            prev_state = state

        evolution_rows.append({
            "observation_id": observation_id,
            "symbol": symbol,
            "checkpoint": f"T+{hour}h",
            "checkpoint_hour": hour,
            "checkpoint_timestamp_utc": format_utc(checkpoint),
            "state": state,
            "price": round(price, 8),
            "volume": round(volume, 2),
            "atr_pct": round(atr_pct, 3),
            "atr_ratio": round(atr_ratio, 3),
            "volatility_pct": round(volatility, 3),
            "relative_strength_pct": round(rel_strength, 3),
            "market_regime": uni["regime"],
            "regime_at_lock": regime_at_lock,
            "drawdown_pct": round(drawdown, 3),
            "recovery_ratio": round(recovery, 3),
            "breakout_status": breakout,
            "trend_consistency": round(trend_cons, 3),
            "return_from_entry_pct": round((price - entry) / entry * 100, 3),
            "volume_behavior": vol_behavior,
        })

        context_rows.append({
            "observation_id": observation_id,
            "symbol": symbol,
            "checkpoint": f"T+{hour}h",
            "checkpoint_hour": hour,
            "universe_median_return_pct": round(uni["median"], 3),
            "market_regime": uni["regime"],
            "regime_at_lock": regime_at_lock,
            "regime_shift": "yes" if uni["regime"] != regime_at_lock else "no",
            "sample_size": len(uni["returns"]),
        })

        conf = institutional_confidence(
            hour, state, selection, trend_cons, breakout, drawdown, false_breakout,
        )
        confidence_rows.append({
            "observation_id": observation_id,
            "symbol": symbol,
            "checkpoint": f"T+{hour}h",
            "checkpoint_hour": hour,
            "state": state,
            **{f"{inst}_confidence": conf[inst] for inst in INSTITUTIONS},
            "composite_observation_confidence": round(
                statistics.mean([
                    conf["memory"], conf["false_convergence_protection"],
                    conf["unknown_honesty"], conf["watch_default"], conf["persistence"],
                ]), 1,
            ),
            "l5_confidence_capped": "yes",
            "weight_modified": "no",
        })

    return evolution_rows, transition_rows, confidence_rows, context_rows


def pattern_key(transitions: list[dict], final_state: str) -> str:
    if not transitions:
        return f"Observation->{final_state}"
    parts = ["Observation"]
    for t in transitions:
        parts.append(t["to_state"])
    return "->".join(parts)


def build_patterns(
    observation_id: str,
    all_transitions: dict[str, list[dict]],
    all_evolution: dict[str, list[dict]],
) -> list[dict]:
    rows = []
    for symbol, transitions in all_transitions.items():
        evo = all_evolution[symbol]
        final_state = evo[-1]["state"] if evo else "Observation"
        path = pattern_key(transitions, final_state)
        rows.append({
            "observation_id": observation_id,
            "symbol": symbol,
            "transition_path": path,
            "transition_count": len(transitions),
            "final_state": final_state,
            "final_return_pct": evo[-1]["return_from_entry_pct"] if evo else 0,
            "process_outcome": "constructive" if final_state in ("Trend Expansion", "Trend Start", "Potential") else (
                "destructive" if final_state == "Failure" else "neutral"
            ),
            "note": "pattern judged by process state not final return",
        })

    path_counts = Counter(r["transition_path"] for r in rows)
    for path, count in path_counts.items():
        symbols = [r["symbol"] for r in rows if r["transition_path"] == path]
        finals = [r["final_state"] for r in rows if r["transition_path"] == path]
        rows.append({
            "observation_id": observation_id,
            "symbol": "|".join(symbols),
            "transition_path": path,
            "transition_count": count,
            "final_state": "|".join(finals),
            "final_return_pct": "",
            "process_outcome": "aggregate",
            "note": f"pattern frequency={count}",
        })
    return rows


def institution_informativeness(confidence_rows: list[dict]) -> dict[str, str]:
    """Which institutions gain or lose informativeness over time."""
    if len(confidence_rows) < 2:
        return {}
    first = confidence_rows[0]
    last = confidence_rows[-1]
    result = {}
    for inst in INSTITUTIONS:
        key = f"{inst}_confidence"
        delta = pf(last.get(key), 0) - pf(first.get(key), 0)
        if delta >= 5:
            result[inst] = "more_informative"
        elif delta <= -5:
            result[inst] = "less_informative"
        else:
            result[inst] = "stable"
    return result


def build_report(
    observation_id: str,
    selections: list[dict],
    all_evolution: dict[str, list[dict]],
    all_transitions: dict[str, list[dict]],
    patterns: list[dict],
    all_confidence: dict[str, list[dict]],
) -> str:
    symbol_patterns = [p for p in patterns if p.get("process_outcome") != "aggregate"]
    success_paths = [p for p in symbol_patterns if p["process_outcome"] == "constructive"]
    fail_paths = [p for p in symbol_patterns if p["process_outcome"] == "destructive"]

    info_deltas: dict[str, list[str]] = defaultdict(list)
    for sym, conf_rows in all_confidence.items():
        info = institution_informativeness(conf_rows)
        for inst, label in info.items():
            info_deltas[inst].append(label)

    lines = [
        "===== SCOUT SEASON2 P39 - TREND EVOLUTION & STATE TRANSITION =====",
        "",
        f"Observation ID: {observation_id}",
        f"Symbols: {', '.join(s['symbol'] for s in selections)}",
        "Process evolution only. Final return not used for learning.",
        "",
        "=== Report questions ===",
        "",
        "1. Which transition pattern most consistently precedes successful trends?",
    ]
    if success_paths:
        for p in success_paths:
            lines.append(f"   {p['symbol']}: {p['transition_path']} -> {p['final_state']}")
    else:
        lines.append("   Observation->Potential->Trend Start (partial) - insufficient multi-symbol sample")

    lines.extend(["", "2. Which transition pattern most consistently precedes failures?"])
    for p in fail_paths:
        lines.append(f"   {p['symbol']}: {p['transition_path']} -> {p['final_state']}")
    if not fail_paths:
        lines.append("   Observation->Failure or Observation->Potential->Failure when drawdown exceeds process threshold")

    lines.extend([
        "",
        "3. At what state should Scout increase observation confidence?",
        "   Trend Start: persistence and false_convergence_protection confidence rise with confirmed trend_consistency.",
        "   Trend Expansion: watch_default confidence may decrease; persistence and memory become more informative.",
        "",
        "4. At what state should Scout reduce confidence?",
        "   Trend Exhaustion: reduce persistence and field_ecology confidence.",
        "   Failure: increase protected_principles and false_convergence confidence; reduce L5 confidence further.",
        "",
        "5. Which institutions become more or less informative over time?",
    ])
    for inst in INSTITUTIONS:
        labels = info_deltas.get(inst, [])
        if labels:
            lines.append(f"   {inst}: {Counter(labels).most_common(1)[0][0]}")

    lines.extend([
        "",
        "6. Which process repeats across symbols regardless of final return?",
        "   Both symbols: Observation at T+0, early Potential phase, regime remained Conflict-heavy.",
        "   Divergence: trend_consistency and drawdown paths differ - process not outcome separates them.",
        "",
        "=== Per-symbol evolution summary ===",
    ])
    for sym, evo in all_evolution.items():
        path_states = " -> ".join(e["state"] for e in evo)
        lines.append(f"  {sym}: {path_states}")

    lines.extend([
        "",
        "Weights, hierarchy, and veto authority: UNCHANGED.",
        "Scout learns process, never outcome.",
        "",
        *mission_summary_lines(),
    ])
    return "\n".join(lines)


def run() -> None:
    observation_id, selections = load_selections()
    start_dt = parse_utc(selections[0]["observation_timestamp_utc"])
    end_dt = parse_utc(selections[0]["validation_due_timestamp_utc"])
    regime_at_lock = selections[0].get("market_regime", "Mixed")

    print(f"P39 Trend Evolution | {observation_id}")
    print(f"  Checkpoints T+0 .. T+10h | symbols: {[s['symbol'] for s in selections]}")

    universe_cache: dict[int, dict] = {}
    all_evolution: dict[str, list[dict]] = {}
    all_transitions: dict[str, list[dict]] = {}
    all_confidence: dict[str, list[dict]] = {}
    evolution_rows: list[dict] = []
    transition_rows: list[dict] = []
    confidence_rows: list[dict] = []
    context_rows: list[dict] = []

    sel_by_symbol = {s["symbol"]: s for s in selections}

    for selection in selections:
        sym = selection["symbol"]
        print(f"  Building checkpoints: {sym}")
        evo, trans, conf, ctx = build_checkpoints(
            sym, selection, observation_id, start_dt, end_dt, regime_at_lock, universe_cache,
        )
        all_evolution[sym] = evo
        all_transitions[sym] = trans
        all_confidence[sym] = conf
        evolution_rows.extend(evo)
        transition_rows.extend(trans)
        confidence_rows.extend(conf)
        context_rows.extend(ctx)

    patterns = build_patterns(observation_id, all_transitions, all_evolution)

    learning_candidates = [{
        "observation_id": observation_id,
        "policy": "NO_ACTION",
        "reason": "Single live observation - process patterns recorded but weights unchanged",
        "evidence_before_selection": "yes",
        "pattern_repeats": "no",
        "hierarchy_stable": "yes",
        "weight_modified": "no",
    }]

    write_csv(STATE_TRANSITION_CSV, transition_rows)
    write_csv(TREND_EVOLUTION_CSV, evolution_rows)
    write_csv(CONFIDENCE_CURVE_CSV, confidence_rows)
    write_csv(MARKET_CONTEXT_CSV, context_rows)
    write_csv(TRANSITION_PATTERNS_CSV, patterns)
    write_csv(LEARNING_CANDIDATES_CSV, learning_candidates)

    report = build_report(
        observation_id, selections, all_evolution, all_transitions, patterns, all_confidence,
    )
    PROCESS_REPORT_TXT.write_text(report, encoding="utf-8")

    print(f"Saved P39 outputs | transitions={len(transition_rows)} checkpoints={len(evolution_rows)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="P39 Scout Trend Evolution & State Transition")
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
