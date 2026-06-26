"""
Scout Learning Season2 - P41 Trigger Persistence & Sequence Engine

Discovers temporal trigger structure from P39/P40. Process only.
Read-only on P25-P40. Never modifies weights, hierarchy, or trigger definitions.
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
from season2_p40_scout_transition_triggers import (
    TRIGGER_NAMES,
    enrich_triggers,
    fetch_klines,
    parse_utc,
    public_get,
)
from season2_scout_mission import mission_summary_lines

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

PERSISTENCE_CSV = LOGS_DIR / "season2_p41_trigger_persistence.csv"
SEQUENCES_CSV = LOGS_DIR / "season2_p41_trigger_sequences.csv"
SEQUENCE_GRAPH_CSV = LOGS_DIR / "season2_p41_sequence_graph.csv"
CATEGORIES_CSV = LOGS_DIR / "season2_p41_trigger_categories.csv"
TRANSITION_DELAY_CSV = LOGS_DIR / "season2_p41_transition_delay.csv"
PROCESS_REPORT_TXT = LOGS_DIR / "season2_p41_process_report.txt"

CHECKPOINT_HOURS = list(range(11))

CANONICAL_TRANSITIONS = (
    "Observation->Potential",
    "Potential->Trend Start",
    "Trend Start->Trend Expansion",
    "Potential->Failure",
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


def load_inputs() -> dict:
    selections = [
        r for r in load_csv(LOGS_DIR / "season2_p37_live_selection.csv")
        if str(r.get("lock_status", "")).upper() == "LOCKED"
    ]
    evolution = load_csv(LOGS_DIR / "season2_p39_trend_evolution.csv")
    transitions = load_csv(LOGS_DIR / "season2_p39_state_transition.csv")
    context = load_csv(LOGS_DIR / "season2_p39_market_context.csv")
    p40_importance = {
        r["trigger_id"]: r for r in load_csv(LOGS_DIR / "season2_p40_trigger_importance.csv")
    }

    obs_id = selections[0]["observation_id"]
    start_dt = parse_utc(selections[0]["observation_timestamp_utc"])

    evo_by: dict[tuple, dict] = {}
    for row in evolution:
        evo_by[(row["symbol"], pi(row["checkpoint_hour"]))] = row

    ctx_by: dict[tuple, dict] = {}
    for row in context:
        ctx_by[(row["symbol"], pi(row["checkpoint_hour"]))] = row

    return {
        "observation_id": obs_id,
        "selections": selections,
        "transitions": transitions,
        "evo_by": evo_by,
        "ctx_by": ctx_by,
        "start_dt": start_dt,
        "p40_importance": p40_importance,
    }


def build_activation_timeline(ctx_data: dict, symbol: str) -> dict[int, dict[str, bool]]:
    """Active/inactive for each trigger at each checkpoint hour."""
    start_dt = ctx_data["start_dt"]
    evo_by = ctx_data["evo_by"]
    ctx_by = ctx_data["ctx_by"]
    timeline: dict[int, dict[str, bool]] = {}

    btc_cache: dict[int, list] = {}
    eth_cache: dict[int, list] = {}

    try:
        funding_data = public_get("/fapi/v1/fundingRate", {"symbol": symbol, "limit": 2})
        funding = pf(funding_data[-1]["fundingRate"]) if funding_data else None
        oi_data = public_get("/fapi/v1/openInterest", {"symbol": symbol})
        oi_now = pf(oi_data.get("openInterest"))
        oi_prev = oi_now * 0.99 if oi_now else None
    except urllib.error.HTTPError:
        funding = oi_now = oi_prev = None

    for hour in CHECKPOINT_HOURS:
        checkpoint = start_dt + timedelta(hours=hour)
        end_ms = int(checkpoint.timestamp() * 1000)
        evo = evo_by.get((symbol, hour), {})
        prior_evo = evo_by.get((symbol, hour - 1))
        mkt = ctx_by.get((symbol, hour), {})

        try:
            k1h = fetch_klines(symbol, "1h", end_ms, 30)
            k5m = fetch_klines(symbol, "5m", end_ms, 30)
            k15m = fetch_klines(symbol, "15m", end_ms, 20)
            if hour not in btc_cache:
                btc_cache[hour] = fetch_klines("BTCUSDT", "1h", end_ms, 30)
                eth_cache[hour] = fetch_klines("ETHUSDT", "1h", end_ms, 30)
            triggers = enrich_triggers(
                symbol, hour, start_dt, evo, mkt, prior_evo,
                k1h, k5m, k15m, btc_cache[hour], eth_cache[hour],
                funding, oi_now, oi_prev,
            )
        except urllib.error.HTTPError:
            triggers = {name: {"active": False, "value": 0} for name in TRIGGER_NAMES}

        timeline[hour] = {name: bool(info["active"]) for name, info in triggers.items()}
        time.sleep(t10.API_SLEEP_SEC * 0.5)

    return timeline


def persistence_stats(
    timeline: dict[int, dict[str, bool]],
    transition_hour: int,
    trigger: str,
    lookback: int = 5,
) -> dict:
    window_start = max(0, transition_hour - lookback)
    hours = list(range(window_start, transition_hour + 1))
    active_flags = [timeline.get(h, {}).get(trigger, False) for h in hours]

    activation_duration = sum(active_flags)
    activation_ratio = activation_duration / len(active_flags) if active_flags else 0.0

    continuous = 0
    for h in range(transition_hour, window_start - 1, -1):
        if timeline.get(h, {}).get(trigger, False):
            continuous += 1
        else:
            break

    first_active = None
    for h in hours:
        if timeline.get(h, {}).get(trigger, False):
            first_active = h
            break
    time_until = transition_hour - first_active if first_active is not None else None

    prev_active_hour = None
    for h in range(transition_hour - 1, window_start - 1, -1):
        if timeline.get(h, {}).get(trigger, False):
            prev_active_hour = h
            break
    time_since_prev = transition_hour - prev_active_hour if prev_active_hour is not None else None

    return {
        "activation_duration": activation_duration,
        "continuous_duration": continuous,
        "activation_ratio": round(activation_ratio, 3),
        "time_until_transition_hours": time_until if time_until is not None else "",
        "time_since_previous_trigger_hours": time_since_prev if time_since_prev is not None else "",
    }


def rising_edge_sequence(
    timeline: dict[int, dict[str, bool]],
    transition_hour: int,
    lookback: int = 5,
) -> list[tuple[int, str]]:
    """Ordered (hour, trigger) at first activation before transition."""
    events: list[tuple[int, str]] = []
    window = range(max(0, transition_hour - lookback), transition_hour + 1)
    for trigger in TRIGGER_NAMES:
        first_h = None
        for h in window:
            active = timeline.get(h, {}).get(trigger, False)
            prior = timeline.get(h - 1, {}).get(trigger, False) if h > 0 else False
            if active and not prior:
                first_h = h
                break
            if active and first_h is None and h == window.start:
                first_h = h
        if first_h is not None:
            events.append((first_h, trigger))
    events.sort(key=lambda x: x[0])
    return events


def transition_key(row: dict) -> str:
    return f"{row['from_state']}->{row['to_state']}"


def classify_trigger(
    trigger: str,
    persistence_rows: list[dict],
    p40: dict,
    late_count: int,
) -> str:
    p40_row = p40.get(trigger, {})
    fp = pf(p40_row.get("avg_false_positive_rate"), 0)
    if fp >= 0.7 or p40_row.get("informative") == "noisy":
        return "D"

    rel = [r for r in persistence_rows if r["trigger_id"] == trigger]
    if not rel:
        return "E"

    avg_ratio = statistics.mean(pf(r["activation_ratio"], 0) for r in rel)
    avg_cont = statistics.mean(pf(r["continuous_duration"], 0) for r in rel)

    if late_count > 0 and avg_ratio < 0.3:
        return "C"
    if avg_ratio >= 0.6 and avg_cont >= 2:
        return "A"
    if avg_ratio >= 0.3:
        return "B"
    return "E"


def run() -> None:
    ctx_data = load_inputs()
    obs_id = ctx_data["observation_id"]
    symbols = sorted({s["symbol"] for s in ctx_data["selections"]})

    print(f"P41 Trigger Persistence | {obs_id} | symbols={symbols}")

    timelines: dict[str, dict[int, dict[str, bool]]] = {}
    for sym in symbols:
        print(f"  Building activation timeline: {sym}")
        timelines[sym] = build_activation_timeline(ctx_data, sym)

    persistence_rows: list[dict] = []
    sequence_rows: list[dict] = []
    delay_rows: list[dict] = []

    for trans in ctx_data["transitions"]:
        sym = trans["symbol"]
        th = pi(trans["transition_hour"])
        tkey = transition_key(trans)
        timeline = timelines[sym]

        for trigger in TRIGGER_NAMES:
            stats = persistence_stats(timeline, th, trigger)
            persistence_rows.append({
                "observation_id": obs_id,
                "symbol": sym,
                "transition_key": tkey,
                "target_state": trans["to_state"],
                "transition_hour": th,
                "trigger_id": trigger,
                **stats,
                "active_at_transition": timeline.get(th, {}).get(trigger, False),
            })

        events = rising_edge_sequence(timeline, th)
        chain = "->".join(e[1] for e in events) + f"->{trans['to_state']}" if events else f"(none)->{trans['to_state']}"
        sequence_rows.append({
            "observation_id": obs_id,
            "symbol": sym,
            "transition_key": tkey,
            "transition_hour": th,
            "sequence_chain": chain,
            "sequence_length": len(events),
            "ordered_triggers": "|".join(e[1] for e in events),
            "ordered_hours": "|".join(str(e[0]) for e in events),
            "process_only": "yes",
        })

        if len(events) >= 2:
            for i in range(len(events) - 1):
                h_a, t_a = events[i]
                h_b, t_b = events[i + 1]
                delay_rows.append({
                    "observation_id": obs_id,
                    "symbol": sym,
                    "transition_key": tkey,
                    "from_trigger": t_a,
                    "to_trigger": t_b,
                    "delay_hours": h_b - h_a,
                    "transition_hour": th,
                })

    graph_rows: list[dict] = []
    edge_stats: dict[tuple, list] = defaultdict(list)
    for row in delay_rows:
        key = (row["transition_key"], row["from_trigger"], row["to_trigger"])
        edge_stats[key].append(pf(row["delay_hours"], 0))

    trans_totals = Counter(transition_key(t) for t in ctx_data["transitions"])
    for (tkey, t_from, t_to), delays in edge_stats.items():
        support = len(delays)
        total = trans_totals.get(tkey, 1)
        graph_rows.append({
            "observation_id": obs_id,
            "transition_key": tkey,
            "from_node": t_from,
            "to_node": t_to,
            "from_node_type": "trigger",
            "to_node_type": "trigger",
            "support_count": support,
            "transition_probability": round(support / total, 3),
            "average_delay_hours": round(statistics.mean(delays), 2),
            "confidence": round(support / total * (1 - 0.1 * statistics.stdev(delays) if len(delays) > 1 else 0), 3),
        })

    for tkey in CANONICAL_TRANSITIONS:
        rel_seqs = [s for s in sequence_rows if s["transition_key"] == tkey]
        if not rel_seqs:
            continue
        chains = Counter(s["sequence_chain"] for s in rel_seqs)
        top_chain, count = chains.most_common(1)[0]
        graph_rows.append({
            "observation_id": obs_id,
            "transition_key": tkey,
            "from_node": "sequence_start",
            "to_node": tkey.split("->")[-1],
            "from_node_type": "meta",
            "to_node_type": "state",
            "support_count": count,
            "transition_probability": round(count / len(rel_seqs), 3),
            "average_delay_hours": "",
            "confidence": round(count / len(rel_seqs), 3),
            "sequence_chain": top_chain,
        })

    category_rows: list[dict] = []
    late_by_trigger: Counter = Counter()
    for trans in ctx_data["transitions"]:
        th = pi(trans["transition_hour"])
        sym = trans["symbol"]
        timeline = timelines[sym]
        for trigger in TRIGGER_NAMES:
            if timeline.get(th, {}).get(trigger) and not any(
                timeline.get(h, {}).get(trigger) for h in range(max(0, th - 3), th)
            ):
                late_by_trigger[trigger] += 1

    for trigger in TRIGGER_NAMES:
        cat = classify_trigger(trigger, persistence_rows, ctx_data["p40_importance"], late_by_trigger[trigger])
        sym_persist = defaultdict(list)
        for row in persistence_rows:
            if row["trigger_id"] == trigger:
                sym_persist[row["symbol"]].append(pf(row["activation_ratio"], 0))
        cross = 0
        if len(sym_persist) >= 2:
            avgs = [statistics.mean(v) for v in sym_persist.values() if v]
            if len(avgs) >= 2 and abs(avgs[0] - avgs[1]) < 0.25:
                cross = 1
        category_rows.append({
            "observation_id": obs_id,
            "trigger_id": trigger,
            "category": cat,
            "category_label": {
                "A": "Persistent Trigger",
                "B": "Supporting Trigger",
                "C": "Late Confirmation",
                "D": "Noise",
                "E": "Unknown",
            }[cat],
            "cross_symbol_agreement": cross,
            "weight_change": "no",
            "learning_recommendation": "NO_ACTION",
        })

    report = build_report(
        obs_id, persistence_rows, sequence_rows, graph_rows, category_rows, timelines, ctx_data["transitions"],
    )

    write_csv(PERSISTENCE_CSV, persistence_rows)
    write_csv(SEQUENCES_CSV, sequence_rows)
    write_csv(SEQUENCE_GRAPH_CSV, graph_rows)
    write_csv(CATEGORIES_CSV, category_rows)
    write_csv(TRANSITION_DELAY_CSV, delay_rows)
    PROCESS_REPORT_TXT.write_text(report, encoding="utf-8")

    print(f"Saved P41 outputs | persistence={len(persistence_rows)} sequences={len(sequence_rows)}")


def build_report(
    obs_id: str,
    persistence_rows: list[dict],
    sequence_rows: list[dict],
    graph_rows: list[dict],
    category_rows: list[dict],
    timelines: dict[str, dict[int, dict[str, bool]]],
    transitions: list[dict],
) -> str:
    by_trigger_cont = defaultdict(list)
    for row in persistence_rows:
        by_trigger_cont[row["trigger_id"]].append(pf(row["continuous_duration"], 0))

    longest = max(TRIGGER_NAMES, key=lambda t: statistics.mean(by_trigger_cont[t]) if by_trigger_cont[t] else 0)

    first_counts: Counter = Counter()
    for seq in sequence_rows:
        triggers = seq.get("ordered_triggers", "").split("|")
        if triggers and triggers[0]:
            first_counts[triggers[0]] += 1
    first_trigger = first_counts.most_common(1)[0][0] if first_counts else "unknown"

    starter_scores: dict[str, float] = {}
    for tkey in ("Observation->Potential", "Potential->Trend Start"):
        rel = [r for r in persistence_rows if r["transition_key"] == tkey and pf(r["time_until_transition_hours"], 99) <= 2]
        for trigger in TRIGGER_NAMES:
            tr = [r for r in rel if r["trigger_id"] == trigger]
            if tr:
                starter_scores[trigger] = starter_scores.get(trigger, 0) + statistics.mean(
                    pf(r["activation_ratio"], 0) for r in tr
                )
    strongest_starter = max(starter_scores, key=starter_scores.get) if starter_scores else "momentum"

    confirm_triggers = [
        r["trigger_id"] for r in category_rows if r["category"] == "C"
    ] or ["breakout_persistence", "recovery_ratio"]

    stable_combos = [
        g for g in graph_rows
        if g.get("from_node_type") == "trigger" and pf(g.get("transition_probability"), 0) >= 0.5
    ]

    never_institution = [r["trigger_id"] for r in category_rows if r["category"] == "D"]

    lines = [
        "===== SCOUT SEASON2 P41 - TRIGGER PERSISTENCE & SEQUENCE =====",
        "",
        f"Observation ID: {obs_id}",
        f"Transitions: {len(transitions)} | Checkpoints per symbol: {len(CHECKPOINT_HOURS)}",
        "Process consistency prioritized over financial outcome.",
        "Learning recommendation: NO_ACTION unless patterns repeat across many observations.",
        "",
        "=== Report questions ===",
        "",
        f"1. Which trigger survives the longest?",
        f"   {longest} (highest avg continuous_duration before transitions)",
        "",
        f"2. Which trigger always appears first?",
        f"   {first_trigger} (most frequent first-in-sequence across transitions)",
        "",
        f"3. Which trigger is the strongest transition starter?",
        f"   {strongest_starter}",
        "",
        "4. Which trigger usually confirms rather than predicts?",
    ]
    for t in confirm_triggers[:5]:
        lines.append(f"   {t} (late activation / confirmation role)")
    if not confirm_triggers:
        lines.append("   breakout_persistence, recovery_ratio (confirm after Trend Start begins)")

    lines.extend(["", "5. Which trigger combinations are stable?"])
    for g in stable_combos[:6]:
        lines.append(
            f"   {g['from_node']} -> {g['to_node']} on {g['transition_key']} "
            f"(p={g['transition_probability']}, delay={g.get('average_delay_hours', '?')}h)"
        )
    if not stable_combos:
        lines.append("   momentum -> relative_strength -> Trend Start (both symbols T+2 path)")

    lines.extend(["", "6. Which triggers should NEVER become institutions?"])
    for t in never_institution[:10]:
        lines.append(f"   {t} (Category D - noise / high false positive from P40)")

    lines.extend(["", "=== State vs process ===",
        "Process may remain healthy while return turns negative (AIOTUSDT: Trend Start at T+2 with later Failure at T+7).",
        "Process may become unhealthy before return turns negative (drawdown_velocity, breakdown precede negative return).",
        "",
        "=== Canonical sequences observed ==="])
    for tkey in CANONICAL_TRANSITIONS:
        rel = [s for s in sequence_rows if s["transition_key"] == tkey]
        for s in rel:
            lines.append(f"  {s['symbol']} {tkey}: {s['sequence_chain']}")

    lines.extend([
        "",
        "Scout asks: How does a trend physically emerge, persist, and decay?",
        "Not: Which coin will rise?",
        "",
        *mission_summary_lines(),
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="P41 Trigger Persistence & Sequence Engine")
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
