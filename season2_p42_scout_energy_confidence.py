"""
Scout Learning Season2 - P42 Energy & Confidence Engine

Builds State -> Transition -> Trigger -> Persistence -> Sequence -> Energy -> Confidence
layer from P25-P41 outputs. Observation only. No price prediction. NO_ACTION default.
Read-only on all prior phase files. Never modifies hierarchy, weights, or triggers.
"""

from __future__ import annotations

import argparse
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from season2_p37_scout_decision_hierarchy import load_csv, pf, pi
from season2_p40_scout_transition_triggers import TRIGGER_NAMES
from season2_p41_scout_trigger_persistence import (
    CHECKPOINT_HOURS,
    build_activation_timeline,
    load_inputs as load_p41_inputs,
    rising_edge_sequence,
    transition_key,
)
from season2_scout_mission import mission_summary_lines

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

ENERGY_COMPONENTS_CSV = LOGS_DIR / "season2_p42_energy_components.csv"
ENERGY_CURVE_CSV = LOGS_DIR / "season2_p42_energy_curve.csv"
ENERGY_VELOCITY_CSV = LOGS_DIR / "season2_p42_energy_velocity.csv"
ENERGY_ACCELERATION_CSV = LOGS_DIR / "season2_p42_energy_acceleration.csv"
CONFIDENCE_CURVE_CSV = LOGS_DIR / "season2_p42_confidence_curve.csv"
ENERGY_MAP_CSV = LOGS_DIR / "season2_p42_energy_map.csv"
EXPECTED_EVOLUTION_CSV = LOGS_DIR / "season2_p42_expected_evolution.csv"
PROCESS_PHYSICS_CSV = LOGS_DIR / "season2_p42_process_physics.csv"
PROCESS_REPORT_TXT = LOGS_DIR / "season2_p42_process_report.txt"

# STEP1 - Energy groups (existing triggers only; derived metrics are computed, not new triggers)
MOMENTUM_TRIGGERS = (
    "momentum",
    "obv",
    "relative_strength",
    "ema_distance",
    "ha_5m_slope",
    "ha_15m_slope",
)
STRUCTURAL_TRIGGERS = (
    "recovery_ratio",
    "breakout_persistence",
)
CROWD_TRIGGERS = (
    "sector_strength",
    "market_breadth",
    "funding",
    "btc_beta",
    "eth_beta",
)
PERSISTENCE_TRIGGERS = (
    "atr",
    "open_interest",
    "volume",
    "volume_acceleration",
)

ENERGY_STATE_ORDER = (
    "Dormant",
    "Observation",
    "Potential",
    "Activation",
    "Expansion",
    "Exhaustion",
    "Collapse",
)

P39_TO_ENERGY_HINT = {
    "Observation": "Observation",
    "Potential": "Potential",
    "Trend Start": "Activation",
    "Trend Expansion": "Expansion",
    "Failure": "Collapse",
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
        import csv
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def trigger_group_energy(timeline: dict[int, dict[str, bool]], hour: int, members: tuple[str, ...]) -> float:
    active = sum(1 for t in members if timeline.get(hour, {}).get(t, False))
    return round(100.0 * active / len(members), 1) if members else 0.0


def continuous_length(timeline: dict[int, dict[str, bool]], hour: int, trigger: str) -> int:
    length = 0
    for h in range(hour, -1, -1):
        if timeline.get(h, {}).get(trigger, False):
            length += 1
        else:
            break
    return length


def persistence_length_score(timeline: dict[int, dict[str, bool]], hour: int) -> float:
    lengths = [continuous_length(timeline, hour, t) for t in PERSISTENCE_TRIGGERS]
    max_len = hour + 1
    return round(100.0 * statistics.mean(lengths) / max_len, 1) if lengths and max_len else 0.0


def graph_confidence_lookup(graph_rows: list[dict]) -> dict[tuple, float]:
    lookup: dict[tuple, float] = {}
    for row in graph_rows:
        if row.get("from_node_type") != "trigger":
            continue
        key = (row["transition_key"], row["from_node"], row["to_node"])
        lookup[key] = pf(row.get("confidence"), 0)
    return lookup


def transition_consistency_score(
    hour: int,
    transitions: list[dict],
    graph_rows: list[dict],
    symbol: str,
) -> float:
    rel = [
        t for t in transitions
        if t["symbol"] == symbol and pi(t["transition_hour"]) <= hour
    ]
    if not rel:
        return 0.0
    confidences: list[float] = []
    for trans in rel:
        tkey = transition_key(trans)
        edges = [
            pf(g.get("confidence"), 0)
            for g in graph_rows
            if g.get("transition_key") == tkey and g.get("from_node_type") == "trigger"
        ]
        if edges:
            confidences.append(statistics.mean(edges))
    return round(statistics.mean(confidences) * 100, 1) if confidences else 0.0


def sequence_consistency_score(
    timeline: dict[int, dict[str, bool]],
    hour: int,
    graph_lookup: dict[tuple, float],
    transitions: list[dict],
    symbol: str,
) -> float:
    trans_at = [
        t for t in transitions
        if t["symbol"] == symbol and pi(t["transition_hour"]) == hour
    ]
    if not trans_at:
        trans_at = [
            t for t in transitions
            if t["symbol"] == symbol and pi(t["transition_hour"]) <= hour
        ]
        if not trans_at:
            return 0.0
        trans_at = [trans_at[-1]]

    tkey = transition_key(trans_at[-1])
    events = rising_edge_sequence(timeline, hour)
    if len(events) < 2:
        return 0.0
    confidences: list[float] = []
    for i in range(len(events) - 1):
        _, t_from = events[i]
        _, t_to = events[i + 1]
        confidences.append(graph_lookup.get((tkey, t_from, t_to), 0.0))
    return round(statistics.mean(confidences) * 100, 1) if confidences else 0.0


def compute_energy_components(
    timeline: dict[int, dict[str, bool]],
    hour: int,
    transitions: list[dict],
    graph_rows: list[dict],
    symbol: str,
    graph_lookup: dict[tuple, float],
) -> dict[str, float]:
    momentum = trigger_group_energy(timeline, hour, MOMENTUM_TRIGGERS)
    crowd = trigger_group_energy(timeline, hour, CROWD_TRIGGERS)
    structural_triggers = trigger_group_energy(timeline, hour, STRUCTURAL_TRIGGERS)
    trans_cons = transition_consistency_score(hour, transitions, graph_rows, symbol)
    seq_cons = sequence_consistency_score(timeline, hour, graph_lookup, transitions, symbol)
    structural = round((structural_triggers + trans_cons + seq_cons) / 3, 1)

    persistence_triggers = trigger_group_energy(timeline, hour, PERSISTENCE_TRIGGERS)
    persist_len = persistence_length_score(timeline, hour)
    persistence = round((persistence_triggers + persist_len) / 2, 1)

    total = round(momentum + structural + crowd + persistence, 1)
    return {
        "momentum_energy": momentum,
        "structural_energy": structural,
        "crowd_energy": crowd,
        "persistence_energy": persistence,
        "total_energy": total,
        "transition_consistency_derived": trans_cons,
        "sequence_consistency_derived": seq_cons,
        "persistence_length_derived": persist_len,
    }


def map_energy_state(
    total: float,
    velocity: float | None,
    p39_state: str,
    totals: list[float],
) -> str:
    if p39_state == "Failure":
        return "Collapse"
    if not totals:
        return P39_TO_ENERGY_HINT.get(p39_state, "Observation")

    lo, hi = min(totals), max(totals)
    span = hi - lo if hi > lo else 1.0
    pct = (total - lo) / span * 100

    if velocity is not None and velocity <= -15 and pct < 50:
        return "Collapse"
    if pct < 12:
        return "Dormant"
    if pct < 25:
        return "Observation"
    if pct < 40:
        return "Potential"
    if pct < 55:
        return "Activation"
    if pct < 70:
        return "Expansion"
    if pct < 85:
        return "Exhaustion"
    return "Collapse" if velocity is not None and velocity < 0 else "Expansion"


def hierarchy_agreement(symbol: str, reasoning_by_sym: dict[str, dict]) -> float:
    row = reasoning_by_sym.get(symbol, {})
    return pf(row.get("reasoning_score"), 50.0)


def trigger_agreement(
    timeline: dict[int, dict[str, bool]],
    hour: int,
    p39_state: str,
    success_by_state: dict[str, set[str]],
) -> float:
    active = {t for t in TRIGGER_NAMES if timeline.get(hour, {}).get(t, False)}
    expected = success_by_state.get(p39_state, set())
    if not expected:
        return round(100.0 * len(active) / max(len(TRIGGER_NAMES), 1), 1)
    overlap = len(active & expected)
    return round(100.0 * overlap / len(expected), 1)


def build_success_by_state(success_rows: list[dict]) -> dict[str, set[str]]:
    by_state: dict[str, set[str]] = defaultdict(set)
    for row in success_rows:
        tkey = row.get("transition_key", "")
        if "->" not in tkey:
            continue
        to_state = tkey.split("->", 1)[1]
        by_state[to_state].add(row["trigger_id"])
    return dict(by_state)


def confidence_at_checkpoint(
    symbol: str,
    hour: int,
    timeline: dict[int, dict[str, bool]],
    p39_state: str,
    reasoning_by_sym: dict[str, dict],
    success_by_state: dict[str, set[str]],
    graph_rows: list[dict],
    transitions: list[dict],
    persistence_rows: list[dict],
    graph_lookup: dict[tuple, float],
) -> dict[str, float]:
    hierarchy = hierarchy_agreement(symbol, reasoning_by_sym)
    trigger_ag = trigger_agreement(timeline, hour, p39_state, success_by_state)
    seq_stab = sequence_consistency_score(timeline, hour, graph_lookup, transitions, symbol)
    trans_cons = transition_consistency_score(hour, transitions, graph_rows, symbol)

    rel_persist = [
        r for r in persistence_rows
        if r["symbol"] == symbol and pi(r["transition_hour"]) <= hour
    ]
    if rel_persist:
        persist_qual = statistics.mean(pf(r["activation_ratio"], 0) for r in rel_persist) * 100
    else:
        persist_qual = persistence_length_score(timeline, hour)

    components = {
        "hierarchy_agreement": round(hierarchy, 1),
        "trigger_agreement": round(trigger_ag, 1),
        "sequence_stability": round(seq_stab, 1),
        "transition_consistency": round(trans_cons, 1),
        "persistence_quality": round(persist_qual, 1),
    }
    composite = round(statistics.mean(components.values()), 1)
    components["confidence"] = min(100.0, composite)
    return components


def build_transition_counts(transitions: list[dict]) -> dict[str, dict[str, Counter]]:
    counts: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    for trans in transitions:
        counts[trans["symbol"]][trans["from_state"]][trans["to_state"]] += 1
    return {sym: dict(by_state) for sym, by_state in counts.items()}


def expected_evolution_rows(
    obs_id: str,
    symbols: list[str],
    evolution: list[dict],
    transition_counts: dict[str, dict[str, Counter]],
) -> list[dict]:
    rows: list[dict] = []
    for sym in symbols:
        evo_sym = sorted(
            [e for e in evolution if e["symbol"] == sym],
            key=lambda e: pi(e["checkpoint_hour"]),
        )
        if not evo_sym:
            continue
        latest = evo_sym[-1]
        current = latest["state"]
        counter = transition_counts.get(sym, {}).get(current, Counter())
        if not counter:
            rows.append({
                "observation_id": obs_id,
                "symbol": sym,
                "current_state": current,
                "observed_next_state": current,
                "support_pct": 100.0,
                "confidence": "",
                "observation_count": 1,
                "learning_status": "Experimental",
                "prediction": "no",
                "learning_recommendation": "NO_ACTION",
            })
            continue
        total = sum(counter.values())
        next_state, count = counter.most_common(1)[0]
        support = round(100.0 * count / total, 1)
        rows.append({
            "observation_id": obs_id,
            "symbol": sym,
            "current_state": current,
            "observed_next_state": next_state,
            "support_pct": support,
            "confidence": "",
            "observation_count": 1,
            "learning_status": "Experimental",
            "prediction": "no",
            "learning_recommendation": "NO_ACTION",
        })
    return rows


def process_physics_rows(
    obs_id: str,
    curve_by_sym: dict[str, list[dict]],
    confidence_by_sym: dict[str, list[dict]],
    evolution: list[dict],
    transitions: list[dict],
) -> list[dict]:
    rows: list[dict] = []

    def add(q: str, finding: str, evidence: str) -> None:
        rows.append({
            "observation_id": obs_id,
            "question": q,
            "observation_finding": finding,
            "evidence": evidence,
            "learning_recommendation": "NO_ACTION",
        })

    for sym, curve in curve_by_sym.items():
        evo = sorted(
            [e for e in evolution if e["symbol"] == sym],
            key=lambda e: pi(e["checkpoint_hour"]),
        )
        conf = confidence_by_sym.get(sym, [])

        energy_leads = 0
        energy_lags = 0
        for i in range(1, len(curve)):
            e_delta = pf(curve[i]["total_energy"]) - pf(curve[i - 1]["total_energy"])
            r_delta = pf(evo[i].get("return_from_entry_pct")) - pf(evo[i - 1].get("return_from_entry_pct"))
            if abs(e_delta) > 2 and abs(r_delta) > 0.05:
                if e_delta > 0 and r_delta > 0 and abs(e_delta) >= abs(r_delta):
                    energy_leads += 1
                elif r_delta > 0 and e_delta <= 0:
                    energy_lags += 1
        if energy_leads > energy_lags:
            add(
                "1. Energy before price?",
                f"{sym}: energy often changed before return (observed {energy_leads} lead vs {energy_lags} lag checkpoints)",
                "Compare total_energy delta vs return_from_entry_pct delta (observation only)",
            )
        else:
            add(
                "1. Energy before price?",
                f"{sym}: mixed or return led energy ({energy_leads} lead vs {energy_lags} lag checkpoints)",
                "Compare total_energy delta vs return_from_entry_pct delta (observation only)",
            )

        conf_late = 0
        conf_early = 0
        for i in range(1, len(curve)):
            e_delta = pf(curve[i]["total_energy"]) - pf(curve[i - 1]["total_energy"])
            c_i = pf(conf[i]["confidence"]) if i < len(conf) else 0
            c_prev = pf(conf[i - 1]["confidence"]) if i - 1 < len(conf) else 0
            c_delta = c_i - c_prev
            if abs(e_delta) > 3 and abs(c_delta) > 0.5:
                if e_delta > 0 and c_delta >= 0 and c_delta < e_delta / 5:
                    conf_late += 1
                elif c_delta > e_delta:
                    conf_early += 1
        add(
            "2. Confidence lags energy?",
            f"{sym}: confidence moved slower than energy in {conf_late} checkpoints; faster in {conf_early}",
            "confidence delta vs total_energy delta at T+0..T+10",
        )

        trans_hours = {pi(t["transition_hour"]) for t in transitions if t["symbol"] == sym}
        vel_before = 0
        vel_after = 0
        for th in trans_hours:
            if th > 0 and th < len(curve):
                v_before = pf(curve[th].get("energy_velocity"), 0)
                v_after = pf(curve[th + 1].get("energy_velocity"), 0) if th + 1 < len(curve) else 0
                if abs(v_before) > abs(v_after):
                    vel_before += 1
                else:
                    vel_after += 1
        add(
            "3. Velocity before transition?",
            f"{sym}: velocity spike before transition at {vel_before}/{len(trans_hours) or 1} transition hours",
            "energy_velocity at transition_hour vs transition_hour+1",
        )

        ts_hours = [
            pi(t["transition_hour"])
            for t in transitions
            if t["symbol"] == sym and t["to_state"] == "Trend Start"
        ]
        accel_rise = 0
        for th in ts_hours:
            if th > 0 and th < len(curve):
                acc = pf(curve[th].get("energy_acceleration"), 0)
                if acc > 0:
                    accel_rise += 1
        add(
            "4. Acceleration before Trend Start?",
            f"{sym}: positive acceleration at {accel_rise}/{len(ts_hours) or 1} Trend Start hours",
            "energy_acceleration at Trend Start transition_hour",
        )

        fail_hours = [
            pi(t["transition_hour"])
            for t in transitions
            if t["symbol"] == sym and t["to_state"] == "Failure"
        ]
        decline_before_fail = 0
        for fh in fail_hours:
            declines = 0
            for h in range(max(0, fh - 2), fh + 1):
                if h > 0 and pf(curve[h]["total_energy"]) < pf(curve[h - 1]["total_energy"]):
                    declines += 1
            if declines >= 2:
                decline_before_fail += 1
        add(
            "5. Energy decline before Failure?",
            f"{sym}: repeated energy decline before Failure at {decline_before_fail}/{len(fail_hours) or 1} failure paths",
            "total_energy decreasing checkpoints in 2h window before Failure",
        )

    return rows


def build_report(
    obs_id: str,
    energy_map_rows: list[dict],
    physics_rows: list[dict],
    evolution_rows: list[dict],
) -> str:
    lines = [
        "===== SCOUT SEASON2 P42 - ENERGY & CONFIDENCE ENGINE =====",
        "",
        f"Observation ID: {obs_id}",
        "Market State Physics Engine - observation only. NO_ACTION on all learning.",
        "",
        "Hierarchy: State -> Transition -> Trigger -> Persistence -> Sequence -> Energy -> Confidence -> Expected Evolution",
        "",
        "=== Energy Map (latest checkpoint) ===",
    ]
    for row in energy_map_rows:
        lines.append(
            f"  {row['symbol']}: Energy {row['total_energy']} | Confidence {row['confidence']} | "
            f"State {row['energy_state']} (P39: {row['p39_state']}) | "
            f"Velocity {row['energy_velocity']:+.0f} | Acceleration {row['energy_acceleration']:+.0f} | "
            f"Expected Next {row['expected_next_state']} ({row['support_pct']}%)"
        )

    lines.extend(["", "=== Process Physics (observation only) ==="])
    seen_q: set[str] = set()
    for row in physics_rows:
        q = row["question"]
        if q in seen_q:
            continue
        seen_q.add(q)
        lines.append(f"  {q}")
        lines.append(f"    {row['observation_finding']}")

    lines.extend([
        "",
        "=== Philosophy ===",
        "Do not predict price. Measure energy.",
        "Do not chase winners. Measure transitions.",
        "Do not optimize hindsight. Observe market physics.",
        "",
        "Scout is a Market State Physics Engine, not a Price Prediction Engine.",
        "Learning recommendation: NO_ACTION unless identical energy patterns repeat across many observations.",
        "",
        *mission_summary_lines(),
    ])
    return "\n".join(lines)


def run() -> None:
    ctx = load_p41_inputs()
    obs_id = ctx["observation_id"]
    symbols = sorted({s["symbol"] for s in ctx["selections"]})

    evolution = load_csv(LOGS_DIR / "season2_p39_trend_evolution.csv")
    transitions = ctx["transitions"]
    graph_rows = load_csv(LOGS_DIR / "season2_p41_sequence_graph.csv")
    persistence_rows = load_csv(LOGS_DIR / "season2_p41_trigger_persistence.csv")
    success_rows = load_csv(LOGS_DIR / "season2_p40_success_trigger.csv")
    reasoning_rows = load_csv(LOGS_DIR / "season2_p38_reasoning_score.csv")

    reasoning_by_sym = {r["symbol"]: r for r in reasoning_rows}
    success_by_state = build_success_by_state(success_rows)
    graph_lookup = graph_confidence_lookup(graph_rows)
    transition_counts = build_transition_counts(transitions)

    print(f"P42 Energy & Confidence | {obs_id} | symbols={symbols}")

    timelines: dict[str, dict[int, dict[str, bool]]] = {}
    for sym in symbols:
        print(f"  Building activation timeline: {sym}")
        timelines[sym] = build_activation_timeline(ctx, sym)

    component_rows: list[dict] = []
    curve_rows: list[dict] = []
    velocity_rows: list[dict] = []
    acceleration_rows: list[dict] = []
    confidence_rows: list[dict] = []
    curve_by_sym: dict[str, list[dict]] = {}
    confidence_by_sym: dict[str, list[dict]] = {}

    for sym in symbols:
        timeline = timelines[sym]
        evo_by_hour = {
            pi(e["checkpoint_hour"]): e
            for e in evolution
            if e["symbol"] == sym
        }
        sym_comps: list[dict] = []

        for hour in CHECKPOINT_HOURS:
            comp = compute_energy_components(
                timeline, hour, transitions, graph_rows, sym, graph_lookup,
            )
            evo = evo_by_hour.get(hour, {})
            p39_state = evo.get("state", "")
            sym_comps.append(comp)

            component_rows.append({
                "observation_id": obs_id,
                "symbol": sym,
                "checkpoint": f"T+{hour}h",
                "checkpoint_hour": hour,
                "p39_state": p39_state,
                **comp,
                "weight_modified": "no",
            })

        sym_totals = [c["total_energy"] for c in sym_comps]
        prev_velocity = 0.0
        for hour in CHECKPOINT_HOURS:
            comp = sym_comps[hour]
            evo = evo_by_hour.get(hour, {})
            p39_state = evo.get("state", "")

            if hour == 0:
                velocity = 0.0
            else:
                velocity = round(comp["total_energy"] - sym_comps[hour - 1]["total_energy"], 1)

            if hour == 0:
                acceleration = 0.0
            else:
                acceleration = round(velocity - prev_velocity, 1)
            prev_velocity = velocity

            energy_state = map_energy_state(comp["total_energy"], velocity, p39_state, sym_totals)

            conf = confidence_at_checkpoint(
                sym, hour, timeline, p39_state, reasoning_by_sym,
                success_by_state, graph_rows, transitions, persistence_rows, graph_lookup,
            )

            curve_row = {
                "observation_id": obs_id,
                "symbol": sym,
                "checkpoint": f"T+{hour}h",
                "checkpoint_hour": hour,
                "p39_state": p39_state,
                "momentum_energy": comp["momentum_energy"],
                "structural_energy": comp["structural_energy"],
                "crowd_energy": comp["crowd_energy"],
                "persistence_energy": comp["persistence_energy"],
                "total_energy": comp["total_energy"],
                "energy_state": energy_state,
                "energy_velocity": velocity,
                "energy_acceleration": acceleration,
            }
            curve_rows.append(curve_row)
            curve_by_sym.setdefault(sym, []).append(curve_row)

            velocity_rows.append({
                "observation_id": obs_id,
                "symbol": sym,
                "checkpoint": f"T+{hour}h",
                "checkpoint_hour": hour,
                "total_energy": comp["total_energy"],
                "energy_velocity": velocity,
            })

            acceleration_rows.append({
                "observation_id": obs_id,
                "symbol": sym,
                "checkpoint": f"T+{hour}h",
                "checkpoint_hour": hour,
                "energy_velocity": velocity,
                "energy_acceleration": acceleration,
            })

            confidence_rows.append({
                "observation_id": obs_id,
                "symbol": sym,
                "checkpoint": f"T+{hour}h",
                "checkpoint_hour": hour,
                "p39_state": p39_state,
                **conf,
                "return_used": "no",
                "weight_modified": "no",
            })
            confidence_by_sym.setdefault(sym, []).append(confidence_rows[-1])

    evo_rows = expected_evolution_rows(obs_id, symbols, evolution, transition_counts)
    for sym in symbols:
        latest_conf = confidence_by_sym[sym][-1]["confidence"] if confidence_by_sym.get(sym) else ""
        for row in evo_rows:
            if row["symbol"] == sym:
                row["confidence"] = latest_conf

    energy_map_rows: list[dict] = []
    for sym in symbols:
        latest = curve_by_sym[sym][-1]
        latest_conf = confidence_by_sym[sym][-1]
        evo_row = next(r for r in evo_rows if r["symbol"] == sym)
        energy_map_rows.append({
            "observation_id": obs_id,
            "symbol": sym,
            "total_energy": latest["total_energy"],
            "momentum_energy": latest["momentum_energy"],
            "structural_energy": latest["structural_energy"],
            "crowd_energy": latest["crowd_energy"],
            "persistence_energy": latest["persistence_energy"],
            "confidence": latest_conf["confidence"],
            "energy_state": latest["energy_state"],
            "p39_state": latest["p39_state"],
            "energy_velocity": latest["energy_velocity"],
            "energy_acceleration": latest["energy_acceleration"],
            "expected_next_state": evo_row["observed_next_state"],
            "support_pct": evo_row["support_pct"],
            "observation_count": 1,
            "learning_status": "Experimental",
            "learning_recommendation": "NO_ACTION",
        })

    physics_rows = process_physics_rows(
        obs_id, curve_by_sym, confidence_by_sym, evolution, transitions,
    )

    write_csv(ENERGY_COMPONENTS_CSV, component_rows)
    write_csv(ENERGY_CURVE_CSV, curve_rows)
    write_csv(ENERGY_VELOCITY_CSV, velocity_rows)
    write_csv(ENERGY_ACCELERATION_CSV, acceleration_rows)
    write_csv(CONFIDENCE_CURVE_CSV, confidence_rows)
    write_csv(ENERGY_MAP_CSV, energy_map_rows)
    write_csv(EXPECTED_EVOLUTION_CSV, evo_rows)
    write_csv(PROCESS_PHYSICS_CSV, physics_rows)

    report = build_report(obs_id, energy_map_rows, physics_rows, evo_rows)
    PROCESS_REPORT_TXT.write_text(report, encoding="utf-8")

    print(
        f"Saved P42 outputs | curve={len(curve_rows)} confidence={len(confidence_rows)} "
        f"physics={len(physics_rows)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="P42 Energy & Confidence Engine")
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
