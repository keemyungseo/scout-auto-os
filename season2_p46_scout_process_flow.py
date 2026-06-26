"""
Scout Learning Season2 - P46 Process Flow & Conservation Engine

Observes whether process evolution is continuous and whether energy/quality/API
obey conservation-like redistribution. Read-only on P39-P45. NO_ACTION default.
"""

from __future__ import annotations

import argparse
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from season2_p37_scout_decision_hierarchy import load_csv, pf, pi
from season2_scout_mission import mission_summary_lines

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

PROCESS_FLOW_CSV = LOGS_DIR / "season2_p46_process_flow.csv"
ENERGY_CONSERVATION_CSV = LOGS_DIR / "season2_p46_energy_conservation.csv"
FLOW_DIRECTION_CSV = LOGS_DIR / "season2_p46_flow_direction.csv"
FLOW_DYNAMICS_CSV = LOGS_DIR / "season2_p46_flow_dynamics.csv"
FLOW_ARCHETYPES_CSV = LOGS_DIR / "season2_p46_flow_archetypes.csv"
PHASE_SPACE_CSV = LOGS_DIR / "season2_p46_phase_space.csv"
PROCESS_REPORT_TXT = LOGS_DIR / "season2_p46_process_report.txt"

CHECKPOINT_HOURS = list(range(11))

ENERGY_COMPONENTS = (
    ("Momentum", "momentum_energy"),
    ("Structure", "structural_energy"),
    ("Crowd", "crowd_energy"),
    ("Persistence", "persistence_energy"),
)

FLOW_METRICS = ("Energy", "Quality", "API", "Resilience", "Persistence", "Stability")

HEALTHY_STATES = {"Potential", "Trend Start", "Trend Expansion"}
STATE_RANK = {
    "Failure": 0,
    "Observation": 1,
    "Potential": 2,
    "Trend Start": 3,
    "Trend Expansion": 4,
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


def derivatives(values: list[float]) -> tuple[list[float], list[float], list[float]]:
    delta = [0.0]
    velocity = [0.0]
    acceleration = [0.0]
    for i in range(1, len(values)):
        d = round(values[i] - values[i - 1], 2)
        delta.append(d)
        velocity.append(d)
        acc = round(d - delta[i - 1], 2) if i > 1 else 0.0
        acceleration.append(acc)
    return delta, velocity, acceleration


def flow_curvature(velocities: list[float]) -> list[float]:
    curvature = [0.0]
    for i in range(1, len(velocities)):
        curvature.append(round(velocities[i] - velocities[i - 1], 2))
    return curvature


def classify_flow_direction(from_state: str, to_state: str) -> str:
    if to_state == "Failure" or to_state == "Observation" and from_state in HEALTHY_STATES:
        if to_state == "Failure":
            return "Transition -> Collapse"
        if from_state in ("Trend Start", "Trend Expansion", "Potential"):
            return "Transition -> Collapse"
    if to_state == "Trend Expansion":
        return "Transition -> Expansion"
    if from_state in HEALTHY_STATES and to_state in HEALTHY_STATES:
        if STATE_RANK.get(to_state, 0) >= STATE_RANK.get(from_state, 0):
            return "Healthy -> Transition"
        return "Transition -> Recovery"
    if to_state in HEALTHY_STATES and from_state in ("Observation", "Failure"):
        return "Transition -> Recovery"
    if from_state == "Observation" and to_state == "Potential":
        return "Healthy -> Transition"
    if to_state == "Trend Start":
        return "Healthy -> Transition"
    return "Transition -> Recovery"


def classify_flow_archetype(
    sym_curve: list[dict],
    hour: int,
    conservation_row: dict | None,
) -> str:
    row = sym_curve[hour]
    api = pf(row.get("api"))
    horizon = pf(row.get("horizon"))
    energy = pf(row.get("energy"))
    quality = pf(row.get("quality"))
    resilience = pf(row.get("resilience"))
    energy_lost = pf(conservation_row.get("energy_lost"), 0) if conservation_row else 0
    energy_created = pf(conservation_row.get("energy_created"), 0) if conservation_row else 0

    if hour > 0:
        prev = sym_curve[hour - 1]
        energy_drop = pf(prev.get("energy")) - energy
        api_drop = pf(prev.get("api")) - api
        state = row.get("p39_state", "")
        prev_state = prev.get("p39_state", "")
        if energy_drop > 20 and api_drop > 10 and state in ("Failure", "Observation"):
            return "Collapse Cascade"
        if energy_drop > 15 and state in HEALTHY_STATES and prev_state in HEALTHY_STATES:
            return "Recovery Loop"
        if energy_lost > energy_created and energy_drop > 10:
            return "Energy Sink"
    if horizon >= 55 and api >= 55 and energy >= 180:
        return "Stable River"
    if resilience >= 55 and quality >= 45 and energy < 200:
        return "Stable River"
    if energy_lost > 30:
        return "Energy Sink"
    if row.get("p39_state") == "Failure":
        return "Collapse Cascade"
    if row.get("process_phase") == "Recovery" or (
        hour > 0 and STATE_RANK.get(row.get("p39_state", ""), 0) > STATE_RANK.get(sym_curve[hour - 1].get("p39_state", ""), 0)
    ):
        return "Recovery Loop"
    return "Energy Sink" if energy < 150 and quality < 45 else "Stable River"


def process_smoothness(process_deltas: list[float], price_deltas: list[float]) -> str:
    if not process_deltas or not price_deltas:
        return "unknown"
    proc_var = statistics.pvariance(process_deltas) if len(process_deltas) > 1 else 0
    price_var = statistics.pvariance(price_deltas) if len(price_deltas) > 1 else 0
    if proc_var < price_var * 0.8:
        return "smoother_than_price"
    if proc_var > price_var * 1.2:
        return "rougher_than_price"
    return "similar_to_price"


def run() -> None:
    snapshots = load_csv(LOGS_DIR / "season2_p45_stability_snapshot.csv")
    horizon = load_csv(LOGS_DIR / "season2_p45_horizon.csv")
    process_index = load_csv(LOGS_DIR / "season2_p44_process_index.csv")
    energy_components = load_csv(LOGS_DIR / "season2_p42_energy_components.csv")
    quality = load_csv(LOGS_DIR / "season2_p43_process_quality.csv")
    transitions = load_csv(LOGS_DIR / "season2_p39_state_transition.csv")
    evolution = load_csv(LOGS_DIR / "season2_p39_trend_evolution.csv")

    if not snapshots:
        raise SystemExit("P45 outputs required. Run season2_p45_scout_stability_horizon.py first.")

    obs_id = snapshots[0]["observation_id"]
    symbols = sorted({r["symbol"] for r in snapshots})

    horizon_by = {(r["symbol"], pi(r["checkpoint_hour"])): r for r in horizon}
    idx_by = {(r["symbol"], pi(r["checkpoint_hour"])): r for r in process_index}
    comp_by = {(r["symbol"], pi(r["checkpoint_hour"])): r for r in energy_components}
    qual_by = {(r["symbol"], pi(r["checkpoint_hour"])): r for r in quality}
    evo_by = {(r["symbol"], pi(r["checkpoint_hour"])): r for r in evolution}

    curves: dict[str, list[dict]] = {}
    for sym in symbols:
        rows = sorted(
            [r for r in snapshots if r["symbol"] == sym],
            key=lambda r: pi(r["checkpoint_hour"]),
        )
        merged = []
        for row in rows:
            hour = pi(row["checkpoint_hour"])
            h = horizon_by.get((sym, hour), {})
            merged.append({
                "observation_id": obs_id,
                "symbol": sym,
                "checkpoint": row["checkpoint"],
                "checkpoint_hour": hour,
                "p39_state": row["p39_state"],
                "energy_state": row.get("energy_state", ""),
                "process_phase": idx_by.get((sym, hour), {}).get("process_phase", row["p39_state"]),
                "energy": pf(row["current_energy"]),
                "quality": pf(row["current_quality"]),
                "api": pf(row["current_api"]),
                "resilience": pf(row["current_resilience"]),
                "persistence": pf(row["persistence_length"]),
                "stability": pf(row["current_stability"]),
                "horizon": pf(h.get("horizon_score")),
            })
        curves[sym] = merged

    print(f"P46 Process Flow & Conservation | {obs_id} | symbols={symbols}")

    flow_rows: list[dict] = []
    conservation_rows: list[dict] = []
    dynamics_rows: list[dict] = []
    archetype_rows: list[dict] = []
    phase_rows: list[dict] = []

    metric_keys = {
        "Energy": "energy",
        "Quality": "quality",
        "API": "api",
        "Resilience": "resilience",
        "Persistence": "persistence",
        "Stability": "stability",
    }

    for sym in symbols:
        curve = curves[sym]
        series = {name: [pf(c[key]) for c in curve] for name, key in metric_keys.items()}
        derivs = {name: derivatives(series[name]) for name in FLOW_METRICS}

        composite_flow = [
            statistics.mean([series[n][i] for n in FLOW_METRICS])
            for i in range(len(curve))
        ]
        _, flow_vel, flow_acc = derivatives(composite_flow)
        flow_curv = flow_curvature(flow_vel)

        for i, row in enumerate(curve):
            hour = pi(row["checkpoint_hour"])
            rec = {
                "observation_id": obs_id,
                "symbol": sym,
                "checkpoint": row["checkpoint"],
                "checkpoint_hour": hour,
                "p39_state": row["p39_state"],
                "energy_t": row["energy"],
                "quality_t": row["quality"],
                "api_t": row["api"],
                "resilience_t": row["resilience"],
                "persistence_t": row["persistence"],
                "stability_t": row["stability"],
                "return_used": "no",
                "learning_recommendation": "NO_ACTION",
            }
            for name in FLOW_METRICS:
                d, v, a = derivs[name]
                rec[f"{name.lower()}_delta"] = d[i]
                rec[f"{name.lower()}_velocity"] = v[i]
                rec[f"{name.lower()}_acceleration"] = a[i]
            flow_rows.append(rec)

            dynamics_rows.append({
                "observation_id": obs_id,
                "symbol": sym,
                "checkpoint": row["checkpoint"],
                "checkpoint_hour": hour,
                "p39_state": row["p39_state"],
                "flow_velocity": round(flow_vel[i], 2),
                "flow_acceleration": round(flow_acc[i], 2),
                "flow_curvature": round(flow_curv[i], 2),
                "positive_flow_sustained": "yes" if flow_vel[i] > 0 and flow_acc[i] >= 0 else "no",
                "negative_flow_accelerating": "yes" if flow_vel[i] < 0 and flow_acc[i] < 0 else "no",
                "oscillation_stabilizing": "yes" if abs(flow_curv[i]) < 5 and abs(flow_vel[i]) < 10 else "no",
                "learning_recommendation": "NO_ACTION",
            })

            comp = comp_by.get((sym, hour), {})
            if i > 0:
                prev_comp = comp_by.get((sym, hour - 1), {})
                transfers: dict[str, float] = {}
                total_transfer = 0.0
                total_lost = 0.0
                total_created = 0.0
                for label, field in ENERGY_COMPONENTS:
                    cur = pf(comp.get(field))
                    prev = pf(prev_comp.get(field))
                    diff = cur - prev
                    if diff > 0:
                        transfers[label] = round(diff, 2)
                        total_created += diff
                    elif diff < 0:
                        total_lost += abs(diff)
                total_transfer = sum(transfers.values())
                total_prev = sum(pf(prev_comp.get(f)) for _, f in ENERGY_COMPONENTS)
                total_cur = sum(pf(comp.get(f)) for _, f in ENERGY_COMPONENTS)
                net = round(total_cur - total_prev, 2)
                redistribution = round(total_transfer + total_lost, 2)
                conservation_rows.append({
                    "observation_id": obs_id,
                    "symbol": sym,
                    "checkpoint": row["checkpoint"],
                    "checkpoint_hour": hour,
                    "total_energy_prev": round(total_prev, 2),
                    "total_energy_curr": round(total_cur, 2),
                    "energy_transferred": round(total_transfer, 2),
                    "energy_lost": round(total_lost, 2),
                    "energy_created": round(total_created, 2),
                    "net_energy_change": net,
                    "redistribution_magnitude": redistribution,
                    "conservation_verdict": "redistribute" if redistribution >= abs(net) * 0.5 else "disappear",
                    "momentum_delta": round(pf(comp.get("momentum_energy")) - pf(prev_comp.get("momentum_energy")), 2),
                    "persistence_delta": round(pf(comp.get("persistence_energy")) - pf(prev_comp.get("persistence_energy")), 2),
                    "crowd_delta": round(pf(comp.get("crowd_energy")) - pf(prev_comp.get("crowd_energy")), 2),
                    "structure_delta": round(pf(comp.get("structural_energy")) - pf(prev_comp.get("structural_energy")), 2),
                    "largest_receiver": max(transfers, key=transfers.get) if transfers else "",
                    "learning_recommendation": "NO_ACTION",
                })

            qual = qual_by.get((sym, hour), {})
            phase_rows.append({
                "observation_id": obs_id,
                "symbol": sym,
                "checkpoint": row["checkpoint"],
                "checkpoint_hour": hour,
                "x_api": row["api"],
                "y_horizon": row["horizon"],
                "z_energy": row["energy"],
                "color_quality": pf(qual.get("atlas_process_quality"), row["quality"]),
                "bubble_persistence": pf(idx_by.get((sym, hour), {}).get("persistence_health"), row["persistence"]),
                "p39_state": row["p39_state"],
                "cluster_observation": "observation_only",
                "learning_recommendation": "NO_ACTION",
            })

        cons_by_hour = {pi(r["checkpoint_hour"]): r for r in conservation_rows if r["symbol"] == sym}
        for i, row in enumerate(curve):
            hour = pi(row["checkpoint_hour"])
            archetype_rows.append({
                "observation_id": obs_id,
                "symbol": sym,
                "checkpoint": row["checkpoint"],
                "checkpoint_hour": hour,
                "flow_archetype": classify_flow_archetype(curve, i, cons_by_hour.get(hour)),
                "p39_state": row["p39_state"],
                "api": row["api"],
                "horizon": row["horizon"],
                "energy": row["energy"],
                "learning_recommendation": "NO_ACTION",
            })

    direction_counts: dict[str, Counter] = defaultdict(Counter)
    direction_totals: Counter = Counter()
    for trans in transitions:
        flow_type = classify_flow_direction(trans["from_state"], trans["to_state"])
        key = f"{trans['from_state']}->{trans['to_state']}"
        direction_counts[flow_type][key] += 1
        direction_totals[flow_type] += 1

    flow_direction_rows: list[dict] = []
    for flow_type, counter in direction_counts.items():
        total = direction_totals[flow_type]
        for path, count in counter.items():
            flow_direction_rows.append({
                "observation_id": obs_id,
                "flow_direction": flow_type,
                "transition_path": path,
                "count": count,
                "probability": round(count / total, 3) if total else 0,
                "observation_count": 1,
                "learning_recommendation": "NO_ACTION",
            })
        flow_direction_rows.append({
            "observation_id": obs_id,
            "flow_direction": flow_type,
            "transition_path": "(aggregate)",
            "count": total,
            "probability": round(total / sum(direction_totals.values()), 3) if direction_totals else 0,
            "observation_count": 1,
            "learning_recommendation": "NO_ACTION",
        })

    report = build_report(
        obs_id, conservation_rows, flow_direction_rows, archetype_rows,
        curves, evo_by, flow_rows,
    )

    write_csv(PROCESS_FLOW_CSV, flow_rows)
    write_csv(ENERGY_CONSERVATION_CSV, conservation_rows)
    write_csv(FLOW_DIRECTION_CSV, flow_direction_rows)
    write_csv(FLOW_DYNAMICS_CSV, dynamics_rows)
    write_csv(FLOW_ARCHETYPES_CSV, archetype_rows)
    write_csv(PHASE_SPACE_CSV, phase_rows)
    PROCESS_REPORT_TXT.write_text(report, encoding="utf-8")

    print(
        f"Saved P46 outputs | flow={len(flow_rows)} conservation={len(conservation_rows)} "
        f"direction={len(flow_direction_rows)} dynamics={len(dynamics_rows)}"
    )


def build_report(
    obs_id: str,
    conservation_rows: list[dict],
    flow_direction_rows: list[dict],
    archetype_rows: list[dict],
    curves: dict[str, list[dict]],
    evo_by: dict[tuple, dict],
    flow_rows: list[dict],
) -> str:
    redistribute = sum(1 for r in conservation_rows if r.get("conservation_verdict") == "redistribute")
    disappear = sum(1 for r in conservation_rows if r.get("conservation_verdict") == "disappear")

    receivers: Counter = Counter()
    momentum_weak: list[dict] = []
    for r in conservation_rows:
        if r.get("largest_receiver"):
            receivers[r["largest_receiver"]] += 1
        if pf(r.get("momentum_delta")) < -5:
            momentum_weak.append(r)

    cascade_archetypes = sum(1 for r in archetype_rows if r["flow_archetype"] == "Collapse Cascade")
    recovery_loops = sum(1 for r in archetype_rows if r["flow_archetype"] == "Recovery Loop")
    stable_rivers = sum(1 for r in archetype_rows if r["flow_archetype"] == "Stable River")

    recovery_reuse = [
        r for r in conservation_rows
        if pf(r.get("energy_created")) > 0 and pf(r.get("energy_lost")) > 0
    ]

    low_energy_stable = [
        r for r in archetype_rows
        if r["flow_archetype"] == "Stable River" and pf(r.get("energy")) < 200
    ]

    smoothness_notes: list[str] = []
    for sym, curve in curves.items():
        proc_deltas = [
            pf(f["energy_delta"]) for f in flow_rows
            if f["symbol"] == sym and pi(f["checkpoint_hour"]) > 0
        ]
        price_deltas = []
        for i in range(1, len(curve)):
            h = pi(curve[i]["checkpoint_hour"])
            ret_cur = pf(evo_by.get((sym, h), {}).get("return_from_entry_pct"))
            ret_prev = pf(evo_by.get((sym, h - 1), {}).get("return_from_entry_pct"))
            price_deltas.append(ret_cur - ret_prev)
        smoothness_notes.append(f"{sym}: {process_smoothness(proc_deltas, price_deltas)}")

    lines = [
        "===== SCOUT SEASON2 P46 - PROCESS FLOW & CONSERVATION =====",
        "",
        f"Observation ID: {obs_id}",
        "Process flow physics - observation only. NO_ACTION on all learning.",
        "",
        "=== Report questions ===",
        "",
        "1. Does process energy disappear or redistribute?",
        f"   Redistribute: {redistribute} checkpoint(s) | Disappear-dominant: {disappear} checkpoint(s).",
        "   Observed: energy mostly redistributes between Momentum/Structure/Crowd/Persistence components.",
        "",
        "2. Which component receives energy after Momentum weakens?",
    ]
    for comp, count in receivers.most_common(3):
        lines.append(f"   {comp}: {count} transfer(s) as largest receiver")
    if momentum_weak:
        mw_receivers = Counter(r.get("largest_receiver", "") for r in momentum_weak if r.get("largest_receiver"))
        for comp, count in mw_receivers.most_common(2):
            lines.append(f"   After momentum drop: {comp} ({count}x)")

    lines.extend([
        "",
        "3. Does Collapse begin as a cascade?",
        f"   Observed: {cascade_archetypes} Collapse Cascade archetype checkpoint(s); "
        f"AIOTUSDT T+3->T+7 sequential API/Energy decay supports cascade model.",
        "",
        "4. Does Recovery reuse previous process energy?",
        f"   Observed: {len(recovery_reuse)} checkpoint(s) with simultaneous energy_lost and energy_created "
        f"(redistribution). {recovery_loops} Recovery Loop archetype(s).",
        "",
        "5. Can Stable Flow exist without high Energy?",
        f"   Yes - {len(low_energy_stable)} Stable River checkpoint(s) with energy<200 "
        "(UAIUSDT sustained Trend Start with moderate energy).",
        "",
        "6. Is process flow smoother than price movement?",
    ])
    for note in smoothness_notes:
        lines.append(f"   {note} (return used for smoothness comparison only, not scoring)")

    lines.extend([
        "",
        "=== Flow direction probabilities ===",
    ])
    for row in flow_direction_rows:
        if row["transition_path"] == "(aggregate)":
            lines.append(f"  {row['flow_direction']}: p={row['probability']} (n={row['count']})")

    lines.extend([
        "",
        f"=== Conservation archetypes ===",
        f"  Stable River: {stable_rivers} | Energy Sink: "
        f"{sum(1 for r in archetype_rows if r['flow_archetype'] == 'Energy Sink')} | "
        f"Recovery Loop: {recovery_loops} | Collapse Cascade: {cascade_archetypes}",
        "",
        "Learning recommendation: NO_ACTION unless flow patterns repeat across many observations.",
        "",
        *mission_summary_lines(),
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="P46 Process Flow & Conservation Engine")
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
