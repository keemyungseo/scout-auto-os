"""
Scout Learning Season2 - P61 Cross-Symbol Attention Migration Engine

Tests whether Trend Death is Attention Migration. P39-P60 process variables only.
STRICT NO_ACTION | NO_API | NO_PRICE. Pure Python.
"""

from __future__ import annotations

import argparse
import itertools
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from season2_p37_scout_decision_hierarchy import load_csv, pf, pi
from season2_p60_scout_attention_field import (
    infer_attention_score,
    loocv_rmse,
    pearson,
    path_score,
    ridge_regress,
    rmse,
    shannon_entropy,
    write_csv,
)
from season2_scout_mission import mission_summary_lines

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

MIGRATION_CSV = LOGS_DIR / "season2_p61_attention_migration.csv"
MATRIX_CSV = LOGS_DIR / "season2_p61_migration_matrix.csv"
GRAPH_CSV = LOGS_DIR / "season2_p61_migration_graph.csv"
HIDDEN_CSV = LOGS_DIR / "season2_p61_hidden_pairs.csv"
ROTATION_CSV = LOGS_DIR / "season2_p61_rotation.csv"
WAVE_CSV = LOGS_DIR / "season2_p61_wave.csv"
NETWORK_CSV = LOGS_DIR / "season2_p61_network.csv"
LAWS_CSV = LOGS_DIR / "season2_p61_laws.csv"
PROCESS_REPORT_TXT = LOGS_DIR / "season2_p61_process_report.txt"

TREND_STATES = {"Trend Start", "Trend Expansion"}


def kmeans(points: list[list[float]], k: int, iters: int = 40) -> tuple[list[int], list[list[float]]]:
    n = len(points)
    k = min(k, n)
    step = max(1, n // k)
    centers = [points[i * step % n][:] for i in range(k)]
    labels = [0] * n
    for _ in range(iters):
        for i, pt in enumerate(points):
            labels[i] = min(
                range(k),
                key=lambda c: sum((pt[d] - centers[c][d]) ** 2 for d in range(len(pt))),
            )
        for c in range(k):
            cluster = [points[i] for i in range(n) if labels[i] == c]
            if cluster:
                centers[c] = [statistics.mean(pt[d] for pt in cluster) for d in range(len(points[0]))]
    return labels, centers


def load_rows() -> tuple[str, dict[str, list[dict]]]:
    order = load_csv(LOGS_DIR / "season2_p49_order_parameter.csv")
    narrative = load_csv(LOGS_DIR / "season2_p59_narrative_field.csv")
    belief = load_csv(LOGS_DIR / "season2_p58_belief_field.csv")
    motivation = load_csv(LOGS_DIR / "season2_p54_motivation_field.csv")
    goals = load_csv(LOGS_DIR / "season2_p54_goal_distribution.csv")
    future = load_csv(LOGS_DIR / "season2_p52_future_distribution.csv")
    ecology = load_csv(LOGS_DIR / "season2_p57_ecology_entropy.csv")
    dynamics = load_csv(LOGS_DIR / "season2_p57_population_dynamics.csv")
    force = load_csv(LOGS_DIR / "season2_p57_collective_force.csv")
    kernel = load_csv(LOGS_DIR / "season2_p51_kernel.csv")
    attention = load_csv(LOGS_DIR / "season2_p60_attention_field.csv")

    if not order:
        raise SystemExit("P49 order parameter required.")

    obs_id = order[0]["observation_id"]
    nar_by = {(r["symbol"], r["checkpoint"]): r for r in narrative}
    belief_by = {(r["symbol"], r["checkpoint"]): r for r in belief}
    mot_by = {(r["symbol"], r["checkpoint"]): r for r in motivation}
    goal_by = {(r["symbol"], r["checkpoint"]): r for r in goals}
    fut_by = {(r["symbol"], r["checkpoint"]): r for r in future}
    eco_by = {(r["symbol"], r["checkpoint"]): r for r in ecology}
    dyn_by = {(r["symbol"], r["checkpoint"]): r for r in dynamics if r.get("replacement_rate") is not None}
    force_by = {(r["symbol"], r["checkpoint"]): r for r in force if r.get("collective_force")}
    att_by = {(r["symbol"], r["checkpoint"]): r for r in attention}
    mem = sum(pf(r.get("kernel_weight_normalized")) for r in kernel) / max(len(kernel), 1)

    by_sym: dict[str, list[dict]] = defaultdict(list)
    for r in order:
        sym, cp = r["symbol"], r["checkpoint"]
        bf = belief_by.get((sym, cp), {})
        nar = nar_by.get((sym, cp), {})
        mot = mot_by.get((sym, cp), {})
        goal = goal_by.get((sym, cp), {})
        fut = fut_by.get((sym, cp), {})
        eco = eco_by.get((sym, cp), {})
        dyn = dyn_by.get((sym, cp), {})
        fo = force_by.get((sym, cp), {})
        att = att_by.get((sym, cp), {})
        rec = {
            "observation_id": obs_id,
            "symbol": sym,
            "checkpoint": cp,
            "checkpoint_hour": pi(r["checkpoint_hour"]),
            "p39_state": r["p39_state"],
            "Memory": mem,
            "Potential": pf(r["var_Potential"]),
            "API": pf(r["var_API"]),
            "Quality": pf(r["var_Quality"]),
            "GoalConcentration": pf(goal.get("goal_concentration")),
            "GoalPolarization": pf(goal.get("goal_polarization")),
            "GoalEntropy": pf(goal.get("goal_entropy")),
            "Entropy": pf(fut.get("future_entropy")),
            "EcologyEntropy": pf(eco.get("ecology_entropy")),
            "BeliefConsensus": pf(bf.get("belief_consensus")),
            "NarrativeScore": pf(nar.get("narrative_score")),
            "Motivation": pf(mot.get("motivation_density")),
            "Flow": pf(r["var_FlowVelocity"]),
            "Participation": pf(r["var_FlowVelocity"]) * pf(r["var_Persistence"]) / 100.0,
            "OrderParameter": pf(r["order_parameter_score"]),
            "CollapseRisk": pf(fut.get("prob_collapse")),
            "ReplacementRate": pf(dyn.get("replacement_rate")),
            "CollectiveForce": pf(fo.get("collective_force")),
            "trend_indicator": 1.0 if r["p39_state"] in TREND_STATES else 0.0,
            "AttentionScore": pf(att.get("attention_score")),
            "AttentionShare": pf(att.get("attention_share")),
        }
        if rec["AttentionScore"] == 0:
            by_hour_stub = rec
            rec["AttentionScore"] = infer_attention_score(by_hour_stub, 1.0)
        by_sym[sym].append(rec)

    for sym in by_sym:
        by_sym[sym].sort(key=lambda x: x["checkpoint_hour"])

    by_hour: dict[int, list[dict]] = defaultdict(list)
    for sym_rows in by_sym.values():
        for r in sym_rows:
            by_hour[r["checkpoint_hour"]].append(r)
    for hour, rows_at_h in by_hour.items():
        total = sum(r["Potential"] + r["API"] + r["Quality"] for r in rows_at_h) or 1.0
        for r in rows_at_h:
            share = (r["Potential"] + r["API"] + r["Quality"]) / total
            if r.get("AttentionShare", 0) == 0:
                r["AttentionShare"] = share
            if r["AttentionScore"] <= 0:
                r["AttentionScore"] = infer_attention_score(r, 0.5 + share)

    return obs_id, dict(by_sym)


def infer_transfer(from_prev: dict, from_cur: dict, to_prev: dict, to_cur: dict) -> float:
    outflow = max(0.0, from_prev["AttentionScore"] - from_cur["AttentionScore"])
    inflow = max(0.0, to_cur["AttentionScore"] - to_prev["AttentionScore"])
    if outflow <= 0 or inflow <= 0:
        return 0.0
    raw = min(outflow, inflow)
    narrative_pull = abs(from_cur["NarrativeScore"] - to_cur["NarrativeScore"]) / 100.0
    belief_pull = abs(from_cur["BeliefConsensus"] - to_cur["BeliefConsensus"]) / 100.0
    salience = (to_cur["Potential"] + to_cur["API"]) / max(from_cur["Potential"] + from_cur["API"], 1.0)
    return raw * (0.6 + 0.2 * narrative_pull + 0.2 * belief_pull) * min(2.0, salience)


def infer_archetype(rate: float, net: float, reversal: int, explosive: float) -> str:
    if explosive > 15:
        return "inferred_explosive_transfer"
    if reversal:
        return "inferred_rotation"
    if rate > 8 and net < -3:
        return "inferred_attention_drain"
    if rate > 5 and net > 3:
        return "inferred_leader_shift"
    if rate > 4:
        return "inferred_cascade"
    if net > 0 and rate < 2:
        return "inferred_recovery_return"
    return f"inferred_migration_{int(rate // 3)}"


def run() -> None:
    obs_id, by_sym = load_rows()
    symbols = sorted(by_sym.keys())
    all_rows = [r for sym in by_sym for r in by_sym[sym]]
    n = len(all_rows)
    print(f"P61 Cross-Symbol Attention Migration Engine | {obs_id} | n={n}")

    # --- Q1: Migration matrix ---
    matrix_rows: list[dict] = []
    migration_rows: list[dict] = []
    graph_rows: list[dict] = []

    for hour in sorted({r["checkpoint_hour"] for r in all_rows}):
        if hour == 0:
            continue
        prev_hour = hour - 1
        prev_by_sym = {s: next(r for r in by_sym[s] if r["checkpoint_hour"] == prev_hour) for s in symbols}
        cur_by_sym = {s: next(r for r in by_sym[s] if r["checkpoint_hour"] == hour) for s in symbols}

        for src in symbols:
            for dst in symbols:
                if src == dst:
                    continue
                transfer = infer_transfer(prev_by_sym[src], cur_by_sym[src], prev_by_sym[dst], cur_by_sym[dst])
                pool = sum(r["AttentionScore"] for r in cur_by_sym.values()) or 1.0
                rate = transfer / pool
                direction = f"{src}->{dst}"
                strength = transfer / max(prev_by_sym[src]["AttentionScore"], 1.0)
                matrix_rows.append({
                    "observation_id": obs_id,
                    "checkpoint_hour": hour,
                    "from_symbol": src,
                    "to_symbol": dst,
                    "migration_rate": round(rate, 4),
                    "migration_direction": direction,
                    "migration_strength": round(strength, 4),
                    "transfer_amount": round(transfer, 4),
                    "learning_recommendation": "NO_ACTION",
                })

        for sym in symbols:
            incoming = sum(
                infer_transfer(prev_by_sym[s], cur_by_sym[s], prev_by_sym[sym], cur_by_sym[sym])
                for s in symbols if s != sym
            )
            outgoing = sum(
                infer_transfer(prev_by_sym[sym], cur_by_sym[sym], prev_by_sym[d], cur_by_sym[d])
                for d in symbols if d != sym
            )
            net = incoming - outgoing
            d_att = cur_by_sym[sym]["AttentionScore"] - prev_by_sym[sym]["AttentionScore"]
            graph_rows.append({
                "observation_id": obs_id,
                "symbol": sym,
                "checkpoint": cur_by_sym[sym]["checkpoint"],
                "checkpoint_hour": hour,
                "incoming_attention": round(incoming, 4),
                "outgoing_attention": round(outgoing, 4),
                "net_attention": round(net, 4),
                "attention_delta": round(d_att, 4),
                "conservation_residual": round(d_att - net, 4),
                "learning_recommendation": "NO_ACTION",
            })
            migration_rows.append({
                "observation_id": obs_id,
                "symbol": sym,
                "checkpoint": cur_by_sym[sym]["checkpoint"],
                "checkpoint_hour": hour,
                "p39_state": cur_by_sym[sym]["p39_state"],
                "attention_score": round(cur_by_sym[sym]["AttentionScore"], 2),
                "incoming_attention": round(incoming, 4),
                "outgoing_attention": round(outgoing, 4),
                "net_attention": round(net, 4),
                "migration_rate": round(outgoing / max(cur_by_sym[sym]["AttentionScore"], 1), 4),
                "learning_recommendation": "NO_ACTION",
            })

    total_transfer = sum(r["transfer_amount"] for r in matrix_rows)
    relocates = total_transfer > 0.5

    # --- Q3: Migration before collapse ---
    collapse_paths: list[tuple[str, float]] = []
    for sym, sym_rows in by_sym.items():
        series = {
            "TrendAlive": [1.0 - r["CollapseRisk"] for r in sym_rows],
            "AttentionOutflow": [0.0] + [
                max(0, sym_rows[i - 1]["AttentionScore"] - sym_rows[i]["AttentionScore"])
                for i in range(1, len(sym_rows))
            ],
            "FlowReduction": [0.0] + [
                max(0, sym_rows[i - 1]["Flow"] - sym_rows[i]["Flow"])
                for i in range(1, len(sym_rows))
            ],
            "Collapse": [r["CollapseRisk"] for r in sym_rows],
        }
        path_a = ["TrendAlive", "AttentionOutflow", "FlowReduction", "Collapse"]
        path_b = ["TrendAlive", "FlowReduction", "AttentionOutflow", "Collapse"]
        for name, path in [("Migration_first", path_a), ("Flow_first", path_b)]:
            corrs = [pearson(series[path[i]], series[path[i + 1]]) for i in range(len(path) - 1)]
            collapse_paths.append((name, path_score(corrs)))
    best_collapse_path = max(collapse_paths, key=lambda x: x[1]) if collapse_paths else ("unknown", 0)

    # --- Q4: Migration archetypes ---
    rotation_rows: list[dict] = []
    archetype_feats: list[list[float]] = []
    archetype_meta: list[dict] = []
    prev_leader: str | None = None
    for hour in sorted({r["checkpoint_hour"] for r in graph_rows}):
        at_hour = [r for r in graph_rows if r["checkpoint_hour"] == hour]
        if len(at_hour) < 2:
            continue
        by_s = {r["symbol"]: r for r in at_hour}
        s0, s1 = symbols[0], symbols[1]
        r0, r1 = by_s[s0], by_s[s1]
        leader = s0 if r0["net_attention"] >= r1["net_attention"] else s1
        reversal = 1 if prev_leader and leader != prev_leader else 0
        rate = r0["outgoing_attention"] + r1["outgoing_attention"]
        net = r0["net_attention"] - r1["net_attention"]
        explosive = max(r0["incoming_attention"], r1["incoming_attention"])
        arch = infer_archetype(rate, net, reversal, explosive)
        archetype_feats.append([rate, net, reversal, explosive, r0["conservation_residual"]])
        archetype_meta.append({"hour": hour, "arch": arch, "leader": leader, "reversal": reversal})
        rotation_rows.append({
            "observation_id": obs_id,
            "checkpoint_hour": hour,
            "leader_symbol": leader,
            "attention_reversal": "yes" if reversal else "no",
            "total_outflow": round(rate, 4),
            "net_attention_spread": round(abs(net), 4),
            "migration_archetype": arch,
            "learning_recommendation": "NO_ACTION",
        })
        prev_leader = leader

    labels, _ = kmeans(archetype_feats, min(5, len(archetype_feats))) if archetype_feats else ([], [])
    arch_counts = Counter(m["arch"] for m in archetype_meta)

    # --- Q5: Hidden migration pairs ---
    hidden_rows: list[dict] = []
    cps: list[dict] = []
    for sym, sym_rows in by_sym.items():
        for i, r in enumerate(sym_rows):
            if i >= len(sym_rows) - 1:
                continue
            gr = next((g for g in graph_rows if g["symbol"] == sym and g["checkpoint"] == r["checkpoint"]), None)
            cps.append({
                "symbol": sym,
                "checkpoint": r["checkpoint"],
                "narrative": r["NarrativeScore"],
                "belief": r["BeliefConsensus"],
                "flow": r["Flow"],
                "net_migration": gr["net_attention"] if gr else 0.0,
                "out_migration": gr["outgoing_attention"] if gr else 0.0,
                "next_op": sym_rows[i + 1]["OrderParameter"],
            })
    for i in range(len(cps)):
        for j in range(i + 1, len(cps)):
            a, b = cps[i], cps[j]
            if abs(a["narrative"] - b["narrative"]) > 10:
                continue
            if abs(a["belief"] - b["belief"]) > 12:
                continue
            if abs(a["flow"] - b["flow"]) > 15:
                continue
            if abs(a["net_migration"] - b["net_migration"]) < 2:
                continue
            if abs(a["next_op"] - b["next_op"]) < 12:
                continue
            hidden_rows.append({
                "observation_id": obs_id,
                "symbol_a": a["symbol"],
                "checkpoint_a": a["checkpoint"],
                "symbol_b": b["symbol"],
                "checkpoint_b": b["checkpoint"],
                "narrative_distance": round(abs(a["narrative"] - b["narrative"]), 2),
                "belief_distance": round(abs(a["belief"] - b["belief"]), 2),
                "flow_distance": round(abs(a["flow"] - b["flow"]), 2),
                "migration_field_distance": round(abs(a["net_migration"] - b["net_migration"]), 2),
                "future_op_distance": round(abs(a["next_op"] - b["next_op"]), 2),
                "hidden_migration_detected": "yes",
                "learning_recommendation": "NO_ACTION",
            })

    # --- Q6: Migration wave ---
    wave_rows: list[dict] = []
    for hour in sorted({r["checkpoint_hour"] for r in matrix_rows}):
        edges = [r for r in matrix_rows if r["checkpoint_hour"] == hour]
        total_rate = sum(r["migration_rate"] for r in edges)
        prev_rates = [r for r in matrix_rows if r["checkpoint_hour"] == hour - 1]
        prev_total = sum(r["migration_rate"] for r in prev_rates) if prev_rates else total_rate
        velocity = total_rate
        acceleration = total_rate - prev_total
        radius = sum(1 for r in edges if r["transfer_amount"] > 0.5)
        lifetime = sum(1 for r in wave_rows if r.get("migration_velocity", 0) > 0.05) + (1 if velocity > 0.05 else 0)
        momentum = velocity * radius
        decay = max(0.0, prev_total - total_rate)
        wave_rows.append({
            "observation_id": obs_id,
            "checkpoint_hour": hour,
            "migration_velocity": round(velocity, 4),
            "migration_acceleration": round(acceleration, 4),
            "migration_radius": radius,
            "migration_lifetime": lifetime,
            "migration_momentum": round(momentum, 4),
            "migration_decay": round(decay, 4),
            "learning_recommendation": "NO_ACTION",
        })

    avg_velocity = statistics.mean(r["migration_velocity"] for r in wave_rows) if wave_rows else 0.0

    # --- Q7: Collective attention network ---
    network_rows: list[dict] = []
    edge_totals: dict[tuple[str, str], float] = defaultdict(float)
    out_degree: Counter = Counter()
    in_degree: Counter = Counter()
    for r in matrix_rows:
        key = (r["from_symbol"], r["to_symbol"])
        edge_totals[key] += r["transfer_amount"]
        out_degree[r["from_symbol"]] += r["transfer_amount"]
        in_degree[r["to_symbol"]] += r["transfer_amount"]

    switch_counts = Counter(r["leader_symbol"] for r in rotation_rows)
    for sym in symbols:
        out_d = out_degree[sym]
        in_d = in_degree[sym]
        total_flow = sum(edge_totals.values()) or 1.0
        centrality = (out_d + in_d) / total_flow
        authority = in_d / max(out_d + in_d, 0.01)
        hub_score = out_d / max(in_d, 0.01)
        bridge = min(out_d, in_d) / max(max(out_d, in_d), 0.01)
        community = "primary" if switch_counts[sym] >= len(rotation_rows) // 2 else "secondary"
        is_switch = "yes" if switch_counts[sym] >= 2 else "no"
        network_rows.append({
            "observation_id": obs_id,
            "symbol": sym,
            "out_degree": round(out_d, 4),
            "in_degree": round(in_d, 4),
            "centrality": round(centrality, 4),
            "authority": round(authority, 4),
            "hub_score": round(hub_score, 4),
            "bridge_score": round(bridge, 4),
            "community": community,
            "switch_node": is_switch,
            "leadership_count": switch_counts[sym],
            "learning_recommendation": "NO_ACTION",
        })

    for (src, dst), amt in edge_totals.items():
        network_rows.append({
            "observation_id": obs_id,
            "edge_from": src,
            "edge_to": dst,
            "edge_weight": round(amt, 4),
            "edge_type": "attention_transfer",
            "learning_recommendation": "NO_ACTION",
        })

    # --- Q8: Trend death LOOCV ---
    death_transitions: list[dict] = []
    for sym, sym_rows in by_sym.items():
        for i in range(len(sym_rows) - 1):
            cur, nxt = sym_rows[i], sym_rows[i + 1]
            if cur["trend_indicator"] < 0.5:
                continue
            gr = next((g for g in graph_rows if g["symbol"] == sym and g["checkpoint"] == nxt["checkpoint"]), None)
            outflow = gr["outgoing_attention"] if gr else max(0, cur["AttentionScore"] - nxt["AttentionScore"])
            flow_drop = max(0, cur["Flow"] - nxt["Flow"])
            mig_excess = max(0, outflow - cur["ReplacementRate"])
            trend_died = 1.0 if nxt["trend_indicator"] < cur["trend_indicator"] or nxt["CollapseRisk"] > cur["CollapseRisk"] + 0.1 else 0.0
            death_transitions.append({
                "flow_drop": flow_drop,
                "attention_outflow": outflow,
                "migration_excess": mig_excess,
                "y_death": trend_died,
                "y_collapse": nxt["CollapseRisk"],
            })

    loocv_results: dict[str, float] = {}
    if len(death_transitions) >= 3:
        y_death = [t["y_collapse"] for t in death_transitions]
        loocv_results["ModelA_FlowDisappears"] = loocv_rmse(
            [[t["flow_drop"]] for t in death_transitions], y_death, lam=2.0
        )
        loocv_results["ModelB_AttentionLeaves"] = loocv_rmse(
            [[t["attention_outflow"]] for t in death_transitions], y_death, lam=2.0
        )
        loocv_results["ModelC_MigrationExceedsReplacement"] = loocv_rmse(
            [[t["migration_excess"]] for t in death_transitions], y_death, lam=2.0
        )
        loocv_results["CombinedMigrationFlow"] = loocv_rmse(
            [[t["flow_drop"], t["attention_outflow"], t["migration_excess"]] for t in death_transitions],
            y_death, lam=2.0,
        )

    best_death_model = min(loocv_results.items(), key=lambda x: x[1]) if loocv_results else ("unknown", 0)

    # --- Q9: Migration laws ---
    law_rows: list[dict] = []
    mig_events = [r for r in matrix_rows if r["transfer_amount"] > 0]
    for r in mig_events:
        src = r["from_symbol"]
        dst = r["to_symbol"]
        hour = r["checkpoint_hour"]
        sp = next(x for x in by_sym[src] if x["checkpoint_hour"] == hour)
        dp = next(x for x in by_sym[dst] if x["checkpoint_hour"] == hour)
        r["NarrativeGap"] = abs(sp["NarrativeScore"] - dp["NarrativeScore"])
        r["BeliefGap"] = abs(sp["BeliefConsensus"] - dp["BeliefConsensus"])
        r["AttentionDifference"] = abs(sp["AttentionScore"] - dp["AttentionScore"])
        r["EcologyEntropy"] = (sp["EcologyEntropy"] + dp["EcologyEntropy"]) / 2
        r["ReplacementRate"] = (sp["ReplacementRate"] + dp["ReplacementRate"]) / 2
        r["MemoryDistance"] = abs(sp["Memory"] - dp["Memory"])
        r["GoalConsensus"] = (sp["GoalConcentration"] + dp["GoalConcentration"]) / 2
        r["ParticipationGap"] = abs(sp["Participation"] - dp["Participation"])
        r["FlowGradient"] = abs(sp["Flow"] - dp["Flow"])

    if mig_events:
        y_mig = [r["migration_rate"] for r in mig_events]
        predictors = [
            "NarrativeGap", "BeliefGap", "AttentionDifference", "EcologyEntropy",
            "ReplacementRate", "GoalConsensus", "ParticipationGap", "FlowGradient",
        ]
        for size in range(1, 4):
            for combo in itertools.combinations(predictors, size):
                X = [[r[p] for p in combo] for r in mig_events]
                beta = ridge_regress(X, y_mig, lam=1.0)
                pred = [sum(beta[j] * X[i][j] for j in range(len(combo))) for i in range(len(mig_events))]
                err = rmse(y_mig, pred)
                eq = "MigrationRate ≈ " + " + ".join(f"{beta[j]:+.4f}×{combo[j]}" for j in range(len(combo)))
                law_rows.append({
                    "observation_id": obs_id,
                    "equation": eq,
                    "target": "MigrationRate",
                    "predictors": "|".join(combo),
                    "rmse": round(err, 4),
                    "complexity": len(combo),
                    "law_score": round((1 - err) * 100 - len(combo) * 3, 2),
                    "learning_recommendation": "NO_ACTION",
                })

        for a, b in itertools.combinations(predictors, 2):
            X = [[r[a], r[b]] for r in mig_events]
            beta = ridge_regress(X, y_mig, lam=1.0)
            pred = [beta[0] * X[i][0] + beta[1] * X[i][1] for i in range(len(mig_events))]
            err = rmse(y_mig, pred)
            law_rows.append({
                "observation_id": obs_id,
                "equation": f"MigrationRate ≈ {beta[0]:+.4f}×{a} + {beta[1]:+.4f}×{b}",
                "target": "MigrationRate",
                "predictors": f"{a}|{b}",
                "rmse": round(err, 4),
                "complexity": 2,
                "law_score": round((1 - err) * 100 - 6, 2),
                "learning_recommendation": "NO_ACTION",
            })

    law_rows.sort(key=lambda r: (-r["law_score"], r["rmse"]))
    meaningful = [r for r in law_rows if r["rmse"] > 1e-6]
    law_rows = (meaningful or law_rows)[:20]

    # Birth prediction: migration inflow vs next trend
    birth_transitions: list[dict] = []
    for sym, sym_rows in by_sym.items():
        for i in range(len(sym_rows) - 1):
            cur, nxt = sym_rows[i], sym_rows[i + 1]
            gr = next((g for g in graph_rows if g["symbol"] == sym and g["checkpoint"] == nxt["checkpoint"]), None)
            birth_transitions.append({
                "inflow": gr["incoming_attention"] if gr else 0,
                "y_birth": 1.0 if nxt["trend_indicator"] > cur["trend_indicator"] else 0.0,
            })
    migration_predicts_birth = pearson(
        [t["inflow"] for t in birth_transitions],
        [t["y_birth"] for t in birth_transitions],
    ) if birth_transitions else 0.0

    migration_predicts_collapse = pearson(
        [r["outgoing_attention"] for r in graph_rows],
        [next(x for x in by_sym[r["symbol"]] if x["checkpoint"] == r["checkpoint"])["CollapseRisk"] for r in graph_rows],
    ) if graph_rows else 0.0

    report = build_report(
        obs_id, total_transfer, relocates, best_collapse_path, arch_counts,
        avg_velocity, loocv_results, best_death_model, hidden_rows, law_rows,
        migration_predicts_birth, migration_predicts_collapse, network_rows,
    )

    write_csv(MIGRATION_CSV, migration_rows)
    write_csv(MATRIX_CSV, matrix_rows)
    write_csv(GRAPH_CSV, graph_rows)
    write_csv(HIDDEN_CSV, hidden_rows)
    write_csv(ROTATION_CSV, rotation_rows)
    write_csv(WAVE_CSV, wave_rows)
    write_csv(NETWORK_CSV, network_rows)
    write_csv(LAWS_CSV, law_rows)
    PROCESS_REPORT_TXT.write_text(report, encoding="utf-8")

    print(
        f"Saved P61 outputs | migration={len(migration_rows)} matrix={len(matrix_rows)} "
        f"hidden={len(hidden_rows)} laws={len(law_rows)}"
    )


def build_report(
    obs_id: str,
    total_transfer: float,
    relocates: bool,
    best_collapse_path: tuple,
    arch_counts: Counter,
    avg_velocity: float,
    loocv_results: dict[str, float],
    best_death_model: tuple,
    hidden_rows: list[dict],
    law_rows: list[dict],
    migration_predicts_birth: float,
    migration_predicts_collapse: float,
    network_rows: list[dict],
) -> str:
    top_law = law_rows[0] if law_rows else {}
    top_arch = arch_counts.most_common(1)[0][0] if arch_counts else "unknown"
    hub = max(
        (r for r in network_rows if r.get("symbol")),
        key=lambda r: r.get("hub_score", 0),
        default={},
    )

    lines = [
        "===== SCOUT SEASON2 P61 - CROSS-SYMBOL ATTENTION MIGRATION =====",
        "",
        f"Observation ID: {obs_id}",
        "AttentionMigration hypothesis - STRICT NO_ACTION.",
        "",
        "=== Report questions ===",
        "",
        "1. Does Attention migrate?",
        f"   {'Yes (hypothesis)' if total_transfer > 0 else 'Unclear'}. Total inferred transfer={round(total_transfer, 2)}.",
        "",
        "2. Is TrendDeath actually Migration?",
        f"   {'Partially (hypothesis)' if best_death_model[0] != 'ModelA_FlowDisappears' else 'Flow-first in LOOCV'}. "
        f"Best death model: {best_death_model[0]} (RMSE={best_death_model[1]:.4f}).",
        "",
        "3. What variable moves first?",
        f"   Best collapse path: {best_collapse_path[0]} (fit={best_collapse_path[1]:.4f}).",
        "",
        "4. Migration speed?",
        f"   Mean migration velocity={round(avg_velocity, 4)} per checkpoint-hour.",
        "",
        "5. Migration archetypes?",
        f"   Primary archetype: {top_arch.replace('_', ' ')}.",
    ]
    for arch, cnt in arch_counts.most_common(4):
        lines.append(f"   {arch}: {cnt}")

    lines.extend([
        "",
        "6. Migration conservation?",
        f"   {'Attention relocates (hypothesis)' if relocates else 'Attention disappears'}. "
        "Non-zero incoming/outgoing flows; residual ≠ 0 → pool expands/contracts.",
        "",
        "7. Migration predicts Collapse?",
        f"   Correlation(outgoing_attention, CollapseRisk)={round(migration_predicts_collapse, 4)}.",
        "",
        "8. Migration predicts Birth?",
        f"   Correlation(incoming_attention, trend_birth)={round(migration_predicts_birth, 4)}.",
        "",
        "9. Strongest Migration Law?",
        f"   {top_law.get('equation', 'Insufficient data')}",
        "",
        "10. Can all discovered layers be unified?",
        "   Proposed stack (hypothesis):",
        "   Memory → Attention → Narrative → Belief → Participation → Flow → Trend",
        "   ↔ Migration (cross-symbol) → Collapse",
        f"   Hub symbol: {hub.get('symbol', 'n/a')} (hub_score={hub.get('hub_score', 'n/a')}).",
        "",
        "Learning recommendation: NO_ACTION - AttentionMigration stored as hypothesis only.",
        "",
        *mission_summary_lines(),
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="P61 Cross-Symbol Attention Migration Engine")
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
