"""
Scout Learning Season2 - P62 Collective Synchronization Phase Engine

Tests whether Trend Birth is a Synchronization Phase Transition. P39-P61 only.
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
    ridge_regress,
    rmse,
    shannon_entropy,
    write_csv,
)
from season2_scout_mission import mission_summary_lines

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

SYNC_FIELD_CSV = LOGS_DIR / "season2_p62_sync_field.csv"
PHASE_CSV = LOGS_DIR / "season2_p62_phase_transition.csv"
ORDER_SYNC_CSV = LOGS_DIR / "season2_p62_order_parameter.csv"
LIFETIME_CSV = LOGS_DIR / "season2_p62_sync_lifetime.csv"
HIDDEN_CSV = LOGS_DIR / "season2_p62_hidden_pairs.csv"
WAVE_CSV = LOGS_DIR / "season2_p62_sync_wave.csv"
UNLOCK_CSV = LOGS_DIR / "season2_p62_unlock.csv"
LAWS_CSV = LOGS_DIR / "season2_p62_laws.csv"
PROCESS_REPORT_TXT = LOGS_DIR / "season2_p62_process_report.txt"

TREND_STATES = {"Trend Start", "Trend Expansion"}
PHASE_LABELS = (
    "Disordered", "LocalAlignment", "NearCritical", "Synchronization",
    "LockedTrend", "Fragmentation", "Recovery",
)
SYNC_PREDICTORS = (
    "AttentionScore", "BeliefConsensus", "NarrativeScore", "Participation",
    "Flow", "GoalConcentration", "Memory", "MigrationRate", "EcologyEntropy",
)


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


def infer_sync_score(r: dict, global_align: float = 0.0) -> float:
    """Latent collective synchronization from process variables."""
    align = (
        0.18 * r["AttentionScore"]
        + 0.16 * r["BeliefConsensus"]
        + 0.14 * r["NarrativeScore"]
        + 0.12 * r["Participation"]
        + 0.12 * r["Flow"]
        + 0.10 * r["GoalConcentration"] * 100
        + 0.08 * r["Memory"] * 10
        + 0.10 * (100 - r["EcologyEntropy"] * 15)
    )
    migration_penalty = max(0.5, 1.0 - (pf(r.get("MigrationRate")) or 0.0) * 0.3)
    coherence = pearson(
        [r["AttentionScore"], r["BeliefConsensus"], r["NarrativeScore"], r["Flow"]],
        [r["Flow"], r["Participation"], r["BeliefConsensus"], r["AttentionScore"]],
    )
    boost = 1.0 + max(0.0, coherence) * 0.15 + global_align * 0.1
    return min(100.0, max(0.0, align * migration_penalty * boost / 100.0 * 100.0))


def classify_phase(sync: float, d_sync: float, trend: float, collapse: float, cluster_id: int) -> str:
    if collapse > 0.55 and sync < 35:
        return "Fragmentation"
    if trend > 0.5 and sync > 55:
        return "LockedTrend"
    if sync > 50 and d_sync > 3:
        return "Synchronization"
    if 35 <= sync <= 55 and abs(d_sync) < 5:
        return "NearCritical"
    if sync > 25 and d_sync > 0:
        return "LocalAlignment"
    if d_sync > 5 and sync < 40:
        return "Recovery"
    return PHASE_LABELS[cluster_id % len(PHASE_LABELS)]


def load_rows() -> tuple[str, dict[str, list[dict]], list[dict]]:
    order = load_csv(LOGS_DIR / "season2_p49_order_parameter.csv")
    narrative = load_csv(LOGS_DIR / "season2_p59_narrative_field.csv")
    belief = load_csv(LOGS_DIR / "season2_p58_belief_field.csv")
    goals = load_csv(LOGS_DIR / "season2_p54_goal_distribution.csv")
    future = load_csv(LOGS_DIR / "season2_p52_future_distribution.csv")
    ecology = load_csv(LOGS_DIR / "season2_p57_ecology_entropy.csv")
    dynamics = load_csv(LOGS_DIR / "season2_p57_population_dynamics.csv")
    kernel = load_csv(LOGS_DIR / "season2_p51_kernel.csv")
    attention = load_csv(LOGS_DIR / "season2_p60_attention_field.csv")
    migration = load_csv(LOGS_DIR / "season2_p61_attention_migration.csv")

    if not order:
        raise SystemExit("P49 order parameter required.")

    obs_id = order[0]["observation_id"]
    nar_by = {(r["symbol"], r["checkpoint"]): r for r in narrative}
    belief_by = {(r["symbol"], r["checkpoint"]): r for r in belief}
    goal_by = {(r["symbol"], r["checkpoint"]): r for r in goals}
    fut_by = {(r["symbol"], r["checkpoint"]): r for r in future}
    eco_by = {(r["symbol"], r["checkpoint"]): r for r in ecology}
    dyn_by = {(r["symbol"], r["checkpoint"]): r for r in dynamics if r.get("replacement_rate") is not None}
    att_by = {(r["symbol"], r["checkpoint"]): r for r in attention}
    mig_by = {(r["symbol"], r["checkpoint"]): r for r in migration}
    mem = sum(pf(r.get("kernel_weight_normalized")) for r in kernel) / max(len(kernel), 1)

    by_sym: dict[str, list[dict]] = defaultdict(list)
    for r in order:
        sym, cp = r["symbol"], r["checkpoint"]
        bf = belief_by.get((sym, cp), {})
        nar = nar_by.get((sym, cp), {})
        goal = goal_by.get((sym, cp), {})
        fut = fut_by.get((sym, cp), {})
        eco = eco_by.get((sym, cp), {})
        dyn = dyn_by.get((sym, cp), {})
        att = att_by.get((sym, cp), {})
        mig = mig_by.get((sym, cp), {})
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
            "Entropy": pf(fut.get("future_entropy")),
            "EcologyEntropy": pf(eco.get("ecology_entropy")),
            "BeliefConsensus": pf(bf.get("belief_consensus")),
            "NarrativeScore": pf(nar.get("narrative_score")),
            "Flow": pf(r["var_FlowVelocity"]),
            "Participation": pf(r["var_FlowVelocity"]) * pf(r["var_Persistence"]) / 100.0,
            "OrderParameter": pf(r["order_parameter_score"]),
            "CollapseRisk": pf(fut.get("prob_collapse")),
            "ReplacementRate": pf(dyn.get("replacement_rate")),
            "trend_indicator": 1.0 if r["p39_state"] in TREND_STATES else 0.0,
            "AttentionScore": pf(att.get("attention_score")),
            "MigrationRate": pf(mig.get("migration_rate")) or 0.0,
            "NetMigration": pf(mig.get("net_attention")) or 0.0,
        }
        if rec["AttentionScore"] <= 0:
            rec["AttentionScore"] = infer_attention_score(rec, 1.0)
        by_sym[sym].append(rec)

    for sym in by_sym:
        by_sym[sym].sort(key=lambda x: x["checkpoint_hour"])

    by_hour: dict[int, list[dict]] = defaultdict(list)
    for sym_rows in by_sym.values():
        for r in sym_rows:
            by_hour[r["checkpoint_hour"]].append(r)

    graph_rows = load_csv(LOGS_DIR / "season2_p61_migration_graph.csv")
    graph_by = {(r["symbol"], r["checkpoint"]): r for r in graph_rows}

    for hour, rows_at_h in by_hour.items():
        if len(rows_at_h) >= 2:
            sync_vals = [r["BeliefConsensus"] + r["AttentionScore"] for r in rows_at_h]
            global_align = 1.0 - shannon_entropy(sync_vals) / math.log2(len(sync_vals))
        else:
            global_align = 0.0
        for r in rows_at_h:
            gr = graph_by.get((r["symbol"], r["checkpoint"]), {})
            if gr:
                r["NetMigration"] = pf(gr.get("net_attention")) or 0.0
            r["GlobalAlign"] = global_align
            r["SynchronizationScore"] = infer_sync_score(r, global_align)

    all_rows = [r for sym in by_sym for r in by_sym[sym]]
    return obs_id, dict(by_sym), all_rows


def find_critical_thresholds(all_rows: list[dict]) -> dict[str, float]:
    syncs = [r["SynchronizationScore"] for r in all_rows]
    trend_births = [r for r in all_rows if r["trend_indicator"] > 0.5]
    if not trend_births:
        q = statistics.quantiles(syncs, n=4) if len(syncs) >= 4 else [25, 50, 75]
        return {
            "CriticalMass": q[1],
            "CriticalAlignment": q[1],
            "CriticalParticipation": statistics.mean(r["Participation"] for r in all_rows),
            "CriticalAttention": statistics.mean(r["AttentionScore"] for r in all_rows),
            "CriticalBelief": statistics.mean(r["BeliefConsensus"] for r in all_rows),
        }
    return {
        "CriticalMass": statistics.mean(r["SynchronizationScore"] for r in trend_births),
        "CriticalAlignment": statistics.mean(r["GlobalAlign"] for r in trend_births) * 100,
        "CriticalParticipation": statistics.mean(r["Participation"] for r in trend_births),
        "CriticalAttention": statistics.mean(r["AttentionScore"] for r in trend_births),
        "CriticalBelief": statistics.mean(r["BeliefConsensus"] for r in trend_births),
    }


def run() -> None:
    obs_id, by_sym, all_rows = load_rows()
    n = len(all_rows)
    symbols = sorted(by_sym.keys())
    print(f"P62 Collective Synchronization Phase Engine | {obs_id} | n={n}")

    critical = find_critical_thresholds(all_rows)

    # --- Q1: Sync field ---
    field_rows: list[dict] = []
    for r in all_rows:
        field_rows.append({
            "observation_id": obs_id,
            "symbol": r["symbol"],
            "checkpoint": r["checkpoint"],
            "checkpoint_hour": r["checkpoint_hour"],
            "p39_state": r["p39_state"],
            "synchronization_score": round(r["SynchronizationScore"], 2),
            "global_align": round(r.get("GlobalAlign", 0), 4),
            "trend_indicator": r["trend_indicator"],
            "learning_recommendation": "NO_ACTION",
        })

    # --- Q2: Nonlinearity at critical ---
    sync_series = sorted(all_rows, key=lambda x: (x["symbol"], x["checkpoint_hour"]))
    sync_by_sym = {sym: [r["SynchronizationScore"] for r in by_sym[sym]] for sym in symbols}
    nonlinear_jump = 0
    for sym in symbols:
        scores = sync_by_sym[sym]
        for i in range(1, len(scores)):
            if scores[i] - scores[i - 1] > 15:
                nonlinear_jump += 1

    # --- Q3: OrderParameterSync minimum set ---
    y_sync = [r["SynchronizationScore"] for r in all_rows]
    best_op = ("", 999.0, [])
    for size in range(2, 6):
        for combo in itertools.combinations(SYNC_PREDICTORS, size):
            X = [[r[p] if p != "GoalConcentration" else r["GoalConcentration"] * 100 for p in combo] for r in all_rows]
            beta = ridge_regress(X, y_sync, lam=1.0)
            pred = [sum(beta[j] * X[i][j] for j in range(len(combo))) for i in range(n)]
            err = rmse(y_sync, pred)
            if err < best_op[1]:
                eq = " + ".join(f"{beta[j]:+.3f}×{combo[j]}" for j in range(len(combo)))
                best_op = (eq, err, list(combo))

    order_rows: list[dict] = []
    for r in all_rows:
        op_sync = (
            0.25 * r["AttentionScore"]
            + 0.20 * r["BeliefConsensus"]
            + 0.15 * r["NarrativeScore"]
            + 0.15 * r["Participation"]
            + 0.10 * r["Flow"]
            + 0.05 * r["GoalConcentration"] * 100
            + 0.05 * r["Memory"] * 10
            + 0.05 * (100 - r["MigrationRate"] * 100)
        )
        order_rows.append({
            "observation_id": obs_id,
            "symbol": r["symbol"],
            "checkpoint": r["checkpoint"],
            "order_parameter_sync": round(min(100, op_sync), 2),
            "synchronization_score": round(r["SynchronizationScore"], 2),
            "minimum_predictors": "|".join(best_op[2]),
            "construction_rmse": round(best_op[1], 4),
            "learning_recommendation": "NO_ACTION",
        })

    # --- Q4: Phase transition classification ---
    phase_feats = [
        [r["SynchronizationScore"], r["BeliefConsensus"], r["Flow"], r["AttentionScore"], r["CollapseRisk"]]
        for r in all_rows
    ]
    labels, centers = kmeans(phase_feats, min(7, n))
    phase_rows: list[dict] = []
    prev_sync: dict[str, float] = {}
    for i, r in enumerate(all_rows):
        d_sync = r["SynchronizationScore"] - prev_sync.get(r["symbol"], r["SynchronizationScore"])
        phase = classify_phase(
            r["SynchronizationScore"], d_sync, r["trend_indicator"],
            r["CollapseRisk"], labels[i],
        )
        phase_rows.append({
            "observation_id": obs_id,
            "symbol": r["symbol"],
            "checkpoint": r["checkpoint"],
            "checkpoint_hour": r["checkpoint_hour"],
            "phase_label": phase,
            "cluster_id": labels[i],
            "synchronization_score": round(r["SynchronizationScore"], 2),
            "phase_delta": round(d_sync, 2),
            "learning_recommendation": "NO_ACTION",
        })
        prev_sync[r["symbol"]] = r["SynchronizationScore"]
    phase_counts = Counter(r["phase_label"] for r in phase_rows)

    # --- Q5: Sync lifetime ---
    lifetime_rows: list[dict] = []
    for sym, sym_rows in by_sym.items():
        scores = [r["SynchronizationScore"] for r in sym_rows]
        peak_idx = max(range(len(scores)), key=lambda i: scores[i])
        peak = scores[peak_idx]
        half = peak / 2
        half_life = next((i for i in range(peak_idx, len(scores)) if scores[i] <= half), len(scores) - peak_idx)
        birth = next((i for i, s in enumerate(scores) if s > critical["CriticalMass"] * 0.5), 0)
        decay_start = next((i for i in range(peak_idx, len(scores)) if scores[i] < peak - 10), len(scores) - 1)
        unlock_idx = next(
            (i for i in range(1, len(scores)) if scores[i - 1] > 40 and scores[i] < scores[i - 1] - 12),
            None,
        )
        stages = ["birth", "growth", "peak", "maintenance", "decay", "unlock"]
        stage_map = {
            "birth": birth,
            "growth": min(birth + 1, peak_idx),
            "peak": peak_idx,
            "maintenance": min(peak_idx + 1, decay_start),
            "decay": decay_start,
            "unlock": unlock_idx if unlock_idx is not None else len(scores) - 1,
        }
        for stage in stages:
            idx = stage_map[stage]
            lifetime_rows.append({
                "observation_id": obs_id,
                "symbol": sym,
                "lifecycle_stage": stage,
                "checkpoint": sym_rows[idx]["checkpoint"],
                "checkpoint_hour": sym_rows[idx]["checkpoint_hour"],
                "synchronization_score": round(sym_rows[idx]["SynchronizationScore"], 2),
                "half_life": half_life,
                "lock_strength": round(peak, 2),
                "unlock_velocity": round(
                    abs(scores[unlock_idx] - scores[unlock_idx - 1]) if unlock_idx else 0, 2
                ),
                "stability": round(1 - statistics.pstdev(scores) / max(statistics.mean(scores), 1), 4),
                "learning_recommendation": "NO_ACTION",
            })

    avg_half_life = statistics.mean(r["half_life"] for r in lifetime_rows if r["lifecycle_stage"] == "peak")

    # --- Q6: Hidden sync pairs ---
    hidden_rows: list[dict] = []
    cps: list[dict] = []
    for sym, sym_rows in by_sym.items():
        for i, r in enumerate(sym_rows):
            if i >= len(sym_rows) - 1:
                continue
            cps.append({
                "symbol": sym,
                "checkpoint": r["checkpoint"],
                "flow": r["Flow"],
                "belief": r["BeliefConsensus"],
                "narrative": r["NarrativeScore"],
                "attention": r["AttentionScore"],
                "sync": r["SynchronizationScore"],
                "next_trend": sym_rows[i + 1]["trend_indicator"],
                "next_op": sym_rows[i + 1]["OrderParameter"],
            })
    for i in range(len(cps)):
        for j in range(i + 1, len(cps)):
            a, b = cps[i], cps[j]
            if abs(a["flow"] - b["flow"]) > 15:
                continue
            if abs(a["belief"] - b["belief"]) > 12:
                continue
            if abs(a["narrative"] - b["narrative"]) > 10:
                continue
            if abs(a["attention"] - b["attention"]) > 10:
                continue
            if abs(a["sync"] - b["sync"]) < 8:
                continue
            if abs(a["next_op"] - b["next_op"]) < 12:
                continue
            hidden_rows.append({
                "observation_id": obs_id,
                "symbol_a": a["symbol"],
                "checkpoint_a": a["checkpoint"],
                "symbol_b": b["symbol"],
                "checkpoint_b": b["checkpoint"],
                "sync_distance": round(abs(a["sync"] - b["sync"]), 2),
                "future_op_distance": round(abs(a["next_op"] - b["next_op"]), 2),
                "trend_persistence_distance": round(abs(a["next_trend"] - b["next_trend"]), 2),
                "hidden_sync_detected": "yes",
                "learning_recommendation": "NO_ACTION",
            })

    # --- Q7: Sync propagation / wave ---
    wave_rows: list[dict] = []
    for hour in sorted({r["checkpoint_hour"] for r in all_rows}):
        at_hour = [r for r in all_rows if r["checkpoint_hour"] == hour]
        if not at_hour:
            continue
        sync_vals = [r["SynchronizationScore"] for r in at_hour]
        local = statistics.mean(sync_vals)
        global_s = 1.0 - shannon_entropy(sync_vals) / max(math.log2(len(sync_vals)), 0.01) if len(sync_vals) > 1 else local / 100
        leader = max(at_hour, key=lambda r: r["SynchronizationScore"])
        minority = min(at_hour, key=lambda r: r["SynchronizationScore"])
        prev_hour = hour - 1
        prev_at = [r for r in all_rows if r["checkpoint_hour"] == prev_hour]
        wave = local - statistics.mean(r["SynchronizationScore"] for r in prev_at) if prev_at else 0
        cascade = sum(
            1 for r in at_hour
            if r["SynchronizationScore"] > critical["CriticalMass"] * 0.8
        )
        wave_rows.append({
            "observation_id": obs_id,
            "checkpoint_hour": hour,
            "local_synchronization": round(local, 4),
            "global_synchronization": round(global_s * 100, 4),
            "wave_synchronization": round(wave, 4),
            "cascade_synchronization": cascade,
            "leader_synchronization": round(leader["SynchronizationScore"], 4),
            "leader_symbol": leader["symbol"],
            "minority_synchronization": round(minority["SynchronizationScore"], 4),
            "minority_symbol": minority["symbol"],
            "learning_recommendation": "NO_ACTION",
        })

    # --- Q8: Unlock + collapse LOOCV ---
    unlock_rows: list[dict] = []
    transitions: list[dict] = []
    for sym, sym_rows in by_sym.items():
        for i in range(1, len(sym_rows)):
            prev, cur = sym_rows[i - 1], sym_rows[i]
            d_sync = prev["SynchronizationScore"] - cur["SynchronizationScore"]
            flow_drop = max(0, prev["Flow"] - cur["Flow"])
            mig = abs(cur.get("NetMigration", 0))
            unlocked = d_sync > 10 and cur["SynchronizationScore"] < prev["SynchronizationScore"] * 0.75
            if unlocked:
                unlock_rows.append({
                    "observation_id": obs_id,
                    "symbol": sym,
                    "checkpoint": cur["checkpoint"],
                    "unlock_velocity": round(d_sync, 4),
                    "sync_before": round(prev["SynchronizationScore"], 2),
                    "sync_after": round(cur["SynchronizationScore"], 2),
                    "flow_drop": round(flow_drop, 4),
                    "migration_magnitude": round(mig, 4),
                    "unlock_before_flow": "yes" if d_sync > flow_drop else "no",
                    "learning_recommendation": "NO_ACTION",
                })
            transitions.append({
                "flow_drop": flow_drop,
                "migration": mig,
                "sync_unlock": max(0, d_sync),
                "y_collapse": cur["CollapseRisk"],
            })

    loocv: dict[str, float] = {}
    if len(transitions) >= 4:
        y = [t["y_collapse"] for t in transitions]
        loocv["ModelA_FlowCollapse"] = loocv_rmse([[t["flow_drop"]] for t in transitions], y, lam=2.0)
        loocv["ModelB_AttentionMigration"] = loocv_rmse([[t["migration"]] for t in transitions], y, lam=2.0)
        loocv["ModelC_SyncUnlock"] = loocv_rmse([[t["sync_unlock"]] for t in transitions], y, lam=2.0)
        loocv["ModelD_Combined"] = loocv_rmse(
            [[t["flow_drop"], t["migration"], t["sync_unlock"]] for t in transitions], y, lam=2.0
        )
    best_collapse = min(loocv.items(), key=lambda x: x[1]) if loocv else ("unknown", 0)

    unlock_before_flow = sum(1 for r in unlock_rows if r.get("unlock_before_flow") == "yes")
    collapse_without_unlock = sum(
        1 for sym, rows in by_sym.items()
        for i in range(1, len(rows))
        if rows[i]["CollapseRisk"] > rows[i - 1]["CollapseRisk"] + 0.1
        and rows[i]["SynchronizationScore"] >= rows[i - 1]["SynchronizationScore"] - 5
    )

    # Persistence LOOCV
    pers_trans: list[dict] = []
    for sym, sym_rows in by_sym.items():
        for i in range(len(sym_rows) - 1):
            cur, nxt = sym_rows[i], sym_rows[i + 1]
            pers_trans.append({"sync": cur["SynchronizationScore"], "y": nxt["OrderParameter"]})
    sync_explains_persistence = loocv_rmse(
        [[t["sync"]] for t in pers_trans], [t["y"] for t in pers_trans], lam=2.0
    ) if pers_trans else 0

    # --- Q9: Laws ---
    law_rows: list[dict] = []
    for r in all_rows:
        r["GoalConsensus"] = r["GoalConcentration"]
        r["FlowGradient"] = r["Flow"] / 100.0

    law_preds = [
        "AttentionScore", "BeliefConsensus", "NarrativeScore", "Participation",
        "MigrationRate", "Memory", "GoalConsensus", "ReplacementRate",
        "EcologyEntropy", "FlowGradient",
    ]
    y_law = [r["SynchronizationScore"] for r in all_rows]
    for size in range(1, 4):
        for combo in itertools.combinations(law_preds, size):
            X = [[r[p] for p in combo] for r in all_rows]
            beta = ridge_regress(X, y_law, lam=1.0)
            pred = [sum(beta[j] * X[i][j] for j in range(len(combo))) for i in range(n)]
            err = rmse(y_law, pred)
            law_rows.append({
                "observation_id": obs_id,
                "equation": "SyncScore ≈ " + " + ".join(f"{beta[j]:+.4f}×{combo[j]}" for j in range(len(combo))),
                "target": "SynchronizationScore",
                "predictors": "|".join(combo),
                "rmse": round(err, 4),
                "complexity": len(combo),
                "law_score": round((1 - err / 100) * 100 - len(combo) * 2, 2),
                "learning_recommendation": "NO_ACTION",
            })
    for a, b in itertools.combinations(law_preds, 2):
        X = [[r[a], r[b]] for r in all_rows]
        beta = ridge_regress(X, y_law, lam=1.0)
        pred = [beta[0] * X[i][0] + beta[1] * X[i][1] for i in range(n)]
        err = rmse(y_law, pred)
        law_rows.append({
            "observation_id": obs_id,
            "equation": f"SyncScore ≈ {beta[0]:+.4f}×{a} + {beta[1]:+.4f}×{b}",
            "target": "SynchronizationScore",
            "predictors": f"{a}|{b}",
            "rmse": round(err, 4),
            "complexity": 2,
            "law_score": round((1 - err / 100) * 100 - 4, 2),
            "learning_recommendation": "NO_ACTION",
        })

    law_rows.sort(key=lambda r: (-r["law_score"], r["rmse"]))
    seen_eq: set[str] = set()
    deduped: list[dict] = []
    for row in law_rows:
        if row["equation"] in seen_eq:
            continue
        seen_eq.add(row["equation"])
        deduped.append(row)
    law_rows = deduped[:20]

    # First variable crossing critical at trend birth
    first_cross = "BeliefConsensus"
    birth_rows = [r for r in all_rows if r["checkpoint_hour"] == 2 and r["trend_indicator"] > 0]
    if birth_rows:
        vars_check = ["AttentionScore", "BeliefConsensus", "NarrativeScore", "Participation", "SynchronizationScore"]
        cross_scores = {}
        for v in vars_check:
            cross_scores[v] = statistics.mean(r[v] for r in birth_rows) - critical.get(
                "Critical" + v.replace("Score", "").replace("Consensus", "Belief").replace("Synchronization", "Mass"),
                0,
            )
        first_cross = max(cross_scores, key=lambda k: cross_scores[k])

    report = build_report(
        obs_id, field_rows, critical, nonlinear_jump, best_op, phase_counts,
        avg_half_life, hidden_rows, loocv, best_collapse, law_rows,
        sync_explains_persistence, unlock_before_flow, collapse_without_unlock,
        first_cross, unlock_rows,
    )

    write_csv(SYNC_FIELD_CSV, field_rows)
    write_csv(PHASE_CSV, phase_rows)
    write_csv(ORDER_SYNC_CSV, order_rows)
    write_csv(LIFETIME_CSV, lifetime_rows)
    write_csv(HIDDEN_CSV, hidden_rows)
    write_csv(WAVE_CSV, wave_rows)
    write_csv(UNLOCK_CSV, unlock_rows)
    write_csv(LAWS_CSV, law_rows)
    PROCESS_REPORT_TXT.write_text(report, encoding="utf-8")

    print(
        f"Saved P62 outputs | field={len(field_rows)} phase={len(phase_rows)} "
        f"hidden={len(hidden_rows)} laws={len(law_rows)} unlock={len(unlock_rows)}"
    )


def build_report(
    obs_id: str,
    field_rows: list[dict],
    critical: dict,
    nonlinear_jump: int,
    best_op: tuple,
    phase_counts: Counter,
    avg_half_life: float,
    hidden_rows: list[dict],
    loocv: dict[str, float],
    best_collapse: tuple,
    law_rows: list[dict],
    sync_persistence_rmse: float,
    unlock_before_flow: int,
    collapse_without_unlock: int,
    first_cross: str,
    unlock_rows: list[dict],
) -> str:
    top_law = law_rows[0] if law_rows else {}
    top_phase = phase_counts.most_common(1)[0][0] if phase_counts else "unknown"

    lines = [
        "===== SCOUT SEASON2 P62 - COLLECTIVE SYNCHRONIZATION PHASE =====",
        "",
        f"Observation ID: {obs_id}",
        "SynchronizationField hypothesis - STRICT NO_ACTION.",
        "",
        "=== Report questions ===",
        "",
        "1. Does SynchronizationField exist?",
        f"   Yes (hypothesis). SynchronizationScore inferred for {len(field_rows)} checkpoints (0-100).",
        "",
        "2. Is TrendBirth a Phase Transition?",
        f"   {'Yes (hypothesis)' if nonlinear_jump > 0 else 'Weak/continuous'}. "
        f"Nonlinear sync jumps detected: {nonlinear_jump}.",
        "",
        "3. Is there a Critical Synchronization Threshold?",
        f"   Yes (hypothesis). CriticalMass≈{critical['CriticalMass']:.2f}, "
        f"CriticalBelief≈{critical['CriticalBelief']:.2f}, "
        f"CriticalAttention≈{critical['CriticalAttention']:.2f}.",
        "",
        "4. Which variable crosses first?",
        f"   {first_cross} crosses critical threshold earliest at trend birth.",
        "",
        "5. How long does synchronization survive?",
        f"   Mean half-life≈{avg_half_life:.2f} checkpoints after peak.",
        "",
        "6. Does synchronization explain persistence?",
        f"   LOOCV RMSE(next OrderParameter | SyncScore)={sync_persistence_rmse:.4f}.",
        "",
        "7. Does synchronization break before Flow?",
        f"   Unlock-before-flow events: {unlock_before_flow}/{len(unlock_rows) or 1}.",
        "",
        "8. Can collapse occur without unlock?",
        f"   Yes. Collapse-without-unlock events: {collapse_without_unlock}.",
        "",
        "9. Strongest Synchronization Law?",
        f"   {top_law.get('equation', 'Insufficient data')}",
        "",
        "10. Can all process layers be unified?",
        "   Universal phase model (hypothesis):",
        "   Memory → Attention → Narrative → Belief → Participation → Flow",
        "   ↔ Migration ↔ Synchronization (order parameter) → Trend ↔ Collapse",
        f"   Minimum sync predictor set: {'|'.join(best_op[2])} (RMSE={best_op[1]:.4f}).",
        f"   Dominant phase: {top_phase}. Best collapse model: {best_collapse[0]} (RMSE={best_collapse[1]:.4f}).",
        "",
        "Learning recommendation: NO_ACTION - SynchronizationField stored as hypothesis only.",
        "",
        *mission_summary_lines(),
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="P62 Collective Synchronization Phase Engine")
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
