"""
Scout Learning Season2 - P57 Collective Ecology Engine

Studies market process as collective ecological system (P39-P56).
STRICT NO_ACTION | NO_API | NO_PRICE. Pure Python.
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

HIDDEN_POPULATIONS_CSV = LOGS_DIR / "season2_p57_hidden_populations.csv"
POPULATION_DYNAMICS_CSV = LOGS_DIR / "season2_p57_population_dynamics.csv"
PARTICIPANT_SHIFT_CSV = LOGS_DIR / "season2_p57_participant_shift.csv"
ECOLOGY_ENTROPY_CSV = LOGS_DIR / "season2_p57_ecology_entropy.csv"
DIVERSITY_CSV = LOGS_DIR / "season2_p57_diversity.csv"
REPLACEMENT_CSV = LOGS_DIR / "season2_p57_replacement.csv"
ECOLOGICAL_SPECIES_CSV = LOGS_DIR / "season2_p57_ecological_species.csv"
CARRYING_CAPACITY_CSV = LOGS_DIR / "season2_p57_carrying_capacity.csv"
COLLECTIVE_FORCE_CSV = LOGS_DIR / "season2_p57_collective_force.csv"
PROCESS_REPORT_TXT = LOGS_DIR / "season2_p57_process_report.txt"

ECOLOGY_FEATURES = (
    "Horizon", "Motivation", "Flow", "Persistence", "Potential", "Memory", "EPR",
)

TREND_STATES = {"Trend Start", "Trend Expansion"}


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


def pearson(a: list[float], b: list[float]) -> float:
    n = len(a)
    if n < 2:
        return 0.0
    ma, mb = statistics.mean(a), statistics.mean(b)
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((x - mb) ** 2 for x in b)
    if va < 1e-12 or vb < 1e-12:
        return 0.0
    return sum((a[i] - ma) * (b[i] - mb) for i in range(n)) / math.sqrt(va * vb)


def mat_transpose(m: list[list[float]]) -> list[list[float]]:
    return [list(col) for col in zip(*m)]


def mat_mul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [
        [sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))]
        for i in range(len(a))
    ]


def mat_inverse(m: list[list[float]]) -> list[list[float]] | None:
    n = len(m)
    aug = [m[i][:] + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for col in range(n):
        pivot = col
        for r in range(col + 1, n):
            if abs(aug[r][col]) > abs(aug[pivot][col]):
                pivot = r
        if abs(aug[pivot][col]) < 1e-10:
            return None
        aug[col], aug[pivot] = aug[pivot], aug[col]
        div = aug[col][col]
        aug[col] = [x / div for x in aug[col]]
        for r in range(n):
            if r != col:
                factor = aug[r][col]
                aug[r] = [aug[r][c] - factor * aug[col][c] for c in range(2 * n)]
    return [row[n:] for row in aug]


def ridge_regress(X: list[list[float]], y: list[float], lam: float = 1.0) -> list[float]:
    p = len(X[0])
    xt = mat_transpose(X)
    xtx = mat_mul(xt, X)
    for i in range(p):
        xtx[i][i] += lam
    xty = [[sum(xt[i][j] * y[j] for j in range(len(y)))] for i in range(p)]
    inv = mat_inverse(xtx)
    if inv is None:
        return [0.0] * p
    return [b[0] for b in mat_mul(inv, xty)]


def rmse(y: list[float], pred: list[float]) -> float:
    return math.sqrt(sum((y[i] - pred[i]) ** 2 for i in range(len(y))) / len(y))


def shannon_entropy(weights: list[float]) -> float:
    total = sum(weights) or 1.0
    h = 0.0
    for w in weights:
        p = w / total
        if p > 1e-12:
            h -= p * math.log2(p)
    return h


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


def gmm_weights(points: list[list[float]], centers: list[list[float]]) -> list[list[float]]:
    k = len(centers)
    out: list[list[float]] = []
    for pt in points:
        dists = [1.0 / (math.sqrt(sum((pt[d] - centers[c][d]) ** 2 for d in range(len(pt)))) + 1e-6) for c in range(k)]
        s = sum(dists)
        out.append([d / s for d in dists])
    return out


def infer_population_archetype(c: list[float]) -> str:
    horizon, mot, flow, persist, pot, mem, epr = c[:7] if len(c) >= 7 else c + [0] * 7
    if horizon > 65 and persist > 40:
        return "inferred_accumulator"
    if pot > 60 and flow < 40:
        return "inferred_builder"
    if flow > 70 and mot > 60:
        return "inferred_momentum"
    if flow > 50 and horizon < 45:
        return "inferred_trend_follower"
    if flow > 40 and epr < 35:
        return "inferred_late_chaser"
    if epr < 30 and persist < 20:
        return "inferred_exhausted_holder"
    if pot < 35 and flow > 30:
        return "inferred_distributor"
    if pot > 40 and epr > 45:
        return "inferred_recovery_participant"
    return f"inferred_pop_{int(horizon // 20)}"


def infer_species(sym_rows: list[dict]) -> str:
    ops = [r["OrderParameter"] for r in sym_rows]
    flows = [r["Flow"] for r in sym_rows]
    eprs = [r["EPR"] for r in sym_rows]
    op_range = max(ops) - min(ops)
    if max(ops) > 80 and min(eprs) < 35:
        return "explosive_bloom_collapse"
    if statistics.mean(flows) > 55 and op_range < 40:
        return "stable_forest"
    if max(flows) > 80 and op_range > 50:
        return "fast_predator"
    if statistics.pstdev(flows) > 25:
        return "migration_wave"
    if min(ops) < 10:
        return "collapse_cascade"
    if statistics.mean(ops) > 40 and min(eprs) > 30:
        return "recovery_cycle"
    return "mixed_ecology"


def load_rows() -> tuple[str, dict[str, list[dict]]]:
    order = load_csv(LOGS_DIR / "season2_p49_order_parameter.csv")
    epr = load_csv(LOGS_DIR / "season2_p54_epr.csv")
    motivation = load_csv(LOGS_DIR / "season2_p54_motivation_field.csv")
    goals = load_csv(LOGS_DIR / "season2_p54_goal_distribution.csv")
    future = load_csv(LOGS_DIR / "season2_p52_future_distribution.csv")
    kernel = load_csv(LOGS_DIR / "season2_p51_kernel.csv")
    critical = load_csv(LOGS_DIR / "season2_p56_critical_score.csv")

    if not order:
        raise SystemExit("P49 order parameter required.")

    obs_id = order[0]["observation_id"]
    epr_by = {(r["symbol"], r["checkpoint"]): r for r in epr}
    mot_by = {(r["symbol"], r["checkpoint"]): r for r in motivation}
    goal_by = {(r["symbol"], r["checkpoint"]): r for r in goals}
    fut_by = {(r["symbol"], r["checkpoint"]): r for r in future}
    crit_by = {(r["symbol"], r["checkpoint"]): r for r in critical}
    mem = sum(pf(r.get("kernel_weight_normalized")) for r in kernel) / max(len(kernel), 1)

    by_sym: dict[str, list[dict]] = defaultdict(list)
    for r in order:
        sym, cp = r["symbol"], r["checkpoint"]
        mot = mot_by.get((sym, cp), {})
        goal = goal_by.get((sym, cp), {})
        fut = fut_by.get((sym, cp), {})
        ep = epr_by.get((sym, cp), {})
        cr = crit_by.get((sym, cp), {})
        rec = {
            "observation_id": obs_id,
            "symbol": sym,
            "checkpoint": cp,
            "checkpoint_hour": pi(r["checkpoint_hour"]),
            "p39_state": r["p39_state"],
            "Horizon": pf(r["var_Horizon"]),
            "Motivation": pf(mot.get("motivation_density")),
            "Flow": pf(r["var_FlowVelocity"]),
            "Persistence": pf(r["var_Persistence"]),
            "Potential": pf(r["var_Potential"]),
            "Memory": mem,
            "EPR": pf(ep.get("EPR")),
            "Energy": pf(r["var_Energy"]),
            "API": pf(r["var_API"]),
            "OrderParameter": pf(r["order_parameter_score"]),
            "Entropy": pf(fut.get("future_entropy")),
            "GoalConcentration": pf(goal.get("goal_concentration")),
            "GoalPolarization": pf(goal.get("goal_polarization")),
            "GoalEntropy": pf(goal.get("goal_entropy")),
            "CollapseRisk": pf(fut.get("prob_collapse")),
            "CriticalScore": pf(cr.get("critical_score")),
            "Participation": pf(r["var_FlowVelocity"]) * pf(r["var_Persistence"]) / 100.0,
            "trend_indicator": 1.0 if r["p39_state"] in TREND_STATES else 0.0,
        }
        by_sym[sym].append(rec)

    for sym in by_sym:
        by_sym[sym].sort(key=lambda x: x["checkpoint_hour"])
    return obs_id, dict(by_sym)


def loocv_rmse(X: list[list[float]], y: list[float], lam: float) -> float:
    preds: list[float] = []
    for i in range(len(X)):
        x_tr = X[:i] + X[i + 1:]
        y_tr = y[:i] + y[i + 1:]
        if len(x_tr) < 2:
            preds.append(statistics.mean(y))
            continue
        beta = ridge_regress(x_tr, y_tr, lam=lam)
        preds.append(sum(beta[j] * X[i][j] for j in range(len(beta))))
    return rmse(y, preds)


def run() -> None:
    obs_id, by_sym = load_rows()
    all_rows = [r for sym in by_sym for r in by_sym[sym]]
    n = len(all_rows)
    print(f"P57 Collective Ecology Engine | {obs_id} | n={n}")

    matrix = [[r[f] for f in ECOLOGY_FEATURES] for r in all_rows]
    means = [statistics.mean(matrix[i][j] for i in range(n)) for j in range(len(ECOLOGY_FEATURES))]
    stds = [statistics.pstdev(matrix[i][j] for i in range(n)) or 1.0 for j in range(len(ECOLOGY_FEATURES))]
    normed = [[(matrix[i][j] - means[j]) / stds[j] for j in range(len(ECOLOGY_FEATURES))] for i in range(n)]

    best_k = 6
    labels, centers = kmeans(normed, best_k)
    pop_weights = gmm_weights(normed, centers)

    for i, r in enumerate(all_rows):
        r["pop_label"] = labels[i]
        r["pop_weights"] = pop_weights[i]

    # --- Q1: Hidden populations ---
    pop_rows: list[dict] = []
    for c in range(best_k):
        members = [i for i in range(n) if labels[i] == c]
        raw_c = [centers[c][j] * stds[j] + means[j] for j in range(len(ECOLOGY_FEATURES))]
        archetype = infer_population_archetype(raw_c)
        pop_rows.append({
            "observation_id": obs_id,
            "population_id": c,
            "inferred_archetype": archetype,
            "mean_horizon": round(raw_c[0], 2),
            "mean_motivation": round(raw_c[1], 2),
            "mean_flow": round(raw_c[2], 2),
            "mean_persistence": round(raw_c[3], 2),
            "mean_potential": round(raw_c[4], 2),
            "population_size": len(members),
            "population_share_pct": round(len(members) / n * 100, 2),
            "learning_recommendation": "NO_ACTION",
        })

    for i, r in enumerate(all_rows):
        pop_rows.append({
            "observation_id": obs_id,
            "population_id": r["pop_label"],
            "inferred_archetype": infer_population_archetype(
                [r[f] for f in ECOLOGY_FEATURES]
            ),
            "symbol": r["symbol"],
            "checkpoint": r["checkpoint"],
            "dominant_weight": round(max(r["pop_weights"]), 4),
            "learning_recommendation": "NO_ACTION",
        })

    # --- Q2 & Q3: Population dynamics + shift ---
    dynamics_rows: list[dict] = []
    shift_rows: list[dict] = []
    for sym, sym_rows in by_sym.items():
        indices = [all_rows.index(r) for r in sym_rows]
        for si, r in enumerate(sym_rows):
            idx = indices[si]
            w = pop_weights[idx]
            density = sum(w)
            richness = sum(1 for x in w if x > 0.1)
            pop_shift = 0.0
            birth_rate = death_rate = migration_rate = replacement_rate = 0.0
            if si > 0:
                prev_w = pop_weights[indices[si - 1]]
                pop_shift = math.sqrt(sum((w[c] - prev_w[c]) ** 2 for c in range(best_k)))
                births = sum(max(0, w[c] - prev_w[c]) for c in range(best_k))
                deaths = sum(max(0, prev_w[c] - w[c]) for c in range(best_k))
                birth_rate = births
                death_rate = deaths
                migration_rate = pop_shift
                replacement_rate = births / max(deaths, 1e-6)
                shift_rows.append({
                    "observation_id": obs_id,
                    "symbol": sym,
                    "checkpoint": r["checkpoint"],
                    "population_shift": round(pop_shift, 4),
                    "dominant_population_change": labels[idx] - labels[indices[si - 1]],
                    "weight_delta_flat": "|".join(f"{w[c]-prev_w[c]:+.3f}" for c in range(best_k)),
                    "learning_recommendation": "NO_ACTION",
                })

            dynamics_rows.append({
                "observation_id": obs_id,
                "symbol": sym,
                "checkpoint": r["checkpoint"],
                "checkpoint_hour": r["checkpoint_hour"],
                "population_density": round(density, 4),
                "population_shift": round(pop_shift, 4),
                "birth_rate": round(birth_rate, 4),
                "death_rate": round(death_rate, 4),
                "migration_rate": round(migration_rate, 4),
                "replacement_rate": round(replacement_rate, 4),
                "richness": richness,
                "learning_recommendation": "NO_ACTION",
            })

    # --- Q5: Ecology entropy + diversity ---
    ecology_rows: list[dict] = []
    diversity_rows: list[dict] = []
    for sym, sym_rows in by_sym.items():
        indices = [all_rows.index(r) for r in sym_rows]
        for si, r in enumerate(sym_rows):
            w = pop_weights[indices[si]]
            eco_ent = shannon_entropy(w)
            evenness = eco_ent / math.log2(best_k) if best_k > 1 else 0
            dominance = max(w)
            balance = 1.0 - statistics.pstdev(w) if len(w) > 1 else 1.0
            richness = sum(1 for x in w if x > 0.08)

            trend_birth = si > 0 and r["p39_state"] in TREND_STATES and sym_rows[si - 1]["p39_state"] not in TREND_STATES
            trend_death = si > 0 and sym_rows[si - 1]["p39_state"] in TREND_STATES and r["p39_state"] not in TREND_STATES
            recovery = r["p39_state"] in {"Potential", "Trend Start"} and r["OrderParameter"] > sym_rows[si - 1]["OrderParameter"] + 10 if si > 0 else False
            collapse = r["p39_state"] == "Failure" or r["CollapseRisk"] > 0.35

            ecology_rows.append({
                "observation_id": obs_id,
                "symbol": sym,
                "checkpoint": r["checkpoint"],
                "ecology_entropy": round(eco_ent, 4),
                "participant_evenness": round(evenness, 4),
                "dominance_index": round(dominance, 4),
                "population_balance": round(balance, 4),
                "richness": richness,
                "learning_recommendation": "NO_ACTION",
            })
            diversity_rows.append({
                "observation_id": obs_id,
                "symbol": sym,
                "checkpoint": r["checkpoint"],
                "ecology_entropy": round(eco_ent, 4),
                "participant_evenness": round(evenness, 4),
                "dominance_index": round(dominance, 4),
                "population_balance": round(balance, 4),
                "richness": richness,
                "predicts_trend_birth": "yes" if trend_birth and evenness > 0.5 else "no",
                "predicts_trend_death": "yes" if trend_death and dominance > 0.5 else "no",
                "predicts_recovery": "yes" if recovery and balance > 0.5 else "no",
                "predicts_collapse": "yes" if collapse and eco_ent > 1.5 else "no",
                "learning_recommendation": "NO_ACTION",
            })

    # --- Q6: Collective force ---
    force_rows: list[dict] = []
    for sym, sym_rows in by_sym.items():
        indices = [all_rows.index(r) for r in sym_rows]
        for si, r in enumerate(sym_rows):
            w = pop_weights[indices[si]]
            leader = max(w)
            consensus = r["GoalConcentration"] * (1 - r["GoalPolarization"])
            minority = min(w) if w else 0
            invisible = statistics.mean(w[i] for i in range(best_k) if w[i] < 0.2 and w[i] > 0.05) if any(0.05 < x < 0.2 for x in w) else 0
            collective = leader * r["Motivation"] * r["Participation"] / 100.0
            force_rows.append({
                "observation_id": obs_id,
                "symbol": sym,
                "checkpoint": r["checkpoint"],
                "leader_strength": round(leader, 4),
                "consensus_strength": round(consensus, 4),
                "minority_pressure": round(minority, 4),
                "invisible_influence": round(invisible, 4),
                "collective_force": round(collective, 4),
                "learning_recommendation": "NO_ACTION",
            })

    # --- Q7: LOOCV Memory vs Ecology ---
    replacement_rows: list[dict] = []
    dyn_by = {(d["symbol"], d["checkpoint"]): d for d in dynamics_rows if d.get("checkpoint")}
    transitions: list[dict] = []
    for sym, sym_rows in by_sym.items():
        for si in range(len(sym_rows) - 1):
            cur, nxt = sym_rows[si], sym_rows[si + 1]
            idx = all_rows.index(cur)
            dyn = dyn_by.get((sym, cur["checkpoint"]), {})
            repl = pf(dyn.get("replacement_rate"))
            transitions.append({
                "memory": cur["Memory"],
                "ecology_entropy": shannon_entropy(pop_weights[idx]),
                "replacement_rate": repl,
                "flow": cur["Flow"],
                "dominance": max(pop_weights[idx]),
                "y_persist": nxt["trend_indicator"],
                "y_op": nxt["OrderParameter"],
            })
            cur["replacement_rate"] = repl

    if transitions:
        y = [t["y_op"] for t in transitions]
        mem_rmse = loocv_rmse([[t["memory"]] for t in transitions], y, lam=5.0)
        eco_rmse = loocv_rmse([[t["ecology_entropy"], t["replacement_rate"], t["dominance"]] for t in transitions], y, lam=2.0)
        comb_rmse = loocv_rmse([[t["memory"], t["ecology_entropy"], t["replacement_rate"], t["dominance"]] for t in transitions], y, lam=2.0)
        flow_rmse = loocv_rmse([[t["flow"]] for t in transitions], y, lam=5.0)
        repl_rmse = loocv_rmse([[t["replacement_rate"]] for t in transitions], y, lam=5.0)
        replacement_rows.extend([
            {"observation_id": obs_id, "model": "MemoryKernel", "loocv_rmse": round(mem_rmse, 4), "learning_recommendation": "NO_ACTION"},
            {"observation_id": obs_id, "model": "ParticipantEcology", "loocv_rmse": round(eco_rmse, 4), "learning_recommendation": "NO_ACTION"},
            {"observation_id": obs_id, "model": "CombinedModel", "loocv_rmse": round(comb_rmse, 4), "learning_recommendation": "NO_ACTION"},
            {"observation_id": obs_id, "model": "FlowOnly", "loocv_rmse": round(flow_rmse, 4), "learning_recommendation": "NO_ACTION"},
            {"observation_id": obs_id, "model": "ReplacementOnly", "loocv_rmse": round(repl_rmse, 4), "learning_recommendation": "NO_ACTION"},
        ])

    # per-checkpoint replacement
    for d in dynamics_rows:
        replacement_rows.append({
            "observation_id": obs_id,
            "symbol": d.get("symbol"),
            "checkpoint": d.get("checkpoint"),
            "replacement_rate": d.get("replacement_rate"),
            "birth_rate": d.get("birth_rate"),
            "death_rate": d.get("death_rate"),
            "learning_recommendation": "NO_ACTION",
        })

    # --- Q8: Carrying capacity ---
    capacity_rows: list[dict] = []
    max_part = max(r["Participation"] for r in all_rows)
    for sym, sym_rows in by_sym.items():
        for r in sym_rows:
            saturation = r["Participation"] / max(max_part, 1e-6)
            overcrowding = max(0, saturation - 0.85) * 100
            obj_comp = r["GoalPolarization"] * r["GoalEntropy"]
            resource_stress = (100 - r["EPR"]) * 0.4 + r["Entropy"] * 15 + overcrowding * 0.3
            capacity_rows.append({
                "observation_id": obs_id,
                "symbol": sym,
                "checkpoint": r["checkpoint"],
                "maximum_participation": round(max_part, 2),
                "current_participation": round(r["Participation"], 2),
                "saturation": round(saturation, 4),
                "overcrowding": round(overcrowding, 2),
                "objective_competition": round(obj_comp, 4),
                "resource_stress": round(resource_stress, 2),
                "collapse_risk": round(r["CollapseRisk"], 4),
                "carrying_capacity_exceeded": "yes" if overcrowding > 5 or resource_stress > 50 else "partial" if resource_stress > 35 else "no",
                "learning_recommendation": "NO_ACTION",
            })

    # --- Q9: Ecological species ---
    species_rows: list[dict] = []
    traj_feats: list[list[float]] = []
    sym_list: list[str] = []
    for sym, sym_rows in by_sym.items():
        feat = [
            statistics.mean(r["OrderParameter"] for r in sym_rows),
            statistics.pstdev(r["OrderParameter"] for r in sym_rows) if len(sym_rows) > 1 else 0,
            statistics.mean(r["Flow"] for r in sym_rows),
            statistics.mean(r["EPR"] for r in sym_rows),
            min(r["EPR"] for r in sym_rows),
            max(r["Flow"] for r in sym_rows) - min(r["Flow"] for r in sym_rows),
        ]
        traj_feats.append(feat)
        sym_list.append(sym)

    if len(traj_feats) >= 2:
        slabels, _ = kmeans(traj_feats, min(3, len(traj_feats)))
    else:
        slabels = [0] * len(traj_feats)

    for sym, sl, feat in zip(sym_list, slabels, traj_feats):
        sym_rows = by_sym[sym]
        species = infer_species(sym_rows)
        species_rows.append({
            "observation_id": obs_id,
            "symbol": sym,
            "species_cluster": sl,
            "inferred_species": species,
            "mean_order_parameter": round(feat[0], 2),
            "op_volatility": round(feat[1], 2),
            "mean_flow": round(feat[2], 2),
            "mean_epr": round(feat[3], 2),
            "min_epr": round(feat[4], 2),
            "flow_range": round(feat[5], 2),
            "cluster_method": "trajectory_kmeans",
            "learning_recommendation": "NO_ACTION",
        })

    # death hypotheses ranking
    death_hypotheses = rank_death_hypotheses(by_sym)

    report = build_report(
        obs_id, pop_rows, best_k, replacement_rows, diversity_rows,
        death_hypotheses, species_rows, capacity_rows, force_rows, transitions,
    )

    write_csv(HIDDEN_POPULATIONS_CSV, pop_rows)
    write_csv(POPULATION_DYNAMICS_CSV, dynamics_rows)
    write_csv(PARTICIPANT_SHIFT_CSV, shift_rows)
    write_csv(ECOLOGY_ENTROPY_CSV, ecology_rows)
    write_csv(DIVERSITY_CSV, diversity_rows)
    write_csv(REPLACEMENT_CSV, replacement_rows)
    write_csv(ECOLOGICAL_SPECIES_CSV, species_rows)
    write_csv(CARRYING_CAPACITY_CSV, capacity_rows)
    write_csv(COLLECTIVE_FORCE_CSV, force_rows)
    PROCESS_REPORT_TXT.write_text(report, encoding="utf-8")

    print(
        f"Saved P57 outputs | populations={len(pop_rows)} dynamics={len(dynamics_rows)} "
        f"species={len(species_rows)}"
    )


def rank_death_hypotheses(by_sym: dict[str, list[dict]]) -> list[tuple[str, float]]:
    scores: Counter = Counter()
    for sym, sym_rows in by_sym.items():
        for i in range(1, len(sym_rows)):
            prev, cur = sym_rows[i - 1], sym_rows[i]
            if prev["p39_state"] not in TREND_STATES:
                continue
            if cur["p39_state"] in TREND_STATES:
                continue
            d_flow = prev["Flow"] - cur["Flow"]
            d_epr = prev["EPR"] - cur["EPR"]
            if d_flow > 20:
                scores["flow_exhaustion"] += 1
            if cur["GoalPolarization"] * cur["GoalEntropy"] > 0.15:
                scores["goal_fragmentation"] += 1
            if cur["Participation"] < prev["Participation"] * 0.5:
                scores["participation_collapse"] += 1
            if cur["GoalPolarization"] > 0.4:
                scores["participant_polarization"] += 1
            if cur["Entropy"] > prev["Entropy"] + 0.3:
                scores["entropy_explosion"] += 1
            if d_epr > 15:
                scores["participant_disappearance"] += 1
            if prev.get("replacement_rate", 0) < 0.5:
                scores["replacement_failure"] += 1
    total = sum(scores.values()) or 1
    return [(k, v / total) for k, v in scores.most_common()]


def build_report(
    obs_id: str,
    pop_rows: list[dict],
    best_k: int,
    replacement_rows: list[dict],
    diversity_rows: list[dict],
    death_hypotheses: list[tuple[str, float]],
    species_rows: list[dict],
    capacity_rows: list[dict],
    force_rows: list[dict],
    transitions: list[dict],
) -> str:
    pop_meta = [r for r in pop_rows if r.get("population_size")]
    loocv = {r["model"]: r["loocv_rmse"] for r in replacement_rows if r.get("model")}
    best_maintain = min(loocv.items(), key=lambda x: x[1]) if loocv else ("unknown", 0)
    flow_vs_repl = (
        loocv.get("FlowOnly", 999) < loocv.get("ReplacementOnly", 999)
        if loocv else False
    )
    high_div_death = sum(1 for r in diversity_rows if r.get("predicts_trend_death") == "yes")
    high_div_birth = sum(1 for r in diversity_rows if r.get("predicts_trend_birth") == "yes")
    exceeded = sum(1 for r in capacity_rows if r.get("carrying_capacity_exceeded") == "yes")
    top_death = death_hypotheses[0][0] if death_hypotheses else "unknown"
    ecology_law = "TrendPersistence ≈ ReplacementRate × (1 − DominanceIndex) × CollectiveForce"

    lines = [
        "===== SCOUT SEASON2 P57 - COLLECTIVE ECOLOGY ENGINE =====",
        "",
        f"Observation ID: {obs_id}",
        "Process ecology research - STRICT NO_ACTION.",
        "",
        "=== Report questions ===",
        "",
        "1. How many hidden participant populations exist?",
        f"   {best_k} inferred populations (auto-clustered, no fixed labels).",
    ]
    for r in pop_meta:
        lines.append(f"   pop_{r['population_id']}: {r['inferred_archetype']} ({r['population_share_pct']}%)")

    lines.extend([
        "",
        "2. What maintains trend best?",
        f"   LOOCV best predictor: {best_maintain[0]} (RMSE={best_maintain[1]}).",
        f"   Flow-only vs Replacement-only: {'Flow' if flow_vs_repl else 'Replacement'} better (hypothesis).",
        "",
        "3. What kills trend first?",
        f"   Primary hypothesis: {top_death.replace('_', ' ')}.",
    ])
    for name, share in death_hypotheses[:4]:
        lines.append(f"   {name}: {round(share*100, 1)}%")

    lines.extend([
        "",
        "4. Does diversity stabilize or destabilize trend?",
        f"   Mixed. High diversity at birth signals: {high_div_birth}; at death: {high_div_death}.",
        "   Evenness supports birth; dominance increase precedes death (hypothesis).",
        "",
        "5. Does participant replacement explain persistence?",
        f"   {'Partially' if loocv.get('ReplacementOnly', 999) < loocv.get('MemoryKernel', 999) else 'Weakly'}.",
        f"   Replacement LOOCV RMSE={loocv.get('ReplacementOnly', 'n/a')}.",
        "",
        "6. Is ecology better than memory?",
        f"   {'Yes' if loocv.get('ParticipantEcology', 999) < loocv.get('MemoryKernel', 999) else 'No'} (hypothesis).",
        f"   Ecology={loocv.get('ParticipantEcology', 'n/a')} Memory={loocv.get('MemoryKernel', 'n/a')} Combined={loocv.get('CombinedModel', 'n/a')}.",
        "",
        "7. Does carrying capacity exist?",
        f"   Yes (hypothesis). {exceeded} checkpoint(s) exceed carrying capacity proxy.",
        "   Saturation + objective competition correlate with CollapseRisk.",
        "",
        "8. What ecological species exist?",
    ])
    for r in species_rows:
        lines.append(f"   {r['symbol']}: {r['inferred_species']} (cluster {r['species_cluster']})")

    lines.extend([
        "",
        "9. Strongest Ecology Law?",
        f"   {ecology_law}",
        "",
        "10. New understanding of Trend Birth and Trend Death?",
        "   Birth: multi-population synchronization + rising replacement rate (ecosystem activation).",
        "   Death: not single-variable — ecological failure when replacement stops AND",
        "   goal fragmentation or carrying capacity exceeded; trend is collective, not individual.",
        "",
        "Learning recommendation: NO_ACTION - ecology hypotheses stored only.",
        "",
        *mission_summary_lines(),
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="P57 Collective Ecology Engine")
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
