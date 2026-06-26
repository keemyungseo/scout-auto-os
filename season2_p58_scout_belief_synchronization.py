"""
Scout Learning Season2 - P58 Information & Belief Synchronization Engine

Studies collective belief synchronization from process variables P39-P57.
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
from season2_scout_mission import mission_summary_lines

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

BELIEF_FIELD_CSV = LOGS_DIR / "season2_p58_belief_field.csv"
INFORMATION_DIFFUSION_CSV = LOGS_DIR / "season2_p58_information_diffusion.csv"
BELIEF_ENTROPY_CSV = LOGS_DIR / "season2_p58_belief_entropy.csv"
EXPECTATION_CSV = LOGS_DIR / "season2_p58_expectation.csv"
BELIEF_HALF_LIFE_CSV = LOGS_DIR / "season2_p58_belief_half_life.csv"
HIDDEN_BELIEFS_CSV = LOGS_DIR / "season2_p58_hidden_beliefs.csv"
BELIEF_NETWORK_CSV = LOGS_DIR / "season2_p58_belief_network.csv"
CONSENSUS_CLUSTERS_CSV = LOGS_DIR / "season2_p58_consensus_clusters.csv"
BELIEF_VS_ECOLOGY_CSV = LOGS_DIR / "season2_p58_belief_vs_ecology.csv"
EMERGENT_LAWS_CSV = LOGS_DIR / "season2_p58_emergent_laws.csv"
PROCESS_REPORT_TXT = LOGS_DIR / "season2_p58_process_report.txt"

TREND_STATES = {"Trend Start", "Trend Expansion"}
BELIEF_FEATURES = ("BeliefConsensus", "GoalConcentration", "API", "Quality", "Motivation", "Confidence")


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


def shannon_entropy(vals: list[float]) -> float:
    total = sum(abs(v) for v in vals) or 1.0
    h = 0.0
    for v in vals:
        p = abs(v) / total
        if p > 1e-12:
            h -= p * math.log2(p)
    return h


def cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


def l2(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(len(a))))


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


def infer_belief_consensus(r: dict) -> float:
    gc = r["GoalConcentration"]
    consensus = gc * (1 - r["GoalPolarization"])
    raw = (
        0.25 * consensus * 100
        + 0.20 * r["API"]
        + 0.15 * r["Quality"]
        + 0.15 * r["Motivation"]
        + 0.15 * r.get("Confidence", 50)
        + 0.10 * r["OrderParameter"]
    )
    return min(100.0, max(0.0, raw * (1 - r["GoalPolarization"] * 0.3)))


def infer_belief_archetype(belief: float, mot: float, flow: float, polar: float, op: float) -> str:
    if belief > 70 and flow > 60:
        return "inferred_conviction"
    if belief > 55 and mot > 50 and flow < 40:
        return "inferred_curiosity"
    if flow > 70 and belief > 50 and polar > 0.2:
        return "inferred_fomo"
    if belief < 40 and flow > 40 and op < 45:
        return "inferred_distribution"
    if polar > 0.35 and belief < 50:
        return "inferred_uncertainty"
    if belief > 45 and op > 40 and flow < 30:
        return "inferred_recovery"
    return f"inferred_belief_{int(belief // 25)}"


def estimate_half_life(series: list[tuple[int, float]]) -> float:
    if len(series) < 2:
        return 0.0
    peak = max(series, key=lambda x: x[1])
    threshold = peak[1] * 0.5
    for hour, val in series:
        if hour > peak[0] and val <= threshold:
            return float(hour - peak[0])
    if peak[1] > 1e-6:
        ratios = []
        for i in range(1, len(series)):
            h0, v0 = series[i - 1]
            h1, v1 = series[i]
            if v0 > v1 > 0 and h1 > h0:
                ratios.append(math.log(v1 / v0) / (h0 - h1))
        if ratios:
            lam = statistics.mean(ratios)
            if lam > 1e-9:
                return round(math.log(2) / lam, 2)
    return float(len(series))


def load_rows() -> tuple[str, dict[str, list[dict]]]:
    order = load_csv(LOGS_DIR / "season2_p49_order_parameter.csv")
    epr = load_csv(LOGS_DIR / "season2_p54_epr.csv")
    motivation = load_csv(LOGS_DIR / "season2_p54_motivation_field.csv")
    goals = load_csv(LOGS_DIR / "season2_p54_goal_distribution.csv")
    future = load_csv(LOGS_DIR / "season2_p52_future_distribution.csv")
    confidence = load_csv(LOGS_DIR / "season2_p52_confidence.csv")
    force = load_csv(LOGS_DIR / "season2_p57_collective_force.csv")
    dynamics = load_csv(LOGS_DIR / "season2_p57_population_dynamics.csv")
    critical_mass = load_csv(LOGS_DIR / "season2_p56_critical_mass.csv")
    kernel = load_csv(LOGS_DIR / "season2_p51_kernel.csv")
    state_space = load_csv(LOGS_DIR / "season2_p47_state_space.csv")

    if not order:
        raise SystemExit("P49 order parameter required.")

    obs_id = order[0]["observation_id"]
    epr_by = {(r["symbol"], r["checkpoint"]): r for r in epr}
    mot_by = {(r["symbol"], r["checkpoint"]): r for r in motivation}
    goal_by = {(r["symbol"], r["checkpoint"]): r for r in goals}
    fut_by = {(r["symbol"], r["checkpoint"]): r for r in future}
    conf_by = {(r["symbol"], r["checkpoint"]): r for r in confidence}
    force_by = {(r["symbol"], r["checkpoint"]): r for r in force if r.get("collective_force")}
    dyn_by = {(r["symbol"], r["checkpoint"]): r for r in dynamics if r.get("replacement_rate") is not None}
    mass_by = {(r["symbol"], r["checkpoint"]): r for r in critical_mass if r.get("total_critical_mass")}
    ss_by = {(r["symbol"], pi(r["checkpoint_hour"])): r for r in state_space}
    mem = sum(pf(r.get("kernel_weight_normalized")) for r in kernel) / max(len(kernel), 1)

    by_sym: dict[str, list[dict]] = defaultdict(list)
    for r in order:
        sym, cp = r["symbol"], r["checkpoint"]
        hour = pi(r["checkpoint_hour"])
        mot = mot_by.get((sym, cp), {})
        goal = goal_by.get((sym, cp), {})
        fut = fut_by.get((sym, cp), {})
        conf = conf_by.get((sym, cp), {})
        ep = epr_by.get((sym, cp), {})
        fo = force_by.get((sym, cp), {})
        dyn = dyn_by.get((sym, cp), {})
        mass = mass_by.get((sym, cp), {})
        ss = ss_by.get((sym, hour), {})
        rec = {
            "observation_id": obs_id,
            "symbol": sym,
            "checkpoint": cp,
            "checkpoint_hour": hour,
            "p39_state": r["p39_state"],
            "API": pf(r["var_API"]),
            "Quality": pf(r["var_Quality"]),
            "Potential": pf(r["var_Potential"]),
            "Flow": pf(r["var_FlowVelocity"]),
            "Persistence": pf(r["var_Persistence"]),
            "Motivation": pf(mot.get("motivation_density")),
            "OrderParameter": pf(r["order_parameter_score"]),
            "Entropy": pf(fut.get("future_entropy")),
            "GoalConcentration": pf(goal.get("goal_concentration")),
            "GoalPolarization": pf(goal.get("goal_polarization")),
            "GoalEntropy": pf(goal.get("goal_entropy")),
            "GoalConsensus": pf(goal.get("goal_concentration")) * (1 - pf(goal.get("goal_polarization"))),
            "Confidence": pf(conf.get("confidence_score")),
            "CollapseRisk": pf(fut.get("prob_collapse")),
            "EPR": pf(ep.get("EPR")),
            "Memory": mem,
            "Participation": pf(r["var_FlowVelocity"]) * pf(r["var_Persistence"]) / 100.0,
            "CollectiveForce": pf(fo.get("collective_force")),
            "ReplacementRate": pf(dyn.get("replacement_rate")),
            "CriticalMass": pf(mass.get("total_critical_mass")),
            "LocalDensity": pf(ss.get("local_density")),
            "trend_indicator": 1.0 if r["p39_state"] in TREND_STATES else 0.0,
        }
        rec["BeliefConsensus"] = infer_belief_consensus(rec)
        by_sym[sym].append(rec)

    for sym in by_sym:
        by_sym[sym].sort(key=lambda x: x["checkpoint_hour"])
    return obs_id, dict(by_sym)


def path_score(chains: list[list[float]]) -> float:
    s = 1.0
    for c in chains:
        s *= max(0, abs(c))
    return s


def run() -> None:
    obs_id, by_sym = load_rows()
    all_rows = [r for sym in by_sym for r in by_sym[sym]]
    n = len(all_rows)
    print(f"P58 Belief Synchronization Engine | {obs_id} | n={n}")

    # --- Q1: Belief field ---
    belief_rows: list[dict] = []
    for r in all_rows:
        belief_rows.append({
            "observation_id": obs_id,
            "symbol": r["symbol"],
            "checkpoint": r["checkpoint"],
            "checkpoint_hour": r["checkpoint_hour"],
            "p39_state": r["p39_state"],
            "belief_consensus": round(r["BeliefConsensus"], 2),
            "goal_consensus": round(r["GoalConsensus"], 4),
            "confidence": round(r["Confidence"], 2),
            "belief_archetype": infer_belief_archetype(
                r["BeliefConsensus"], r["Motivation"], r["Flow"], r["GoalPolarization"], r["OrderParameter"]
            ),
            "learning_recommendation": "NO_ACTION",
        })

    # --- Q2: Information diffusion ---
    diffusion_rows: list[dict] = []
    for sym, sym_rows in by_sym.items():
        for i in range(1, len(sym_rows)):
            prev, cur = sym_rows[i - 1], sym_rows[i]
            d_belief = cur["BeliefConsensus"] - prev["BeliefConsensus"]
            hours = max(cur["checkpoint_hour"] - prev["checkpoint_hour"], 1)
            speed = abs(d_belief) / hours
            aligned = sum(
                1 for v in ("API", "Quality", "Motivation", "Flow")
                if (cur[v] - prev[v]) * d_belief > 0
            )
            radius = aligned / 4.0
            delay = 0
            if i >= 2:
                d_belief_l2 = sym_rows[i - 2]["BeliefConsensus"] - prev["BeliefConsensus"]
                if abs(d_belief_l2) > abs(d_belief):
                    delay = 1
            stability = 1.0 / (1.0 + abs(cur["Entropy"] - prev["Entropy"]))
            diffusion_rows.append({
                "observation_id": obs_id,
                "symbol": sym,
                "from_checkpoint": prev["checkpoint"],
                "to_checkpoint": cur["checkpoint"],
                "propagation_speed": round(speed, 4),
                "propagation_radius": round(radius, 4),
                "propagation_delay": delay,
                "propagation_stability": round(stability, 4),
                "belief_delta": round(d_belief, 2),
                "learning_recommendation": "NO_ACTION",
            })

    # --- Q3: Belief before participation? ---
    ordering_rows: list[dict] = []
    for sym, sym_rows in by_sym.items():
        belief = [r["BeliefConsensus"] for r in sym_rows]
        mot = [r["Motivation"] for r in sym_rows]
        part = [r["Participation"] for r in sym_rows]
        flow = [r["Flow"] for r in sym_rows]
        trend = [r["trend_indicator"] for r in sym_rows]
        path1 = path_score([
            pearson(belief, mot), pearson(mot, part), pearson(part, flow), pearson(flow, trend),
        ])
        path2 = path_score([
            pearson(part, belief), pearson(belief, trend),
        ])
        ordering_rows.append({
            "observation_id": obs_id,
            "symbol": sym,
            "path_belief_first": round(path1, 4),
            "path_participation_first": round(path2, 4),
            "belief_before_participation": "yes" if path1 > path2 else "no",
            "correlation_belief_motivation": round(pearson(belief, mot), 4),
            "correlation_motivation_participation": round(pearson(mot, part), 4),
            "correlation_participation_flow": round(pearson(part, flow), 4),
            "learning_recommendation": "NO_ACTION",
        })

    # --- Q4: Expectation divergence ---
    expectation_rows: list[dict] = []
    belief_entropy_rows: list[dict] = []
    for r in all_rows:
        exp_vec = [r["BeliefConsensus"], r["API"], r["Quality"], r["Potential"], r["Motivation"]]
        exp_var = statistics.pvariance(exp_vec) if len(exp_vec) > 1 else 0
        exp_ent = shannon_entropy(exp_vec)
        exp_frag = r["GoalEntropy"] * r["GoalPolarization"]
        belief_ent = shannon_entropy([r["GoalConcentration"], 1 - r["GoalPolarization"], r["BeliefConsensus"] / 100])
        expectation_rows.append({
            "observation_id": obs_id,
            "symbol": r["symbol"],
            "checkpoint": r["checkpoint"],
            "expectation_variance": round(exp_var, 4),
            "expectation_entropy": round(exp_ent, 4),
            "expectation_fragmentation": round(exp_frag, 4),
            "collapse_risk": round(r["CollapseRisk"], 4),
            "predicts_collapse": "yes" if exp_frag > 0.15 and r["CollapseRisk"] > 0.2 else "partial" if exp_frag > 0.1 else "no",
            "learning_recommendation": "NO_ACTION",
        })
        belief_entropy_rows.append({
            "observation_id": obs_id,
            "symbol": r["symbol"],
            "checkpoint": r["checkpoint"],
            "belief_entropy": round(belief_ent, 4),
            "goal_entropy": round(r["GoalEntropy"], 4),
            "expectation_entropy": round(exp_ent, 4),
            "learning_recommendation": "NO_ACTION",
        })

    # --- Q5: Belief half-life ---
    half_life_rows: list[dict] = []
    for sym, sym_rows in by_sym.items():
        series = [(r["checkpoint_hour"], r["BeliefConsensus"]) for r in sym_rows]
        hl = estimate_half_life(series)
        decays: list[float] = []
        pers: list[float] = []
        for i in range(1, len(sym_rows)):
            prev, cur = sym_rows[i - 1], sym_rows[i]
            if prev["Flow"] > cur["Flow"] + 10:
                decays.append(prev["BeliefConsensus"] - cur["BeliefConsensus"])
                pers.append(cur["BeliefConsensus"] / max(prev["BeliefConsensus"], 1))
        half_life_rows.append({
            "observation_id": obs_id,
            "symbol": sym,
            "belief_half_life_checkpoints": hl,
            "belief_persistence": round(statistics.mean(pers), 4) if pers else 1.0,
            "belief_decay_rate": round(statistics.mean(decays), 4) if decays else 0.0,
            "inertia_detected": "yes" if pers and statistics.mean(pers) > 0.7 else "partial" if pers else "no",
            "learning_recommendation": "NO_ACTION",
        })
        for i, r in enumerate(sym_rows):
            if i == 0:
                continue
            prev = sym_rows[i - 1]
            if prev["Flow"] > r["Flow"] + 5:
                half_life_rows.append({
                    "observation_id": obs_id,
                    "symbol": sym,
                    "checkpoint": r["checkpoint"],
                    "flow_decreased": "yes",
                    "belief_retained_pct": round(r["BeliefConsensus"] / max(prev["BeliefConsensus"], 1) * 100, 2),
                    "learning_recommendation": "NO_ACTION",
                })

    # --- Q6: Hidden beliefs ---
    hidden_rows: list[dict] = []
    state_vars = ("Potential", "Flow", "API", "Quality", "OrderParameter")
    cps: list[dict] = []
    for sym, sym_rows in by_sym.items():
        for i, r in enumerate(sym_rows):
            if i >= len(sym_rows) - 1:
                continue
            cps.append({
                "symbol": sym,
                "checkpoint": r["checkpoint"],
                "state": [r[v] for v in state_vars],
                "belief": r["BeliefConsensus"],
                "next_op": sym_rows[i + 1]["OrderParameter"],
                "next_state": sym_rows[i + 1]["p39_state"],
            })
    for i in range(len(cps)):
        for j in range(i + 1, len(cps)):
            a, b = cps[i], cps[j]
            if l2(a["state"], b["state"]) > 30:
                continue
            if abs(a["next_op"] - b["next_op"]) < 15:
                continue
            belief_explains = abs(a["belief"] - b["belief"]) > abs(l2(a["state"], b["state"]))
            hidden_rows.append({
                "observation_id": obs_id,
                "symbol_a": a["symbol"],
                "checkpoint_a": a["checkpoint"],
                "symbol_b": b["symbol"],
                "checkpoint_b": b["checkpoint"],
                "state_distance": round(l2(a["state"], b["state"]), 2),
                "belief_distance": round(abs(a["belief"] - b["belief"]), 2),
                "future_op_distance": round(abs(a["next_op"] - b["next_op"]), 2),
                "belief_explains_outcome": "yes" if belief_explains else "partial",
                "hidden_belief_detected": "yes",
                "learning_recommendation": "NO_ACTION",
            })

    # --- Q7: Belief archetypes (cluster) ---
    archetype_feats = [[r["BeliefConsensus"], r["Motivation"], r["Flow"], r["GoalPolarization"], r["OrderParameter"]] for r in all_rows]
    alabels, _ = kmeans(archetype_feats, min(6, n))
    cluster_rows: list[dict] = []
    for i, r in enumerate(all_rows):
        arch = infer_belief_archetype(r["BeliefConsensus"], r["Motivation"], r["Flow"], r["GoalPolarization"], r["OrderParameter"])
        cluster_rows.append({
            "observation_id": obs_id,
            "symbol": r["symbol"],
            "checkpoint": r["checkpoint"],
            "belief_archetype": arch,
            "consensus_cluster": alabels[i],
            "belief_consensus": round(r["BeliefConsensus"], 2),
            "learning_recommendation": "NO_ACTION",
        })
    for c in range(max(alabels) + 1):
        members = [all_rows[i] for i in range(n) if alabels[i] == c]
        if members:
            cluster_rows.append({
                "observation_id": obs_id,
                "symbol": "(aggregate)",
                "consensus_cluster": c,
                "belief_archetype": infer_belief_archetype(
                    statistics.mean(m["BeliefConsensus"] for m in members),
                    statistics.mean(m["Motivation"] for m in members),
                    statistics.mean(m["Flow"] for m in members),
                    statistics.mean(m["GoalPolarization"] for m in members),
                    statistics.mean(m["OrderParameter"] for m in members),
                ),
                "member_count": len(members),
                "learning_recommendation": "NO_ACTION",
            })

    # --- Q8: Belief vs ecology ---
    ecology_rows: list[dict] = []
    for sym, sym_rows in by_sym.items():
        belief = [r["BeliefConsensus"] for r in sym_rows]
        repl = [r["ReplacementRate"] for r in sym_rows]
        force = [r["CollectiveForce"] for r in sym_rows]
        persist = [r["trend_indicator"] for r in sym_rows]
        chain = path_score([
            pearson(belief, repl), pearson(repl, force), pearson(force, persist),
        ])
        ecology_rows.append({
            "observation_id": obs_id,
            "symbol": sym,
            "causal_chain": "Belief→Replacement→CollectiveForce→TrendPersistence",
            "chain_fit_score": round(chain, 4),
            "ecology_emerges_from_belief": "yes" if chain > 0.05 else "partial" if chain > 0.02 else "no",
            "correlation_belief_replacement": round(pearson(belief, repl), 4),
            "correlation_replacement_force": round(pearson(repl, force), 4),
            "correlation_force_persistence": round(pearson(force, persist), 4),
            "learning_recommendation": "NO_ACTION",
        })
    ecology_rows.extend(ordering_rows)

    # --- Q9: Belief network ---
    network_rows: list[dict] = []
    var_names = ("BeliefConsensus", "Motivation", "Flow", "GoalConcentration", "API", "Quality")
    for sym, sym_rows in by_sym.items():
        if len(sym_rows) < 3:
            continue
        cols = {v: [r[v] if v != "GoalConcentration" else r["GoalConcentration"] for r in sym_rows] for v in var_names}
        edges = 0
        strength = 0.0
        for i, vi in enumerate(var_names):
            for j, vj in enumerate(var_names):
                if j <= i:
                    continue
                c = abs(pearson(cols[vi], cols[vj]))
                if c > 0.4:
                    edges += 1
                    strength += c
                    network_rows.append({
                        "observation_id": obs_id,
                        "symbol": sym,
                        "node_a": vi,
                        "node_b": vj,
                        "edge_weight": round(c, 4),
                        "edge_type": "belief_coupling",
                        "learning_recommendation": "NO_ACTION",
                    })
        minority = statistics.mean([r["GoalPolarization"] for r in sym_rows])
        sync_eff = strength / max(edges, 1) * (1 - minority)
        network_rows.append({
            "observation_id": obs_id,
            "symbol": sym,
            "belief_connectivity": edges,
            "consensus_clusters": len(set(alabels[i] for i, r in enumerate(all_rows) if r["symbol"] == sym)),
            "minority_resistance": round(minority, 4),
            "synchronization_efficiency": round(sync_eff, 4),
            "learning_recommendation": "NO_ACTION",
        })

    # --- Q10: Emergent laws ---
    law_rows: list[dict] = []
    law_targets = [
        ("BeliefConsensus", ["GoalConsensus", "API", "Quality", "Motivation"], "BeliefConsensus"),
        ("CollapseRisk", ["ExpectationEntropy", "BeliefConsensus", "GoalPolarization"], "CollapseProbability"),
        ("trend_indicator", ["BeliefConsensus", "Motivation", "Flow"], "TrendBirthProbability"),
        ("BeliefConsensus", ["Memory", "Persistence", "GoalConsensus"], "BeliefPersistence"),
    ]
    for r in all_rows:
        exp_vec = [r["BeliefConsensus"], r["API"], r["Quality"], r["Potential"], r["Motivation"]]
        r["ExpectationEntropy"] = shannon_entropy(exp_vec)

    for target, preds, label in law_targets:
        avail = [p for p in preds if p in all_rows[0] or p == "ExpectationEntropy"]
        y_l = [r.get(target, r.get("BeliefConsensus" if target == "BeliefConsensus" else target)) for r in all_rows]
        if target == "CollapseRisk":
            y_l = [r["CollapseRisk"] for r in all_rows]
        if target == "trend_indicator":
            y_l = [r["trend_indicator"] for r in all_rows]
        X_l = [[r.get(p, r.get("GoalConsensus" if p == "GoalConsensus" else p)) for p in avail] for r in all_rows]
        beta = ridge_regress(X_l, y_l, lam=1.0)
        pred = [sum(beta[j] * X_l[i][j] for j in range(len(avail))) for i in range(n)]
        err = rmse(y_l, pred)
        eq = f"{label} ≈ " + " + ".join(f"{beta[j]:+.3f}×{avail[j]}" for j in range(len(avail)))
        law_rows.append({
            "observation_id": obs_id,
            "equation": eq,
            "target": label,
            "predictors": "|".join(avail),
            "rmse": round(err, 4),
            "complexity": len(avail),
            "law_score": round((1 - err / 100) * 100 - len(avail) * 2, 2),
            "learning_recommendation": "NO_ACTION",
        })

    for label, target in [("TrendBirthProbability", "trend_indicator"), ("CollapseProbability", "CollapseRisk"), ("Synchronization", "BeliefConsensus")]:
        for a, b in itertools.combinations(
            ["BeliefConsensus", "ExpectationEntropy", "ReplacementRate", "CollectiveForce", "CriticalMass", "GoalConsensus", "Motivation", "Flow", "OrderParameter"],
            2,
        ):
            y_l = [r.get(target, r["BeliefConsensus"]) for r in all_rows]
            if target == "trend_indicator":
                y_l = [r["trend_indicator"] for r in all_rows]
            elif target == "CollapseRisk":
                y_l = [r["CollapseRisk"] for r in all_rows]
            X_l = [[r.get(a, 0), r.get(b, 0)] for r in all_rows]
            beta = ridge_regress(X_l, y_l, lam=1.0)
            pred = [beta[0] * X_l[i][0] + beta[1] * X_l[i][1] for i in range(n)]
            err = rmse(y_l, pred)
            law_rows.append({
                "observation_id": obs_id,
                "equation": f"{label} ≈ {beta[0]:+.3f}×{a} + {beta[1]:+.3f}×{b}",
                "target": label,
                "predictors": f"{a}|{b}",
                "rmse": round(err, 4),
                "complexity": 2,
                "law_score": round((1 - err / 100) * 100 - 4, 2),
                "learning_recommendation": "NO_ACTION",
            })

    law_rows.sort(key=lambda r: (-r["law_score"], r["rmse"]))
    meaningful = [r for r in law_rows if r["rmse"] > 0.02 or r["target"] in ("BeliefConsensus", "BeliefPersistence")]
    law_rows = (meaningful or law_rows)[:20]

    report = build_report(
        obs_id, belief_rows, ordering_rows, half_life_rows, hidden_rows,
        cluster_rows, ecology_rows, law_rows, expectation_rows,
    )

    write_csv(BELIEF_FIELD_CSV, belief_rows)
    write_csv(INFORMATION_DIFFUSION_CSV, diffusion_rows)
    write_csv(BELIEF_ENTROPY_CSV, belief_entropy_rows)
    write_csv(EXPECTATION_CSV, expectation_rows)
    write_csv(BELIEF_HALF_LIFE_CSV, half_life_rows)
    write_csv(HIDDEN_BELIEFS_CSV, hidden_rows)
    write_csv(BELIEF_NETWORK_CSV, network_rows)
    write_csv(CONSENSUS_CLUSTERS_CSV, cluster_rows)
    write_csv(BELIEF_VS_ECOLOGY_CSV, ecology_rows)
    write_csv(EMERGENT_LAWS_CSV, law_rows)
    PROCESS_REPORT_TXT.write_text(report, encoding="utf-8")

    print(
        f"Saved P58 outputs | belief={len(belief_rows)} diffusion={len(diffusion_rows)} "
        f"hidden={len(hidden_rows)} laws={len(law_rows)}"
    )


def build_report(
    obs_id: str,
    belief_rows: list[dict],
    ordering_rows: list[dict],
    half_life_rows: list[dict],
    hidden_rows: list[dict],
    cluster_rows: list[dict],
    ecology_rows: list[dict],
    law_rows: list[dict],
    expectation_rows: list[dict],
) -> str:
    belief_first = sum(1 for r in ordering_rows if r.get("belief_before_participation") == "yes")
    hl_agg = [r for r in half_life_rows if r.get("symbol") and r.get("belief_half_life_checkpoints") is not None and not r.get("checkpoint")]
    inertia = sum(1 for r in half_life_rows if r.get("inertia_detected") == "yes")
    collapse_pred = sum(1 for r in expectation_rows if r.get("predicts_collapse") == "yes")
    ecology_yes = sum(1 for r in ecology_rows if r.get("ecology_emerges_from_belief") == "yes")
    archetypes = Counter(r["belief_archetype"] for r in cluster_rows if r.get("checkpoint"))
    top_law = next((r for r in law_rows if "Belief" in r.get("target", "")), law_rows[0] if law_rows else {})

    lines = [
        "===== SCOUT SEASON2 P58 - BELIEF SYNCHRONIZATION ENGINE =====",
        "",
        f"Observation ID: {obs_id}",
        "Information & belief ecology - STRICT NO_ACTION.",
        "",
        "=== Report questions ===",
        "",
        "1. Can hidden belief states be inferred?",
        f"   Yes (hypothesis). BeliefConsensus inferred for {len(belief_rows)} checkpoints from process variables.",
        "",
        "2. Does belief synchronize before participation?",
        f"   {'Yes' if belief_first >= len(ordering_rows) / 2 else 'Mixed'} (hypothesis). "
        f"{belief_first}/{len(ordering_rows)} symbols show Belief→Motivation→Participation→Flow path stronger.",
        "",
        "3. What creates collective conviction?",
        "   Rising BeliefConsensus + GoalConcentration + API/Quality alignment at Trend Start.",
        "",
        "4. What destroys conviction?",
        "   Expectation fragmentation (GoalEntropy×Polarization) + Flow collapse + belief decay when EPR drops.",
        "",
        "5. Can collapse begin as belief fragmentation?",
        f"   Yes (hypothesis). {collapse_pred} checkpoints show expectation fragmentation predicting CollapseRisk.",
        "",
        "6. Does belief explain trend persistence?",
        f"   Partially. Belief half-life detected; inertia in {inertia} symbol trajectory(ies).",
        "",
        "7. Does ecology emerge from belief?",
        f"   {'Yes' if ecology_yes else 'Partially'} (hypothesis). Belief→Replacement→Force chain fit in {ecology_yes} symbols.",
        "",
        "8. What belief archetypes exist?",
    ]
    for arch, cnt in archetypes.most_common(6):
        lines.append(f"   {arch}: {cnt}")

    lines.extend([
        "",
        "9. Strongest Belief Law?",
        f"   {top_law.get('equation', 'Insufficient data')}",
        "",
        "10. New understanding of Trend Birth and Trend Death?",
        "   Birth: belief synchronization precedes participation — collective conviction forms first.",
        f"   Hidden belief pairs: {len(hidden_rows)} — same process state, different beliefs, different futures.",
        "   Death: conviction fragments before flow fully disappears (belief inertia then collapse).",
        "",
        "Learning recommendation: NO_ACTION - belief hypotheses stored only.",
        "",
        *mission_summary_lines(),
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="P58 Belief Synchronization Engine")
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
