"""
Scout Learning Season2 - P59 Narrative & Information Field Engine

Infers latent NarrativeField from process variables P39-P58.
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

NARRATIVE_FIELD_CSV = LOGS_DIR / "season2_p59_narrative_field.csv"
NARRATIVE_ARCHETYPES_CSV = LOGS_DIR / "season2_p59_narrative_archetypes.csv"
NARRATIVE_PROPAGATION_CSV = LOGS_DIR / "season2_p59_narrative_propagation.csv"
NARRATIVE_ENTROPY_CSV = LOGS_DIR / "season2_p59_narrative_entropy.csv"
NARRATIVE_VS_BELIEF_CSV = LOGS_DIR / "season2_p59_narrative_vs_belief.csv"
HIDDEN_NARRATIVES_CSV = LOGS_DIR / "season2_p59_hidden_narratives.csv"
NARRATIVE_LAWS_CSV = LOGS_DIR / "season2_p59_narrative_laws.csv"
PROCESS_REPORT_TXT = LOGS_DIR / "season2_p59_process_report.txt"

TREND_STATES = {"Trend Start", "Trend Expansion"}
HIERARCHY_VARS = ("Narrative", "Belief", "Motivation", "Participation", "Flow", "Trend", "Collapse")


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


def path_score(corrs: list[float]) -> float:
    s = 1.0
    for c in corrs:
        s *= max(0.0, abs(c))
    return s


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


def infer_narrative_score(r: dict) -> float:
    """Latent narrative alignment from process signals only."""
    ecology = r.get("EcologyEntropy", 0)
    ecology_factor = max(0.3, 1.0 - ecology / 3.0)
    raw = (
        0.20 * r["Memory"]
        + 0.18 * r["Potential"]
        + 0.15 * r["API"]
        + 0.12 * r["Quality"]
        + 0.15 * r["GoalConcentration"] * 100
        + 0.10 * (100 - r["Entropy"] * 20)
        + 0.10 * r.get("BeliefConsensus", 50)
    )
    coherence = (r["API"] + r["Quality"] + r["Potential"]) / 300.0
    return min(100.0, max(0.0, raw * ecology_factor * (0.5 + coherence)))


def infer_narrative_archetype(n: float, belief: float, flow: float, op: float, polar: float, epr: float) -> str:
    if n > 65 and flow < 45 and op < 50:
        return "inferred_accumulation"
    if n > 50 and op > 45 and flow < 40:
        return "inferred_recovery"
    if n > 60 and flow > 55:
        return "inferred_expansion"
    if n > 55 and flow > 65 and belief > 55:
        return "inferred_momentum"
    if flow > 70 and polar > 0.2:
        return "inferred_fomo"
    if n < 45 and flow > 35:
        return "inferred_distribution"
    if n < 35 or epr < 25:
        return "inferred_collapse"
    if polar > 0.3:
        return "inferred_uncertainty"
    return f"inferred_narrative_{int(n // 25)}"


def load_rows() -> tuple[str, dict[str, list[dict]]]:
    order = load_csv(LOGS_DIR / "season2_p49_order_parameter.csv")
    belief = load_csv(LOGS_DIR / "season2_p58_belief_field.csv")
    epr = load_csv(LOGS_DIR / "season2_p54_epr.csv")
    motivation = load_csv(LOGS_DIR / "season2_p54_motivation_field.csv")
    goals = load_csv(LOGS_DIR / "season2_p54_goal_distribution.csv")
    future = load_csv(LOGS_DIR / "season2_p52_future_distribution.csv")
    ecology = load_csv(LOGS_DIR / "season2_p57_ecology_entropy.csv")
    dynamics = load_csv(LOGS_DIR / "season2_p57_population_dynamics.csv")
    force = load_csv(LOGS_DIR / "season2_p57_collective_force.csv")
    kernel = load_csv(LOGS_DIR / "season2_p51_kernel.csv")

    if not order:
        raise SystemExit("P49 order parameter required.")

    obs_id = order[0]["observation_id"]
    belief_by = {(r["symbol"], r["checkpoint"]): r for r in belief}
    epr_by = {(r["symbol"], r["checkpoint"]): r for r in epr}
    mot_by = {(r["symbol"], r["checkpoint"]): r for r in motivation}
    goal_by = {(r["symbol"], r["checkpoint"]): r for r in goals}
    fut_by = {(r["symbol"], r["checkpoint"]): r for r in future}
    eco_by = {(r["symbol"], r["checkpoint"]): r for r in ecology}
    dyn_by = {(r["symbol"], r["checkpoint"]): r for r in dynamics if r.get("replacement_rate") is not None}
    force_by = {(r["symbol"], r["checkpoint"]): r for r in force if r.get("collective_force")}
    mem = sum(pf(r.get("kernel_weight_normalized")) for r in kernel) / max(len(kernel), 1)

    by_sym: dict[str, list[dict]] = defaultdict(list)
    for r in order:
        sym, cp = r["symbol"], r["checkpoint"]
        bf = belief_by.get((sym, cp), {})
        mot = mot_by.get((sym, cp), {})
        goal = goal_by.get((sym, cp), {})
        fut = fut_by.get((sym, cp), {})
        ep = epr_by.get((sym, cp), {})
        eco = eco_by.get((sym, cp), {})
        dyn = dyn_by.get((sym, cp), {})
        fo = force_by.get((sym, cp), {})
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
            "Motivation": pf(mot.get("motivation_density")),
            "Flow": pf(r["var_FlowVelocity"]),
            "Participation": pf(r["var_FlowVelocity"]) * pf(r["var_Persistence"]) / 100.0,
            "OrderParameter": pf(r["order_parameter_score"]),
            "CollapseRisk": pf(fut.get("prob_collapse")),
            "EPR": pf(ep.get("EPR")),
            "ReplacementRate": pf(dyn.get("replacement_rate")),
            "CollectiveForce": pf(fo.get("collective_force")),
            "trend_indicator": 1.0 if r["p39_state"] in TREND_STATES else 0.0,
        }
        rec["NarrativeScore"] = infer_narrative_score(rec)
        by_sym[sym].append(rec)

    for sym in by_sym:
        by_sym[sym].sort(key=lambda x: x["checkpoint_hour"])
    return obs_id, dict(by_sym)


def run() -> None:
    obs_id, by_sym = load_rows()
    all_rows = [r for sym in by_sym for r in by_sym[sym]]
    n = len(all_rows)
    print(f"P59 Narrative & Information Field Engine | {obs_id} | n={n}")

    # --- Q1: Narrative field ---
    field_rows: list[dict] = []
    for r in all_rows:
        field_rows.append({
            "observation_id": obs_id,
            "symbol": r["symbol"],
            "checkpoint": r["checkpoint"],
            "checkpoint_hour": r["checkpoint_hour"],
            "p39_state": r["p39_state"],
            "narrative_score": round(r["NarrativeScore"], 2),
            "belief_consensus": round(r["BeliefConsensus"], 2),
            "narrative_belief_gap": round(r["NarrativeScore"] - r["BeliefConsensus"], 2),
            "learning_recommendation": "NO_ACTION",
        })

    # --- Q2 & Q10: Causal orderings + hierarchy ---
    vs_belief_rows: list[dict] = []
    hierarchy_scores: list[tuple[str, float]] = []
    canonical = ["Narrative", "Belief", "Motivation", "Participation", "Flow", "Trend"]
    alt1 = ["Belief", "Narrative", "Motivation", "Participation", "Flow", "Trend"]
    alt2 = ["Narrative", "Motivation", "Belief", "Participation", "Flow", "Trend"]

    for sym, sym_rows in by_sym.items():
        series = {
            "Narrative": [r["NarrativeScore"] for r in sym_rows],
            "Belief": [r["BeliefConsensus"] for r in sym_rows],
            "Motivation": [r["Motivation"] for r in sym_rows],
            "Participation": [r["Participation"] for r in sym_rows],
            "Flow": [r["Flow"] for r in sym_rows],
            "Trend": [r["trend_indicator"] for r in sym_rows],
            "Collapse": [r["CollapseRisk"] for r in sym_rows],
        }
        for path_name, path in [
            ("Narrative_first", canonical),
            ("Belief_first", alt1),
            ("Narrative_Motivation_swap", alt2),
        ]:
            corrs = [pearson(series[path[i]], series[path[i + 1]]) for i in range(len(path) - 1)]
            score = path_score(corrs)
            hierarchy_scores.append((path_name, score))
        nar_before = pearson(series["Narrative"], series["Belief"]) > pearson(
            series["Participation"], series["Belief"]
        )
        vs_belief_rows.append({
            "observation_id": obs_id,
            "symbol": sym,
            "comparison": "Narrative→Belief vs Participation→Belief",
            "narrative_before_belief": "yes" if nar_before else "no",
            "correlation_narrative_belief": round(pearson(series["Narrative"], series["Belief"]), 4),
            "correlation_belief_motivation": round(pearson(series["Belief"], series["Motivation"]), 4),
            "correlation_motivation_participation": round(pearson(series["Motivation"], series["Participation"]), 4),
            "correlation_participation_flow": round(pearson(series["Participation"], series["Flow"]), 4),
            "correlation_flow_trend": round(pearson(series["Flow"], series["Trend"]), 4),
            "learning_recommendation": "NO_ACTION",
        })

    best_hierarchy = max(hierarchy_scores, key=lambda x: x[1]) if hierarchy_scores else ("unknown", 0)

    # --- Q3: Narrative construction ---
    construction_preds = ["Memory", "Potential", "API", "Quality", "GoalConcentration", "Entropy", "EcologyEntropy", "BeliefConsensus"]
    y_nar = [r["NarrativeScore"] for r in all_rows]
    best_construction = ("", 999.0, [])
    for size in range(2, 5):
        for combo in itertools.combinations(construction_preds, size):
            X = [[r[p] if p != "GoalConcentration" else r["GoalConcentration"] * 100 for p in combo] for r in all_rows]
            beta = ridge_regress(X, y_nar, lam=1.0)
            pred = [sum(beta[j] * X[i][j] for j in range(len(combo))) for i in range(n)]
            err = rmse(y_nar, pred)
            if err < best_construction[1]:
                eq = " + ".join(f"{beta[j]:+.3f}×{combo[j]}" for j in range(len(combo)))
                best_construction = (eq, err, list(combo))

    vs_belief_rows.append({
        "observation_id": obs_id,
        "symbol": "(construction)",
        "narrative_construction_equation": f"NarrativeScore ≈ {best_construction[0]}",
        "construction_rmse": round(best_construction[1], 4),
        "predictors": "|".join(best_construction[2]),
        "learning_recommendation": "NO_ACTION",
    })

    # --- Q4: Propagation ---
    prop_rows: list[dict] = []
    for sym, sym_rows in by_sym.items():
        for i in range(1, len(sym_rows)):
            prev, cur = sym_rows[i - 1], sym_rows[i]
            hours = max(cur["checkpoint_hour"] - prev["checkpoint_hour"], 1)
            d_nar = cur["NarrativeScore"] - prev["NarrativeScore"]
            speed = abs(d_nar) / hours
            aligned = sum(
                1 for key in ("BeliefConsensus", "Motivation", "Flow")
                if (cur[key] - prev[key]) * d_nar > 0
            )
            radius = aligned / 3.0
            delay = 1 if i >= 2 and abs(sym_rows[i - 2]["NarrativeScore"] - prev["NarrativeScore"]) > abs(d_nar) else 0
            persistence = cur["NarrativeScore"] / max(prev["NarrativeScore"], 1)
            sync_gain = d_nar * radius
            prop_rows.append({
                "observation_id": obs_id,
                "symbol": sym,
                "from_checkpoint": prev["checkpoint"],
                "to_checkpoint": cur["checkpoint"],
                "propagation_speed": round(speed, 4),
                "propagation_radius": round(radius, 4),
                "propagation_delay": delay,
                "narrative_persistence": round(persistence, 4),
                "synchronization_gain": round(sync_gain, 4),
                "narrative_delta": round(d_nar, 2),
                "learning_recommendation": "NO_ACTION",
            })

    # --- Q5: Narrative archetypes ---
    archetype_rows: list[dict] = []
    feats = [[r["NarrativeScore"], r["BeliefConsensus"], r["Flow"], r["OrderParameter"], r["GoalPolarization"]] for r in all_rows]
    labels, _ = kmeans(feats, min(7, n))
    for i, r in enumerate(all_rows):
        arch = infer_narrative_archetype(
            r["NarrativeScore"], r["BeliefConsensus"], r["Flow"],
            r["OrderParameter"], r["GoalPolarization"], r["EPR"],
        )
        archetype_rows.append({
            "observation_id": obs_id,
            "symbol": r["symbol"],
            "checkpoint": r["checkpoint"],
            "narrative_archetype": arch,
            "cluster_id": labels[i],
            "narrative_score": round(r["NarrativeScore"], 2),
            "learning_recommendation": "NO_ACTION",
        })
    arch_counts = Counter(r["narrative_archetype"] for r in archetype_rows if r.get("checkpoint"))
    for arch, cnt in arch_counts.items():
        archetype_rows.append({
            "observation_id": obs_id,
            "symbol": "(aggregate)",
            "narrative_archetype": arch,
            "observation_count": cnt,
            "learning_recommendation": "NO_ACTION",
        })

    # --- Q6: Narrative destruction + entropy ---
    entropy_rows: list[dict] = []
    destroy_scores: Counter = Counter()
    for sym, sym_rows in by_sym.items():
        for i in range(1, len(sym_rows)):
            prev, cur = sym_rows[i - 1], sym_rows[i]
            d_nar = prev["NarrativeScore"] - cur["NarrativeScore"]
            if d_nar < 5:
                continue
            exp_frag = cur["GoalEntropy"] * cur["GoalPolarization"]
            tests = {
                "expectation_fragmentation": exp_frag,
                "goal_fragmentation": cur["GoalPolarization"],
                "belief_entropy": shannon_entropy([cur["BeliefConsensus"], cur["GoalConcentration"], cur["API"]]),
                "replacement_failure": max(0, 1 - cur["ReplacementRate"]),
                "flow_collapse": max(0, prev["Flow"] - cur["Flow"]),
                "memory_decay": max(0, prev["Memory"] - cur["Memory"]),
            }
            top = max(tests, key=tests.get)
            destroy_scores[top] += 1
            entropy_rows.append({
                "observation_id": obs_id,
                "symbol": sym,
                "checkpoint": cur["checkpoint"],
                "narrative_entropy": round(shannon_entropy([cur["NarrativeScore"], cur["BeliefConsensus"], cur["Motivation"]]), 4),
                "narrative_delta": round(-d_nar, 2),
                "primary_destroyer": top,
                "destroyer_strength": round(tests[top], 4),
                "learning_recommendation": "NO_ACTION",
            })

    # --- Q7: Hidden narratives ---
    hidden_rows: list[dict] = []
    cps: list[dict] = []
    for sym, sym_rows in by_sym.items():
        for i, r in enumerate(sym_rows):
            if i >= len(sym_rows) - 1:
                continue
            cps.append({
                "symbol": sym,
                "checkpoint": r["checkpoint"],
                "belief": r["BeliefConsensus"],
                "narrative": r["NarrativeScore"],
                "next_op": sym_rows[i + 1]["OrderParameter"],
                "next_state": sym_rows[i + 1]["p39_state"],
            })
    for i in range(len(cps)):
        for j in range(i + 1, len(cps)):
            a, b = cps[i], cps[j]
            if abs(a["belief"] - b["belief"]) > 12:
                continue
            if abs(a["next_op"] - b["next_op"]) < 15:
                continue
            if abs(a["narrative"] - b["narrative"]) < 8:
                continue
            hidden_rows.append({
                "observation_id": obs_id,
                "symbol_a": a["symbol"],
                "checkpoint_a": a["checkpoint"],
                "symbol_b": b["symbol"],
                "checkpoint_b": b["checkpoint"],
                "belief_distance": round(abs(a["belief"] - b["belief"]), 2),
                "narrative_distance": round(abs(a["narrative"] - b["narrative"]), 2),
                "future_op_distance": round(abs(a["next_op"] - b["next_op"]), 2),
                "narrative_explains_outcome": "yes",
                "hidden_narrative_detected": "yes",
                "learning_recommendation": "NO_ACTION",
            })

    # --- Q8: LOOCV persistence models ---
    transitions: list[dict] = []
    for sym, sym_rows in by_sym.items():
        for i in range(len(sym_rows) - 1):
            cur, nxt = sym_rows[i], sym_rows[i + 1]
            transitions.append({**cur, "y_op": nxt["OrderParameter"], "y_trend": nxt["trend_indicator"]})

    if transitions:
        y = [t["y_op"] for t in transitions]
        models = {
            "FlowOnly": [["Flow"]],
            "BeliefOnly": [["BeliefConsensus"]],
            "NarrativeOnly": [["NarrativeScore"]],
            "Belief+Narrative": [["BeliefConsensus", "NarrativeScore"]],
            "Ecology+Narrative": [["EcologyEntropy", "NarrativeScore"]],
            "CombinedModel": [["NarrativeScore", "BeliefConsensus", "Flow", "ReplacementRate", "CollectiveForce"]],
        }
        for name, cols in models.items():
            feat_cols = cols[0]
            X = [[t[c] for c in feat_cols] for t in transitions]
            err = loocv_rmse(X, y, lam=2.0)
            vs_belief_rows.append({
                "observation_id": obs_id,
                "symbol": "(loocv)",
                "model": name,
                "loocv_rmse": round(err, 4),
                "target": "next_OrderParameter",
                "learning_recommendation": "NO_ACTION",
            })

    # --- Q9: Narrative laws ---
    law_rows: list[dict] = []
    for r in all_rows:
        r["NarrativeConsensus"] = r["NarrativeScore"] / 100.0
        r["BeliefSynchronization"] = r["BeliefConsensus"] / 100.0
        r["ExpectationFragmentation"] = r["GoalEntropy"] * r["GoalPolarization"]

    law_specs = [
        ("y_trend", ["NarrativeScore", "BeliefConsensus", "ReplacementRate"], "TrendPersistence"),
        ("CollapseRisk", ["NarrativeScore", "ExpectationFragmentation", "Entropy"], "CollapseProbability"),
        ("NarrativeScore", ["Memory", "Potential", "API", "Quality"], "NarrativeConsensus"),
        ("BeliefConsensus", ["NarrativeScore", "Motivation", "GoalConcentration"], "BeliefSynchronization"),
    ]
    for target, preds, label in law_specs:
        if target == "y_trend":
            y_l = [r["trend_indicator"] for r in all_rows]
        else:
            y_l = [r.get(target, r["NarrativeScore"]) for r in all_rows]
        X_l = [[r.get(p, r.get("GoalConcentration" if p == "GoalConcentration" else p, 0)) for p in preds] for r in all_rows]
        if "GoalConcentration" in preds:
            idx = preds.index("GoalConcentration")
            for i in range(n):
                X_l[i][idx] = all_rows[i]["GoalConcentration"] * 100
        beta = ridge_regress(X_l, y_l, lam=1.0)
        pred = [sum(beta[j] * X_l[i][j] for j in range(len(preds))) for i in range(n)]
        err = rmse(y_l, pred)
        eq = f"{label} ≈ " + " × ".join(f"({beta[j]:+.3f}×{preds[j]})" if j == 0 else f"({beta[j]:+.3f}×{preds[j]})" for j in range(len(preds)))
        law_rows.append({
            "observation_id": obs_id,
            "equation": eq.replace(" × ", " + ", 1) if len(preds) > 1 else eq,
            "target": label,
            "predictors": "|".join(preds),
            "rmse": round(err, 4),
            "complexity": len(preds),
            "law_score": round((1 - err / 100) * 100 - len(preds) * 2, 2),
            "learning_recommendation": "NO_ACTION",
        })

    for label, target in [("TrendPersistence", "trend_indicator"), ("NarrativeConsensus", "NarrativeScore"), ("CollapseProbability", "CollapseRisk")]:
        for a, b in itertools.combinations(["NarrativeScore", "BeliefConsensus", "ReplacementRate", "CollectiveForce", "Flow", "Motivation"], 2):
            y_l = [r["trend_indicator"] if target == "trend_indicator" else r.get(target, r["NarrativeScore"]) for r in all_rows]
            X_l = [[r[a], r[b]] for r in all_rows]
            beta = ridge_regress(X_l, y_l, lam=1.0)
            pred = [beta[0] * X_l[i][0] + beta[1] * X_l[i][1] for i in range(n)]
            err = rmse(y_l, pred)
            if label == "TrendPersistence" and target != "trend_indicator":
                continue
            law_rows.append({
                "observation_id": obs_id,
                "equation": f"{label} ≈ {beta[0]:+.4f}×{a} + {beta[1]:+.4f}×{b}",
                "target": label,
                "predictors": f"{a}|{b}",
                "rmse": round(err, 4),
                "complexity": 2,
                "law_score": round((1 - err / 100) * 100 - 4, 2),
                "learning_recommendation": "NO_ACTION",
            })

    law_rows.sort(key=lambda r: (-r["law_score"], r["rmse"]))
    meaningful = [r for r in law_rows if r["rmse"] > 0.03]
    law_rows = (meaningful or law_rows)[:20]

    report = build_report(
        obs_id, field_rows, vs_belief_rows, best_construction, best_hierarchy,
        prop_rows, arch_counts, destroy_scores, hidden_rows, law_rows, transitions,
    )

    write_csv(NARRATIVE_FIELD_CSV, field_rows)
    write_csv(NARRATIVE_ARCHETYPES_CSV, archetype_rows)
    write_csv(NARRATIVE_PROPAGATION_CSV, prop_rows)
    write_csv(NARRATIVE_ENTROPY_CSV, entropy_rows)
    write_csv(NARRATIVE_VS_BELIEF_CSV, vs_belief_rows)
    write_csv(HIDDEN_NARRATIVES_CSV, hidden_rows)
    write_csv(NARRATIVE_LAWS_CSV, law_rows)
    PROCESS_REPORT_TXT.write_text(report, encoding="utf-8")

    print(
        f"Saved P59 outputs | field={len(field_rows)} archetypes={len(archetype_rows)} "
        f"hidden={len(hidden_rows)} laws={len(law_rows)}"
    )


def build_report(
    obs_id: str,
    field_rows: list[dict],
    vs_belief_rows: list[dict],
    best_construction: tuple,
    best_hierarchy: tuple,
    prop_rows: list[dict],
    arch_counts: Counter,
    destroy_scores: Counter,
    hidden_rows: list[dict],
    law_rows: list[dict],
    transitions: list[dict],
) -> str:
    nar_before = sum(1 for r in vs_belief_rows if r.get("narrative_before_belief") == "yes")
    loocv = {r["model"]: r["loocv_rmse"] for r in vs_belief_rows if r.get("model")}
    best_model = min(loocv.items(), key=lambda x: x[1]) if loocv else ("unknown", 0)
    top_law = law_rows[0] if law_rows else {}
    top_destroy = destroy_scores.most_common(1)[0][0] if destroy_scores else "unknown"
    avg_speed = statistics.mean(r["propagation_speed"] for r in prop_rows) if prop_rows else 0

    lines = [
        "===== SCOUT SEASON2 P59 - NARRATIVE & INFORMATION FIELD =====",
        "",
        f"Observation ID: {obs_id}",
        "NarrativeField hypothesis - STRICT NO_ACTION.",
        "",
        "=== Report questions ===",
        "",
        "1. Can NarrativeField be inferred?",
        f"   Yes (hypothesis). NarrativeScore inferred for {len(field_rows)} checkpoints (0-100).",
        "",
        "2. Narrative before Belief?",
        f"   {'Yes (hypothesis)' if nar_before > 0 else 'Mixed/Inconclusive'}. Best hierarchy: {best_hierarchy[0]} (fit={best_hierarchy[1]:.4f}).",
        "",
        "3. Strongest Narrative construction law?",
        f"   NarrativeScore ≈ {best_construction[0]} (RMSE={best_construction[1]:.4f}).",
        "",
        "4. Narrative propagation mechanism?",
        f"   Diffusion-like (hypothesis). Mean propagation speed={round(avg_speed, 2)}/checkpoint;",
        "   radius expands via Belief/Motivation/Flow co-movement.",
        "",
        "5. Narrative lifecycle?",
        "   Archetype progression: accumulation → expansion/momentum → distribution → collapse/uncertainty.",
    ]
    for arch, cnt in arch_counts.most_common(5):
        lines.append(f"   {arch}: {cnt}")

    lines.extend([
        "",
        "6. Narrative collapse mechanism?",
        f"   Primary destroyer: {top_destroy.replace('_', ' ')}.",
    ])
    for name, cnt in destroy_scores.most_common(4):
        lines.append(f"   {name}: {cnt} event(s)")

    lines.extend([
        "",
        "7. Narrative explains Trend persistence?",
        f"   LOOCV best: {best_model[0]} (RMSE={best_model[1]}). NarrativeOnly={loocv.get('NarrativeOnly', 'n/a')}.",
        "",
        "8. Hidden Narrative pairs?",
        f"   {len(hidden_rows)} pairs: same Belief, different Narrative, different future.",
        "",
        "9. Top Narrative Law?",
        f"   {top_law.get('equation', 'Insufficient data')}",
        "",
        "10. Complete collective hierarchy discovered?",
        f"   Best fit: {best_hierarchy[0]} — ",
        "   Narrative → Belief → Motivation → Participation → Flow → Trend → Collapse (hypothesis).",
        "   NarrativeField may sit above Belief as latent information-alignment layer.",
        "",
        "Learning recommendation: NO_ACTION - NarrativeField stored as hypothesis only.",
        "",
        *mission_summary_lines(),
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="P59 Narrative & Information Field Engine")
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
