"""
Scout Learning Season2 - P60 Attention & Competition Field Engine

Infers latent AttentionField from process variables P39-P59.
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

ATTENTION_FIELD_CSV = LOGS_DIR / "season2_p60_attention_field.csv"
ATTENTION_PROPAGATION_CSV = LOGS_DIR / "season2_p60_attention_propagation.csv"
ATTENTION_ENTROPY_CSV = LOGS_DIR / "season2_p60_attention_entropy.csv"
ATTENTION_COMPETITION_CSV = LOGS_DIR / "season2_p60_attention_competition.csv"
ATTENTION_SWITCH_CSV = LOGS_DIR / "season2_p60_attention_switch.csv"
HIDDEN_ATTENTION_CSV = LOGS_DIR / "season2_p60_hidden_attention_pairs.csv"
ATTENTION_LAWS_CSV = LOGS_DIR / "season2_p60_attention_laws.csv"
PROCESS_REPORT_TXT = LOGS_DIR / "season2_p60_process_report.txt"

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


def shannon_entropy(vals: list[float]) -> float:
    total = sum(abs(v) for v in vals) or 1.0
    h = 0.0
    for v in vals:
        p = abs(v) / total
        if p > 1e-12:
            h -= p * math.log2(p)
    return h


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


def infer_attention_score(r: dict, competition_factor: float = 1.0) -> float:
    """Latent attention allocation from process signals only."""
    ecology_penalty = max(0.25, 1.0 - r.get("EcologyEntropy", 0) / 2.8)
    entropy_penalty = max(0.35, 1.0 - r.get("Entropy", 0) * 0.12)
    concentration = r.get("GoalConcentration", 0.5) * 100
    salience = (r["Potential"] + r["API"] + r["Quality"]) / 300.0
    raw = (
        0.24 * r["Potential"]
        + 0.20 * r["API"]
        + 0.16 * r["Quality"]
        + 0.14 * concentration
        + 0.10 * r.get("Memory", 0) * 8.0
        + 0.08 * r.get("Motivation", 0)
        + 0.08 * (100 - min(r.get("BeliefConsensus", 50), 100))
    )
    return min(100.0, max(0.0, raw * ecology_penalty * entropy_penalty * salience * competition_factor))


def load_rows() -> tuple[str, dict[str, list[dict]]]:
    order = load_csv(LOGS_DIR / "season2_p49_order_parameter.csv")
    narrative = load_csv(LOGS_DIR / "season2_p59_narrative_field.csv")
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
    nar_by = {(r["symbol"], r["checkpoint"]): r for r in narrative}
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
        nar = nar_by.get((sym, cp), {})
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
            "NarrativeScore": pf(nar.get("narrative_score")),
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
        by_sym[sym].append(rec)

    for sym in by_sym:
        by_sym[sym].sort(key=lambda x: x["checkpoint_hour"])

    # Cross-symbol competition at each hour
    by_hour: dict[int, list[dict]] = defaultdict(list)
    for sym_rows in by_sym.values():
        for r in sym_rows:
            by_hour[r["checkpoint_hour"]].append(r)

    for hour, rows_at_h in by_hour.items():
        total_salience = sum(r["Potential"] + r["API"] + r["Quality"] for r in rows_at_h) or 1.0
        for r in rows_at_h:
            share = (r["Potential"] + r["API"] + r["Quality"]) / total_salience
            r["AttentionShare"] = share
            r["AttentionScore"] = infer_attention_score(r, competition_factor=0.5 + share)

    return obs_id, dict(by_sym)


def hierarchy_paths() -> list[tuple[str, list[str]]]:
    base = ["Attention", "Narrative", "Belief", "Motivation", "Participation", "Flow", "Trend", "Collapse"]
    paths = [("Attention_first", base)]
    swaps = [
        ("Narrative_first", ["Narrative", "Attention", "Belief", "Motivation", "Participation", "Flow", "Trend", "Collapse"]),
        ("Belief_first", ["Belief", "Attention", "Narrative", "Motivation", "Participation", "Flow", "Trend", "Collapse"]),
        ("Motivation_first", ["Motivation", "Attention", "Narrative", "Belief", "Participation", "Flow", "Trend", "Collapse"]),
        ("Flow_first", ["Flow", "Participation", "Motivation", "Belief", "Narrative", "Attention", "Trend", "Collapse"]),
        ("Attention_Narrative_swap", ["Narrative", "Belief", "Attention", "Motivation", "Participation", "Flow", "Trend", "Collapse"]),
    ]
    return paths + swaps


def run() -> None:
    obs_id, by_sym = load_rows()
    all_rows = [r for sym in by_sym for r in by_sym[sym]]
    n = len(all_rows)
    print(f"P60 Attention & Competition Field Engine | {obs_id} | n={n}")

    # --- Q1: Attention field ---
    field_rows: list[dict] = []
    for r in all_rows:
        field_rows.append({
            "observation_id": obs_id,
            "symbol": r["symbol"],
            "checkpoint": r["checkpoint"],
            "checkpoint_hour": r["checkpoint_hour"],
            "p39_state": r["p39_state"],
            "attention_score": round(r["AttentionScore"], 2),
            "attention_share": round(r.get("AttentionShare", 0), 4),
            "narrative_score": round(r["NarrativeScore"], 2),
            "attention_narrative_gap": round(r["AttentionScore"] - r["NarrativeScore"], 2),
            "learning_recommendation": "NO_ACTION",
        })

    # --- Q2 & Q10: Hierarchy ---
    hierarchy_scores: list[tuple[str, float]] = []
    attention_before_narrative_count = 0
    hierarchy_detail_rows: list[dict] = []

    for sym, sym_rows in by_sym.items():
        series = {
            "Attention": [r["AttentionScore"] for r in sym_rows],
            "Narrative": [r["NarrativeScore"] for r in sym_rows],
            "Belief": [r["BeliefConsensus"] for r in sym_rows],
            "Motivation": [r["Motivation"] for r in sym_rows],
            "Participation": [r["Participation"] for r in sym_rows],
            "Flow": [r["Flow"] for r in sym_rows],
            "Trend": [r["trend_indicator"] for r in sym_rows],
            "Collapse": [r["CollapseRisk"] for r in sym_rows],
        }
        att_before = pearson(series["Attention"], series["Narrative"]) > pearson(
            series["Flow"], series["Narrative"]
        )
        if att_before:
            attention_before_narrative_count += 1
        for path_name, path in hierarchy_paths():
            corrs = [pearson(series[path[i]], series[path[i + 1]]) for i in range(len(path) - 1)]
            score = path_score(corrs)
            hierarchy_scores.append((path_name, score, sym))
        hierarchy_detail_rows.append({
            "observation_id": obs_id,
            "symbol": sym,
            "attention_before_narrative": "yes" if att_before else "no",
            "correlation_attention_narrative": round(pearson(series["Attention"], series["Narrative"]), 4),
            "correlation_narrative_belief": round(pearson(series["Narrative"], series["Belief"]), 4),
            "correlation_belief_motivation": round(pearson(series["Belief"], series["Motivation"]), 4),
            "correlation_motivation_participation": round(pearson(series["Motivation"], series["Participation"]), 4),
            "correlation_participation_flow": round(pearson(series["Participation"], series["Flow"]), 4),
            "correlation_flow_trend": round(pearson(series["Flow"], series["Trend"]), 4),
            "learning_recommendation": "NO_ACTION",
        })

    path_totals: Counter = Counter()
    for path_name, score, _ in hierarchy_scores:
        path_totals[path_name] += score
    best_hierarchy = max(path_totals.items(), key=lambda x: x[1]) if path_totals else ("unknown", 0)

    # --- Q3: Attention construction ---
    construction_preds = [
        "Memory", "Potential", "API", "Quality", "Entropy", "EcologyEntropy",
        "BeliefConsensus", "NarrativeScore", "Motivation", "Participation",
    ]
    y_att = [r["AttentionScore"] for r in all_rows]
    best_construction = ("", 999.0, [])
    for size in range(2, 5):
        for combo in itertools.combinations(construction_preds, size):
            X = [[r[p] for p in combo] for r in all_rows]
            beta = ridge_regress(X, y_att, lam=1.0)
            pred = [sum(beta[j] * X[i][j] for j in range(len(combo))) for i in range(n)]
            err = rmse(y_att, pred)
            if err < best_construction[1]:
                eq = " + ".join(f"{beta[j]:+.3f}×{combo[j]}" for j in range(len(combo)))
                best_construction = (eq, err, list(combo))

    # --- Q4: Propagation ---
    prop_rows: list[dict] = []
    for sym, sym_rows in by_sym.items():
        for i in range(1, len(sym_rows)):
            prev, cur = sym_rows[i - 1], sym_rows[i]
            hours = max(cur["checkpoint_hour"] - prev["checkpoint_hour"], 1)
            d_att = cur["AttentionScore"] - prev["AttentionScore"]
            speed = abs(d_att) / hours
            aligned = sum(
                1 for key in ("NarrativeScore", "BeliefConsensus", "Motivation", "Flow")
                if (cur[key] - prev[key]) * d_att > 0
            )
            radius = aligned / 4.0
            persistence = cur["AttentionScore"] / max(prev["AttentionScore"], 1)
            amplification = d_att * radius if d_att > 0 else 0.0
            decay = abs(d_att) if d_att < 0 else 0.0
            transfer = 0.0
            hour = cur["checkpoint_hour"]
            others = [r for s, rows in by_sym.items() if s != sym for r in rows if r["checkpoint_hour"] == hour]
            if others:
                transfer = sum(o["AttentionScore"] - prev["AttentionScore"] for o in others) / len(others)
            prop_rows.append({
                "observation_id": obs_id,
                "symbol": sym,
                "from_checkpoint": prev["checkpoint"],
                "to_checkpoint": cur["checkpoint"],
                "propagation_speed": round(speed, 4),
                "propagation_radius": round(radius, 4),
                "persistence": round(persistence, 4),
                "amplification": round(amplification, 4),
                "decay": round(decay, 4),
                "transfer": round(transfer, 4),
                "attention_delta": round(d_att, 2),
                "learning_recommendation": "NO_ACTION",
            })

    # --- Q5: Attention conservation & competition ---
    competition_rows: list[dict] = []
    conservation_scores: list[float] = []
    for hour in sorted({r["checkpoint_hour"] for r in all_rows}):
        at_hour = [r for r in all_rows if r["checkpoint_hour"] == hour]
        if len(at_hour) < 2:
            continue
        scores = [r["AttentionScore"] for r in at_hour]
        total = sum(scores)
        shares = [s / total if total else 0 for s in scores]
        entropy = shannon_entropy(scores)
        competition_index = 1.0 - entropy / math.log2(len(scores)) if len(scores) > 1 else 0.0
        conservation_scores.append(total)
        for i, r in enumerate(at_hour):
            competition_rows.append({
                "observation_id": obs_id,
                "checkpoint_hour": hour,
                "symbol": r["symbol"],
                "attention_score": round(r["AttentionScore"], 2),
                "attention_share": round(shares[i], 4),
                "attention_competition_index": round(competition_index, 4),
                "total_attention_pool": round(total, 2),
                "learning_recommendation": "NO_ACTION",
            })

    pool_std = statistics.pstdev(conservation_scores) if len(conservation_scores) > 1 else 0.0
    pool_mean = statistics.mean(conservation_scores) if conservation_scores else 0.0
    conserved = pool_std / max(pool_mean, 1) < 0.25

    # --- Q6: Narrative competition / attention switching ---
    switch_rows: list[dict] = []
    symbols = sorted(by_sym.keys())
    if len(symbols) >= 2:
        sym_a, sym_b = symbols[0], symbols[1]
        rows_a = by_sym[sym_a]
        rows_b = by_sym[sym_b]
        prev_leader = None
        for i in range(min(len(rows_a), len(rows_b))):
            ra, rb = rows_a[i], rows_b[i]
            leader = sym_a if ra["AttentionScore"] >= rb["AttentionScore"] else sym_b
            margin = abs(ra["AttentionScore"] - rb["AttentionScore"])
            mode = "coexistence"
            if margin > 25:
                mode = "winner_take_all"
            elif margin > 10:
                mode = "dominance"
            elif margin < 3:
                mode = "cycling" if prev_leader and prev_leader != leader else "coexistence"
            switched = prev_leader is not None and leader != prev_leader
            if switched:
                mode = "switching"
            switch_rows.append({
                "observation_id": obs_id,
                "checkpoint": ra["checkpoint"],
                "checkpoint_hour": ra["checkpoint_hour"],
                "leader_symbol": leader,
                "attention_margin": round(margin, 2),
                "competition_mode": mode,
                "attention_switched": "yes" if switched else "no",
                "narrative_a": round(ra["NarrativeScore"], 2),
                "narrative_b": round(rb["NarrativeScore"], 2),
                "attention_a": round(ra["AttentionScore"], 2),
                "attention_b": round(rb["AttentionScore"], 2),
                "learning_recommendation": "NO_ACTION",
            })
            prev_leader = leader

    switch_counts = Counter(r["competition_mode"] for r in switch_rows)

    # --- Q7: Attention collapse ---
    entropy_rows: list[dict] = []
    destroy_scores: Counter = Counter()
    for sym, sym_rows in by_sym.items():
        for i in range(1, len(sym_rows)):
            prev, cur = sym_rows[i - 1], sym_rows[i]
            d_att = prev["AttentionScore"] - cur["AttentionScore"]
            if d_att < 4:
                continue
            tests = {
                "entropy": cur["Entropy"],
                "belief_fragmentation": shannon_entropy([cur["BeliefConsensus"], cur["GoalConcentration"], cur["API"]]),
                "goal_fragmentation": cur["GoalPolarization"],
                "replacement_failure": max(0, 1 - cur["ReplacementRate"]),
                "flow_exhaustion": max(0, prev["Flow"] - cur["Flow"]),
                "narrative_decay": max(0, prev["NarrativeScore"] - cur["NarrativeScore"]),
                "memory_decay": max(0, prev["Memory"] - cur["Memory"]),
            }
            top = max(tests, key=tests.get)
            destroy_scores[top] += 1
            entropy_rows.append({
                "observation_id": obs_id,
                "symbol": sym,
                "checkpoint": cur["checkpoint"],
                "attention_entropy": round(shannon_entropy([cur["AttentionScore"], cur["NarrativeScore"], cur["BeliefConsensus"]]), 4),
                "attention_delta": round(-d_att, 2),
                "primary_destroyer": top,
                "destroyer_strength": round(tests[top], 4),
                "learning_recommendation": "NO_ACTION",
            })

    # --- Q8: Hidden attention pairs ---
    hidden_rows: list[dict] = []
    cps: list[dict] = []
    for sym, sym_rows in by_sym.items():
        for i, r in enumerate(sym_rows):
            if i >= len(sym_rows) - 1:
                continue
            cps.append({
                "symbol": sym,
                "checkpoint": r["checkpoint"],
                "narrative": r["NarrativeScore"],
                "attention": r["AttentionScore"],
                "next_op": sym_rows[i + 1]["OrderParameter"],
                "next_state": sym_rows[i + 1]["p39_state"],
            })
    for i in range(len(cps)):
        for j in range(i + 1, len(cps)):
            a, b = cps[i], cps[j]
            if abs(a["narrative"] - b["narrative"]) > 12:
                continue
            if abs(a["attention"] - b["attention"]) < 6:
                continue
            if abs(a["next_op"] - b["next_op"]) < 10:
                continue
            hidden_rows.append({
                "observation_id": obs_id,
                "symbol_a": a["symbol"],
                "checkpoint_a": a["checkpoint"],
                "symbol_b": b["symbol"],
                "checkpoint_b": b["checkpoint"],
                "narrative_distance": round(abs(a["narrative"] - b["narrative"]), 2),
                "attention_distance": round(abs(a["attention"] - b["attention"]), 2),
                "future_op_distance": round(abs(a["next_op"] - b["next_op"]), 2),
                "attention_explains_outcome": "yes",
                "hidden_attention_detected": "yes",
                "learning_recommendation": "NO_ACTION",
            })

    # --- Q9: Attention laws ---
    law_rows: list[dict] = []
    for r in all_rows:
        r["AttentionConsensus"] = r["AttentionScore"] / 100.0
        r["NarrativeStrength"] = r["NarrativeScore"] / 100.0
        r["OneMinusEntropy"] = max(0, 1 - r["Entropy"] * 0.1)

    law_specs = [
        ("trend_indicator", ["AttentionScore", "NarrativeScore", "ReplacementRate"], "TrendPersistence"),
        ("CollapseRisk", ["AttentionScore", "Entropy", "ReplacementRate"], "CollapseProbability"),
        ("AttentionScore", ["Potential", "API", "Quality", "OneMinusEntropy"], "AttentionConsensus"),
        ("NarrativeScore", ["AttentionScore", "BeliefConsensus", "Motivation"], "NarrativeStrength"),
    ]
    for target, preds, label in law_specs:
        y_l = [r[target] for r in all_rows]
        X_l = [[r[p] for p in preds] for r in all_rows]
        beta = ridge_regress(X_l, y_l, lam=1.0)
        pred = [sum(beta[j] * X_l[i][j] for j in range(len(preds))) for i in range(n)]
        err = rmse(y_l, pred)
        eq = f"{label} ≈ " + " + ".join(f"{beta[j]:+.4f}×{preds[j]}" for j in range(len(preds)))
        law_rows.append({
            "observation_id": obs_id,
            "equation": eq,
            "target": label,
            "predictors": "|".join(preds),
            "rmse": round(err, 4),
            "complexity": len(preds),
            "law_score": round((1 - err / 100) * 100 - len(preds) * 2, 2),
            "learning_recommendation": "NO_ACTION",
        })

    for label, target in [
        ("TrendPersistence", "trend_indicator"),
        ("AttentionConsensus", "AttentionScore"),
        ("CollapseProbability", "CollapseRisk"),
    ]:
        pool = ["AttentionScore", "NarrativeScore", "ReplacementRate", "CollectiveForce", "Flow", "OneMinusEntropy"]
        for a, b in itertools.combinations(pool, 2):
            y_l = [r[target] for r in all_rows]
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
        obs_id, field_rows, best_construction, best_hierarchy,
        attention_before_narrative_count, prop_rows, competition_rows,
        conserved, pool_std, switch_counts, destroy_scores, hidden_rows,
        law_rows, hierarchy_detail_rows,
    )

    write_csv(ATTENTION_FIELD_CSV, field_rows)
    write_csv(ATTENTION_PROPAGATION_CSV, prop_rows)
    write_csv(ATTENTION_ENTROPY_CSV, entropy_rows)
    write_csv(ATTENTION_COMPETITION_CSV, competition_rows)
    write_csv(ATTENTION_SWITCH_CSV, switch_rows)
    write_csv(HIDDEN_ATTENTION_CSV, hidden_rows)
    write_csv(ATTENTION_LAWS_CSV, law_rows)
    PROCESS_REPORT_TXT.write_text(report, encoding="utf-8")

    print(
        f"Saved P60 outputs | field={len(field_rows)} prop={len(prop_rows)} "
        f"competition={len(competition_rows)} hidden={len(hidden_rows)} laws={len(law_rows)}"
    )


def build_report(
    obs_id: str,
    field_rows: list[dict],
    best_construction: tuple,
    best_hierarchy: tuple,
    att_before_count: int,
    prop_rows: list[dict],
    competition_rows: list[dict],
    conserved: bool,
    pool_std: float,
    switch_counts: Counter,
    destroy_scores: Counter,
    hidden_rows: list[dict],
    law_rows: list[dict],
    hierarchy_detail_rows: list[dict],
) -> str:
    avg_speed = statistics.mean(r["propagation_speed"] for r in prop_rows) if prop_rows else 0
    avg_comp = statistics.mean(r["attention_competition_index"] for r in competition_rows) if competition_rows else 0
    top_law = law_rows[0] if law_rows else {}
    top_destroy = destroy_scores.most_common(1)[0][0] if destroy_scores else "unknown"
    hierarchy_desc = {
        "Attention_first": "Attention → Narrative → Belief → Motivation → Participation → Flow → Trend → Collapse",
        "Narrative_first": "Narrative → Attention → Belief → Motivation → Participation → Flow → Trend → Collapse",
        "Belief_first": "Belief → Attention → Narrative → Motivation → Participation → Flow → Trend → Collapse",
        "Flow_first": "Flow → Participation → Motivation → Belief → Narrative → Attention → Trend → Collapse",
    }
    best_path = hierarchy_desc.get(best_hierarchy[0], best_hierarchy[0])
    top_mode = switch_counts.most_common(1)[0][0] if switch_counts else "unknown"

    lines = [
        "===== SCOUT SEASON2 P60 - ATTENTION & COMPETITION FIELD =====",
        "",
        f"Observation ID: {obs_id}",
        "AttentionField hypothesis - STRICT NO_ACTION.",
        "",
        "=== Report questions ===",
        "",
        "1. Can AttentionField be inferred?",
        f"   Yes (hypothesis). AttentionScore inferred for {len(field_rows)} checkpoints (0-100).",
        "",
        "2. Attention before Narrative?",
        f"   {'Yes (hypothesis)' if att_before_count > 0 else 'Mixed/Inconclusive'}. "
        f"Best hierarchy: {best_hierarchy[0]} (fit={best_hierarchy[1]:.4f}).",
        "",
        "3. Strongest Attention construction law?",
        f"   AttentionScore ≈ {best_construction[0]} (RMSE={best_construction[1]:.4f}).",
        "",
        "4. Attention propagation mechanism?",
        f"   Competitive diffusion (hypothesis). Mean speed={round(avg_speed, 2)}/checkpoint;",
        "   amplification when Narrative/Belief/Motivation/Flow co-align; transfer between symbols.",
        "",
        "5. Attention competition?",
        f"   Mean competition index={round(avg_comp, 4)}. Primary mode: {top_mode.replace('_', ' ')}.",
    ]
    for mode, cnt in switch_counts.most_common(4):
        lines.append(f"   {mode}: {cnt} checkpoint(s)")

    lines.extend([
        "",
        "6. Attention conservation?",
        f"   {'Partially conserved (hypothesis)' if conserved else 'Not strictly conserved'}. "
        f"Pool std={round(pool_std, 2)} across checkpoint hours.",
        "",
        "7. Attention collapse mechanism?",
        f"   Primary destroyer: {top_destroy.replace('_', ' ')}.",
    ])
    for name, cnt in destroy_scores.most_common(4):
        lines.append(f"   {name}: {cnt} event(s)")

    lines.extend([
        "",
        "8. Hidden Attention pairs?",
        f"   {len(hidden_rows)} pairs: same Narrative, different Attention, different future.",
        "",
        "9. Strongest Attention Law?",
        f"   {top_law.get('equation', 'Insufficient data')}",
        "",
        "10. Complete process hierarchy discovered?",
        f"   Best fit: {best_hierarchy[0]} (aggregate fit={best_hierarchy[1]:.4f}) — ",
        f"   {best_path} (hypothesis).",
        "   AttentionField may sit above Narrative as limited competitive resource layer.",
        "",
        "Learning recommendation: NO_ACTION - AttentionField stored as hypothesis only.",
        "",
        *mission_summary_lines(),
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="P60 Attention & Competition Field Engine")
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
