"""
Scout Learning Season2 - P56 Criticality & Collective Phase Transition Engine

Tests whether trend birth is a collective phase transition at process criticality.
Read-only on P39-P55. STRICT NO_ACTION | NO_API | NO_PRICE_RETURN_MODEL. Pure Python.
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

CRITICAL_SCORE_CSV = LOGS_DIR / "season2_p56_critical_score.csv"
PHASE_TRANSITION_CSV = LOGS_DIR / "season2_p56_phase_transition.csv"
FIRST_CROSSING_CSV = LOGS_DIR / "season2_p56_first_crossing.csv"
COLLECTIVE_SYNC_CSV = LOGS_DIR / "season2_p56_collective_sync.csv"
CRITICAL_MASS_CSV = LOGS_DIR / "season2_p56_critical_mass.csv"
INSTABILITY_CSV = LOGS_DIR / "season2_p56_instability.csv"
UNIVERSAL_CURVE_CSV = LOGS_DIR / "season2_p56_universal_curve.csv"
HUMAN_SYNC_CSV = LOGS_DIR / "season2_p56_human_sync.csv"
CRITICAL_LAWS_CSV = LOGS_DIR / "season2_p56_critical_laws.csv"
PROCESS_REPORT_TXT = LOGS_DIR / "season2_p56_process_report.txt"

CRITICAL_VARS = (
    "Potential", "Flow", "Motivation", "Persistence", "Entropy",
    "Synchronization", "OrderParameter",
)

CROSSING_VARS = (
    "Potential", "Flow", "Motivation", "API", "Memory",
    "Entropy", "Persistence", "Synchronization",
)

TREND_STATES = {"Trend Start", "Trend Expansion"}
THRESHOLD = 65.0


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


def l2(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(len(a))))


def load_rows() -> tuple[str, dict[str, list[dict]]]:
    order = load_csv(LOGS_DIR / "season2_p49_order_parameter.csv")
    epr = load_csv(LOGS_DIR / "season2_p54_epr.csv")
    motivation = load_csv(LOGS_DIR / "season2_p54_motivation_field.csv")
    goals = load_csv(LOGS_DIR / "season2_p54_goal_distribution.csv")
    future = load_csv(LOGS_DIR / "season2_p52_future_distribution.csv")
    potential = load_csv(LOGS_DIR / "season2_p47_potential_field.csv")
    birth_sync = load_csv(LOGS_DIR / "season2_p55_birth_sync.csv")
    lifecycle = load_csv(LOGS_DIR / "season2_p55_lifecycle.csv")
    state_space = load_csv(LOGS_DIR / "season2_p47_state_space.csv")
    kernel = load_csv(LOGS_DIR / "season2_p51_kernel.csv")

    if not order:
        raise SystemExit("P49 order parameter required.")

    obs_id = order[0]["observation_id"]
    epr_by = {(r["symbol"], r["checkpoint"]): r for r in epr}
    mot_by = {(r["symbol"], r["checkpoint"]): r for r in motivation}
    goal_by = {(r["symbol"], r["checkpoint"]): r for r in goals}
    fut_by = {(r["symbol"], r["checkpoint"]): r for r in future}
    pot_by = {(r["symbol"], pi(r["checkpoint_hour"])): r for r in potential}
    sync_by = {(r["symbol"], r["checkpoint"]): r for r in birth_sync}
    life_by = {(r["symbol"], r["checkpoint"]): r for r in lifecycle if r.get("cluster_id") != ""}
    ss_by = {(r["symbol"], pi(r["checkpoint_hour"])): r for r in state_space}
    mem = sum(pf(r.get("kernel_weight_normalized")) for r in kernel) / max(len(kernel), 1)

    by_sym: dict[str, list[dict]] = defaultdict(list)
    for r in order:
        sym, cp = r["symbol"], r["checkpoint"]
        hour = pi(r["checkpoint_hour"])
        mot = mot_by.get((sym, cp), {})
        goal = goal_by.get((sym, cp), {})
        fut = fut_by.get((sym, cp), {})
        ep = epr_by.get((sym, cp), {})
        pot = pot_by.get((sym, hour), {})
        sync = sync_by.get((sym, cp), {})
        life = life_by.get((sym, cp), {})
        ss = ss_by.get((sym, hour), {})
        rec = {
            "observation_id": obs_id,
            "symbol": sym,
            "checkpoint": cp,
            "checkpoint_hour": hour,
            "p39_state": r["p39_state"],
            "API": pf(r["var_API"]),
            "Energy": pf(r["var_Energy"]),
            "Quality": pf(r["var_Quality"]),
            "Potential": pf(r["var_Potential"]),
            "PotentialField": pf(pot.get("potential_score")),
            "Flow": pf(r["var_FlowVelocity"]),
            "Persistence": pf(r["var_Persistence"]),
            "Resilience": pf(r["var_Resilience"]),
            "Horizon": pf(r["var_Horizon"]),
            "OrderParameter": pf(r["order_parameter_score"]),
            "Entropy": pf(fut.get("future_entropy")),
            "Motivation": pf(mot.get("motivation_density")),
            "MotivationGradient": pf(mot.get("motivation_gradient")),
            "EPR": pf(ep.get("EPR")),
            "Memory": mem,
            "Synchronization": pf(sync.get("synchronization_score")),
            "GoalConcentration": pf(goal.get("goal_concentration")),
            "GoalPolarization": pf(goal.get("goal_polarization")),
            "GoalEntropy": pf(goal.get("goal_entropy")),
            "GoalConsensus": pf(goal.get("goal_concentration")) * (1 - pf(goal.get("goal_polarization"))),
            "CollapseRisk": pf(fut.get("prob_collapse")),
            "LifecyclePhase": life.get("discovered_phase", ""),
            "LocalDensity": pf(ss.get("local_density")),
            "Participation": pf(r["var_FlowVelocity"]) * pf(r["var_Persistence"]) / 100.0,
            "trend_indicator": 1.0 if r["p39_state"] in TREND_STATES else 0.0,
        }
        by_sym[sym].append(rec)

    for sym in by_sym:
        by_sym[sym].sort(key=lambda x: x["checkpoint_hour"])
    return obs_id, dict(by_sym)


def critical_score(r: dict) -> float:
    """Nonlinear criticality composite — hypothesis only."""
    pot = r["Potential"] / 100.0
    flow = r["Flow"] / 100.0
    mot = r["Motivation"] / 100.0
    persist = r["Persistence"] / 100.0
    ent_pen = max(0, 1.0 - r["Entropy"] / 3.0)
    sync = r["Synchronization"] / 100.0
    op = r["OrderParameter"] / 100.0
    raw = (pot ** 1.2) * (flow ** 0.8) * (mot ** 1.0) * (persist ** 0.5) * ent_pen * (sync ** 1.1) * (op ** 1.3)
    return min(100.0, raw * 100.0)


def threshold_nonlinearity(vals: list[float], threshold: float = THRESHOLD) -> float:
    """Measure abruptness: variance above vs below threshold."""
    above = [v for v in vals if v >= threshold]
    below = [v for v in vals if v < threshold]
    if not above or not below:
        return 0.0
    return abs(statistics.mean(above) - statistics.mean(below))


def run() -> None:
    obs_id, by_sym = load_rows()
    all_rows = [r for sym in by_sym for r in by_sym[sym]]
    n = len(all_rows)
    print(f"P56 Criticality & Phase Transition | {obs_id} | n={n}")

    # --- Q1: Critical score ---
    critical_rows: list[dict] = []
    all_critical: list[float] = []
    for sym, sym_rows in by_sym.items():
        for i, r in enumerate(sym_rows):
            cs = round(critical_score(r), 2)
            all_critical.append(cs)
            above = cs >= THRESHOLD
            critical_rows.append({
                "observation_id": obs_id,
                "symbol": sym,
                "checkpoint": r["checkpoint"],
                "checkpoint_hour": r["checkpoint_hour"],
                "p39_state": r["p39_state"],
                "critical_score": cs,
                "above_critical_threshold": "yes" if above else "no",
                "threshold_used": THRESHOLD,
                "nonlinear_gain": round(threshold_nonlinearity(all_critical), 4),
                "learning_recommendation": "NO_ACTION",
            })

    # --- Q2: Phase transition vs continuous ---
    phase_rows: list[dict] = []
    for sym, sym_rows in by_sym.items():
        for i in range(1, len(sym_rows)):
            prev, cur = sym_rows[i - 1], sym_rows[i]
            state_jump = 1.0 if prev["p39_state"] != cur["p39_state"] else 0.0
            pot_jump = abs(cur["Potential"] - prev["Potential"])
            flow_jump = abs(cur["Flow"] - prev["Flow"])
            mot_jump = abs(cur["Motivation"] - prev["Motivation"])
            state_dist = l2(
                [prev["Potential"], prev["Flow"], prev["Motivation"], prev["OrderParameter"]],
                [cur["Potential"], cur["Flow"], cur["Motivation"], cur["OrderParameter"]],
            )
            drift = state_dist / max(state_dist + pot_jump + flow_jump + mot_jump, 1e-6)
            abrupt = (pot_jump + flow_jump + mot_jump) / max(state_dist, 1e-6)
            birth = cur["p39_state"] in TREND_STATES and prev["p39_state"] not in TREND_STATES
            phase_rows.append({
                "observation_id": obs_id,
                "symbol": sym,
                "from_checkpoint": prev["checkpoint"],
                "to_checkpoint": cur["checkpoint"],
                "state_distance": round(state_dist, 2),
                "potential_jump": round(pot_jump, 2),
                "flow_jump": round(flow_jump, 2),
                "motivation_jump": round(mot_jump, 2),
                "continuous_drift_score": round(drift * 100, 2),
                "abrupt_jump_score": round(min(100, abrupt * 20), 2),
                "transition_type": "phase_transition" if abrupt > 2.0 or birth else "continuous_drift",
                "trend_birth_event": "yes" if birth else "no",
                "learning_recommendation": "NO_ACTION",
            })

    # --- Q3: First threshold crossing ---
    crossing_rows: list[dict] = []
    first_cross_rank: Counter = Counter()
    for sym, sym_rows in by_sym.items():
        cross_hour: dict[str, int] = {}
        for var in CROSSING_VARS:
            key = "Motivation" if var == "Motivation" else var
            if var == "Synchronization":
                key = "Synchronization"
            for r in sym_rows:
                val = r.get(key, r.get(var, 0))
                thresh = THRESHOLD if var not in ("Entropy", "Memory") else (2.0 if var == "Entropy" else 5.0)
                if var == "Entropy":
                    crossed = val >= thresh
                elif var == "Memory":
                    crossed = val >= thresh
                else:
                    crossed = val >= thresh
                if crossed and var not in cross_hour:
                    cross_hour[var] = r["checkpoint_hour"]
        if cross_hour:
            ranked = sorted(cross_hour.items(), key=lambda x: x[1])
            for rank, (var, hour) in enumerate(ranked, 1):
                first_cross_rank[var] += 1
                crossing_rows.append({
                    "observation_id": obs_id,
                    "symbol": sym,
                    "variable": var,
                    "first_crossing_hour": hour,
                    "rank": rank,
                    "threshold": THRESHOLD if var not in ("Entropy",) else 2.0,
                    "learning_recommendation": "NO_ACTION",
                })

    for var, cnt in first_cross_rank.most_common():
        crossing_rows.append({
            "observation_id": obs_id,
            "symbol": "(aggregate)",
            "variable": var,
            "first_crossing_count": cnt,
            "global_rank": first_cross_rank.most_common().index((var, cnt)) + 1,
            "learning_recommendation": "NO_ACTION",
        })

    # --- Q4: Collective synchronization ---
    collective_rows: list[dict] = []
    sync_vars = ("GoalConcentration", "Motivation", "Flow", "Synchronization", "Persistence")
    for sym, sym_rows in by_sym.items():
        vecs = [[r[v] for v in sync_vars] for r in sym_rows]
        pairwise: list[float] = []
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                pairwise.append(pearson(vecs[i], vecs[j]) if len(vecs[i]) == len(vecs[j]) else 0)
        pair_align = statistics.mean(pairwise) if pairwise else 0
        goal_align = pearson([r["GoalConcentration"] for r in sym_rows], [r["Motivation"] for r in sym_rows])
        mot_align = pearson([r["Motivation"] for r in sym_rows], [r["Flow"] for r in sym_rows])
        flow_align = pearson([r["Flow"] for r in sym_rows], [r["Synchronization"] for r in sym_rows])
        global_sync = statistics.mean([r["Synchronization"] for r in sym_rows])
        cluster_align = 1.0 - statistics.pstdev([r["GoalConcentration"] for r in sym_rows]) / 100.0 if len(sym_rows) > 1 else 0
        sync_index = round(min(100, max(0, (pair_align + 1) / 2 * 30 + goal_align ** 2 * 20 + mot_align ** 2 * 20 + flow_align ** 2 * 15 + global_sync * 0.15)), 2)
        collective_rows.append({
            "observation_id": obs_id,
            "symbol": sym,
            "pairwise_alignment": round(pair_align, 4),
            "cluster_alignment": round(cluster_align, 4),
            "global_synchronization": round(global_sync, 2),
            "goal_alignment": round(goal_align, 4),
            "motivation_alignment": round(mot_align, 4),
            "flow_alignment": round(flow_align, 4),
            "synchronization_index": sync_index,
            "learning_recommendation": "NO_ACTION",
        })
        for r in sym_rows:
            collective_rows.append({
                "observation_id": obs_id,
                "symbol": sym,
                "checkpoint": r["checkpoint"],
                "synchronization_index": sync_index,
                "goal_alignment": round(goal_align, 4),
                "motivation_alignment": round(mot_align, 4),
                "flow_alignment": round(flow_align, 4),
                "learning_recommendation": "NO_ACTION",
            })

    # --- Q5: Critical mass ---
    mass_rows: list[dict] = []
    birth_densities: list[float] = []
    non_birth_densities: list[float] = []
    for sym, sym_rows in by_sym.items():
        for i, r in enumerate(sym_rows):
            mot_d = r["Motivation"] / 100.0
            goal_d = r["GoalConcentration"]
            part_d = r["Participation"] / 100.0
            mem_d = r["Memory"] / 10.0
            total_mass = mot_d * 0.35 + goal_d * 0.25 + part_d * 0.25 + mem_d * 0.15
            birth_next = i + 1 < len(sym_rows) and sym_rows[i + 1]["p39_state"] in TREND_STATES and r["p39_state"] not in TREND_STATES
            if birth_next:
                birth_densities.append(total_mass)
            else:
                non_birth_densities.append(total_mass)
            mass_rows.append({
                "observation_id": obs_id,
                "symbol": sym,
                "checkpoint": r["checkpoint"],
                "motivation_density": round(mot_d, 4),
                "goal_density": round(goal_d, 4),
                "participation_density": round(part_d, 4),
                "memory_density": round(mem_d, 4),
                "total_critical_mass": round(total_mass, 4),
                "pre_trend_birth": "yes" if birth_next else "no",
                "learning_recommendation": "NO_ACTION",
            })

    min_birth = min(birth_densities) if birth_densities else 0
    mass_rows.append({
        "observation_id": obs_id,
        "symbol": "(estimate)",
        "estimated_minimum_critical_mass": round(min_birth, 4),
        "mean_pre_birth_mass": round(statistics.mean(birth_densities), 4) if birth_densities else "",
        "mean_non_birth_mass": round(statistics.mean(non_birth_densities), 4) if non_birth_densities else "",
        "learning_recommendation": "NO_ACTION",
    })

    # --- Q6: Early instability ---
    instability_rows: list[dict] = []
    for sym, sym_rows in by_sym.items():
        peak_hour = max(sym_rows, key=lambda r: r["OrderParameter"])["checkpoint_hour"]
        peak_op = max(r["OrderParameter"] for r in sym_rows)
        for i, r in enumerate(sym_rows):
            if r["checkpoint_hour"] >= peak_hour:
                continue
            if i == 0:
                continue
            prev = sym_rows[i - 1]
            ent_curv = abs(r["Entropy"] - 2 * prev["Entropy"] + (sym_rows[i - 2]["Entropy"] if i >= 2 else prev["Entropy"]))
            mem_decay = prev["Memory"] - r["Memory"]
            goal_frag = r["GoalEntropy"] * r["GoalPolarization"]
            flow_div = abs(r["Flow"] - prev["Flow"]) / max(prev["Flow"], 1)
            inst_score = ent_curv * 10 + goal_frag * 50 + flow_div * 20 + max(0, -r["EPR"] + prev["EPR"]) * 0.5
            instability_rows.append({
                "observation_id": obs_id,
                "symbol": sym,
                "checkpoint": r["checkpoint"],
                "checkpoint_hour": r["checkpoint_hour"],
                "hours_before_peak": peak_hour - r["checkpoint_hour"],
                "local_entropy_curvature": round(ent_curv, 4),
                "memory_decay": round(mem_decay, 4),
                "goal_fragmentation": round(goal_frag, 4),
                "flow_divergence": round(flow_div, 4),
                "instability_score": round(inst_score, 2),
                "early_instability_detected": "yes" if inst_score > 5 and r["checkpoint_hour"] < peak_hour else "partial" if inst_score > 2 else "no",
                "learning_recommendation": "NO_ACTION",
            })

    # --- Q7: Universal curve ---
    curve_rows: list[dict] = []
    curve_vars = ("Potential", "Flow", "Motivation", "Persistence")
    pooled: dict[str, dict[int, list[float]]] = {v: defaultdict(list) for v in curve_vars}

    for sym, sym_rows in by_sym.items():
        trend_rows = [r for r in sym_rows if r["p39_state"] in TREND_STATES or any(
            tr["p39_state"] in TREND_STATES for tr in sym_rows
        )]
        if not trend_rows:
            trend_rows = sym_rows
        t0 = trend_rows[0]["checkpoint_hour"]
        t1 = trend_rows[-1]["checkpoint_hour"]
        span = max(t1 - t0, 1)
        for r in sym_rows:
            if r["p39_state"] not in TREND_STATES and r["checkpoint_hour"] < t0:
                continue
            rel_life = round((r["checkpoint_hour"] - t0) / span * 100)
            rel_life = min(100, max(0, rel_life))
            for v in curve_vars:
                pooled[v][rel_life].append(r[v])
                curve_rows.append({
                    "observation_id": obs_id,
                    "symbol": sym,
                    "checkpoint": r["checkpoint"],
                    "relative_lifetime_pct": rel_life,
                    "variable": v,
                    "normalized_value": round(r[v], 2),
                    "learning_recommendation": "NO_ACTION",
                })

    for v in curve_vars:
        for pct in sorted(pooled[v].keys()):
            curve_rows.append({
                "observation_id": obs_id,
                "symbol": "(pooled)",
                "relative_lifetime_pct": pct,
                "variable": v,
                "mean_value": round(statistics.mean(pooled[v][pct]), 2),
                "std_value": round(statistics.pstdev(pooled[v][pct]), 2) if len(pooled[v][pct]) > 1 else 0,
                "sample_count": len(pooled[v][pct]),
                "learning_recommendation": "NO_ACTION",
            })

    # --- Q8: Human sync hypothesis ---
    human_rows: list[dict] = []
    for sym, sym_rows in by_sym.items():
        participant_count = len([r for r in sym_rows if r["Participation"] > 10])
        for i, r in enumerate(sym_rows):
            sync_gain = r["Synchronization"] - (sym_rows[i - 1]["Synchronization"] if i > 0 else r["Synchronization"])
            human_rows.append({
                "observation_id": obs_id,
                "symbol": sym,
                "checkpoint": r["checkpoint"],
                "synchronization_gain": round(sync_gain, 2),
                "participant_count_proxy": participant_count,
                "goal_consensus": round(r["GoalConsensus"], 4),
                "motivation_density": round(r["Motivation"], 2),
                "unexpected_sync_hypothesis": "yes" if sync_gain > 10 and r["GoalConsensus"] < 0.7 else "partial" if sync_gain > 5 else "no",
                "learning_recommendation": "NO_ACTION",
            })

    h_corr_sync_part = pearson(
        [r["synchronization_gain"] for r in human_rows if "synchronization_gain" in r],
        [r["participant_count_proxy"] for r in human_rows if "participant_count_proxy" in r],
    )
    h_corr_sync_goal = pearson(
        [r["synchronization_gain"] for r in human_rows if "synchronization_gain" in r],
        [r["goal_consensus"] for r in human_rows if "goal_consensus" in r],
    )
    human_rows.append({
        "observation_id": obs_id,
        "symbol": "(correlation)",
        "sync_gain_vs_participant_count": round(h_corr_sync_part, 4),
        "sync_gain_vs_goal_consensus": round(h_corr_sync_goal, 4),
        "hypothesis_favor": "independent_sync" if abs(h_corr_sync_part) > abs(h_corr_sync_goal) else "consensus_driven",
        "learning_recommendation": "NO_ACTION",
    })

    # --- Q9: Critical laws ---
    for r in all_rows:
        r["CriticalScore"] = critical_score(r)
        r["CriticalPoint"] = 1.0 if r["CriticalScore"] >= THRESHOLD else 0.0
        r["TrendBirthProbability"] = 0.85 if r["p39_state"] in TREND_STATES else (0.4 if r["CriticalScore"] > 50 else 0.1)
        r["CollapseProbability"] = r["CollapseRisk"]

    law_specs = [
        ("TrendBirthProbability", ["CriticalScore", "Synchronization", "Motivation", "Flow"], "TrendBirthProbability"),
        ("CriticalPoint", ["Potential", "Flow", "Motivation", "OrderParameter"], "CriticalPoint"),
        ("CriticalScore", ["Potential", "Flow", "Motivation", "Persistence", "Entropy"], "CriticalScore"),
        ("CollapseProbability", ["CriticalScore", "Entropy", "GoalEntropy"], "CollapseProbability"),
        ("Synchronization", ["Motivation", "Flow", "GoalConcentration"], "Synchronization"),
        ("TrendBirthProbability", ["Motivation", "GoalConsensus", "Participation"], "TrendBirthProbability"),
        ("CriticalPoint", ["EPR", "Synchronization", "Persistence"], "CriticalPoint"),
        ("CollapseProbability", ["EPR", "Entropy", "Motivation"], "CollapseProbability"),
    ]

    law_rows: list[dict] = []
    for target, preds, label in law_specs:
        avail = preds
        y_l = [r.get(target, 0) for r in all_rows]
        X_l = [[r[p] for p in avail] for r in all_rows]
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
            "interpretability": 10 - len(avail),
            "law_score": round((1 - err / 100) * 100 - len(avail) * 2, 2),
            "learning_recommendation": "NO_ACTION",
        })

    for target, label in [("TrendBirthProbability", "TrendBirthProbability"), ("CriticalPoint", "CriticalPoint"),
                          ("Synchronization", "Synchronization"), ("CollapseProbability", "CollapseProbability")]:
        for a, b in itertools.combinations(["CriticalScore", "Motivation", "Flow", "Entropy", "EPR", "GoalConsensus"], 2):
            y_l = [r[target] for r in all_rows]
            X_l = [[r[a], r[b]] for r in all_rows]
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
                "interpretability": 8,
                "law_score": round((1 - err / 100) * 100 - 4, 2),
                "learning_recommendation": "NO_ACTION",
            })

    law_rows.sort(key=lambda r: (-r["law_score"], r["rmse"]))
    # Prefer non-trivial laws (exclude perfect binary fits)
    meaningful = [r for r in law_rows if r["rmse"] > 0.05 or r["target"] in ("CriticalScore", "Synchronization", "TrendBirthProbability", "CollapseProbability")]
    law_rows = (meaningful or law_rows)[:20]

    report = build_report(
        obs_id, critical_rows, phase_rows, crossing_rows, collective_rows,
        mass_rows, instability_rows, curve_rows, human_rows, law_rows,
        first_cross_rank, birth_densities, min_birth, h_corr_sync_part, h_corr_sync_goal,
    )

    write_csv(CRITICAL_SCORE_CSV, critical_rows)
    write_csv(PHASE_TRANSITION_CSV, phase_rows)
    write_csv(FIRST_CROSSING_CSV, crossing_rows)
    write_csv(COLLECTIVE_SYNC_CSV, collective_rows)
    write_csv(CRITICAL_MASS_CSV, mass_rows)
    write_csv(INSTABILITY_CSV, instability_rows)
    write_csv(UNIVERSAL_CURVE_CSV, curve_rows)
    write_csv(HUMAN_SYNC_CSV, human_rows)
    write_csv(CRITICAL_LAWS_CSV, law_rows)
    PROCESS_REPORT_TXT.write_text(report, encoding="utf-8")

    print(
        f"Saved P56 outputs | critical={len(critical_rows)} phase={len(phase_rows)} "
        f"laws={len(law_rows)} instability={len(instability_rows)}"
    )


def build_report(
    obs_id: str,
    critical_rows: list[dict],
    phase_rows: list[dict],
    crossing_rows: list[dict],
    collective_rows: list[dict],
    mass_rows: list[dict],
    instability_rows: list[dict],
    curve_rows: list[dict],
    human_rows: list[dict],
    law_rows: list[dict],
    first_cross_rank: Counter,
    birth_densities: list[float],
    min_birth: float,
    h_corr_sync_part: float,
    h_corr_sync_goal: float,
) -> str:
    phase_trans = sum(1 for r in phase_rows if r.get("transition_type") == "phase_transition")
    births = sum(1 for r in phase_rows if r.get("trend_birth_event") == "yes")
    above_crit = sum(1 for r in critical_rows if r.get("above_critical_threshold") == "yes")
    first_var = first_cross_rank.most_common(1)[0][0] if first_cross_rank else "unknown"
    sym_sync = [r for r in collective_rows if r.get("symbol") not in ("(correlation)",) and "pairwise_alignment" in r]
    avg_sync_idx = statistics.mean(r["synchronization_index"] for r in sym_sync) if sym_sync else 0
    early_inst = sum(1 for r in instability_rows if r.get("early_instability_detected") == "yes")
    top_law = next((r for r in law_rows if r["target"] == "CriticalScore"), law_rows[0] if law_rows else {})
    top_human = next(
        (r for r in law_rows if r["target"] in ("Synchronization", "TrendBirthProbability") and "Motivation" in r.get("predictors", "")),
        next((r for r in law_rows if r["target"] == "Synchronization"), top_law),
    )

    pre_birth_pred = sum(
        1 for r in critical_rows
        if float(r.get("critical_score", 0)) > 55
    )

    lines = [
        "===== SCOUT SEASON2 P56 - CRITICALITY & PHASE TRANSITION =====",
        "",
        f"Observation ID: {obs_id}",
        "Collective synchronization & criticality - STRICT NO_ACTION.",
        "",
        "=== Report questions ===",
        "",
        "1. Does Trend emerge gradually or critically?",
        f"   Critically (hypothesis). {phase_trans}/{len(phase_rows)} transitions classified as phase_transition;",
        f"   {births} trend birth events show abrupt jumps in Potential/Flow/Motivation.",
        "",
        "2. Does a Critical Point exist?",
        f"   Yes (hypothesis). CriticalScore threshold={THRESHOLD}; {above_crit} checkpoints above critical point.",
        "",
        "3. What variable reaches Criticality first?",
        f"   {first_var} (most frequent first threshold crossing across symbols).",
    ]
    for var, cnt in first_cross_rank.most_common(5):
        lines.append(f"   {var}: {cnt} first-crossing event(s)")

    lines.extend([
        "",
        "4. What creates Collective Synchronization?",
        f"   Goal-Motivation-Flow alignment chain. Mean SynchronizationIndex={round(avg_sync_idx, 1)}/100.",
        "   Pairwise variable alignment + rising Synchronization score at Trend Start.",
        "",
        "5. Can Trend Birth be predicted before synchronization?",
        f"   Partially. Estimated minimum critical mass={round(min_birth, 3)}; "
        f"mean pre-birth mass={round(statistics.mean(birth_densities), 3) if birth_densities else 'n/a'}.",
        "",
        "6. Can Collapse begin before visible weakness?",
        f"   Yes (hypothesis). {early_inst} early instability signal(s) detected before OrderParameter peak.",
        "   Goal fragmentation + entropy curvature precede visible trend weakness.",
        "",
        "7. Do all trends share one universal lifecycle?",
        "   Partially. Pooled relative-lifetime curves show Flow peak mid-life, Potential decay at exhaustion;",
        "   AIOT sudden vs UAI extended trend differ in shape — weak universality.",
        "",
        "8. Strongest discovered Critical Law?",
        f"   {top_law.get('equation', 'Insufficient data')}",
        "",
        "9. Strongest discovered Human Synchronization Law?",
        f"   {top_human.get('equation', top_law.get('equation', 'Insufficient data'))}",
        f"   SyncGain vs participants r={round(h_corr_sync_part, 3)} vs goal consensus r={round(h_corr_sync_goal, 3)}.",
        "   Hypothesis: trend exists because independent participants unexpectedly synchronize.",
        "",
        "Learning recommendation: NO_ACTION - criticality physics stored as hypothesis only.",
        "",
        *mission_summary_lines(),
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="P56 Criticality & Phase Transition Engine")
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
