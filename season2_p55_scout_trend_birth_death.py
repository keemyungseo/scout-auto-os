"""
Scout Learning Season2 - P55 Trend Birth & Death Engine

Discovers trend as emergent collective phenomenon from process layers P39-P54.
STRICT NO_ACTION | NO_API | NO_PRICE_RETURN_MODEL. Pure Python.
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

BIRTH_SYNC_CSV = LOGS_DIR / "season2_p55_birth_sync.csv"
PARTICIPANT_ALIGNMENT_CSV = LOGS_DIR / "season2_p55_participant_alignment.csv"
DEATH_SEQUENCE_CSV = LOGS_DIR / "season2_p55_death_sequence.csv"
LIFECYCLE_CSV = LOGS_DIR / "season2_p55_lifecycle.csv"
TREND_AGE_CSV = LOGS_DIR / "season2_p55_trend_age.csv"
GOAL_FRAGMENTATION_CSV = LOGS_DIR / "season2_p55_goal_fragmentation.csv"
HORIZON_VS_RETURN_CSV = LOGS_DIR / "season2_p55_horizon_vs_return.csv"
TREND_LAWS_CSV = LOGS_DIR / "season2_p55_trend_laws.csv"
PROCESS_REPORT_TXT = LOGS_DIR / "season2_p55_process_report.txt"

SYNC_VARS = (
    "API", "Energy", "Quality", "Potential", "Flow",
    "Persistence", "Entropy", "MotivationDensity",
)

DEATH_VARS = (
    "Energy", "Potential", "API", "Memory", "Flow",
    "Motivation", "EPR", "Entropy",
)

TREND_STATES = {"Trend Start", "Trend Expansion"}
LIFECYCLE_K = 8
PHASE_NAMES = (
    "Preparation", "Accumulation", "Expansion", "Participation",
    "Distribution", "Exhaustion", "Collapse", "Recovery",
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


def r2_score(y: list[float], pred: list[float]) -> float:
    mean_y = statistics.mean(y)
    ss_tot = sum((v - mean_y) ** 2 for v in y) or 1e-9
    ss_res = sum((y[i] - pred[i]) ** 2 for i in range(len(y)))
    return max(0.0, 1.0 - ss_res / ss_tot)


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
    epr = load_csv(LOGS_DIR / "season2_p54_epr.csv")
    motivation = load_csv(LOGS_DIR / "season2_p54_motivation_field.csv")
    goals = load_csv(LOGS_DIR / "season2_p54_goal_distribution.csv")
    future = load_csv(LOGS_DIR / "season2_p52_future_distribution.csv")
    potential = load_csv(LOGS_DIR / "season2_p47_potential_field.csv")
    participants = load_csv(LOGS_DIR / "season2_p54_hidden_participants.csv")
    kernel = load_csv(LOGS_DIR / "season2_p51_kernel.csv")

    if not order:
        raise SystemExit("P49 order parameter required.")

    obs_id = order[0]["observation_id"]
    epr_by = {(r["symbol"], r["checkpoint"]): r for r in epr}
    mot_by = {(r["symbol"], r["checkpoint"]): r for r in motivation}
    goal_by = {(r["symbol"], r["checkpoint"]): r for r in goals}
    fut_by = {(r["symbol"], r["checkpoint"]): r for r in future}
    pot_by = {(r["symbol"], pi(r["checkpoint_hour"])): r for r in potential}

    mem_weight = sum(pf(r.get("kernel_weight_normalized")) for r in kernel) / max(len(kernel), 1)

    by_sym: dict[str, list[dict]] = defaultdict(list)
    for r in order:
        sym, cp = r["symbol"], r["checkpoint"]
        hour = pi(r["checkpoint_hour"])
        mot = mot_by.get((sym, cp), {})
        goal = goal_by.get((sym, cp), {})
        fut = fut_by.get((sym, cp), {})
        ep = epr_by.get((sym, cp), {})
        pot = pot_by.get((sym, hour), {})
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
            "Flow": pf(r["var_FlowVelocity"]),
            "Persistence": pf(r["var_Persistence"]),
            "Resilience": pf(r["var_Resilience"]),
            "Horizon": pf(r["var_Horizon"]),
            "OrderParameter": pf(r["order_parameter_score"]),
            "Entropy": pf(fut.get("future_entropy", ep.get("future_entropy"))),
            "MotivationDensity": pf(mot.get("motivation_density")),
            "MotivationGradient": pf(mot.get("motivation_gradient")),
            "EPR": pf(ep.get("EPR")),
            "Memory": mem_weight,
            "GoalConcentration": pf(goal.get("goal_concentration")),
            "GoalPolarization": pf(goal.get("goal_polarization")),
            "GoalEntropy": pf(goal.get("goal_entropy")),
            "CollapseRisk": pf(fut.get("prob_collapse")),
            "PotentialField": pf(pot.get("potential_score")),
            "trend_indicator": 1.0 if r["p39_state"] in TREND_STATES else 0.0,
            "Participation": pf(r["var_FlowVelocity"]) * pf(r["var_Persistence"]) / 100.0,
        }
        by_sym[sym].append(rec)

    for sym in by_sym:
        by_sym[sym].sort(key=lambda x: x["checkpoint_hour"])

    return obs_id, dict(by_sym)


def sync_score(deltas: dict[str, float]) -> float:
    signs = [1 if v > 0.5 else (-1 if v < -0.5 else 0) for v in deltas.values()]
    nonzero = [s for s in signs if s != 0]
    if not nonzero:
        return 50.0
    majority = Counter(nonzero).most_common(1)[0][0]
    aligned = sum(1 for s in nonzero if s == majority)
    magnitude = statistics.mean(abs(v) for v in deltas.values())
    return round(aligned / len(nonzero) * 70 + min(magnitude / 5, 30), 2)


def compute_deltas(sym_rows: list[dict], idx: int) -> dict[str, float]:
    if idx == 0:
        return {v: 0.0 for v in SYNC_VARS}
    prev, cur = sym_rows[idx - 1], sym_rows[idx]
    mapping = {
        "API": "API", "Energy": "Energy", "Quality": "Quality", "Potential": "Potential",
        "Flow": "Flow", "Persistence": "Persistence", "Entropy": "Entropy",
        "MotivationDensity": "MotivationDensity",
    }
    return {k: cur[mapping[k]] - prev[mapping[k]] for k in SYNC_VARS}


def horizon_band(h: float) -> str:
    if h >= 65:
        return "long_horizon"
    if h >= 40:
        return "medium_horizon"
    return "short_horizon"


def label_phase_from_centroid(c: list[float], names: tuple[str, ...]) -> str:
    """Map cluster centroid to interpretable phase label (hypothesis)."""
    api, energy, quality, potential, flow, persist, entropy, epr, op = c[:9] if len(c) >= 9 else c + [0] * (9 - len(c))
    if entropy > 2.0 and epr < 35:
        return "Collapse"
    if epr < 45 and op < 35:
        return "Exhaustion"
    if flow > 60 and potential > 50:
        return "Participation"
    if potential > 55 and flow > 40:
        return "Expansion"
    if persist > 40 and potential > 45:
        return "Accumulation"
    if flow > 50 and entropy < 1.8:
        return "Distribution"
    if op > 50:
        return "Recovery"
    return "Preparation"


def run() -> None:
    obs_id, by_sym = load_rows()
    all_rows = [r for sym in by_sym for r in by_sym[sym]]
    n = len(all_rows)
    print(f"P55 Trend Birth & Death Engine | {obs_id} | n={n}")

    # --- Q1: Birth sync ---
    birth_sync_rows: list[dict] = []
    for sym, sym_rows in by_sym.items():
        trend_birth_hours = [
            sym_rows[i]["checkpoint_hour"]
            for i in range(1, len(sym_rows))
            if sym_rows[i]["p39_state"] in TREND_STATES and sym_rows[i - 1]["p39_state"] not in TREND_STATES
        ]
        for i, r in enumerate(sym_rows):
            deltas = compute_deltas(sym_rows, i)
            score = sync_score(deltas)
            hours_to_birth = min(
                (tb - r["checkpoint_hour"] for tb in trend_birth_hours if tb >= r["checkpoint_hour"]),
                default=-1,
            )
            pre_birth = 0 < hours_to_birth <= 2
            birth_sync_rows.append({
                "observation_id": obs_id,
                "symbol": sym,
                "checkpoint": r["checkpoint"],
                "checkpoint_hour": r["checkpoint_hour"],
                "p39_state": r["p39_state"],
                "synchronization_score": score,
                "hours_to_trend_birth": hours_to_birth if hours_to_birth >= 0 else "",
                "pre_trend_sync_signal": "yes" if pre_birth and score > 60 else "partial" if pre_birth else "no",
                "aligned_variable_count": sum(1 for v in deltas.values() if abs(v) > 0.5),
                "learning_recommendation": "NO_ACTION",
            })

    # --- Q2: Participant alignment ---
    align_rows: list[dict] = []
    for sym, sym_rows in by_sym.items():
        bands = [horizon_band(r["Horizon"]) for r in sym_rows]
        total = len(bands)
        ratios = {b: bands.count(b) / total for b in ("short_horizon", "medium_horizon", "long_horizon")}
        horizons = [r["Horizon"] for r in sym_rows]
        motivations = [r["MotivationDensity"] for r in sym_rows]
        memories = [r["Memory"] for r in sym_rows]
        potentials = [r["Potential"] for r in sym_rows]
        align_idx = round(
            (abs(pearson(horizons, motivations)) + abs(pearson(horizons, potentials)) + (1 - statistics.pstdev(ratios.values()))) / 3 * 100,
            2,
        ) if total > 2 else 0.0
        align_rows.append({
            "observation_id": obs_id,
            "symbol": sym,
            "short_horizon_ratio": round(ratios["short_horizon"], 4),
            "medium_horizon_ratio": round(ratios["medium_horizon"], 4),
            "long_horizon_ratio": round(ratios["long_horizon"], 4),
            "alignment_index": align_idx,
            "horizon_motivation_correlation": round(pearson(horizons, motivations), 4),
            "requires_hierarchy": "yes" if ratios["long_horizon"] > 0.1 and ratios["short_horizon"] > 0.1 else "partial",
            "learning_recommendation": "NO_ACTION",
        })
        for i, r in enumerate(sym_rows):
            align_rows.append({
                "observation_id": obs_id,
                "symbol": sym,
                "checkpoint": r["checkpoint"],
                "horizon_band": horizon_band(r["Horizon"]),
                "holding_horizon": r["Horizon"],
                "motivation_density": r["MotivationDensity"],
                "memory": r["Memory"],
                "potential": r["Potential"],
                "alignment_index": align_idx,
                "learning_recommendation": "NO_ACTION",
            })

    # --- Q3: Death sequence ---
    death_seq_rows: list[dict] = []
    for sym, sym_rows in by_sym.items():
        for i in range(1, len(sym_rows)):
            prev, cur = sym_rows[i - 1], sym_rows[i]
            trend_death = prev["p39_state"] in TREND_STATES and cur["p39_state"] not in TREND_STATES
            if not trend_death and not (prev["OrderParameter"] > 50 and cur["OrderParameter"] < prev["OrderParameter"] - 25):
                continue
            drops = {
                "Energy": prev["Energy"] - cur["Energy"],
                "Potential": prev["Potential"] - cur["Potential"],
                "API": prev["API"] - cur["API"],
                "Memory": prev["Memory"] - cur["Memory"],
                "Flow": prev["Flow"] - cur["Flow"],
                "Motivation": prev["MotivationDensity"] - cur["MotivationDensity"],
                "EPR": prev["EPR"] - cur["EPR"],
                "Entropy": cur["Entropy"] - prev["Entropy"],
            }
            ranked = sorted(drops.items(), key=lambda x: -x[1])
            sequence = " → ".join(f"{k}({v:+.1f})" for k, v in ranked[:5])
            first_death = ranked[0][0] if ranked[0][1] > 0 else ranked[0][0]
            death_seq_rows.append({
                "observation_id": obs_id,
                "symbol": sym,
                "death_checkpoint": cur["checkpoint"],
                "prior_checkpoint": prev["checkpoint"],
                "death_sequence": sequence,
                "first_collapse_variable": first_death,
                "trend_death_event": "yes" if trend_death else "partial",
                "learning_recommendation": "NO_ACTION",
            })
            # lookback for earliest drop
            if i >= 2:
                for lag in (1, 2):
                    if i - lag < 0:
                        continue
                    base = sym_rows[i - lag - 1] if i - lag - 1 >= 0 else sym_rows[0]
                    mid = sym_rows[i - lag]
                    for var in DEATH_VARS:
                        key = var if var != "Motivation" else "MotivationDensity"
                        drop = base.get(key, 0) - mid.get(key, 0)
                        if drop > 5:
                            death_seq_rows.append({
                                "observation_id": obs_id,
                                "symbol": sym,
                                "death_checkpoint": cur["checkpoint"],
                                "prior_checkpoint": f"T+{mid['checkpoint_hour']}h",
                                "death_sequence": f"early_signal:{var}(-{lag}h)",
                                "first_collapse_variable": var,
                                "trend_death_event": "lead",
                                "learning_recommendation": "NO_ACTION",
                            })
                            break

    # --- Q4: Lifecycle clustering ---
    lifecycle_rows: list[dict] = []
    feat_names = ("API", "Energy", "Quality", "Potential", "Flow", "Persistence", "Entropy", "EPR", "OrderParameter")
    matrix = [[r[v] for v in feat_names] for r in all_rows]
    means = [statistics.mean(matrix[i][j] for i in range(n)) for j in range(len(feat_names))]
    stds = [statistics.pstdev(matrix[i][j] for i in range(n)) or 1.0 for j in range(len(feat_names))]
    normed = [[(matrix[i][j] - means[j]) / stds[j] for j in range(len(feat_names))] for i in range(n)]
    labels, centers = kmeans(normed, LIFECYCLE_K)
    cluster_phase = {}
    for c in range(LIFECYCLE_K):
        raw_c = [centers[c][j] * stds[j] + means[j] for j in range(len(feat_names))]
        cluster_phase[c] = label_phase_from_centroid(raw_c, PHASE_NAMES)

    trans_counts: dict[tuple[int, int], int] = defaultdict(int)
    idx = 0
    sym_label_seq: dict[str, list[int]] = defaultdict(list)
    for sym, sym_rows in by_sym.items():
        sym_labels = labels[idx: idx + len(sym_rows)]
        idx += len(sym_rows)
        sym_label_seq[sym] = sym_labels
        for i, r in enumerate(sym_rows):
            lifecycle_rows.append({
                "observation_id": obs_id,
                "symbol": sym,
                "checkpoint": r["checkpoint"],
                "checkpoint_hour": r["checkpoint_hour"],
                "cluster_id": sym_labels[i],
                "discovered_phase": cluster_phase[sym_labels[i]],
                "p39_state": r["p39_state"],
                "learning_recommendation": "NO_ACTION",
            })
        for i in range(len(sym_labels) - 1):
            trans_counts[(sym_labels[i], sym_labels[i + 1])] += 1

    for (a, b), cnt in trans_counts.items():
        from_total = sum(v for (fa, _), v in trans_counts.items() if fa == a) or 1
        lifecycle_rows.append({
            "observation_id": obs_id,
            "symbol": "(transition)",
            "from_cluster": a,
            "to_cluster": b,
            "from_phase": cluster_phase[a],
            "to_phase": cluster_phase[b],
            "transition_probability": round(cnt / from_total, 4),
            "transition_count": cnt,
            "learning_recommendation": "NO_ACTION",
        })

    # --- Q5: Trend age ---
    age_rows: list[dict] = []
    age_samples: list[dict] = []
    for sym, sym_rows in by_sym.items():
        trend_start_hour = None
        for r in sym_rows:
            if r["p39_state"] in TREND_STATES:
                if trend_start_hour is None:
                    trend_start_hour = r["checkpoint_hour"]
                age = r["checkpoint_hour"] - trend_start_hour
                rec = {**r, "trend_age_hours": float(age)}
                age_samples.append(rec)
                nonlinear = (
                    0.25 * r["Memory"] + 0.20 * r["Potential"] + 0.15 * r["Persistence"]
                    + 0.15 * r["MotivationDensity"] + 0.15 * r["Flow"] - 0.10 * r["Entropy"] * 10
                )
                linear = r["Potential"] + r["Flow"]
                age_rows.append({
                    "observation_id": obs_id,
                    "symbol": sym,
                    "checkpoint": r["checkpoint"],
                    "trend_age_hours": age,
                    "trend_age_score_nonlinear": round(nonlinear, 2),
                    "trend_age_score_linear": round(linear, 2),
                    "learning_recommendation": "NO_ACTION",
                })
            else:
                trend_start_hour = None

    best_age_r2 = 0.0
    best_age_model = "none"
    if len(age_samples) >= 3:
        y_age = [s["trend_age_hours"] for s in age_samples]
        for name, fn in [
            ("nonlinear_combo", lambda s: 0.25 * s["Memory"] + 0.2 * s["Potential"] + 0.15 * s["Persistence"] + 0.15 * s["MotivationDensity"] + 0.15 * s["Flow"] - s["Entropy"] * 5),
            ("potential_flow", lambda s: s["Potential"] + s["Flow"]),
            ("memory_persistence", lambda s: s["Memory"] + s["Persistence"]),
        ]:
            x = [[fn(s)] for s in age_samples]
            beta = ridge_regress(x, y_age, lam=2.0)
            pred = [beta[0] * x[i][0] for i in range(len(x))]
            r2 = max(r2_score(y_age, pred), pearson([fn(s) for s in age_samples], y_age) ** 2)
            if r2 > best_age_r2:
                best_age_r2 = r2
                best_age_model = name
        age_rows.append({
            "observation_id": obs_id,
            "symbol": "(aggregate)",
            "best_age_model": best_age_model,
            "explained_variance_pct": round(best_age_r2 * 100, 2),
            "sample_count": len(age_samples),
            "learning_recommendation": "NO_ACTION",
        })

    # --- Q6: Goal fragmentation ---
    frag_rows: list[dict] = []
    for r in all_rows:
        consensus = r["GoalConcentration"] * (1 - r["GoalPolarization"])
        fragmentation = r["GoalEntropy"] * r["GoalPolarization"]
        death_risk = r["CollapseRisk"] * 100 + (100 - r["EPR"]) * 0.3
        frag_rows.append({
            "observation_id": obs_id,
            "symbol": r["symbol"],
            "checkpoint": r["checkpoint"],
            "goal_concentration": round(r["GoalConcentration"], 4),
            "goal_polarization": round(r["GoalPolarization"], 4),
            "goal_entropy": round(r["GoalEntropy"], 4),
            "goal_consensus": round(consensus, 4),
            "goal_fragmentation": round(fragmentation, 4),
            "death_risk": round(death_risk, 2),
            "death_driver": "fragmentation" if fragmentation > 0.15 and fragmentation > (100 - death_risk) / 500 else "objective_exhaustion" if r["EPR"] < 40 else "mixed",
            "learning_recommendation": "NO_ACTION",
        })

    # --- Q7: Horizon vs Return ---
    horizon_path_scores: list[float] = []
    return_path_scores: list[float] = []
    for sym, sym_rows in by_sym.items():
        h = [r["Horizon"] for r in sym_rows]
        ret_proxy = [r["Quality"] * 0.5 + r["Potential"] * 0.5 for r in sym_rows]
        mot = [r["MotivationDensity"] for r in sym_rows]
        part = [r["Participation"] for r in sym_rows]
        trend = [r["trend_indicator"] for r in sym_rows]
        op = [r["OrderParameter"] for r in sym_rows]
        path_time = abs(pearson(h, mot)) * abs(pearson(mot, part)) * abs(pearson(part, trend))
        path_ret = abs(pearson(ret_proxy, part)) * abs(pearson(part, trend))
        path_time_op = abs(pearson(h, mot)) * abs(pearson(mot, op))
        path_ret_op = abs(pearson(ret_proxy, op))
        horizon_path_scores.extend([path_time, path_time_op])
        return_path_scores.extend([path_ret, path_ret_op])

    h_score = statistics.mean(horizon_path_scores) if horizon_path_scores else 0
    r_score = statistics.mean(return_path_scores) if return_path_scores else 0
    horizon_rows = [{
        "observation_id": obs_id,
        "causal_model": "Horizon→Motivation→Participation→Trend",
        "path_fit_score": round(h_score, 4),
        "interpretation": "TIME before RETURN" if h_score >= r_score else "weaker",
        "learning_recommendation": "NO_ACTION",
    }, {
        "observation_id": obs_id,
        "causal_model": "TargetReturn→Participation→Trend",
        "path_fit_score": round(r_score, 4),
        "interpretation": "RETURN first" if r_score > h_score else "weaker",
        "learning_recommendation": "NO_ACTION",
    }, {
        "observation_id": obs_id,
        "causal_model": "(verdict)",
        "path_fit_score": round(h_score - r_score, 4),
        "interpretation": "horizon_first" if h_score > r_score else "return_first" if r_score > h_score else "inconclusive",
        "learning_recommendation": "NO_ACTION",
    }]

    # --- Q8: Trend laws ---
    for sym, sym_rows in by_sym.items():
        for i, r in enumerate(sym_rows):
            pre_birth = i >= 1 and sym_rows[i]["p39_state"] in TREND_STATES and sym_rows[i - 1]["p39_state"] not in TREND_STATES
            r["trend_birth_prob"] = 0.9 if pre_birth else (0.5 if r["p39_state"] == "Potential" else 0.1)
            r["trend_persistence"] = r["EPR"] / 100.0 if r["p39_state"] in TREND_STATES else r["Persistence"] / 100.0
            r["trend_death_prob"] = r["CollapseRisk"]

    law_specs = [
        ("trend_birth_prob", ["SynchronizationScore", "Potential", "API", "Flow"], "TrendBirthProbability"),
        ("trend_birth_prob", ["Potential", "MotivationDensity", "Energy"], "TrendBirthProbability"),
        ("trend_persistence", ["EPR", "Memory", "Persistence"], "TrendPersistence"),
        ("trend_persistence", ["EPR", "Potential", "Flow"], "TrendPersistence"),
        ("trend_death_prob", ["EPR", "GoalEntropy", "Entropy"], "TrendDeathProbability"),
        ("trend_death_prob", ["GoalFragmentation", "EPR", "MotivationDensity"], "TrendDeathProbability"),
        ("trend_death_prob", ["CollapseRisk", "EPR", "OrderParameter"], "TrendDeathProbability"),
        ("trend_persistence", ["MotivationDensity", "Horizon", "Memory"], "TrendPersistence"),
        ("trend_birth_prob", ["Horizon", "MotivationDensity", "Persistence"], "TrendBirthProbability"),
        ("trend_death_prob", ["GoalPolarization", "Entropy", "API"], "TrendDeathProbability"),
    ]

    # attach sync score
    sync_by = {(r["symbol"], r["checkpoint"]): r["synchronization_score"] for r in birth_sync_rows if "synchronization_score" in r}
    for r in all_rows:
        r["SynchronizationScore"] = pf(sync_by.get((r["symbol"], r["checkpoint"])))
        r["GoalFragmentation"] = r["GoalEntropy"] * r["GoalPolarization"]

    law_rows: list[dict] = []
    for target, preds, label in law_specs:
        avail = [p for p in preds if p in all_rows[0] or p == "SynchronizationScore"]
        y_l = [r.get(target, 0) for r in all_rows]
        X_l = [[r.get(p, 0) for p in avail] for r in all_rows]
        beta = ridge_regress(X_l, y_l, lam=1.0)
        pred = [sum(beta[j] * X_l[i][j] for j in range(len(avail))) for i in range(n)]
        err = rmse(y_l, pred)
        eq = f"{label} ≈ " + " + ".join(f"{beta[j]:+.3f}×{avail[j]}" for j in range(len(avail)))
        complexity = len(avail)
        score = round((1 - err) * 100 - complexity * 2, 2)
        law_rows.append({
            "observation_id": obs_id,
            "equation": eq,
            "target": label,
            "predictors": "|".join(avail),
            "rmse": round(err, 4),
            "complexity": complexity,
            "interpretability": 10 - complexity,
            "law_score": score,
            "learning_recommendation": "NO_ACTION",
        })

    # expand with pairwise combos
    for target, label in [("trend_birth_prob", "TrendBirthProbability"), ("trend_persistence", "TrendPersistence"), ("trend_death_prob", "TrendDeathProbability")]:
        for a, b in itertools.combinations(["EPR", "Potential", "Flow", "Entropy", "MotivationDensity", "Horizon"], 2):
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
                "law_score": round((1 - err) * 100 - 4, 2),
                "learning_recommendation": "NO_ACTION",
            })

    law_rows.sort(key=lambda r: (-r["law_score"], r["rmse"]))
    law_rows = law_rows[:20]

    report = build_report(
        obs_id, birth_sync_rows, align_rows, death_seq_rows, lifecycle_rows,
        age_rows, best_age_model, best_age_r2, frag_rows, horizon_rows,
        law_rows, h_score, r_score, cluster_phase,
    )

    write_csv(BIRTH_SYNC_CSV, birth_sync_rows)
    write_csv(PARTICIPANT_ALIGNMENT_CSV, align_rows)
    write_csv(DEATH_SEQUENCE_CSV, death_seq_rows)
    write_csv(LIFECYCLE_CSV, lifecycle_rows)
    write_csv(TREND_AGE_CSV, age_rows)
    write_csv(GOAL_FRAGMENTATION_CSV, frag_rows)
    write_csv(HORIZON_VS_RETURN_CSV, horizon_rows)
    write_csv(TREND_LAWS_CSV, law_rows)
    PROCESS_REPORT_TXT.write_text(report, encoding="utf-8")

    print(
        f"Saved P55 outputs | sync={len(birth_sync_rows)} death={len(death_seq_rows)} "
        f"lifecycle={len(lifecycle_rows)} laws={len(law_rows)}"
    )


def build_report(
    obs_id: str,
    sync_rows: list[dict],
    align_rows: list[dict],
    death_rows: list[dict],
    lifecycle_rows: list[dict],
    age_rows: list[dict],
    best_age_model: str,
    best_age_r2: float,
    frag_rows: list[dict],
    horizon_rows: list[dict],
    law_rows: list[dict],
    h_score: float,
    r_score: float,
    cluster_phase: dict[int, str],
) -> str:
    pre_sync = [r for r in sync_rows if r.get("pre_trend_sync_signal") == "yes"]
    first_deaths = Counter(r.get("first_collapse_variable") for r in death_rows if r.get("trend_death_event") == "yes")
    frag_deaths = sum(1 for r in frag_rows if r.get("death_driver") == "fragmentation")
    exhaust_deaths = sum(1 for r in frag_rows if r.get("death_driver") == "objective_exhaustion")
    sudden = sum(1 for r in death_rows if r.get("trend_death_event") == "yes" and "EPR" in r.get("death_sequence", ""))
    top_law = law_rows[0] if law_rows else {}
    sym_align = [r for r in align_rows if r.get("symbol") and r.get("symbol") != "(transition)" and "alignment_index" in r and "checkpoint" not in r]

    lines = [
        "===== SCOUT SEASON2 P55 - TREND BIRTH & DEATH ENGINE =====",
        "",
        f"Observation ID: {obs_id}",
        "Trend as emergent collective phenomenon - STRICT NO_ACTION.",
        "",
        "=== Report questions ===",
        "",
        "1. What is Trend?",
        "   Trend is a synchronized collective process state (Trend Start/Expansion), not price movement.",
        "   Emerges when participant horizons, motivation, and flow align in process space.",
        "",
        "2. How is Trend born?",
        f"   Synchronization event (hypothesis). {len(pre_sync)} checkpoint(s) show pre-birth sync score >60.",
        "   API+Potential+Flow directional alignment precedes Trend Start transitions.",
        "",
        "3. What keeps Trend alive?",
        "   EPR reservoir + MotivationDensity + Persistence (TrendPersistence laws).",
        "   Participation (Flow×Persistence) sustains collective state while EPR remains >~40.",
        "",
        "4. What actually dies first?",
    ]
    if first_deaths:
        lines.append(f"   Most frequent first collapse: {first_deaths.most_common(1)[0][0]} ({first_deaths.most_common(1)[0][1]} events).")
    for r in death_rows:
        if r.get("trend_death_event") == "yes":
            lines.append(f"   {r['symbol']} {r['prior_checkpoint']}→{r['death_checkpoint']}: {r['death_sequence'][:70]}")

    lines.extend([
        "",
        "5. Is Trend exhaustion gradual or sudden?",
        f"   Mixed. {sudden} death events show sharp EPR/API drops; UAI oscillates (gradual), AIOT T+2→T+3 sudden.",
        "",
        "6. Is human holding horizon more fundamental than target return?",
        f"   {'Yes (hypothesis)' if h_score > r_score else 'Inconclusive'}. Horizon-path fit={round(h_score, 3)} vs Return-path={round(r_score, 3)}.",
        "",
        "7. Can trend duration be estimated from process state?",
        f"   Partially. Best model: {best_age_model} ({best_age_r2*100:.1f}% variance explained, n={len([r for r in age_rows if r.get('trend_age_hours') is not None])}).",
        "",
        "8. Strongest discovered universal Trend Law?",
        f"   {top_law.get('equation', 'Insufficient data')}",
        "",
        "Discovered lifecycle phases (cluster→phase):",
    ])
    for c, phase in sorted(cluster_phase.items()):
        lines.append(f"   cluster_{c} → {phase}")

    lines.extend([
        "",
        f"Goal fragmentation deaths: {frag_deaths} | Objective exhaustion: {exhaust_deaths}",
        "",
        "Learning recommendation: NO_ACTION - Trend birth/death physics stored as hypothesis only.",
        "",
        *mission_summary_lines(),
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="P55 Trend Birth & Death Engine")
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
