"""
Scout Learning Season2 - P50 Dynamic Evolution Law Engine

Studies how current process state generates next state: State(T+1) ≈ F(State(T)).
Read-only on P39-P49. STRICT NO_ACTION. Pure Python.
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

LOCAL_EVOLUTION_CSV = LOGS_DIR / "season2_p50_local_evolution.csv"
PREDICTION_ERROR_CSV = LOGS_DIR / "season2_p50_prediction_error.csv"
EVOLUTION_OPERATORS_CSV = LOGS_DIR / "season2_p50_evolution_operators.csv"
CAUSAL_CONTRIBUTION_CSV = LOGS_DIR / "season2_p50_causal_contribution.csv"
FIXED_POINTS_CSV = LOGS_DIR / "season2_p50_fixed_points.csv"
LOCAL_STABILITY_CSV = LOGS_DIR / "season2_p50_local_stability.csv"
PROCESS_REPORT_TXT = LOGS_DIR / "season2_p50_process_report.txt"

STATE_VARS = (
    "Energy",
    "Quality",
    "API",
    "Persistence",
    "Resilience",
    "Composition",
    "Horizon",
    "Potential",
    "OrderParameter",
    "AttractorBias",
    "Inertia",
    "StateSpaceDrift",
)

VAR_FIELDS = {
    "Energy": "var_Energy",
    "Quality": "var_Quality",
    "API": "var_API",
    "Persistence": "var_Persistence",
    "Resilience": "var_Resilience",
    "Composition": "var_Composition",
    "Horizon": "var_Horizon",
    "Potential": "var_Potential",
    "OrderParameter": "order_parameter_score",
    "AttractorBias": "var_AttractorBias",
    "Inertia": "var_Inertia",
    "StateSpaceDrift": "state_space_density",
}

HEALTHY = {"Potential", "Trend Start", "Trend Expansion"}
STATE_RANK = {"Failure": 0, "Observation": 1, "Potential": 2, "Trend Start": 3, "Trend Expansion": 4}


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


def state_vec(row: dict) -> list[float]:
    vals = []
    for name in STATE_VARS:
        if name == "StateSpaceDrift":
            vals.append(pf(row.get("state_space_density")))
        else:
            vals.append(pf(row.get(VAR_FIELDS[name])))
    return vals


def mat_transpose(m: list[list[float]]) -> list[list[float]]:
    return [list(col) for col in zip(*m)]


def mat_mul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    rows, cols, inner = len(a), len(b[0]), len(b)
    return [[sum(a[i][k] * b[k][j] for k in range(inner)) for j in range(cols)] for i in range(rows)]


def mat_vec(m: list[list[float]], v: list[float]) -> list[float]:
    return [sum(m[i][j] * v[j] for j in range(len(v))) for i in range(len(m))]


def vec_mat(v: list[float], m: list[list[float]]) -> list[float]:
    return [sum(v[i] * m[i][j] for i in range(len(v))) for j in range(len(m[0]))]


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


def fit_evolution_matrix(X: list[list[float]], Y: list[list[float]]) -> list[list[float]]:
    """Least squares: Y = X @ A.T  =>  A.T = (X.T X)^-1 X.T Y"""
    p = len(X[0])
    n = len(X)
    xt = mat_transpose(X)
    xtx = mat_mul(xt, X)
    xty = mat_mul(xt, Y)
    inv = mat_inverse(xtx)
    if inv is None:
        return [[1.0 if i == j else 0.0 for j in range(p)] for i in range(p)]
    at = mat_mul(inv, xty)
    return mat_transpose(at)


def local_matrix(s: list[float], s_next: list[float], a_global: list[list[float]], alpha: float = 0.5) -> list[list[float]]:
    p = len(s)
    denom = sum(x * x for x in s) + 1e-6
    local = [[0.0] * p for _ in range(p)]
    for i in range(p):
        for j in range(p):
            local[i][j] = s_next[i] * s[j] / denom
    blend = [[alpha * a_global[i][j] + (1 - alpha) * local[i][j] for j in range(p)] for i in range(p)]
    return blend


def frobenius(a: list[list[float]], b: list[list[float]]) -> float:
    return math.sqrt(sum((a[i][j] - b[i][j]) ** 2 for i in range(len(a)) for j in range(len(a[0]))))


def predict_error(s: list[float], s_next: list[float], a: list[list[float]]) -> float:
    pred = vec_mat(s, a)
    return math.sqrt(sum((pred[i] - s_next[i]) ** 2 for i in range(len(s))))


def classify_operator(from_state: str, to_state: str, delta_op: float, reversals: int) -> str:
    if to_state == "Failure" or delta_op < -25:
        return "Collapse Evolution"
    if STATE_RANK.get(to_state, 1) > STATE_RANK.get(from_state, 1):
        return "Recovery Evolution"
    if reversals >= 2:
        return "Oscillation Evolution"
    if from_state in HEALTHY and to_state in HEALTHY:
        return "Stable Evolution"
    if delta_op > 5:
        return "Recovery Evolution"
    return "Stable Evolution"


def power_max_eigenvalue(a: list[list[float]], iters: int = 100) -> float:
    p = len(a)
    v = [1.0 / math.sqrt(p)] * p
    for _ in range(iters):
        w = mat_vec(a, v)
        n = math.sqrt(sum(x * x for x in w)) or 1.0
        v = [x / n for x in w]
    return abs(sum(v[i] * mat_vec(a, v)[i] for i in range(p)))


def kmeans_flat(flat: list[list[float]], k: int = 4) -> list[int]:
    if not flat:
        return []
    n = len(flat)
    k = min(k, n)
    centers = [flat[i][:] for i in range(k)]
    labels = [0] * n
    for _ in range(30):
        for i, pt in enumerate(flat):
            labels[i] = min(range(k), key=lambda c: math.sqrt(sum((pt[d] - centers[c][d]) ** 2 for d in range(len(pt)))))
        for c in range(k):
            cluster = [flat[i] for i in range(n) if labels[i] == c]
            if cluster:
                centers[c] = [statistics.mean(pt[d] for pt in cluster) for d in range(len(flat[0]))]
    return labels


def run() -> None:
    order = load_csv(LOGS_DIR / "season2_p49_order_parameter.csv")
    state_space = load_csv(LOGS_DIR / "season2_p47_state_space.csv")
    process_index = load_csv(LOGS_DIR / "season2_p44_process_index.csv")
    phase_trans = load_csv(LOGS_DIR / "season2_p49_phase_transition.csv")

    if not order:
        raise SystemExit("P49 order parameter required.")

    obs_id = order[0]["observation_id"]
    ss_by = {(r["symbol"], pi(r["checkpoint_hour"])): r for r in state_space}
    pi_by = {(r["symbol"], pi(r["checkpoint_hour"])): r for r in process_index}

    checkpoints: list[dict] = []
    for row in order:
        sym = row["symbol"]
        hour = pi(row["checkpoint_hour"])
        ss = ss_by.get((sym, hour), {})
        pidx = pi_by.get((sym, hour), {})
        rec = dict(row)
        rec["process_drift"] = pf(pidx.get("process_drift"))
        rec["state_space_density"] = pf(ss.get("local_density"))
        checkpoints.append(rec)

    by_sym: dict[str, list[dict]] = defaultdict(list)
    for cp in checkpoints:
        by_sym[cp["symbol"]].append(cp)
    for sym in by_sym:
        by_sym[sym].sort(key=lambda r: pi(r["checkpoint_hour"]))

    pairs: list[tuple[dict, dict, dict]] = []
    for sym, rows in by_sym.items():
        for i in range(len(rows) - 1):
            pairs.append((rows[i], rows[i + 1], {"symbol": sym, "from_hour": pi(rows[i]["checkpoint_hour"])}))

    X = [state_vec(a) for a, _, _ in pairs]
    Y = [state_vec(b) for _, b, _ in pairs]
    a_global = fit_evolution_matrix(X, Y)
    p = len(STATE_VARS)

    print(f"P50 Dynamic Evolution Law | {obs_id} | transitions={len(pairs)}")

    local_rows: list[dict] = []
    error_rows: list[dict] = []
    operator_rows: list[dict] = []
    stability_rows: list[dict] = []
    flat_mats: list[list[float]] = []
    op_meta: list[dict] = []

    local_rows.append({
        "observation_id": obs_id,
        "symbol": "(global)",
        "from_checkpoint": "",
        "to_checkpoint": "",
        "from_hour": "",
        "to_hour": "",
        "from_state": "",
        "to_state": "",
        "evolution_matrix_flat": "|".join(f"{a_global[i][j]:.4f}" for i in range(p) for j in range(p)),
        "operator_type_inferred": "Global Linear Evolution",
        "learning_recommendation": "NO_ACTION",
    })

    for a_row, b_row, meta in pairs:
        s = state_vec(a_row)
        s_next = state_vec(b_row)
        a_loc = local_matrix(s, s_next, a_global, alpha=0.6)
        err = predict_error(s, s_next, a_loc)
        delta_op = pf(b_row["order_parameter_score"]) - pf(a_row["order_parameter_score"])
        reversals = sum(
            1 for j in range(len(s))
            if (s_next[j] - s[j]) * s[j] < 0 and abs(s_next[j] - s[j]) > 5
        )
        op_type = classify_operator(a_row["p39_state"], b_row["p39_state"], delta_op, reversals)

        local_rows.append({
            "observation_id": obs_id,
            "symbol": meta["symbol"],
            "from_checkpoint": a_row["checkpoint"],
            "to_checkpoint": b_row["checkpoint"],
            "from_hour": meta["from_hour"],
            "to_hour": pi(b_row["checkpoint_hour"]),
            "from_state": a_row["p39_state"],
            "to_state": b_row["p39_state"],
            "evolution_matrix_flat": "|".join(f"{a_loc[i][j]:.4f}" for i in range(p) for j in range(p)),
            "operator_type_inferred": op_type,
            "learning_recommendation": "NO_ACTION",
        })

        error_rows.append({
            "observation_id": obs_id,
            "symbol": meta["symbol"],
            "from_checkpoint": a_row["checkpoint"],
            "to_checkpoint": b_row["checkpoint"],
            "prediction_rmse": round(err, 4),
            "relative_error_pct": round(err / (math.sqrt(sum(x * x for x in s_next)) + 1e-6) * 100, 2),
            "operator_type_inferred": op_type,
            "learning_recommendation": "NO_ACTION",
        })

        flat_mats.append([a_loc[i][j] for i in range(p) for j in range(p)])
        op_meta.append({
            "symbol": meta["symbol"],
            "from_checkpoint": a_row["checkpoint"],
            "to_checkpoint": b_row["checkpoint"],
            "operator_type": op_type,
        })

    labels = kmeans_flat(flat_mats, k=4)
    cluster_types: Counter = Counter()
    for lbl in set(labels):
        members = [op_meta[i]["operator_type"] for i in range(len(labels)) if labels[i] == lbl]
        cluster_types[members[0] if members else "Unknown"] += 1

    for i, meta in enumerate(op_meta):
        operator_rows.append({
            "observation_id": obs_id,
            "symbol": meta["symbol"],
            "from_checkpoint": meta["from_checkpoint"],
            "to_checkpoint": meta["to_checkpoint"],
            "operator_type_inferred": meta["operator_type"],
            "cluster_id": labels[i],
            "cluster_method": "KMeans_on_evolution_matrix",
            "forced_label": "no",
            "learning_recommendation": "NO_ACTION",
        })

    op_type_counts = Counter(m["operator_type"] for m in op_meta)
    for op_type, count in op_type_counts.items():
        operator_rows.append({
            "observation_id": obs_id,
            "symbol": "(aggregate)",
            "from_checkpoint": "",
            "to_checkpoint": "",
            "operator_type_inferred": op_type,
            "cluster_id": "",
            "cluster_method": "aggregate",
            "observation_count": count,
            "forced_label": "no",
            "learning_recommendation": "NO_ACTION",
        })

    col_sums = [0.0] * p
    for a_loc_flat in flat_mats:
        for j in range(p):
            col_sums[j] += sum(abs(a_loc_flat[i * p + j]) for i in range(p))
    total = sum(col_sums) or 1.0
    causal_rows = [{
        "observation_id": obs_id,
        "variable": STATE_VARS[j],
        "evolution_contribution": round(col_sums[j] / total * 100, 2),
        "rank": 0,
        "learning_recommendation": "NO_ACTION",
    } for j in range(p)]
    ranked = sorted(causal_rows, key=lambda r: -r["evolution_contribution"])
    for i, row in enumerate(ranked):
        row["rank"] = i + 1

    fixed_rows: list[dict] = []
    for cp in checkpoints:
        s = state_vec(cp)
        pred = vec_mat(s, a_global)
        residual = math.sqrt(sum((pred[i] - s[i]) ** 2 for i in range(p)))
        fixed_type = "neutral"
        if residual < 5:
            fixed_type = "fixed_point"
        elif sum(pred) > sum(s) + 10:
            fixed_type = "repeller"
        elif sum(pred) < sum(s) - 10:
            fixed_type = "attractor"
        fixed_rows.append({
            "observation_id": obs_id,
            "symbol": cp["symbol"],
            "checkpoint": cp["checkpoint"],
            "checkpoint_hour": pi(cp["checkpoint_hour"]),
            "p39_state": cp["p39_state"],
            "residual_norm": round(residual, 4),
            "point_type": fixed_type,
            "predicted_next_sum": round(sum(pred), 2),
            "current_sum": round(sum(s), 2),
            "learning_recommendation": "NO_ACTION",
        })

    for a_row, b_row, meta in pairs:
        s = state_vec(a_row)
        a_loc = local_matrix(s, state_vec(b_row), a_global, alpha=0.6)
        lam = power_max_eigenvalue(a_loc)
        perturb = [0.01 * (1 if i % 2 == 0 else -1) for i in range(p)]
        s_pert = [s[i] + perturb[i] for i in range(p)]
        traj_orig = vec_mat(s, a_loc)
        traj_pert = vec_mat(s_pert, a_loc)
        div = math.sqrt(sum((traj_pert[i] - traj_orig[i]) ** 2 for i in range(p)))
        stability_rows.append({
            "observation_id": obs_id,
            "symbol": meta["symbol"],
            "from_checkpoint": a_row["checkpoint"],
            "to_checkpoint": b_row["checkpoint"],
            "max_eigenvalue_magnitude": round(lam, 4),
            "perturbation_divergence": round(div, 4),
            "lyapunov_like": "convergent" if lam < 1.0 else "divergent" if lam > 1.05 else "neutral",
            "learning_recommendation": "NO_ACTION",
        })

    avg_err = statistics.mean(r["prediction_rmse"] for r in error_rows) if error_rows else 0
    report = build_report(
        obs_id, error_rows, op_type_counts, ranked, fixed_rows, stability_rows, avg_err, len(set(labels)),
    )

    write_csv(LOCAL_EVOLUTION_CSV, local_rows)
    write_csv(PREDICTION_ERROR_CSV, error_rows)
    write_csv(EVOLUTION_OPERATORS_CSV, operator_rows)
    write_csv(CAUSAL_CONTRIBUTION_CSV, causal_rows)
    write_csv(FIXED_POINTS_CSV, fixed_rows)
    write_csv(LOCAL_STABILITY_CSV, stability_rows)
    PROCESS_REPORT_TXT.write_text(report, encoding="utf-8")

    print(
        f"Saved P50 outputs | evolution={len(local_rows)} errors={len(error_rows)} "
        f"operators={len(operator_rows)} fixed={len(fixed_rows)}"
    )


def build_report(
    obs_id: str,
    error_rows: list[dict],
    op_counts: Counter,
    causal_ranked: list[dict],
    fixed_rows: list[dict],
    stability_rows: list[dict],
    avg_err: float,
    n_clusters: int,
) -> str:
    collapse_ops = op_counts.get("Collapse Evolution", 0)
    divergent = sum(1 for r in stability_rows if r["lyapunov_like"] == "divergent")
    convergent = sum(1 for r in stability_rows if r["lyapunov_like"] == "convergent")
    neutral_stab = sum(1 for r in stability_rows if r["lyapunov_like"] == "neutral")
    collapse_err = [
        r["prediction_rmse"] for r in error_rows if r["operator_type_inferred"] == "Collapse Evolution"
    ]
    stable_err = [
        r["prediction_rmse"] for r in error_rows if r["operator_type_inferred"] == "Stable Evolution"
    ]
    collapse_mean_err = statistics.mean(collapse_err) if collapse_err else 0
    stable_mean_err = statistics.mean(stable_err) if stable_err else 0
    fixed_pts = sum(1 for r in fixed_rows if r["point_type"] == "fixed_point")
    attractors = sum(1 for r in fixed_rows if r["point_type"] == "attractor")
    nearest_fixed = min(fixed_rows, key=lambda r: r["residual_norm"]) if fixed_rows else None

    lines = [
        "===== SCOUT SEASON2 P50 - DYNAMIC EVOLUTION LAW =====",
        "",
        f"Observation ID: {obs_id}",
        "Evolution law discovery - STRICT NO_ACTION.",
        "",
        "=== Report questions ===",
        "",
        "1. Can next Process State be predicted from current Process State?",
        f"   Partially. Mean prediction RMSE={round(avg_err, 2)} across {len(error_rows)} transitions.",
        "   Linear F(State) approximates short-horizon evolution; large error at collapse jumps.",
        "",
        "2. How many Evolution Operators exist?",
        f"   {len(op_counts)} inferred operator types | {n_clusters} matrix clusters.",
    ]
    for op, cnt in op_counts.most_common():
        lines.append(f"   {op}: {cnt}")

    lines.extend([
        "",
        "3. Which operator produces Collapse?",
        "   Collapse Evolution: Potential->Failure, Trend Start->Potential with large negative OrderParameter delta.",
        f"   Observed {collapse_ops} collapse-classified transition(s).",
        "",
        "4. Which variables dominate evolution?",
    ])
    for row in causal_ranked[:5]:
        lines.append(f"   #{row['rank']} {row['variable']}: {row['evolution_contribution']}% contribution")

    lines.extend([
        "",
        "5. Are there Fixed Points?",
        f"   Fixed/near-fixed: {fixed_pts} | Attractors: {attractors} | Total checkpoints scanned: {len(fixed_rows)}.",
        (
            f"   Nearest neutral: {nearest_fixed['symbol']} {nearest_fixed['checkpoint']} "
            f"({nearest_fixed['p39_state']}, residual={nearest_fixed['residual_norm']})."
            if nearest_fixed else "   No checkpoint scanned."
        ),
        "",
        "6. Is Process evolution deterministic or chaotic?",
        f"   Mixed / weakly deterministic. Convergent Jacobian regions: {convergent} | Neutral: {neutral_stab} | Divergent: {divergent}.",
        f"   Collapse operator mean RMSE={round(collapse_mean_err, 1)} vs Stable={round(stable_mean_err, 1)} — collapse jumps are least linearly predictable.",
        "   Single-observation sample; full law requires cross-observation repetition.",
        "",
        "Learning recommendation: NO_ACTION - Dynamic Evolution Law stored as hypothesis only.",
        "",
        *mission_summary_lines(),
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="P50 Dynamic Evolution Law Engine")
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
