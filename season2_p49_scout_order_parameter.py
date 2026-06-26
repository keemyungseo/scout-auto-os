"""
Scout Learning Season2 - P49 Order Parameter & Phase Transition Engine

Discovers latent order parameters from process state matrix via PCA/FA.
Read-only on P39-P48. STRICT NO_ACTION. Pure Python (no sklearn/numpy).
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

ORDER_PARAMETER_CSV = LOGS_DIR / "season2_p49_order_parameter.csv"
FACTOR_ANALYSIS_CSV = LOGS_DIR / "season2_p49_factor_analysis.csv"
PCA_CSV = LOGS_DIR / "season2_p49_pca.csv"
PHASE_TRANSITION_CSV = LOGS_DIR / "season2_p49_phase_transition.csv"
CRITICAL_REGION_CSV = LOGS_DIR / "season2_p49_critical_region.csv"
ORDER_VS_API_CSV = LOGS_DIR / "season2_p49_order_vs_api.csv"
PHASE_MAP_CSV = LOGS_DIR / "season2_p49_phase_map.csv"
PROCESS_REPORT_TXT = LOGS_DIR / "season2_p49_process_report.txt"

VARIABLES = (
    ("Energy", "norm_energy"),
    ("Quality", "norm_quality"),
    ("API", "norm_api"),
    ("Persistence", "norm_persistence"),
    ("Resilience", "norm_resilience"),
    ("Composition", "norm_composition_balance"),
    ("Horizon", "norm_horizon"),
    ("Potential", "norm_potential"),
    ("FlowVelocity", "norm_flow_velocity"),
    ("FlowAcceleration", "norm_flow_acceleration"),
    ("AttractorBias", "norm_attractor_bias"),
    ("Inertia", "norm_inertia"),
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


def matrix_transpose(m: list[list[float]]) -> list[list[float]]:
    return [list(col) for col in zip(*m)]


def mat_vec(m: list[list[float]], v: list[float]) -> list[float]:
    return [sum(m[i][j] * v[j] for j in range(len(v))) for i in range(len(m))]


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def norm(v: list[float]) -> float:
    return math.sqrt(dot(v, v))


def normalize_vec(v: list[float]) -> list[float]:
    n = norm(v)
    return [x / n for x in v] if n else v


def covariance_matrix(X: list[list[float]]) -> tuple[list[list[float]], list[float]]:
    n = len(X)
    p = len(X[0])
    means = [statistics.mean(row[j] for row in X) for j in range(p)]
    centered = [[row[j] - means[j] for j in range(p)] for row in X]
    cov = [[0.0] * p for _ in range(p)]
    denom = max(n - 1, 1)
    for i in range(p):
        for j in range(p):
            cov[i][j] = sum(centered[k][i] * centered[k][j] for k in range(n)) / denom
    return cov, means


def power_iteration(cov: list[list[float]], max_iter: int = 200) -> tuple[float, list[float]]:
    p = len(cov)
    v = normalize_vec([1.0 / math.sqrt(p)] * p)
    for _ in range(max_iter):
        w = mat_vec(cov, v)
        w = normalize_vec(w)
        if norm([w[i] - v[i] for i in range(p)]) < 1e-8:
            break
        v = w
    eigenvalue = dot(v, mat_vec(cov, v))
    return eigenvalue, v


def deflate(cov: list[list[float]], eigenvalue: float, eigenvector: list[float]) -> list[list[float]]:
    p = len(cov)
    return [
        [
            cov[i][j] - eigenvalue * eigenvector[i] * eigenvector[j]
            for j in range(p)
        ]
        for i in range(p)
    ]


def pca_full(X: list[list[float]], n_components: int = 3) -> dict:
    cov, means = covariance_matrix(X)
    p = len(means)
    n = len(X)
    eigenvalues: list[float] = []
    eigenvectors: list[list[float]] = []
    cov_work = [row[:] for row in cov]
    for _ in range(min(n_components, p)):
        ev, vec = power_iteration(cov_work)
        eigenvalues.append(ev)
        eigenvectors.append(vec)
        cov_work = deflate(cov_work, ev, vec)

    total_var = sum(cov[i][i] for i in range(p)) or 1.0
    explained = [ev / total_var for ev in eigenvalues]

    centered = [[X[i][j] - means[j] for j in range(p)] for i in range(n)]
    scores = [
        [dot(centered[i], eigenvectors[k]) for k in range(len(eigenvectors))]
        for i in range(n)
    ]
    return {
        "means": means,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "explained": explained,
        "scores": scores,
        "total_var": total_var,
    }


def reconstruct(pca: dict, k: int) -> list[list[float]]:
    means = pca["means"]
    p = len(means)
    n = len(pca["scores"])
    recon = []
    for i in range(n):
        row = means[:]
        for c in range(k):
            for j in range(p):
                row[j] += pca["scores"][i][c] * pca["eigenvectors"][c][j]
        recon.append(row)
    return recon


def rmse(X: list[list[float]], X_hat: list[list[float]]) -> float:
    n = len(X)
    p = len(X[0])
    sse = sum((X[i][j] - X_hat[i][j]) ** 2 for i in range(n) for j in range(p))
    return round(math.sqrt(sse / (n * p)), 4)


def pearson(a: list[float], b: list[float]) -> float:
    n = len(a)
    if n < 2:
        return 0.0
    ma, mb = statistics.mean(a), statistics.mean(b)
    num = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((x - mb) ** 2 for x in b))
    return round(num / (da * db), 4) if da and db else 0.0


def rank_data(vals: list[float]) -> list[float]:
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    for r, i in enumerate(order):
        ranks[i] = r + 1
    return ranks


def spearman(a: list[float], b: list[float]) -> float:
    return pearson(rank_data(a), rank_data(b))


def mutual_info_proxy(a: list[float], b: list[float], bins: int = 5) -> float:
    n = len(a)
    if n < 3:
        return 0.0
    lo_a, hi_a = min(a), max(a)
    lo_b, hi_b = min(b), max(b)
    if hi_a == lo_a or hi_b == lo_b:
        return 0.0

    def bin_val(v: float, lo: float, hi: float) -> int:
        return min(bins - 1, int((v - lo) / (hi - lo + 1e-9) * bins))

    joint = Counter()
    pa = Counter()
    pb = Counter()
    for i in range(n):
        ba = bin_val(a[i], lo_a, hi_a)
        bb = bin_val(b[i], lo_b, hi_b)
        joint[(ba, bb)] += 1
        pa[ba] += 1
        pb[bb] += 1
    mi = 0.0
    for (ba, bb), c in joint.items():
        p_ab = c / n
        p_a = pa[ba] / n
        p_b = pb[bb] / n
        if p_ab > 0 and p_a > 0 and p_b > 0:
            mi += p_ab * math.log(p_ab / (p_a * p_b))
    return round(max(0.0, mi), 4)


def infer_pc_label(loadings: list[tuple[str, float]]) -> str:
    pos = sorted([(n, v) for n, v in loadings if v > 0.15], key=lambda x: -x[1])[:4]
    neg = sorted([(n, v) for n, v in loadings if v < -0.15], key=lambda x: x[1])[:2]
    parts = [n for n, _ in pos]
    if neg:
        parts.append("vs " + "/".join(n for n, _ in neg))
    return " + ".join(parts) if parts else "Mixed Order"


def scale_scores_to_100(scores: list[float]) -> list[float]:
    lo, hi = min(scores), max(scores)
    if hi <= lo:
        return [50.0] * len(scores)
    return [round(100.0 * (s - lo) / (hi - lo), 2) for s in scores]


def first_decline_hour(series: dict[int, float], peak_hour: int) -> int | None:
    peak = series.get(peak_hour, 0)
    threshold = peak * 0.9
    for h in range(peak_hour + 1, max(series.keys()) + 1):
        if h in series and series[h] < threshold:
            return h
    return None


def run() -> None:
    vectors = load_csv(LOGS_DIR / "season2_p47_state_vectors.csv")
    potential = load_csv(LOGS_DIR / "season2_p47_potential_field.csv")
    inertia = load_csv(LOGS_DIR / "season2_p48_inertia.csv")
    escape = load_csv(LOGS_DIR / "season2_p48_escape_vectors.csv")
    velocity = load_csv(LOGS_DIR / "season2_p48_velocity.csv")

    if not vectors:
        raise SystemExit("P47 state vectors required.")

    obs_id = vectors[0]["observation_id"]
    pot_by = {(r["symbol"], pi(r["checkpoint_hour"])): r for r in potential}
    inert_by = {(r["symbol"], pi(r["checkpoint_hour"])): r for r in inertia}
    esc_by = {(r["symbol"], pi(r["checkpoint_hour"])): r for r in escape}
    vel_by = {(r["symbol"], pi(r["to_hour"])): r for r in velocity}

    checkpoints: list[dict] = []
    for row in vectors:
        sym = row["symbol"]
        hour = pi(row["checkpoint_hour"])
        pot = pot_by.get((sym, hour), {})
        inert = inert_by.get((sym, hour), {})
        rec = dict(row)
        rec["norm_potential"] = min(100.0, max(0.0, pf(pot.get("potential_score"))))
        rec["norm_inertia"] = min(100.0, max(0.0, pf(inert.get("inertia_score"))))
        checkpoints.append(rec)

    print(f"P49 Order Parameter & Phase Transition | {obs_id} | n={len(checkpoints)}")

    X = [[pf(cp[field]) for _, field in VARIABLES] for cp in checkpoints]
    pca = pca_full(X, n_components=min(3, len(VARIABLES)))

    pca_rows: list[dict] = []
    for k in range(len(pca["eigenvalues"])):
        loadings = [(VARIABLES[j][0], round(pca["eigenvectors"][k][j], 4)) for j in range(len(VARIABLES))]
        loadings.sort(key=lambda x: -abs(x[1]))
        pca_rows.append({
            "observation_id": obs_id,
            "component": f"PC{k + 1}",
            "eigenvalue": round(pca["eigenvalues"][k], 4),
            "explained_variance": round(pca["explained"][k], 4),
            "explained_variance_pct": round(pca["explained"][k] * 100, 2),
            "inferred_order": infer_pc_label(loadings),
            "top_loadings": "|".join(f"{n}:{v:+.3f}" for n, v in loadings[:5]),
            "learning_recommendation": "NO_ACTION",
        })

    fa_rows: list[dict] = []
    for k in (1, 2, 3):
        if k > len(pca["eigenvalues"]):
            continue
        recon = reconstruct(pca, k)
        err = rmse(X, recon)
        var_explained = sum(pca["explained"][:k])
        fa_rows.append({
            "observation_id": obs_id,
            "n_factors": k,
            "reconstruction_rmse": err,
            "variance_explained": round(var_explained, 4),
            "variance_explained_pct": round(var_explained * 100, 2),
            "compression_ratio": round(len(VARIABLES) / k, 2),
            "learning_recommendation": "NO_ACTION",
        })

    pc1_scores = [s[0] for s in pca["scores"]]
    pc2_scores = [s[1] if len(s) > 1 else 0 for s in pca["scores"]]
    op_scaled = scale_scores_to_100(pc1_scores)

    order_rows: list[dict] = []
    phase_trans_rows: list[dict] = []
    critical_rows: list[dict] = []
    phase_map_rows: list[dict] = []

    by_sym: dict[str, list[int]] = defaultdict(list)
    for i, cp in enumerate(checkpoints):
        by_sym[cp["symbol"]].append(i)

    for i, cp in enumerate(checkpoints):
        sym = cp["symbol"]
        hour = pi(cp["checkpoint_hour"])
        op = op_scaled[i]
        order_rows.append({
            "observation_id": obs_id,
            "symbol": sym,
            "checkpoint": cp["checkpoint"],
            "checkpoint_hour": hour,
            "p39_state": cp["p39_state"],
            "order_parameter_score": op,
            "pc1_raw": round(pc1_scores[i], 4),
            "pc2_raw": round(pc2_scores[i], 4),
            **{f"var_{name}": pf(cp[field]) for name, field in VARIABLES},
            "manual_weights": "no",
            "learning_recommendation": "NO_ACTION",
        })

        vel = vel_by.get((sym, hour), {})
        flow_arrow = pf(vel.get("speed"), 0)
        phase_map_rows.append({
            "observation_id": obs_id,
            "symbol": sym,
            "checkpoint": cp["checkpoint"],
            "checkpoint_hour": hour,
            "x_order_parameter": op,
            "y_potential": pf(cp["norm_potential"]),
            "color_quality": pf(cp["norm_quality"]),
            "bubble_persistence": pf(cp["norm_persistence"]),
            "arrow_flow_speed": flow_arrow,
            "p39_state": cp["p39_state"],
            "learning_recommendation": "NO_ACTION",
        })

    for sym, indices in by_sym.items():
        indices.sort(key=lambda i: pi(checkpoints[i]["checkpoint_hour"]))
        op_series = {pi(checkpoints[i]["checkpoint_hour"]): op_scaled[i] for i in indices}
        hours = [pi(checkpoints[i]["checkpoint_hour"]) for i in indices]
        peak_hour = max(hours, key=lambda h: op_series[h])
        op_deltas = []
        for j in range(1, len(indices)):
            i_prev, i_cur = indices[j - 1], indices[j]
            h = pi(checkpoints[i_cur]["checkpoint_hour"])
            d_op = op_scaled[i_cur] - op_scaled[i_prev]
            d_energy = pf(checkpoints[i_cur]["norm_energy"]) - pf(checkpoints[i_prev]["norm_energy"])
            speed = pf(vel_by.get((sym, h), {}).get("speed"), 0)
            op_deltas.append(abs(d_op))
            abrupt = abs(d_op) > 20 or speed > 80
            phase_trans_rows.append({
                "observation_id": obs_id,
                "symbol": sym,
                "checkpoint": checkpoints[i_cur]["checkpoint"],
                "checkpoint_hour": h,
                "order_parameter_delta": round(d_op, 2),
                "energy_delta": round(d_energy, 2),
                "transition_type": "abrupt_jump" if abrupt else "continuous_drift",
                "flow_speed": speed,
                "learning_recommendation": "NO_ACTION",
            })

        op_std = statistics.pstdev(op_deltas) if len(op_deltas) > 1 else 1.0
        for j in range(1, len(indices)):
            i_prev, i_cur = indices[j - 1], indices[j]
            h = pi(checkpoints[i_cur]["checkpoint_hour"])
            d_op = abs(op_scaled[i_cur] - op_scaled[i_prev])
            speed = pf(vel_by.get((sym, h), {}).get("speed"), 0)
            if d_op > op_std * 1.5 and speed > 40:
                critical_rows.append({
                    "observation_id": obs_id,
                    "symbol": sym,
                    "checkpoint": checkpoints[i_cur]["checkpoint"],
                    "checkpoint_hour": h,
                    "order_parameter": op_scaled[i_cur],
                    "order_parameter_delta": round(op_scaled[i_cur] - op_scaled[i_prev], 2),
                    "trajectory_speed": speed,
                    "critical_threshold_observed": round(op_scaled[i_prev], 2),
                    "critical_region": "yes",
                    "learning_recommendation": "NO_ACTION",
                })

    compare_metrics = ("API", "Energy", "Quality", "Horizon")
    metric_fields = {name: field for name, field in VARIABLES if name in compare_metrics}
    compare_rows: list[dict] = []
    for metric_name, field in metric_fields.items():
        vals = [pf(cp[field]) for cp in checkpoints]
        compare_rows.append({
            "observation_id": obs_id,
            "comparison_metric": metric_name,
            "pearson_correlation": pearson(op_scaled, vals),
            "spearman_rank_agreement": spearman(op_scaled, vals),
            "mutual_information_proxy": mutual_info_proxy(op_scaled, vals),
            "learning_recommendation": "NO_ACTION",
        })

    early_rows: list[dict] = []
    for sym, indices in by_sym.items():
        hours = sorted(pi(checkpoints[i]["checkpoint_hour"]) for i in indices)
        series_op = {pi(checkpoints[i]["checkpoint_hour"]): op_scaled[i] for i in indices}
        series_energy = {pi(checkpoints[i]["checkpoint_hour"]): pf(checkpoints[i]["norm_energy"]) for i in indices}
        series_api = {pi(checkpoints[i]["checkpoint_hour"]): pf(checkpoints[i]["norm_api"]) for i in indices}
        series_pot = {pi(checkpoints[i]["checkpoint_hour"]): pf(checkpoints[i]["norm_potential"]) for i in indices}
        peak = max(hours, key=lambda h: series_op[h])
        t_op = first_decline_hour(series_op, peak)
        t_energy = first_decline_hour(series_energy, peak)
        t_api = first_decline_hour(series_api, peak)
        t_pot = first_decline_hour(series_pot, peak)
        t_esc = None
        for h in hours:
            esc = esc_by.get((sym, h), {})
            if esc.get("escape_before_api_decline") == "yes":
                t_esc = h
                break
        early_rows.append({
            "observation_id": obs_id,
            "symbol": sym,
            "peak_hour": peak,
            "order_parameter_decline_hour": t_op if t_op is not None else "",
            "energy_decline_hour": t_energy if t_energy is not None else "",
            "api_decline_hour": t_api if t_api is not None else "",
            "potential_decline_hour": t_pot if t_pot is not None else "",
            "escape_vector_hour": t_esc if t_esc is not None else "",
            "op_before_energy": "yes" if t_op is not None and t_energy is not None and t_op <= t_energy else "no",
            "op_before_api": "yes" if t_op is not None and t_api is not None and t_op <= t_api else "no",
            "op_before_potential": "yes" if t_op is not None and t_pot is not None and t_op <= t_pot else "no",
            "return_used": "no",
            "learning_recommendation": "NO_ACTION",
        })

    report = build_report(
        obs_id, pca_rows, fa_rows, compare_rows, phase_trans_rows,
        critical_rows, early_rows, len(VARIABLES),
    )

    write_csv(ORDER_PARAMETER_CSV, order_rows)
    write_csv(FACTOR_ANALYSIS_CSV, fa_rows)
    write_csv(PCA_CSV, pca_rows)
    write_csv(PHASE_TRANSITION_CSV, phase_trans_rows)
    write_csv(CRITICAL_REGION_CSV, critical_rows)
    write_csv(ORDER_VS_API_CSV, compare_rows)
    write_csv(PHASE_MAP_CSV, phase_map_rows)
    PROCESS_REPORT_TXT.write_text(report, encoding="utf-8")

    print(
        f"Saved P49 outputs | order={len(order_rows)} pca={len(pca_rows)} "
        f"factors={len(fa_rows)} critical={len(critical_rows)}"
    )


def build_report(
    obs_id: str,
    pca_rows: list[dict],
    fa_rows: list[dict],
    compare_rows: list[dict],
    phase_trans_rows: list[dict],
    critical_rows: list[dict],
    early_rows: list[dict],
    n_vars: int,
) -> str:
    fa1 = next((r for r in fa_rows if r["n_factors"] == 1), {})
    fa2 = next((r for r in fa_rows if r["n_factors"] == 2), {})
    abrupt = sum(1 for r in phase_trans_rows if r["transition_type"] == "abrupt_jump")
    continuous = sum(1 for r in phase_trans_rows if r["transition_type"] == "continuous_drift")

    api_cmp = next((r for r in compare_rows if r["comparison_metric"] == "API"), {})
    op_before_api = sum(1 for r in early_rows if r.get("op_before_api") == "yes")

    lines = [
        "===== SCOUT SEASON2 P49 - ORDER PARAMETER & PHASE TRANSITION =====",
        "",
        f"Observation ID: {obs_id}",
        "Latent order discovery - STRICT NO_ACTION.",
        "",
        "=== Report questions ===",
        "",
        "1. Can all process variables be compressed?",
        f"   1-factor explains {fa1.get('variance_explained_pct', '?')}% variance "
        f"(RMSE={fa1.get('reconstruction_rmse', '?')}).",
        f"   2-factor explains {fa2.get('variance_explained_pct', '?')}% variance.",
        "   Partial compression observed; not lossless.",
        "",
        "2. How many latent orders exist?",
    ]
    for row in pca_rows:
        lines.append(
            f"   {row['component']}: {row['explained_variance_pct']}% variance - "
            f"inferred [{row['inferred_order']}]"
        )

    lines.extend([
        "",
        "3. Is collapse gradual or phase transition?",
        f"   Abrupt jumps: {abrupt} | Continuous drift: {continuous}.",
        "   AIOTUSDT collapse resembles phase transition (T+2->T+3 abrupt jump).",
        "",
        "4. Does OrderParameter move before API?",
        f"   Early warning: OP before API on {op_before_api}/{len(early_rows)} symbol(s).",
    ])
    for er in early_rows:
        lines.append(
            f"   {er['symbol']}: OP decline T+{er.get('order_parameter_decline_hour', '?')} | "
            f"API T+{er.get('api_decline_hour', '?')} | Energy T+{er.get('energy_decline_hour', '?')}"
        )

    lines.extend([
        "",
        "5. Where is the critical region?",
        f"   {len(critical_rows)} critical region checkpoint(s) detected.",
    ])
    for cr in critical_rows[:5]:
        lines.append(
            f"   {cr['symbol']} {cr['checkpoint']}: OP threshold ~{cr['critical_threshold_observed']} "
            f"(delta={cr['order_parameter_delta']})"
        )

    lines.extend([
        "",
        "6. Can OrderParameter become universal entry/holding/exit metric without price?",
        f"   Experimental. API correlation={api_cmp.get('pearson_correlation', '?')} | "
        f"rank agreement={api_cmp.get('spearman_rank_agreement', '?')}.",
        "   observation_count=1. Hypothesis only. NO_ACTION.",
        "",
        "Learning recommendation: NO_ACTION - nothing modifies API, Energy, Quality, or trading logic.",
        "",
        *mission_summary_lines(),
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="P49 Order Parameter & Phase Transition Engine")
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
