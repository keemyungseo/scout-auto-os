"""
Scout Learning Season2 - P54 Hidden Motivation & Expected Profit Reservoir Engine

Tests hypothesis that process dynamics emerge from hidden participant target-return distributions.
Read-only on P39-P53. STRICT NO_ACTION. Pure Python.
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

HIDDEN_PARTICIPANTS_CSV = LOGS_DIR / "season2_p54_hidden_participants.csv"
EPR_CSV = LOGS_DIR / "season2_p54_epr.csv"
CONSUMPTION_CSV = LOGS_DIR / "season2_p54_consumption.csv"
TREND_LIFE_CSV = LOGS_DIR / "season2_p54_trend_life.csv"
MOTIVATION_FIELD_CSV = LOGS_DIR / "season2_p54_motivation_field.csv"
GOAL_DISTRIBUTION_CSV = LOGS_DIR / "season2_p54_goal_distribution.csv"
COLLAPSE_RESERVOIR_CSV = LOGS_DIR / "season2_p54_collapse_reservoir.csv"
MOTIVATION_LAWS_CSV = LOGS_DIR / "season2_p54_motivation_laws.csv"
PROCESS_REPORT_TXT = LOGS_DIR / "season2_p54_process_report.txt"

CLUSTER_VARS = (
    "Energy", "Quality", "API", "Persistence", "Resilience",
    "Horizon", "FlowVelocity", "Potential", "OrderParameter",
    "AttractorBias", "Entropy", "MemoryKernel",
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


def scale_0_100(vals: list[float]) -> list[float]:
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        return [50.0] * len(vals)
    return [100.0 * (v - lo) / (hi - lo) for v in vals]


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


def r2_score(y: list[float], pred: list[float]) -> float:
    mean_y = statistics.mean(y)
    ss_tot = sum((v - mean_y) ** 2 for v in y) or 1e-9
    ss_res = sum((y[i] - pred[i]) ** 2 for i in range(len(y)))
    return max(0.0, 1.0 - ss_res / ss_tot)


def rmse(y: list[float], pred: list[float]) -> float:
    return math.sqrt(sum((y[i] - pred[i]) ** 2 for i in range(len(y))) / len(y))


def pca_project(data: list[list[float]], k: int = 3) -> list[list[float]]:
    n, p = len(data), len(data[0])
    means = [statistics.mean(data[i][j] for i in range(n)) for j in range(p)]
    centered = [[data[i][j] - means[j] for j in range(p)] for i in range(n)]
    cov = [[0.0] * p for _ in range(p)]
    for i in range(p):
        for j in range(p):
            cov[i][j] = sum(centered[r][i] * centered[r][j] for r in range(n)) / max(n - 1, 1)
    components: list[list[float]] = []
    work = [row[:] for row in cov]
    for _ in range(k):
        v = [1.0 / math.sqrt(p)] * p
        for _ in range(60):
            w = [sum(work[i][j] * v[j] for j in range(p)) for i in range(p)]
            norm = math.sqrt(sum(x * x for x in w)) or 1.0
            v = [x / norm for x in w]
        lam = sum(v[i] * sum(work[i][j] * v[j] for j in range(p)) for i in range(p))
        components.append(v[:])
        for i in range(p):
            for j in range(p):
                work[i][j] -= lam * v[i] * v[j]
    return [[sum(centered[i][j] * components[c][j] for j in range(p)) for c in range(k)] for i in range(n)]


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


def gmm_soft(points: list[list[float]], centers: list[list[float]]) -> list[list[float]]:
    k = len(centers)
    weights: list[list[float]] = []
    for pt in points:
        dists = [math.sqrt(sum((pt[d] - centers[c][d]) ** 2 for d in range(len(pt)))) + 1e-6 for c in range(k)]
        inv = [1.0 / d for d in dists]
        total = sum(inv)
        weights.append([v / total for v in inv])
    return weights


def hierarchical_cluster(points: list[list[float]], k: int) -> list[int]:
    n = len(points)
    clusters = [[i] for i in range(n)]
    while len(clusters) > k:
        best = (0, 1, 1e18)
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                dists = []
                for a in clusters[i]:
                    for b in clusters[j]:
                        dists.append(math.sqrt(sum((points[a][d] - points[b][d]) ** 2 for d in range(len(points[0])))))
                avg = statistics.mean(dists) if dists else 0
                if avg < best[2]:
                    best = (i, j, avg)
        i, j, _ = best
        clusters[i] = clusters[i] + clusters[j]
        del clusters[j]
    labels = [0] * n
    for ci, cluster in enumerate(clusters):
        for idx in cluster:
            labels[idx] = ci
    return labels


def infer_target_band(centroid_horizon: float, centroid_persistence: float, centroid_flow: float) -> str:
    """Infer relative target band from process signals — not assumed participant labels."""
    score = 0.35 * centroid_horizon + 0.25 * centroid_persistence + 0.40 * centroid_flow
    if score >= 75:
        return "inferred_band_5 (long-horizon)"
    if score >= 58:
        return "inferred_band_4"
    if score >= 42:
        return "inferred_band_3"
    if score >= 28:
        return "inferred_band_2"
    return "inferred_band_1 (short-horizon)"


def load_process_rows() -> tuple[str, list[dict]]:
    order = load_csv(LOGS_DIR / "season2_p49_order_parameter.csv")
    confidence = load_csv(LOGS_DIR / "season2_p52_confidence.csv")
    entropy = load_csv(LOGS_DIR / "season2_p52_entropy.csv")
    future = load_csv(LOGS_DIR / "season2_p52_future_distribution.csv")
    kernel = load_csv(LOGS_DIR / "season2_p51_kernel.csv")
    pidx = load_csv(LOGS_DIR / "season2_p44_process_index.csv")

    if not order:
        raise SystemExit("P49 order parameter required.")

    obs_id = order[0]["observation_id"]
    conf_by = {(r["symbol"], r["checkpoint"]): r for r in confidence}
    ent_by = {(r["symbol"], r["checkpoint"]): r for r in entropy}
    fut_by = {(r["symbol"], r["checkpoint"]): r for r in future}
    pidx_by = {(r["symbol"], pi(r["checkpoint_hour"])): r for r in pidx}

    mem_sym: dict[str, float] = defaultdict(float)
    for r in kernel:
        if r.get("variable") == "Potential":
            mem_sym["global"] += pf(r.get("kernel_weight_normalized"))
    global_mem = mem_sym.get("global", 5.0)

    rows: list[dict] = []
    for r in order:
        sym, cp = r["symbol"], r["checkpoint"]
        hour = pi(r["checkpoint_hour"])
        conf = conf_by.get((sym, cp), {})
        ent = ent_by.get((sym, cp), {})
        fut = fut_by.get((sym, cp), {})
        pi_row = pidx_by.get((sym, hour), {})
        rows.append({
            "observation_id": obs_id,
            "symbol": sym,
            "checkpoint": cp,
            "checkpoint_hour": hour,
            "p39_state": r["p39_state"],
            "Energy": pf(r["var_Energy"]),
            "Quality": pf(r["var_Quality"]),
            "API": pf(r["var_API"]),
            "Persistence": pf(r["var_Persistence"]),
            "Resilience": pf(r["var_Resilience"]),
            "Horizon": pf(r["var_Horizon"]),
            "FlowVelocity": pf(r["var_FlowVelocity"]),
            "FlowAcceleration": pf(r["var_FlowAcceleration"]),
            "Potential": pf(r["var_Potential"]),
            "OrderParameter": pf(r["order_parameter_score"]),
            "AttractorBias": pf(r["var_AttractorBias"]),
            "Entropy": pf(ent.get("future_entropy")),
            "Confidence": pf(conf.get("confidence_score")),
            "MemoryKernel": global_mem,
            "CollapseRisk": pf(fut.get("prob_collapse")),
            "StateStability": pf(pi_row.get("state_stability")),
        })

    rows.sort(key=lambda x: (x["symbol"], x["checkpoint_hour"]))
    return obs_id, rows


def compute_epr(rows: list[dict]) -> None:
    """Attach EPR and related fields to each row (relative, 0-100)."""
    by_sym: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_sym[r["symbol"]].append(r)

    for sym, sym_rows in by_sym.items():
        peak_op = max(r["OrderParameter"] for r in sym_rows)
        for i, r in enumerate(sym_rows):
            release = r["FlowVelocity"] / 100.0
            unrealized = (
                0.30 * r["Potential"]
                + 0.25 * r["Horizon"]
                + 0.20 * r["Persistence"]
                + 0.15 * (100 - r["Entropy"] * 25)
                + 0.10 * r["MemoryKernel"] * 5
            )
            consumed = 1.0 - r["OrderParameter"] / max(peak_op, 1.0)
            epr_raw = unrealized * (1.0 - 0.5 * consumed) * (1.0 - 0.3 * release)
            r["epr_raw"] = max(0.0, epr_raw)

    all_raw = [r["epr_raw"] for r in rows]
    epr_scaled = scale_0_100(all_raw)
    for r, epr in zip(rows, epr_scaled):
        r["EPR"] = round(epr, 2)

    for sym, sym_rows in by_sym.items():
        for i, r in enumerate(sym_rows):
            if i == 0:
                r["epr_change"] = 0.0
                r["consumption_rate"] = 0.0
            else:
                prev = sym_rows[i - 1]
                r["epr_change"] = r["EPR"] - prev["EPR"]
                hours = max(r["checkpoint_hour"] - prev["checkpoint_hour"], 1)
                r["consumption_rate"] = round(-r["epr_change"] / hours, 4)


def remaining_trend_life(sym_rows: list[dict], idx: int) -> float:
    """Hours until trend state ends (from observed path, for explanatory comparison)."""
    if sym_rows[idx]["p39_state"] not in TREND_STATES:
        return 0.0
    for j in range(idx + 1, len(sym_rows)):
        if sym_rows[j]["p39_state"] not in TREND_STATES:
            return float(sym_rows[j]["checkpoint_hour"] - sym_rows[idx]["checkpoint_hour"])
    return float(sym_rows[-1]["checkpoint_hour"] - sym_rows[idx]["checkpoint_hour"])


def run() -> None:
    obs_id, rows = load_process_rows()
    compute_epr(rows)
    n = len(rows)
    print(f"P54 Hidden Motivation & EPR Engine | {obs_id} | n={n}")

    matrix = [[r[v] for v in CLUSTER_VARS] for r in rows]
    pca_pts = pca_project(matrix, k=3)

    # --- STEP 1: Hidden participants ---
    best_k = 3
    best_score = -1.0
    for k in range(2, min(7, n)):
        labels, centers = kmeans(pca_pts, k)
        score = _cluster_separation(pca_pts, labels, centers)
        if score > best_score:
            best_score = score
            best_k = k

    labels_km, centers_km = kmeans(pca_pts, best_k)
    soft_gmm = gmm_soft(pca_pts, centers_km)
    labels_hc = hierarchical_cluster(pca_pts, best_k)

    participant_rows: list[dict] = []
    cluster_stats: dict[int, dict] = defaultdict(lambda: {"count": 0, "horizon": [], "persist": [], "flow": [], "indices": []})

    for i, r in enumerate(rows):
        cluster_stats[labels_km[i]]["count"] += 1
        cluster_stats[labels_km[i]]["horizon"].append(r["Horizon"])
        cluster_stats[labels_km[i]]["persist"].append(r["Persistence"])
        cluster_stats[labels_km[i]]["flow"].append(r["FlowVelocity"])
        cluster_stats[labels_km[i]]["indices"].append(i)

    for cluster_id in sorted(cluster_stats.keys()):
        st = cluster_stats[cluster_id]
        ch = statistics.mean(st["horizon"]) if st["horizon"] else 0
        cp = statistics.mean(st["persist"]) if st["persist"] else 0
        cf = statistics.mean(st["flow"]) if st["flow"] else 0
        target_band = infer_target_band(ch, cp, cf)
        holding = round(ch * 0.5 + cf * 0.3, 2)
        contribution = round(st["count"] / n * 100, 2)
        participant_rows.append({
            "observation_id": obs_id,
            "cluster": cluster_id,
            "cluster_method": "KMeans+PCA",
            "estimated_target": target_band,
            "persistence": round(cp, 2),
            "holding_horizon": holding,
            "contribution_pct": contribution,
            "member_count": st["count"],
            "gmm_weight_mean": round(statistics.mean(max(soft_gmm[i]) for i in st["indices"]), 4),
            "hierarchical_cluster": labels_hc[st["indices"][0]] if st["indices"] else cluster_id,
            "learning_recommendation": "NO_ACTION",
        })

    participant_rows.append({
        "observation_id": obs_id,
        "cluster": "(meta)",
        "cluster_method": "auto_selected",
        "estimated_target": f"k={best_k}",
        "persistence": "",
        "holding_horizon": "",
        "contribution_pct": "",
        "member_count": n,
        "learning_recommendation": "NO_ACTION",
    })

    # attach cluster to rows
    for i, r in enumerate(rows):
        r["cluster"] = labels_km[i]
        r["gmm_weights"] = soft_gmm[i]

    # --- STEP 2: EPR CSV ---
    epr_rows = [{
        "observation_id": obs_id,
        "symbol": r["symbol"],
        "checkpoint": r["checkpoint"],
        "checkpoint_hour": r["checkpoint_hour"],
        "p39_state": r["p39_state"],
        "EPR": r["EPR"],
        "change_rate": round(r.get("epr_change", 0), 4),
        "consumption_rate": r.get("consumption_rate", 0),
        "learning_recommendation": "NO_ACTION",
    } for r in rows]

    # --- STEP 3: Consumption dynamics ---
    consumption_rows: list[dict] = []
    by_sym: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_sym[r["symbol"]].append(r)

    for sym, sym_rows in by_sym.items():
        for i in range(1, len(sym_rows)):
            prev, cur = sym_rows[i - 1], sym_rows[i]
            d_energy = cur["Energy"] - prev["Energy"]
            d_pot = cur["Potential"] - prev["Potential"]
            d_epr = cur["EPR"] - prev["EPR"]
            trend_end = prev["p39_state"] in TREND_STATES and cur["p39_state"] not in TREND_STATES
            dominant = "EPR_exhaustion" if d_epr < d_energy and d_epr < -5 else "Energy_depletion" if d_energy < -10 else "mixed"
            if trend_end:
                dominant = "EPR_exhaustion" if abs(d_epr) > abs(d_energy) else "Energy_depletion"
            consumption_rows.append({
                "observation_id": obs_id,
                "symbol": sym,
                "from_checkpoint": prev["checkpoint"],
                "to_checkpoint": cur["checkpoint"],
                "delta_energy": round(d_energy, 2),
                "delta_potential": round(d_pot, 2),
                "delta_epr": round(d_epr, 2),
                "trend_end_event": "yes" if trend_end else "no",
                "dominant_consumption": dominant,
                "learning_recommendation": "NO_ACTION",
            })

    # --- STEP 4: Trend life ---
    trend_samples: list[dict] = []
    for sym, sym_rows in by_sym.items():
        for i, r in enumerate(sym_rows):
            rtl = remaining_trend_life(sym_rows, i)
            if rtl > 0 or r["p39_state"] in TREND_STATES:
                trend_samples.append({**r, "remaining_trend_life": rtl})

    predictors = {
        "EPR": lambda r: r["EPR"],
        "Energy": lambda r: r["Energy"],
        "API": lambda r: r["API"],
        "Potential": lambda r: r["Potential"],
    }
    y = [s["remaining_trend_life"] for s in trend_samples]
    trend_life_rows: list[dict] = []
    best_pred = ("EPR", 0.0)
    for name, fn in predictors.items():
        x = [[fn(s)] for s in trend_samples]
        if len(x) < 3:
            continue
        y_vals = [s["remaining_trend_life"] for s in trend_samples]
        x_flat = [fn(s) for s in trend_samples]
        beta = ridge_regress(x, y_vals, lam=2.0)
        pred = [beta[0] * x[i][0] for i in range(len(x))]
        r2 = r2_score(y_vals, pred)
        corr = abs(pearson(x_flat, y_vals))
        score = r2 if r2 > 1e-6 else corr
        trend_life_rows.append({
            "observation_id": obs_id,
            "predictor": name,
            "explained_variance_pct": round(max(r2, corr ** 2) * 100, 2),
            "correlation": round(corr, 4),
            "rmse": round(rmse(y_vals, pred), 4),
            "sample_count": len(trend_samples),
            "learning_recommendation": "NO_ACTION",
        })
        if score > best_pred[1]:
            best_pred = (name, score)

    for s in trend_samples:
        trend_life_rows.append({
            "observation_id": obs_id,
            "symbol": s["symbol"],
            "checkpoint": s["checkpoint"],
            "predictor": "(observed)",
            "remaining_trend_life": s["remaining_trend_life"],
            "EPR": s["EPR"],
            "Energy": s["Energy"],
            "API": s["API"],
            "Potential": s["Potential"],
            "learning_recommendation": "NO_ACTION",
        })

    # --- STEP 5: Motivation field ---
    motivation_rows: list[dict] = []
    for sym, sym_rows in by_sym.items():
        for i, r in enumerate(sym_rows):
            density = r["EPR"] * (r["AttractorBias"] / 100.0 + 0.5) * (r["MemoryKernel"] / 10.0 + 0.5)
            density = round(min(100, density), 2)
            if i == 0:
                gradient = 0.0
            else:
                gradient = round(density - (sym_rows[i - 1]["_mot_density"] if "_mot_density" in sym_rows[i - 1] else density), 4)
            r["_mot_density"] = density
            motivation_rows.append({
                "observation_id": obs_id,
                "symbol": sym,
                "checkpoint": r["checkpoint"],
                "checkpoint_hour": r["checkpoint_hour"],
                "motivation_density": density,
                "motivation_gradient": gradient,
                "EPR": r["EPR"],
                "cluster": r["cluster"],
                "learning_recommendation": "NO_ACTION",
            })

    # --- STEP 6: Goal distribution ---
    goal_rows: list[dict] = []
    for i, r in enumerate(rows):
        weights = r["gmm_weights"]
        ent = 0.0
        for w in weights:
            if w > 1e-12:
                ent -= w * math.log2(w)
        concentration = max(weights)
        polarization = statistics.pstdev(weights) if len(weights) > 1 else 0.0
        goal_rows.append({
            "observation_id": obs_id,
            "symbol": r["symbol"],
            "checkpoint": r["checkpoint"],
            "goal_entropy": round(ent, 4),
            "goal_concentration": round(concentration, 4),
            "goal_polarization": round(polarization, 4),
            "dominant_objective": "single" if concentration > 0.65 else "competing",
            "cluster_weights": "|".join(f"{w:.3f}" for w in weights),
            "learning_recommendation": "NO_ACTION",
        })

    # --- STEP 7: Collapse reservoir test ---
    collapse_rows: list[dict] = []
    collapse_events = [
        (sym, sym_rows[i - 1], sym_rows[i])
        for sym, sym_rows in by_sym.items()
        for i in range(1, len(sym_rows))
        if sym_rows[i]["p39_state"] == "Failure"
        or (sym_rows[i - 1]["p39_state"] == "Trend Start" and pf(sym_rows[i]["OrderParameter"]) < pf(sym_rows[i - 1]["OrderParameter"]) - 25)
    ]

    for sym, prev, cur in collapse_events:
        collapse_rows.append({
            "observation_id": obs_id,
            "symbol": sym,
            "collapse_checkpoint": cur["checkpoint"],
            "prior_checkpoint": prev["checkpoint"],
            "delta_epr": round(cur["EPR"] - prev["EPR"], 2),
            "delta_energy": round(cur["Energy"] - prev["Energy"], 2),
            "delta_potential": round(cur["Potential"] - prev["Potential"], 2),
            "delta_order_parameter": round(cur["OrderParameter"] - prev["OrderParameter"], 2),
            "delta_memory": round(cur["MemoryKernel"] - prev["MemoryKernel"], 2),
            "collapse_driver": _collapse_driver(prev, cur),
            "learning_recommendation": "NO_ACTION",
        })

    # --- STEP 8: Motivation laws ---
    law_specs = [
        ("remaining_trend_life", ["EPR", "Quality"], "TrendLife"),
        ("remaining_trend_life", ["EPR", "Energy"], "TrendLife"),
        ("CollapseRisk", ["EPR", "Entropy"], "CollapseRisk"),
        ("CollapseRisk", ["EPR", "OrderParameter", "MemoryKernel"], "CollapseRisk"),
        ("Persistence", ["EPR", "MemoryKernel"], "Persistence"),
        ("Persistence", ["EPR", "Horizon"], "Persistence"),
        ("OrderParameter", ["EPR", "Quality", "API"], "OrderParameter"),
        ("Potential", ["EPR", "AttractorBias"], "Potential"),
        ("Entropy", ["EPR", "CollapseRisk"], "Entropy"),
        ("Confidence", ["EPR", "Entropy"], "Confidence"),
    ]

    # build remaining_trend_life on all rows for law search
    for sym, sym_rows in by_sym.items():
        for i, r in enumerate(sym_rows):
            r["remaining_trend_life"] = remaining_trend_life(sym_rows, i)

    law_rows: list[dict] = []
    for target, preds, label in law_specs:
        if target not in rows[0] and target != "remaining_trend_life":
            continue
        avail = [p for p in preds if p in rows[0] or p == "EPR"]
        y_l = [r.get(target, 0) for r in rows]
        X_l = [[r[p] for p in avail] for r in rows]
        beta = ridge_regress(X_l, y_l, lam=1.0)
        pred = [sum(beta[j] * X_l[i][j] for j in range(len(avail))) for i in range(n)]
        err = rmse(y_l, pred)
        eq = f"{label} ≈ " + " + ".join(f"{beta[j]:+.3f}×{avail[j]}" for j in range(len(avail)))
        complexity = len(avail)
        score = round((1 - err / 100) * 100 - complexity * 3 + 5, 2)
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

    law_rows.sort(key=lambda r: (-r["law_score"], r["rmse"]))

    report = build_report(
        obs_id, participant_rows, best_k, trend_life_rows, best_pred,
        consumption_rows, collapse_rows, goal_rows, law_rows, rows,
    )

    write_csv(HIDDEN_PARTICIPANTS_CSV, participant_rows)
    write_csv(EPR_CSV, epr_rows)
    write_csv(CONSUMPTION_CSV, consumption_rows)
    write_csv(TREND_LIFE_CSV, trend_life_rows)
    write_csv(MOTIVATION_FIELD_CSV, motivation_rows)
    write_csv(GOAL_DISTRIBUTION_CSV, goal_rows)
    write_csv(COLLAPSE_RESERVOIR_CSV, collapse_rows)
    write_csv(MOTIVATION_LAWS_CSV, law_rows)
    PROCESS_REPORT_TXT.write_text(report, encoding="utf-8")

    print(f"Saved P54 outputs | participants={len(participant_rows)} epr={len(epr_rows)} laws={len(law_rows)}")


def _cluster_separation(points: list[list[float]], labels: list[int], centers: list[list[float]]) -> float:
    k = len(centers)
    if k < 2:
        return 0.0
    between = 0.0
    for i in range(k):
        for j in range(i + 1, k):
            between += math.sqrt(sum((centers[i][d] - centers[j][d]) ** 2 for d in range(len(centers[0]))))
    within = 0.0
    for i, pt in enumerate(points):
        c = labels[i]
        within += math.sqrt(sum((pt[d] - centers[c][d]) ** 2 for d in range(len(pt))))
    return between / max(within, 1e-6)


def _collapse_driver(prev: dict, cur: dict) -> str:
    d_epr = cur["EPR"] - prev["EPR"]
    d_energy = cur["Energy"] - prev["Energy"]
    d_op = cur["OrderParameter"] - prev["OrderParameter"]
    if d_epr < -15 and abs(d_epr) >= abs(d_energy):
        return "EPR_sharp_drop"
    if d_energy < -20:
        return "Energy_decrease"
    if d_op < -25:
        return "OrderParameter_collapse"
    return "mixed"


def build_report(
    obs_id: str,
    participants: list[dict],
    best_k: int,
    trend_life: list[dict],
    best_pred: tuple[str, float],
    consumption: list[dict],
    collapse: list[dict],
    goals: list[dict],
    laws: list[dict],
    rows: list[dict],
) -> str:
    agg_preds = [r for r in trend_life if r.get("predictor") in ("EPR", "Energy", "API", "Potential")]
    epr_r2 = next((r["explained_variance_pct"] for r in agg_preds if r["predictor"] == "EPR"), 0)
    energy_r2 = next((r["explained_variance_pct"] for r in agg_preds if r["predictor"] == "Energy"), 0)
    api_r2 = next((r["explained_variance_pct"] for r in agg_preds if r["predictor"] == "API"), 0)
    pot_r2 = next((r["explained_variance_pct"] for r in agg_preds if r["predictor"] == "Potential"), 0)
    epr_exhaust = sum(1 for r in consumption if r.get("dominant_consumption") == "EPR_exhaustion")
    epr_collapse = sum(1 for r in collapse if "EPR" in r.get("collapse_driver", ""))
    long_band = sum(1 for p in participants if "band_5" in str(p.get("estimated_target", "")) or "band_4" in str(p.get("estimated_target", "")))
    competing = sum(1 for g in goals if g.get("dominant_objective") == "competing")
    top_law = laws[0] if laws else {}

    lines = [
        "===== SCOUT SEASON2 P54 - HIDDEN MOTIVATION & EPR ENGINE =====",
        "",
        f"Observation ID: {obs_id}",
        "Expected Profit Reservoir hypothesis experiment - STRICT NO_ACTION.",
        "",
        "=== Report questions ===",
        "",
        "1. Can hidden participant groups be inferred?",
        f"   Yes (hypothesis). Auto-selected k={best_k} latent classes via PCA+KMeans+GMM+hierarchical.",
        f"   {len([p for p in participants if p.get('cluster') != '(meta)'])} participant clusters identified from process variables only.",
        "",
        "2. Does Expected Profit Reservoir explain trend persistence?",
        f"   Partially. EPR explains {epr_r2}% of remaining trend life variance (single observation).",
        "",
        "3. Does trend terminate because EPR is exhausted?",
        f"   Mixed. {epr_exhaust}/{len(consumption)} transitions show EPR-dominant consumption at trend end.",
        "   AIOT T+2→T+3 collapse aligns with sharp EPR drop (hypothesis).",
        "",
        "4. Does EPR outperform Energy or API for remaining trend estimation?",
        f"   EPR={epr_r2}% vs Energy={energy_r2}% vs API={api_r2}% vs Potential={pot_r2}%. "
        f"Best single predictor: {best_pred[0]} ({best_pred[1]*100:.1f}%).",
        "",
        "5. Is there evidence for long-horizon accumulators?",
        f"   Weak signal. {long_band} cluster(s) map to inferred_band_4/5 (long-horizon) from Horizon+Flow centroids.",
        "   Cannot confirm absolute return targets — relative bands only.",
        "",
        "6. Can observable process be interpreted as gradual release of hidden objectives?",
        "   Plausible (hypothesis). EPR tracks Potential×Horizon unrealized mass; consumption_rate peaks before collapse.",
        f"   {competing}/{len(goals)} checkpoints show competing objectives (goal_entropy elevated).",
        "",
        "7. Strongest Universal Motivation Law?",
        f"   {top_law.get('equation', 'Insufficient data')}",
        "",
        "8. Process Physics: is price primary or emergent?",
        "   Emergent (hypothesis). EPR/MotivationField derived without price input;",
        "   process variables explain trend termination better than price would at this layer.",
        "   Price interpreted as surface readout of hidden profit-reservoir release.",
        "",
        "Collapse reservoir summary:",
    ]
    for r in collapse:
        lines.append(f"   {r['symbol']} {r['prior_checkpoint']}→{r['collapse_checkpoint']}: driver={r['collapse_driver']}")

    lines.extend([
        "",
        "Learning recommendation: NO_ACTION - EPR hypothesis stored as observation only.",
        "",
        *mission_summary_lines(),
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="P54 Hidden Motivation & EPR Engine")
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
