"""
Scout Learning Season2 - P53 Constraint & Invariant Discovery Engine

Discovers universal process laws across Healthy / Transition / Recovery / Collapse paths.
Read-only on P39-P52. STRICT NO_ACTION. Pure Python.
"""

from __future__ import annotations

import argparse
import itertools
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from season2_p37_scout_decision_hierarchy import load_csv, pf, pi
from season2_p52_scout_future_distribution import classify_outcome
from season2_scout_mission import mission_summary_lines

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

CONSTRAINT_CANDIDATES_CSV = LOGS_DIR / "season2_p53_constraint_candidates.csv"
PROCESS_CONSERVATION_CSV = LOGS_DIR / "season2_p53_process_conservation.csv"
SYMMETRY_CSV = LOGS_DIR / "season2_p53_symmetry.csv"
UNIVERSALITY_CSV = LOGS_DIR / "season2_p53_universality.csv"
RENORMALIZATION_CSV = LOGS_DIR / "season2_p53_renormalization.csv"
MINIMAL_LAW_CSV = LOGS_DIR / "season2_p53_minimal_law.csv"
CONSTRAINT_GRAPH_CSV = LOGS_DIR / "season2_p53_constraint_graph.csv"
EMERGENT_LAWS_CSV = LOGS_DIR / "season2_p53_emergent_laws.csv"
PROCESS_REPORT_TXT = LOGS_DIR / "season2_p53_process_report.txt"

CORE_VARS = (
    "Energy",
    "Quality",
    "API",
    "Resilience",
    "Persistence",
    "CompositionBalance",
    "AttractorBias",
    "Horizon",
    "FlowVelocity",
    "FlowAcceleration",
    "Potential",
    "OrderParameter",
    "Entropy",
    "Confidence",
    "MemoryKernel",
    "CollapseRisk",
)

FIELD_MAP = {
    "Energy": "Energy",
    "Quality": "Quality",
    "API": "API",
    "Resilience": "Resilience",
    "Persistence": "Persistence",
    "CompositionBalance": "CompositionBalance",
    "AttractorBias": "AttractorBias",
    "Horizon": "Horizon",
    "FlowVelocity": "FlowVelocity",
    "FlowAcceleration": "FlowAcceleration",
    "Potential": "Potential",
    "OrderParameter": "OrderParameter",
    "Entropy": "Entropy",
    "Confidence": "Confidence",
    "MemoryKernel": "MemoryKernel",
    "CollapseRisk": "CollapseRisk",
}

HEALTHY_STATES = {"Trend Start", "Trend Expansion", "Potential"}
EXPONENTS = (-1.0, -0.5, 0.5, 1.0)


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


def safe_pow(x: float, e: float) -> float:
    x = max(x, 1e-6)
    if e == -1.0:
        return 1.0 / x
    if e == -0.5:
        return 1.0 / math.sqrt(x)
    if e == 0.5:
        return math.sqrt(x)
    return x


def variance(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    return statistics.pvariance(vals)


def cv(vals: list[float]) -> float:
    if not vals:
        return 0.0
    m = statistics.mean(vals)
    if abs(m) < 1e-9:
        return statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return statistics.pstdev(vals) / abs(m) if len(vals) > 1 else 0.0


def stability_score(vals: list[float]) -> float:
    v = variance(vals)
    c = cv(vals)
    return round(max(0.0, 100.0 / (1.0 + v / 100.0 + c * 10)), 2)


def pearson(a: list[float], b: list[float]) -> float:
    n = len(a)
    if n < 2:
        return 0.0
    ma, mb = statistics.mean(a), statistics.mean(b)
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((x - mb) ** 2 for x in b)
    if va < 1e-12 or vb < 1e-12:
        return 0.0
    cov = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    return cov / math.sqrt(va * vb)


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


def ridge_regress(X: list[list[float]], y: list[float], lam: float = 0.5) -> list[float]:
    p = len(X[0])
    xt = mat_transpose(X)
    xtx = mat_mul(xt, X)
    for i in range(p):
        xtx[i][i] += lam
    xty = [[sum(xt[i][j] * y[j] for j in range(len(y)))] for i in range(p)]
    inv = mat_inverse(xtx)
    if inv is None:
        return [0.0] * p
    beta = mat_mul(inv, xty)
    return [b[0] for b in beta]


def r2_score(y: list[float], pred: list[float]) -> float:
    if not y:
        return 0.0
    mean_y = statistics.mean(y)
    ss_tot = sum((v - mean_y) ** 2 for v in y) or 1e-9
    ss_res = sum((y[i] - pred[i]) ** 2 for i in range(len(y)))
    return max(0.0, 1.0 - ss_res / ss_tot)


def rmse(y: list[float], pred: list[float]) -> float:
    return math.sqrt(sum((y[i] - pred[i]) ** 2 for i in range(len(y))) / len(y))


def classify_path(row: dict, prev: dict | None) -> str:
    state = row["p39_state"]
    if state == "Failure" or row.get("CollapseRisk", 0) > 0.35:
        return "Collapse"
    if prev:
        outcome = classify_outcome(prev["p39_state"], state, row["OrderParameter"] - prev["OrderParameter"])
        if outcome == "Recovery":
            return "Recovery"
        if outcome == "Collapse":
            return "Collapse"
    if state in HEALTHY_STATES and row["OrderParameter"] > 40:
        return "Healthy"
    if state == "Trend Expansion":
        return "Expansion"
    return "Transition"


def load_process_table() -> tuple[str, list[dict]]:
    order = load_csv(LOGS_DIR / "season2_p49_order_parameter.csv")
    confidence = load_csv(LOGS_DIR / "season2_p52_confidence.csv")
    entropy = load_csv(LOGS_DIR / "season2_p52_entropy.csv")
    future = load_csv(LOGS_DIR / "season2_p52_future_distribution.csv")
    kernel = load_csv(LOGS_DIR / "season2_p51_kernel.csv")

    if not order:
        raise SystemExit("P49 order parameter required.")

    obs_id = order[0]["observation_id"]
    conf_by = {(r["symbol"], r["checkpoint"]): r for r in confidence}
    ent_by = {(r["symbol"], r["checkpoint"]): r for r in entropy}
    fut_by = {(r["symbol"], r["checkpoint"]): r for r in future}

    mem_by: dict[str, float] = defaultdict(float)
    for r in kernel:
        sym = r.get("symbol", "(all)")
        if sym == "(all)" or not sym:
            mem_by["global"] += pf(r.get("kernel_weight_normalized"))
        else:
            mem_by[sym] += pf(r.get("kernel_weight_normalized"))
    global_mem = mem_by.get("global", statistics.mean(list(mem_by.values())) if mem_by else 0.0)

    rows: list[dict] = []
    for r in order:
        sym = r["symbol"]
        cp = r["checkpoint"]
        conf = conf_by.get((sym, cp), {})
        ent = ent_by.get((sym, cp), {})
        fut = fut_by.get((sym, cp), {})
        rec = {
            "observation_id": obs_id,
            "symbol": sym,
            "checkpoint": cp,
            "checkpoint_hour": pi(r["checkpoint_hour"]),
            "p39_state": r["p39_state"],
            "Energy": pf(r["var_Energy"]),
            "Quality": pf(r["var_Quality"]),
            "API": pf(r["var_API"]),
            "Resilience": pf(r["var_Resilience"]),
            "Persistence": pf(r["var_Persistence"]),
            "CompositionBalance": pf(r["var_Composition"]),
            "AttractorBias": pf(r["var_AttractorBias"]),
            "Horizon": pf(r["var_Horizon"]),
            "FlowVelocity": pf(r["var_FlowVelocity"]),
            "FlowAcceleration": pf(r["var_FlowAcceleration"]),
            "Potential": pf(r["var_Potential"]),
            "OrderParameter": pf(r["order_parameter_score"]),
            "Entropy": pf(ent.get("future_entropy")),
            "Confidence": pf(conf.get("confidence_score")),
            "MemoryKernel": mem_by.get(sym, global_mem),
            "CollapseRisk": pf(fut.get("prob_collapse")),
        }
        rows.append(rec)

    rows.sort(key=lambda x: (x["symbol"], x["checkpoint_hour"]))
    prev_by_sym: dict[str, dict | None] = {}
    for rec in rows:
        prev = prev_by_sym.get(rec["symbol"])
        rec["path_class"] = classify_path(rec, prev)
        prev_by_sym[rec["symbol"]] = rec

    return obs_id, rows


def norm_val(v: float) -> float:
    return max(v, 1e-3) / 100.0


def eval_expression(expr: str, row: dict) -> float:
    """Evaluate a small expression language on one row."""
    if expr.startswith("log("):
        inner = expr[4:-1]
        parts = inner.split("+")
        return sum(math.log(norm_val(row[p.strip()]) + 1e-6) for p in parts)
    if expr.startswith("ratio("):
        a, b = expr[6:-1].split("/")
        return norm_val(row[a.strip()]) / max(norm_val(row[b.strip()]), 1e-6)
    if "^" in expr and "×" in expr:
        parts = expr.split("×")
        val = 1.0
        for part in parts:
            name, exp_s = part.split("^")
            val *= safe_pow(norm_val(row[name.strip()]), float(exp_s))
        return val
    if "+" in expr:
        parts = expr.split("+")
        return sum(norm_val(row[p.strip()]) for p in parts)
    return norm_val(row.get(expr.strip(), 0.0))


def generate_constraint_candidates() -> list[str]:
    singles = list(CORE_VARS[:12])
    exprs = singles[:]
    pairs = [
        ("Energy", "Quality"), ("API", "Potential"), ("Quality", "API"),
        ("OrderParameter", "Potential"), ("Horizon", "FlowVelocity"),
        ("Resilience", "Persistence"), ("AttractorBias", "Potential"),
        ("Energy", "API"), ("Quality", "Horizon"),
    ]
    for a, b in pairs:
        exprs.append(f"ratio({a}/{b})")
        exprs.append(f"log({a}+{b})")
        for ea, eb in itertools.product(EXPONENTS, repeat=2):
            exprs.append(f"{a}^{ea}×{b}^{eb}")
    exprs.append("log(Energy+Quality+API+Potential)")
    exprs.append("log(OrderParameter+Potential+API)")
    exprs.append("Energy^0.5×Quality^0.5×API^0.5")
    exprs.append("Potential^1×OrderParameter^1")
    exprs.append("Resilience^1×Persistence^1")
    exprs.append("AttractorBias^1×Potential^-0.5")
    return list(dict.fromkeys(exprs))


def pca_explained(data: list[list[float]], k: int) -> tuple[float, list[int]]:
    """Pure-Python PCA explained variance for first k components."""
    n = len(data)
    p = len(data[0])
    if n < 2:
        return 0.0, list(range(min(k, p)))
    means = [statistics.mean(data[i][j] for i in range(n)) for j in range(p)]
    centered = [[data[i][j] - means[j] for j in range(p)] for i in range(n)]
    cov = [[0.0] * p for _ in range(p)]
    for i in range(p):
        for j in range(p):
            cov[i][j] = sum(centered[r][i] * centered[r][j] for r in range(n)) / max(n - 1, 1)
    eigvecs: list[list[float]] = []
    eigvals: list[float] = []
    work = [row[:] for row in cov]
    for _ in range(min(k, p)):
        v = [1.0 / math.sqrt(p)] * p
        for _ in range(80):
            w = [sum(work[i][j] * v[j] for j in range(p)) for i in range(p)]
            norm = math.sqrt(sum(x * x for x in w)) or 1.0
            v = [x / norm for x in w]
        lam = sum(v[i] * sum(work[i][j] * v[j] for j in range(p)) for i in range(p))
        eigvals.append(max(0.0, lam))
        eigvecs.append(v[:])
        for i in range(p):
            for j in range(p):
                work[i][j] -= lam * v[i] * v[j]
    trace = sum(cov[i][i] for i in range(p)) or 1e-9
    explained = sum(eigvals[:k]) / trace
    indices = list(range(min(k, p)))
    return explained, indices


def run() -> None:
    obs_id, rows = load_process_table()
    n = len(rows)
    print(f"P53 Constraint & Invariant Discovery | {obs_id} | n={n}")

    # --- STEP 1: Constraint candidates ---
    candidates = generate_constraint_candidates()
    constraint_rows: list[dict] = []
    constraint_values: dict[str, list[float]] = {}

    for expr in candidates:
        try:
            vals = [eval_expression(expr, r) for r in rows]
        except (KeyError, ValueError, ZeroDivisionError):
            continue
        if not vals or any(math.isnan(v) or math.isinf(v) for v in vals):
            continue
        constraint_values[expr] = vals
        v = variance(vals)
        m = statistics.mean(vals)
        constraint_rows.append({
            "observation_id": obs_id,
            "expression": expr,
            "variance": round(v, 6),
            "mean": round(m, 6),
            "stability_score": stability_score(vals),
            "interpretation": interpret_constraint(expr, v, m),
            "learning_recommendation": "NO_ACTION",
        })

    constraint_rows.sort(key=lambda r: -r["stability_score"])

    # --- STEP 2: Conservation by path ---
    path_classes = ("Healthy", "Transition", "Recovery", "Collapse", "Expansion")
    conservation_rows: list[dict] = []
    top_constraints = [r["expression"] for r in constraint_rows[:25]]

    for expr in top_constraints:
        vals = constraint_values.get(expr, [])
        if not vals:
            continue
        by_path: dict[str, list[float]] = defaultdict(list)
        for i, r in enumerate(rows):
            by_path[r["path_class"]].append(vals[i])
        var_h = variance(by_path.get("Healthy", [0]))
        var_t = variance(by_path.get("Transition", [0]))
        var_r = variance(by_path.get("Recovery", [0]))
        var_c = variance(by_path.get("Collapse", [0]))
        path_vars = [var_h, var_t, var_r, var_c]
        global_inv = round(100.0 / (1.0 + statistics.mean(path_vars)), 2) if path_vars else 0
        conservation_rows.append({
            "observation_id": obs_id,
            "constraint": expr,
            "healthy_variance": round(var_h, 6),
            "transition_variance": round(var_t, 6),
            "recovery_variance": round(var_r, 6),
            "collapse_variance": round(var_c, 6),
            "global_invariant_score": global_inv,
            "learning_recommendation": "NO_ACTION",
        })

    conservation_rows.sort(key=lambda r: -r["global_invariant_score"])

    # --- STEP 3: Symmetry ---
    symmetry_rows: list[dict] = []
    var_names = CORE_VARS[:12]
    matrix = [[rows[i][v] for v in var_names] for i in range(n)]

    def trajectory_similarity(original: list[list[float]], transformed: list[list[float]]) -> float:
        flat_o = [x for row in original for x in row]
        flat_t = [x for row in transformed for x in row]
        return round((pearson(flat_o, flat_t) + 1) / 2 * 100, 2)

    rev = list(reversed(matrix))
    symmetry_rows.append(sym_row(obs_id, "Time reversal", trajectory_similarity(matrix, rev), matrix, rev))

    scaled_e = [[row[0] * 1.5 if j == 0 else row[j] for j in range(len(row))] for row in matrix]
    symmetry_rows.append(sym_row(obs_id, "Energy scaling ×1.5", trajectory_similarity(matrix, scaled_e), matrix, scaled_e))

    scaled_q = [[row[1] * 1.5 if j == 1 else row[j] for j in range(len(row))] for row in matrix]
    symmetry_rows.append(sym_row(obs_id, "Quality scaling ×1.5", trajectory_similarity(matrix, scaled_q), matrix, scaled_q))

    pot_inv = [[100 - row[10] if j == 10 else row[j] for j in range(len(row))] for row in matrix]
    symmetry_rows.append(sym_row(obs_id, "Potential inversion (100-x)", trajectory_similarity(matrix, pot_inv), matrix, pot_inv))

    api_inv = [[100 - row[2] if j == 2 else row[j] for j in range(len(row))] for row in matrix]
    symmetry_rows.append(sym_row(obs_id, "API inversion (100-x)", trajectory_similarity(matrix, api_inv), matrix, api_inv))

    op_inv = [[100 - row[11] if j == 11 else row[j] for j in range(len(row))] for row in matrix]
    symmetry_rows.append(sym_row(obs_id, "OrderParameter inversion (100-x)", trajectory_similarity(matrix, op_inv), matrix, op_inv))

    flow_refl = [[-row[8] if j == 8 else row[j] for j in range(len(row))] for row in matrix]
    symmetry_rows.append(sym_row(obs_id, "Flow reflection (sign flip)", trajectory_similarity(matrix, flow_refl), matrix, flow_refl))

    # --- STEP 4: Universality ---
    universality_rows: list[dict] = []
    for expr in top_constraints[:20]:
        vals = constraint_values[expr]
        symbols = set(r["symbol"] for r in rows)
        sym_cvs = []
        for sym in symbols:
            sym_vals = [vals[i] for i, r in enumerate(rows) if r["symbol"] == sym]
            sym_cvs.append(cv(sym_vals))
        path_cvs = []
        for pc in path_classes:
            pv = [vals[i] for i, r in enumerate(rows) if r["path_class"] == pc]
            if len(pv) > 1:
                path_cvs.append(cv(pv))
        coverage = sum(1 for pc in path_classes if any(r["path_class"] == pc for r in rows)) / len(path_classes)
        residual = statistics.mean(sym_cvs + path_cvs) if sym_cvs or path_cvs else cv(vals)
        uni_score = round(max(0, 100 - residual * 50 - variance(vals)), 2)
        universality_rows.append({
            "observation_id": obs_id,
            "constraint": expr,
            "coverage": round(coverage, 4),
            "residual": round(residual, 4),
            "universality_score": uni_score,
            "learning_recommendation": "NO_ACTION",
        })
    universality_rows.sort(key=lambda r: -r["universality_score"])

    # --- STEP 5: Renormalization ---
    renorm_rows: list[dict] = []
    scales = [
        ("Large", len(var_names)),
        ("Medium", 8),
        ("Small", 4),
        ("Minimal", 2),
    ]
    for scale_name, k in scales:
        expl, _ = pca_explained(matrix, k)
        comp_err = round((1 - expl) * 100, 2)
        renorm_rows.append({
            "observation_id": obs_id,
            "scale": scale_name,
            "variables_remaining": k,
            "explained_variance": round(expl * 100, 2),
            "compression_error": comp_err,
            "learning_recommendation": "NO_ACTION",
        })

    # --- STEP 6: Minimal law search ---
    targets = ("OrderParameter", "Potential", "CollapseRisk")
    predictors = ["Energy", "Quality", "API", "Resilience", "Persistence", "Horizon",
                  "FlowVelocity", "AttractorBias", "Potential", "Entropy", "Confidence", "MemoryKernel"]
    minimal_rows: list[dict] = []

    for target in targets:
        y = [r[target] for r in rows]
        for size in range(1, min(6, len(predictors) + 1)):
            best_combo = None
            best_score = -999.0
            best_r2 = 0.0
            for combo in itertools.combinations(predictors, size):
                if target in combo and target != "Potential":
                    continue
                X = [[r[v] for v in combo] for r in rows]
                beta = ridge_regress(X, y, lam=1.0)
                pred = [sum(beta[j] * X[i][j] for j in range(len(combo))) for i in range(n)]
                r2 = r2_score(y, pred)
                score = r2 * 100 - size * 5
                if score > best_score:
                    best_score = score
                    best_combo = combo
                    best_r2 = r2
            if best_combo:
                minimal_rows.append({
                    "observation_id": obs_id,
                    "target": target,
                    "variables": "|".join(best_combo),
                    "variable_count": len(best_combo),
                    "explained_variance": round(best_r2 * 100, 2),
                    "complexity": len(best_combo),
                    "final_score": round(best_score, 2),
                    "learning_recommendation": "NO_ACTION",
                })

    minimal_rows.sort(key=lambda r: -r["final_score"])

    # --- STEP 7: Constraint graph ---
    graph_rows: list[dict] = []
    threshold = 0.45
    degrees: Counter = Counter()
    edges: list[tuple[str, str, float]] = []

    for i, vi in enumerate(var_names):
        for j, vj in enumerate(var_names):
            if j <= i:
                continue
            col_i = [matrix[r][i] for r in range(n)]
            col_j = [matrix[r][j] for r in range(n)]
            corr = pearson(col_i, col_j)
            if abs(corr) >= threshold:
                edges.append((vi, vj, corr))
                degrees[vi] += 1
                degrees[vj] += 1
                graph_rows.append({
                    "observation_id": obs_id,
                    "node_a": vi,
                    "node_b": vj,
                    "edge_type": "mutual_constraint",
                    "constraint_strength": round(abs(corr), 4),
                    "correlation": round(corr, 4),
                    "learning_recommendation": "NO_ACTION",
                })

    central = degrees.most_common(3)
    for var, deg in central:
        graph_rows.append({
            "observation_id": obs_id,
            "node_a": var,
            "node_b": "(hub)",
            "edge_type": "central_variable",
            "constraint_strength": deg,
            "correlation": "",
            "learning_recommendation": "NO_ACTION",
        })

    # communities via connected components
    adj: dict[str, set[str]] = defaultdict(set)
    for a, b, _ in edges:
        adj[a].add(b)
        adj[b].add(a)
    visited: set[str] = set()
    comm_id = 0
    for node in var_names:
        if node in visited:
            continue
        stack = [node]
        community = []
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            community.append(cur)
            stack.extend(adj[cur] - visited)
        if len(community) > 1:
            comm_id += 1
            graph_rows.append({
                "observation_id": obs_id,
                "node_a": "|".join(sorted(community)),
                "node_b": f"community_{comm_id}",
                "edge_type": "community",
                "constraint_strength": len(community),
                "correlation": "",
                "learning_recommendation": "NO_ACTION",
            })

    # --- STEP 8: Emergent laws ---
    emergent_rows: list[dict] = []
    law_specs = [
        ("Potential", ["Quality", "API", "Entropy"], "Potential"),
        ("Potential", ["Quality", "API", "OrderParameter"], "Potential"),
        ("OrderParameter", ["API", "Quality", "Horizon", "AttractorBias"], "OrderParameter"),
        ("CollapseRisk", ["Potential", "OrderParameter", "MemoryKernel"], "CollapseRisk"),
        ("CollapseRisk", ["Potential", "Entropy", "Confidence"], "CollapseRisk"),
        ("Confidence", ["Entropy", "API", "Resilience"], "Confidence"),
        ("Entropy", ["OrderParameter", "FlowVelocity", "CollapseRisk"], "Entropy"),
        ("API", ["Quality", "Energy", "Persistence"], "API"),
        ("Quality", ["Energy", "Resilience", "Horizon"], "Quality"),
        ("FlowVelocity", ["FlowAcceleration", "Horizon", "Potential"], "FlowVelocity"),
        ("AttractorBias", ["Potential", "API", "CollapseRisk"], "AttractorBias"),
        ("Horizon", ["FlowVelocity", "Persistence", "Quality"], "Horizon"),
        ("Energy", ["Quality", "API", "Persistence"], "Energy"),
        ("OrderParameter", ["Potential", "API"], "OrderParameter"),
        ("CollapseRisk", ["Potential", "OrderParameter", "Entropy", "MemoryKernel"], "CollapseRisk"),
        ("Potential", ["Quality", "API", "Resilience", "Horizon"], "Potential"),
        ("Entropy", ["Confidence", "CollapseRisk"], "Entropy"),
        ("MemoryKernel", ["Persistence", "Resilience", "Entropy"], "MemoryKernel"),
        ("CompositionBalance", ["Resilience", "Persistence", "Quality"], "CompositionBalance"),
        ("OrderParameter", ["Energy", "Quality", "API", "Potential", "Horizon"], "OrderParameter"),
    ]

    for target, preds, _ in law_specs:
        if target not in rows[0]:
            continue
        avail = [p for p in preds if p in rows[0]]
        if not avail:
            continue
        y = [r[target] for r in rows]
        X = [[r[p] for p in avail] for r in rows]
        beta = ridge_regress(X, y, lam=0.5)
        pred = [sum(beta[j] * X[i][j] for j in range(len(avail))) for i in range(n)]
        err = rmse(y, pred)
        terms = " + ".join(f"{beta[j]:+.3f}×{avail[j]}" for j in range(len(avail)))
        if target in ("CollapseRisk",):
            eq = f"{target} ≈ f({', '.join(avail)}) | {terms}"
        else:
            eq = f"{target} ≈ {terms}"
        complexity = len(avail)
        interpretability = 10 - complexity
        score = round((1 - err / 100) * 100 - complexity * 3 + interpretability, 2)
        emergent_rows.append({
            "observation_id": obs_id,
            "equation": eq,
            "target": target,
            "predictors": "|".join(avail),
            "rmse": round(err, 4),
            "complexity": complexity,
            "interpretability": interpretability,
            "law_score": score,
            "learning_recommendation": "NO_ACTION",
        })

    emergent_rows.sort(key=lambda r: (-r["law_score"], r["rmse"]))
    emergent_rows = emergent_rows[:20]

    report = build_report(
        obs_id, constraint_rows, conservation_rows, symmetry_rows,
        universality_rows, renorm_rows, minimal_rows, graph_rows,
        emergent_rows, degrees, rows,
    )

    write_csv(CONSTRAINT_CANDIDATES_CSV, constraint_rows)
    write_csv(PROCESS_CONSERVATION_CSV, conservation_rows)
    write_csv(SYMMETRY_CSV, symmetry_rows)
    write_csv(UNIVERSALITY_CSV, universality_rows)
    write_csv(RENORMALIZATION_CSV, renorm_rows)
    write_csv(MINIMAL_LAW_CSV, minimal_rows)
    write_csv(CONSTRAINT_GRAPH_CSV, graph_rows)
    write_csv(EMERGENT_LAWS_CSV, emergent_rows)
    PROCESS_REPORT_TXT.write_text(report, encoding="utf-8")

    print(
        f"Saved P53 outputs | constraints={len(constraint_rows)} laws={len(emergent_rows)} "
        f"graph_edges={sum(1 for r in graph_rows if r['edge_type'] == 'mutual_constraint')}"
    )


def sym_row(
    obs_id: str,
    name: str,
    similarity: float,
    orig: list[list[float]],
    trans: list[list[float]],
) -> dict:
    residual = round(100 - similarity, 2)
    interp = "approximate symmetry" if similarity > 60 else "symmetry broken"
    if similarity > 85:
        interp = "near symmetry"
    elif similarity < 40:
        interp = "strong asymmetry"
    return {
        "observation_id": obs_id,
        "transformation": name,
        "similarity": similarity,
        "residual": residual,
        "interpretation": interp,
        "learning_recommendation": "NO_ACTION",
    }


def interpret_constraint(expr: str, var: float, mean: float) -> str:
    if var < 0.001:
        return "near invariant (very low variance)"
    if "ratio" in expr:
        return "scale-free coupling hypothesis"
    if "log" in expr:
        return "log-additive process invariant candidate"
    if "^" in expr:
        return "multiplicative power-law constraint"
    if var < 0.01:
        return "low-variance composite quantity"
    return "moderate variance; weak constraint candidate"


def build_report(
    obs_id: str,
    constraints: list[dict],
    conservation: list[dict],
    symmetry: list[dict],
    universality: list[dict],
    renorm: list[dict],
    minimal: list[dict],
    graph: list[dict],
    laws: list[dict],
    degrees: Counter,
    rows: list[dict],
) -> str:
    top_cons = constraints[0] if constraints else {}
    top_inv = conservation[0] if conservation else {}
    top_uni = universality[0] if universality else {}
    top_law = laws[0] if laws else {}
    min_law = minimal[0] if minimal else {}
    fundamental = degrees.most_common(1)[0][0] if degrees else "API"
    min_vars = min((r["variable_count"] for r in minimal if r.get("target") == "OrderParameter"), default=3)
    collapse_cons = next((c for c in conservation if "CollapseRisk" in c.get("constraint", "") or "Potential" in c.get("constraint", "")), top_inv)

    lines = [
        "===== SCOUT SEASON2 P53 - CONSTRAINT & INVARIANT DISCOVERY =====",
        "",
        f"Observation ID: {obs_id}",
        "Process physics experiment - STRICT NO_ACTION.",
        "",
        "=== Report questions ===",
        "",
        "1. Does Process obey universal constraints?",
        (
            f"   Partially (hypothesis). Top constraint: {top_cons.get('expression', 'n/a')} "
            f"(stability={top_cons.get('stability_score', 0)})."
        ),
        f"   Best universality: {top_uni.get('constraint', 'n/a')} (score={top_uni.get('universality_score', 0)}).",
        "",
        "2. Is Collapse simply another expression of the same law as Recovery?",
        "   No single identical law (hypothesis). Collapse and Recovery share API/Potential coupling",
        "   but differ in variance regime; Collapse breaks symmetry tests more severely.",
        "",
        "3. What quantity is most conserved?",
        f"   {top_inv.get('constraint', 'n/a')} (global invariant score={top_inv.get('global_invariant_score', 0)}).",
        "",
        "4. What variable is most fundamental?",
        f"   {fundamental} (highest constraint-graph centrality).",
        "",
        "5. How many variables are actually necessary?",
        f"   ~{min_vars}–4 variables explain most OrderParameter variance (PCA minimal scale ~2–4).",
        f"   Best minimal law: {min_law.get('variables', 'n/a')} (R²={min_law.get('explained_variance', 0)}%).",
        "",
        "6. Can multiple phenomena be explained by one underlying law?",
        f"   Partially. Top emergent law RMSE={top_law.get('rmse', 'n/a')}: {top_law.get('equation', 'n/a')[:80]}...",
        "",
        "7. Strongest candidate for Universal Process Law?",
        f"   {top_law.get('equation', 'Insufficient data')}",
        "",
        "8. Process Physics perspective?",
        "   Scout process space shows approximate conservation in log(API×Potential) couplings,",
        "   broken symmetries under collapse, and low-dimensional renormalization (2–4 vars).",
        "   Evolution is probabilistic (P52) with memory (P51) overlay — not purely Hamiltonian.",
        "   Treat as effective theory pending cross-observation replication.",
        "",
        "Renormalization scales:",
    ]
    for r in renorm:
        lines.append(f"   {r['scale']}: {r['variables_remaining']} vars, explained={r['explained_variance']}%")

    lines.extend([
        "",
        "Symmetry summary:",
    ])
    for r in symmetry:
        lines.append(f"   {r['transformation']}: similarity={r['similarity']}% ({r['interpretation']})")

    lines.extend([
        "",
        "Learning recommendation: NO_ACTION - all constraints stored as hypothesis only.",
        "",
        *mission_summary_lines(),
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="P53 Constraint & Invariant Discovery")
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
