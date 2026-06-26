"""Feature importance for DNA predictor."""

from __future__ import annotations

import statistics

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _feature_names(rows: list[dict]) -> list[str]:
    return sorted(rows[0]["x"].keys()) if rows else []


def _matrix(rows: list[dict], names: list[str]) -> np.ndarray:
    return np.array([[float(r["x"].get(n, 0)) for n in names] for r in rows], dtype=float)


def compute_importance(rows: list[dict], y: np.ndarray) -> list[dict]:
    names = _feature_names(rows)
    if not names or len(rows) < 10:
        return []

    X = _matrix(rows, names)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    mi = mutual_info_classif(Xs, y, random_state=42)
    mi_map = {n: float(v) for n, v in zip(names, mi)}

    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    lr.fit(Xs, y)
    coef_map = {n: float(abs(c)) for n, c in zip(names, lr.coef_.ravel())}

    rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    rf.fit(Xs, y)
    rf_map = {n: float(v) for n, v in zip(names, rf.feature_importances_)}

    perm = permutation_importance(rf, Xs, y, n_repeats=10, random_state=42, n_jobs=1)
    perm_map = {n: float(v) for n, v in zip(names, perm.importances_mean)}

    shap_map: dict[str, float] = {}
    try:
        import shap
        explainer = shap.Explainer(rf, Xs)
        sv = explainer(Xs)
        mean_abs = np.abs(sv.values).mean(axis=0)
        shap_map = {n: float(v) for n, v in zip(names, mean_abs)}
    except Exception:
        shap_map = {n: 0.0 for n in names}

    out: list[dict] = []
    for n in names:
        out.append({
            "feature": n,
            "information_gain": round(mi_map.get(n, 0), 6),
            "logistic_coef_abs": round(coef_map.get(n, 0), 6),
            "rf_importance": round(rf_map.get(n, 0), 6),
            "permutation_importance": round(perm_map.get(n, 0), 6),
            "shap_mean_abs": round(shap_map.get(n, 0), 6),
        })

    def _rank_key(d: dict, key: str) -> float:
        return float(d.get(key, 0))

    for key in ("information_gain", "shap_mean_abs", "permutation_importance", "rf_importance", "logistic_coef_abs"):
        ranked = sorted(out, key=lambda d: -_rank_key(d, key))
        for i, row in enumerate(ranked):
            row[f"rank_{key}"] = i + 1

    composite = []
    for row in out:
        ranks = [
            row.get("rank_information_gain", 999),
            row.get("rank_shap_mean_abs", 999),
            row.get("rank_permutation_importance", 999),
            row.get("rank_rf_importance", 999),
        ]
        row["composite_rank"] = round(statistics.mean(ranks), 2)
        composite.append(row)
    composite.sort(key=lambda d: d["composite_rank"])
    for i, row in enumerate(composite):
        row["overall_rank"] = i + 1
    return composite


def top_features_for_class(
    importance_rows: list[dict],
    rows: list[dict],
    class_label: int,
    top_n: int = 20,
) -> list[dict]:
    """Features that best separate class vs rest (mean diff * composite importance)."""
    names = _feature_names(rows)
    sub = [r for r in rows if r["cluster_id"] == class_label]
    rest = [r for r in rows if r["cluster_id"] != class_label]
    if not sub or not rest:
        return []

    imp_map = {r["feature"]: r for r in importance_rows}
    scored: list[dict] = []
    for n in names:
        a = statistics.mean(float(r["x"].get(n, 0)) for r in sub)
        b = statistics.mean(float(r["x"].get(n, 0)) for r in rest)
        imp = imp_map.get(n, {})
        scored.append({
            "feature": n,
            "class_mean": round(a, 4),
            "other_mean": round(b, 4),
            "mean_diff": round(a - b, 4),
            "overall_rank": imp.get("overall_rank", 999),
            "information_gain": imp.get("information_gain", 0),
            "shap_mean_abs": imp.get("shap_mean_abs", 0),
        })
    scored.sort(key=lambda d: (-abs(d["mean_diff"]) * (1.0 / max(d["overall_rank"], 1))))
    return scored[:top_n]
