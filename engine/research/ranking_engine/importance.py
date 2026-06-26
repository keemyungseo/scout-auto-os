"""Feature importance — gain, permutation, SHAP."""

from __future__ import annotations

import numpy as np
from sklearn.inspection import permutation_importance

from scout_auto_os.engine.research.ranking_engine.constants import RANDOM_SEED
from scout_auto_os.engine.research.ranking_engine.models import RankingModelBundle, predict_scores


def gain_importance(bundle: RankingModelBundle) -> list[dict]:
    model = bundle.model
    names = bundle.feature_names
    if hasattr(model, "feature_importances_"):
        imps = model.feature_importances_
    elif hasattr(model, "get_feature_importance"):
        imps = model.get_feature_importance()
    elif hasattr(model, "feature_importances"):
        imps = model.feature_importances()
    elif hasattr(model, "coef_"):
        imps = np.abs(np.ravel(model.coef_))
    else:
        return []
    imps = np.ravel(imps)
    if len(imps) != len(names):
        return []
    pairs = sorted(zip(names, imps), key=lambda x: -float(x[1]))
    return [
        {"feature": n, "gain_importance": round(float(v), 6), "rank": i + 1}
        for i, (n, v) in enumerate(pairs[:50])
    ]


def permutation_importance_rows(
    bundle: RankingModelBundle,
    rows: list[dict],
    max_samples: int = 1500,
) -> list[dict]:
    if len(rows) > max_samples:
        rows = rows[:max_samples]
    X = np.array([[float(r["x"].get(n, 0.0)) for n in bundle.feature_names] for r in rows])
    y = np.array([float(r["return_2h"]) for r in rows])
    if bundle.scaler is not None:
        X = bundle.scaler.transform(X)

    def scorer(est, Xv, _y):
        if bundle.kind == "classifier":
            p = est.predict_proba(Xv)
            return p[:, 1] if p.shape[1] > 1 else p[:, 0]
        return est.predict(Xv)

    try:
        result = permutation_importance(
            bundle.model, X, y, n_repeats=5, random_state=RANDOM_SEED, n_jobs=1,
        )
    except Exception:
        return []
    order = np.argsort(-result.importances_mean)
    return [
        {
            "feature": bundle.feature_names[i],
            "permutation_importance": round(float(result.importances_mean[i]), 6),
            "rank": rank + 1,
        }
        for rank, i in enumerate(order[:30])
    ]


def shap_rows(bundle: RankingModelBundle, rows: list[dict], max_samples: int = 400) -> list[dict]:
    try:
        import shap
    except ImportError:
        return []
    if len(rows) > max_samples:
        rows = rows[:max_samples]
    X = np.array([[float(r["x"].get(n, 0.0)) for n in bundle.feature_names] for r in rows])
    if bundle.scaler is not None:
        X = bundle.scaler.transform(X)
    try:
        if bundle.kind == "ranker" and hasattr(bundle.model, "predict"):
            explainer = shap.TreeExplainer(bundle.model)
            sv = explainer.shap_values(X)
            if isinstance(sv, list):
                sv = sv[0]
            mean_abs = np.abs(sv).mean(axis=0)
        else:
            explainer = shap.Explainer(bundle.model, X)
            vals = explainer(X)
            mean_abs = np.abs(vals.values).mean(axis=0)
    except Exception:
        return []
    order = np.argsort(-mean_abs)
    return [
        {
            "feature": bundle.feature_names[i],
            "shap_mean_abs": round(float(mean_abs[i]), 6),
            "rank": rank + 1,
        }
        for rank, i in enumerate(order[:30])
    ]


def merge_importance(
    gain: list[dict],
    perm: list[dict],
    shap: list[dict],
) -> list[dict]:
    feats = {g["feature"] for g in gain} | {p["feature"] for p in perm} | {s["feature"] for s in shap}
    gain_m = {g["feature"]: g.get("gain_importance", 0) for g in gain}
    perm_m = {p["feature"]: p.get("permutation_importance", 0) for p in perm}
    shap_m = {s["feature"]: s.get("shap_mean_abs", 0) for s in shap}
    rows = []
    for f in feats:
        score = gain_m.get(f, 0) * 0.4 + perm_m.get(f, 0) * 0.3 + shap_m.get(f, 0) * 0.3
        rows.append({
            "feature": f,
            "gain_importance": gain_m.get(f, 0),
            "permutation_importance": perm_m.get(f, 0),
            "shap_mean_abs": shap_m.get(f, 0),
            "combined_score": round(score, 6),
        })
    rows.sort(key=lambda x: -x["combined_score"])
    for i, r in enumerate(rows[:20], 1):
        r["importance_rank"] = i
    return rows[:20]
