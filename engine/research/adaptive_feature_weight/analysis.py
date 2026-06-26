"""Feature interaction and stability analysis."""

from __future__ import annotations

import statistics
from collections import defaultdict
from itertools import combinations

import numpy as np

from scout_auto_os.engine.research.adaptive_feature_weight.constants import INTERACTION_TOP_N


def interaction_matrix(
    rows: list[dict],
    feat_names: list[str],
    condition_id: str,
    top_n: int = INTERACTION_TOP_N,
) -> list[dict]:
    if len(rows) < 20:
        return []
    y = np.array([float(r["return_2h"]) for r in rows])
    feat_imp: dict[str, float] = {}
    for f in feat_names:
        x = np.array([float(r["x"].get(f, 0)) for r in rows])
        if np.std(x) < 1e-9:
            continue
        feat_imp[f] = abs(float(np.corrcoef(x, y)[0, 1]))
    top_feats = [f for f, _ in sorted(feat_imp.items(), key=lambda x: -x[1])[:top_n]]

    out: list[dict] = []
    for f1, f2 in combinations(top_feats, 2):
        x1 = np.array([float(r["x"].get(f1, 0)) for r in rows])
        x2 = np.array([float(r["x"].get(f2, 0)) for r in rows])
        inter = x1 * x2
        if np.std(inter) < 1e-9:
            continue
        corr = float(np.corrcoef(inter, y)[0, 1])
        out.append({
            "condition_id": condition_id,
            "feature_a": f1,
            "feature_b": f2,
            "interaction_strength": round(abs(corr), 6),
            "signed_corr": round(corr, 6),
        })
    out.sort(key=lambda x: -x["interaction_strength"])
    return out[:50]


def feature_stability(
    conditional_rows: list[dict],
    feat_names: list[str],
) -> list[dict]:
    by_feat: dict[str, list[float]] = defaultdict(list)
    for r in conditional_rows:
        by_feat[r["feature"]].append(float(r.get("combined_score", 0)))

    out: list[dict] = []
    for f in feat_names:
        vals = by_feat.get(f, [])
        if len(vals) < 2:
            continue
        mu = statistics.mean(vals)
        sd = statistics.pstdev(vals)
        out.append({
            "feature": f,
            "mean_importance": round(mu, 6),
            "std_across_conditions": round(sd, 6),
            "stability_score": round(mu / (sd + 1e-9), 4),
            "condition_count": len(vals),
        })
    out.sort(key=lambda x: -x["stability_score"])
    return out


def feature_drift(
    train_imp: list[dict],
    blind_imp: list[dict],
) -> list[dict]:
    train_m = {(r["condition_id"], r["feature"]): float(r.get("combined_score", 0)) for r in train_imp}
    blind_m = {(r["condition_id"], r["feature"]): float(r.get("combined_score", 0)) for r in blind_imp}
    keys = set(train_m) | set(blind_m)
    out: list[dict] = []
    for key in keys:
        cid, feat = key
        t = train_m.get(key, 0)
        b = blind_m.get(key, 0)
        out.append({
            "condition_id": cid,
            "feature": feat,
            "train_importance": round(t, 6),
            "blind_importance": round(b, 6),
            "drift": round(b - t, 6),
            "drift_pct": round((b - t) / abs(t or 1e-9) * 100, 2),
        })
    out.sort(key=lambda x: -abs(x["drift"]))
    return out[:200]
