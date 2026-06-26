"""Adaptive feature weighting — uniform vs conditional (frozen model)."""

from __future__ import annotations

import numpy as np

from scout_auto_os.engine.research.adaptive_feature_weight.constants import WEIGHT_CLIP
from scout_auto_os.engine.research.adaptive_feature_weight.scan_conditions import primary_condition
from scout_auto_os.engine.research.ranking_engine.models import RankingModelBundle, predict_scores


def _blend_weights(
    w_global: dict[str, float],
    w_cond: dict[str, float],
    feat_names: list[str],
) -> np.ndarray:
    ratios = []
    for f in feat_names:
        g = w_global.get(f, 1e-9)
        c = w_cond.get(f, g)
        ratios.append(c / max(g, 1e-9))
    arr = np.array(ratios, dtype=float)
    arr = arr / (arr.mean() or 1.0)
    return np.clip(arr, WEIGHT_CLIP[0], WEIGHT_CLIP[1])


def predict_uniform(bundle: RankingModelBundle, rows: list[dict]) -> np.ndarray:
    return predict_scores(bundle, rows)


def predict_adaptive(
    bundle: RankingModelBundle,
    rows: list[dict],
    scan_tags: list[str],
    w_global: dict[str, float],
    w_cond_map: dict[str, dict[str, float]],
) -> np.ndarray:
    feat_names = bundle.feature_names
    primary = primary_condition(scan_tags)
    w_cond = w_cond_map.get(primary, w_global)
    ratios = _blend_weights(w_global, w_cond, feat_names)

    X = np.array([[float(r["x"].get(n, 0.0)) for n in feat_names] for r in rows], dtype=float)
    X_adj = X * ratios
    if bundle.scaler is not None:
        X_adj = bundle.scaler.transform(X_adj)
    if bundle.kind == "classifier":
        proba = bundle.model.predict_proba(X_adj)
        return proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
    return bundle.model.predict(X_adj)
