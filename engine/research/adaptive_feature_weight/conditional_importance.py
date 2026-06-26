"""Conditional feature importance — per scan state."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from scout_auto_os.engine.research.adaptive_feature_weight.constants import MIN_CONDITION_SAMPLES
from scout_auto_os.engine.research.ranking_engine.importance import (
    gain_importance,
    merge_importance,
    permutation_importance_rows,
    shap_rows,
)
from scout_auto_os.engine.research.ranking_engine.models import RankingModelBundle


def compute_conditional_importance(
    bundle: RankingModelBundle,
    train_rows: list[dict],
    scan_conditions: dict[str, list[str]],
    condition_id: str,
) -> list[dict]:
    scans_with = {s for s, tags in scan_conditions.items() if condition_id in tags}
    subset = [r for r in train_rows if r["scan_kst"] in scans_with]
    if len(subset) < MIN_CONDITION_SAMPLES:
        return []

    gain = gain_importance(bundle)
    perm = permutation_importance_rows(bundle, subset, max_samples=min(800, len(subset)))
    shap = shap_rows(bundle, subset, max_samples=min(300, len(subset)))
    merged = merge_importance(gain, perm, shap)

    for row in merged:
        row["condition_id"] = condition_id
        row["sample_count"] = len(subset)
        row["scan_count"] = len(scans_with)
    return merged


def importance_to_weight_map(importance_rows: list[dict]) -> dict[str, float]:
    if not importance_rows:
        return {}
    scores = {r["feature"]: float(r.get("combined_score", 0)) for r in importance_rows}
    total = sum(scores.values()) or 1.0
    return {k: v / total for k, v in scores.items()}


def build_all_conditional_importance(
    bundle: RankingModelBundle,
    train_rows: list[dict],
    scan_conditions: dict[str, list[str]],
    conditions: list[str],
) -> tuple[list[dict], dict[str, dict[str, float]]]:
    all_rows: list[dict] = []
    weight_maps: dict[str, dict[str, float]] = {}
    for cid in conditions:
        rows = compute_conditional_importance(bundle, train_rows, scan_conditions, cid)
        if rows:
            all_rows.extend(rows)
            weight_maps[cid] = importance_to_weight_map(rows)
    return all_rows, weight_maps
