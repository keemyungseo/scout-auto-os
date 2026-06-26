"""Temporal vs snapshot analysis."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from scout_auto_os.engine.research.ranking_engine.importance import (
    gain_importance,
    merge_importance,
    permutation_importance_rows,
    shap_rows,
)
from scout_auto_os.engine.research.ranking_engine.models import RankingModelBundle


def importance_analysis(
    bundle: RankingModelBundle,
    rows: list[dict],
) -> tuple[list[dict], dict]:
    gain = gain_importance(bundle)
    perm = permutation_importance_rows(bundle, rows)
    shap = shap_rows(bundle, rows)
    merged = merge_importance(gain, perm, shap)

    level_totals = {"absolute": 0.0, "delta": 0.0, "accel": 0.0, "other": 0.0}
    for r in merged:
        f = r["feature"]
        score = float(r.get("combined_score", 0))
        if "_delta" in f or f.endswith("_delta"):
            level_totals["delta"] += score
        elif "_accel" in f:
            level_totals["accel"] += score
        elif "_current" in f or f.startswith("dna_") or f.startswith("entry_"):
            level_totals["absolute"] += score
        else:
            level_totals["other"] += score

    total = sum(level_totals.values()) or 1.0
    level_pct = {k: round(v / total * 100, 2) for k, v in level_totals.items()}

    delta_vs_abs: list[dict] = []
    by_base: dict[str, dict[str, float]] = defaultdict(lambda: {"absolute": 0.0, "delta": 0.0})
    for r in merged:
        f = r["feature"]
        score = float(r.get("combined_score", 0))
        if f.startswith("ts_") and "_delta" in f:
            base = f.replace("ts_", "").replace("_delta", "")
            by_base[base]["delta"] += score
        elif f.startswith("ts_") and "_current" in f:
            base = f.replace("ts_", "").replace("_current", "")
            by_base[base]["absolute"] += score
        elif not f.startswith("ts_") and not f.startswith("ctx_"):
            by_base[f]["absolute"] += score

    for base, scores in sorted(by_base.items(), key=lambda x: -(x[1]["delta"] + x[1]["absolute"])):
        d, a = scores["delta"], scores["absolute"]
        if d + a < 1e-9:
            continue
        delta_vs_abs.append({
            "base_feature": base,
            "delta_importance": round(d, 6),
            "absolute_importance": round(a, 6),
            "delta_wins": d > a,
            "delta_share_pct": round(d / (d + a) * 100, 2),
        })

    delta_vs_abs.sort(key=lambda x: -x["delta_importance"])
    return merged, {"level_pct": level_pct, "delta_vs_absolute": delta_vs_abs[:30]}


def statistical_significance(baseline_avg: float, temporal_avg: float, n: int) -> dict:
    """Simple paired approximation — probabilistic, not formal H-test on small n."""
    diff = temporal_avg - baseline_avg
    se = abs(baseline_avg) * 0.15 / max(n ** 0.5, 1)
    z = diff / se if se > 1e-9 else 0.0
    return {
        "lift_pct": round(diff / abs(baseline_avg or 0.01) * 100, 2),
        "diff": round(diff, 4),
        "approx_z": round(z, 4),
        "significant_hypothesis": abs(z) > 1.96 and diff > 0,
    }
