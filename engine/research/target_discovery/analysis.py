"""Label ranking, learnability, and feature-importance shift analysis."""

from __future__ import annotations

from scout_auto_os.engine.research.ranking_engine.importance import (
    gain_importance,
    merge_importance,
    permutation_importance_rows,
    shap_rows,
)
from scout_auto_os.engine.research.ranking_engine.models import RankingModelBundle
from scout_auto_os.engine.research.target_discovery.label_builder import LabelSpec, apply_label, label_learnability


def rank_labels(blind_results: list[dict], baseline_id: str) -> list[dict]:
    rows = [r for r in blind_results if not r.get("error")]
    ranked = sorted(
        rows,
        key=lambda x: (
            -float(x.get("avg_return_2h", 0)),
            -float(x.get("rank_ndcg5", 0)),
            -float(x.get("sharpe", 0)),
        ),
    )
    out: list[dict] = []
    baseline_avg = next(
        (float(r.get("avg_return_2h", 0)) for r in rows if r.get("label_id") == baseline_id),
        0.0,
    )
    for i, r in enumerate(ranked, 1):
        avg = float(r.get("avg_return_2h", 0))
        out.append({
            **{k: r.get(k) for k in (
                "label_id", "label_name", "category", "rank_key",
                "avg_return_2h", "top2_avg_return_2h", "top5_avg_return_2h",
                "win_rate", "sharpe", "rank_ndcg5", "rank_p5",
            )},
            "label_rank": i,
            "vs_baseline_pct": round((avg - baseline_avg) / abs(baseline_avg or 0.01) * 100, 2),
            "beats_baseline": avg > baseline_avg,
        })
    return out


def learnability_report(train_rows: list[dict], specs: list[LabelSpec]) -> list[dict]:
    return [label_learnability(train_rows, spec) for spec in specs]


def feature_importance_for_label(
    bundle: RankingModelBundle,
    train_rows: list[dict],
    spec: LabelSpec,
    sample_n: int = 800,
) -> list[dict]:
    labeled = apply_label(train_rows[:sample_n], spec)
    gain = gain_importance(bundle)
    perm = permutation_importance_rows(bundle, labeled)
    shap = shap_rows(bundle, labeled[: min(300, len(labeled))])
    merged = merge_importance(gain, perm, shap)
    for r in merged:
        r["label_id"] = spec.label_id
    return merged


def importance_shift(
    baseline_imp: list[dict],
    candidate_imp: list[dict],
    label_id: str,
    top_n: int = 20,
) -> list[dict]:
    base_map = {r["feature"]: float(r.get("combined_score", 0)) for r in baseline_imp}
    cand_map = {r["feature"]: float(r.get("combined_score", 0)) for r in candidate_imp}
    features = set(base_map) | set(cand_map)
    rows: list[dict] = []
    for f in features:
        b = base_map.get(f, 0.0)
        c = cand_map.get(f, 0.0)
        rows.append({
            "label_id": label_id,
            "feature": f,
            "baseline_importance": round(b, 6),
            "label_importance": round(c, 6),
            "delta": round(c - b, 6),
        })
    rows.sort(key=lambda x: -abs(x["delta"]))
    return rows[:top_n]


def statistical_vs_baseline(baseline_avg: float, candidate_avg: float, n: int) -> dict:
    diff = candidate_avg - baseline_avg
    se = abs(baseline_avg) * 0.15 / max(n ** 0.5, 1)
    z = diff / se if se > 1e-9 else 0.0
    return {
        "lift_pct": round(diff / abs(baseline_avg or 0.01) * 100, 2),
        "diff": round(diff, 4),
        "approx_z": round(z, 4),
        "significant_hypothesis": abs(z) > 1.96 and diff > 0,
    }
