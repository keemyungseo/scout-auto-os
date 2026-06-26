"""Feature shift vs Long + label ranking."""

from __future__ import annotations

from scout_auto_os.engine.research.ranking_engine.importance import (
    gain_importance,
    merge_importance,
    permutation_importance_rows,
    shap_rows,
)
from scout_auto_os.engine.research.ranking_engine.models import RankingModelBundle
from scout_auto_os.engine.research.short_constitution.label_builder import ShortLabelSpec, apply_short_label


def rank_short_labels(results: list[dict], baseline_id: str) -> list[dict]:
    rows = [r for r in results if not r.get("error")]
    ranked = sorted(
        rows,
        key=lambda x: (-float(x.get("avg_return_2h", 0)), -float(x.get("rank_ndcg5", 0))),
    )
    base_avg = next((float(r.get("avg_return_2h", 0)) for r in rows if r.get("label_id") == baseline_id), 0)
    out = []
    for i, r in enumerate(ranked, 1):
        avg = float(r.get("avg_return_2h", 0))
        out.append({
            "label_rank": i,
            "label_id": r.get("label_id"),
            "label_name": r.get("label_name"),
            "category": r.get("category"),
            "avg_return_2h": avg,
            "top2_avg_return_2h": r.get("top2_avg_return_2h"),
            "win_rate": r.get("win_rate"),
            "sharpe": r.get("sharpe"),
            "rank_ndcg5": r.get("rank_ndcg5"),
            "rank_p5": r.get("rank_p5"),
            "vs_baseline_pct": round((avg - base_avg) / abs(base_avg or 0.01) * 100, 2),
            "beats_baseline": avg > base_avg,
        })
    return out


def feature_importance(bundle: RankingModelBundle, rows: list[dict], spec: ShortLabelSpec) -> list[dict]:
    labeled = apply_short_label(rows[:800], spec)
    merged = merge_importance(
        gain_importance(bundle),
        permutation_importance_rows(bundle, labeled),
        shap_rows(bundle, labeled[:300]),
    )
    for r in merged:
        r["direction"] = "short"
    return merged


def feature_shift_vs_long(
    long_imp: list[dict],
    short_imp: list[dict],
    top_n: int = 25,
) -> tuple[list[dict], dict]:
    lmap = {r["feature"]: float(r.get("combined_score") or 0) for r in long_imp}
    smap = {r["feature"]: float(r.get("combined_score") or 0) for r in short_imp}
    features = set(lmap) | set(smap)

    long_top = {
        r["feature"]
        for r in sorted(long_imp, key=lambda x: -float(x.get("combined_score") or 0))[:15]
    }
    short_top = {
        r["feature"]
        for r in sorted(short_imp, key=lambda x: -float(x.get("combined_score") or 0))[:15]
    }
    shared = long_top & short_top
    long_only = long_top - short_top
    short_only = short_top - long_top

    rows = []
    for f in features:
        rows.append({
            "feature": f,
            "long_importance": round(lmap.get(f, 0), 6),
            "short_importance": round(smap.get(f, 0), 6),
            "delta_short_minus_long": round(smap.get(f, 0) - lmap.get(f, 0), 6),
            "in_long_top15": f in long_top,
            "in_short_top15": f in short_top,
        })
    rows.sort(key=lambda x: -abs(x["delta_short_minus_long"]))

    summary = {
        "top15_overlap": len(shared),
        "top15_overlap_pct": round(len(shared) / 15 * 100, 2),
        "long_only_top15": sorted(long_only)[:10],
        "short_only_top15": sorted(short_only)[:10],
        "independent_structure": len(shared) < 8,
    }
    return rows[:top_n], summary


def load_long_importance(data_dir) -> list[dict]:
    import csv
    from pathlib import Path

    p = Path(data_dir) / "constitution_validation" / "feature_importance.csv"
    if not p.exists():
        p = Path(data_dir).parent / "research_bundle" / "reports" / "constitution_feature_importance_v1.csv"
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))
