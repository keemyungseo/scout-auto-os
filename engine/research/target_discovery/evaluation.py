"""Train and evaluate CatBoost ranker per label candidate."""

from __future__ import annotations

from collections import defaultdict

from scout_auto_os.engine.research.ranking_engine.features import feature_matrix
from scout_auto_os.engine.research.ranking_engine.metrics import evaluate_strategy_on_blind
from scout_auto_os.engine.research.ranking_engine.models import RankingModelBundle, predict_scores, train_model
from scout_auto_os.engine.research.ranking_engine.validation import split_validation_rows, walk_forward_validation
from scout_auto_os.engine.research.target_discovery.constants import BASELINE_MODEL
from scout_auto_os.engine.research.target_discovery.label_builder import LabelSpec, apply_label


def _eval_rows(blind_rows: list[dict]) -> list[dict]:
    """Use baseline ground truth for ranking-quality metrics."""
    return [
        {
            **r,
            "outcome_rank": r["baseline_outcome_rank"],
            "relevance": r["baseline_relevance"],
        }
        for r in blind_rows
    ]


def evaluate_label_candidate(
    train_rows: list[dict],
    blind_rows: list[dict],
    feat_names: list[str],
    spec: LabelSpec,
    model_name: str = BASELINE_MODEL,
) -> tuple[dict, list[dict], RankingModelBundle | None]:
    labeled_train = apply_label(train_rows, spec)
    labeled_blind = apply_label(blind_rows, spec)

    try:
        bundle = train_model(model_name, labeled_train, feat_names)
    except Exception as exc:
        return {"label_id": spec.label_id, "error": str(exc)}, [], None

    def score_fn(row, peers, b=bundle):
        return float(predict_scores(b, [row])[0])

    _, blind_metrics, _ = evaluate_strategy_on_blind(_eval_rows(labeled_blind), score_fn)
    row = {
        "label_id": spec.label_id,
        "label_name": spec.name,
        "category": spec.category,
        "rank_key": spec.rank_key,
        "model": model_name,
        "split": "blind",
        **blind_metrics,
    }

    gen_rows: list[dict] = [{**row, "split_type": "blind_holdout"}]
    for val in split_validation_rows(_eval_rows(labeled_blind), feat_names, model_name, bundle):
        gen_rows.append({
            "label_id": spec.label_id,
            "label_name": spec.name,
            "category": spec.category,
            **val,
        })

    wf = walk_forward_validation(labeled_train + labeled_blind, feat_names, model_name)
    for val in wf:
        gen_rows.append({
            "label_id": spec.label_id,
            "label_name": spec.name,
            "category": spec.category,
            **val,
        })

    return row, gen_rows, bundle


def regime_breakdown(
    blind_rows: list[dict],
    bundle: RankingModelBundle,
    spec: LabelSpec,
) -> list[dict]:
    from scout_auto_os.engine.research.zero_base.validation import classify_regime

    labeled = apply_label(blind_rows, spec)

    def score_fn(row, peers, b=bundle):
        return float(predict_scores(b, [row])[0])

    by_scan: dict[str, list[dict]] = defaultdict(list)
    for r in labeled:
        by_scan[r["scan_kst"]].append(r)

    out: list[dict] = []
    for scan, rows in sorted(by_scan.items()):
        regime = classify_regime([
            {"features": {k.replace("dna_", ""): v for k, v in (rows[0].get("x") or {}).items() if k.startswith("dna_")}},
        ])
        _, metrics, _ = evaluate_strategy_on_blind(_eval_rows(rows), score_fn)
        out.append({"label_id": spec.label_id, "regime": regime, "scan": scan, **metrics})
    return out
