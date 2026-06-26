"""Walk-forward and split validation."""

from __future__ import annotations

import statistics
from collections import defaultdict

from scout_auto_os.engine.research.directional.dna.validator import split_scans
from scout_auto_os.engine.research.ranking_engine.metrics import evaluate_strategy_on_blind
from scout_auto_os.engine.research.ranking_engine.models import RankingModelBundle, predict_scores, train_model


def _month_key(s: str) -> str:
    return s[:7]


def _week_key(s: str) -> str:
    return s[:10]


def walk_forward_validation(
    rows: list[dict],
    feat_names: list[str],
    model_name: str,
    train_ratio: float = 0.7,
) -> list[dict]:
    scans = sorted({r["scan_kst"] for r in rows})
    train_scans, _ = split_scans(scans, train_ratio)
    train_cut = train_scans[-1]
    holdout = [r for r in rows if r["scan_kst"] > train_cut]
    if not holdout:
        return []

    train_all = [r for r in rows if r["scan_kst"] <= train_cut]
    bundle = train_model(model_name, train_all, feat_names)

    def score_fn(row, peers):
        sc = predict_scores(bundle, [row])[0]
        return float(sc)

    _, metrics, _ = evaluate_strategy_on_blind(holdout, score_fn)
    return [{"split": "walk_forward", "model": model_name, **metrics}]


def split_validation_rows(
    rows: list[dict],
    feat_names: list[str],
    model_name: str,
    bundle: RankingModelBundle,
) -> list[dict]:
    out: list[dict] = []
    scans = sorted({r["scan_kst"] for r in rows})

    def score_fn(row, peers):
        return float(predict_scores(bundle, [row])[0])

    for split_type, grouper in (("monthly", _month_key), ("weekly", _week_key)):
        buckets: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            buckets[grouper(r["scan_kst"])].append(r)
        for key, chunk in sorted(buckets.items()):
            if len(chunk) < 20:
                continue
            _, metrics, _ = evaluate_strategy_on_blind(chunk, score_fn)
            out.append({"split_type": split_type, "fold_id": key, "model": model_name, **metrics})
    return out
