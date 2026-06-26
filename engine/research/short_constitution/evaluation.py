"""Short label evaluation + ranking engine."""

from __future__ import annotations

from collections import defaultdict

from scout_auto_os.engine.research.ranking_engine.features import feature_matrix
from scout_auto_os.engine.research.ranking_engine.metrics import (
    aggregate_ranking_quality,
    equity_mdd,
    evaluate_strategy_on_blind,
    ranking_quality_per_scan,
    sharpe,
    sortino,
)
from scout_auto_os.engine.research.ranking_engine.models import RankingModelBundle, predict_scores, train_model
from scout_auto_os.engine.research.ranking_engine.validation import split_validation_rows, walk_forward_validation
from scout_auto_os.engine.research.short_constitution.constants import MODEL
from scout_auto_os.engine.research.short_constitution.label_builder import ShortLabelSpec, apply_short_label


def _eval_rows(blind_rows: list[dict]) -> list[dict]:
    return [
        {
            **r,
            "outcome_rank": r["baseline_outcome_rank"],
            "relevance": r["baseline_relevance"],
            "return_2h": r["short_return_2h"],
        }
        for r in blind_rows
    ]


def profit_factor(returns: list[float]) -> float:
    gains = sum(r for r in returns if r > 0)
    losses = abs(sum(r for r in returns if r < 0))
    if losses < 1e-9:
        return round(gains, 4) if gains > 0 else 0.0
    return round(gains / losses, 4)


def extended_metrics(picks: list[dict], split_name: str) -> dict:
    if not picks:
        return {"split": split_name, "trade_count": 0}
    rets = [float(p.get("short_return_2h", 0)) for p in picks]
    top2 = [float(p.get("short_return_2h", 0)) for p in picks if int(p.get("rank", 99)) <= 2]
    top5 = [float(p.get("short_return_2h", 0)) for p in picks if int(p.get("rank", 99)) <= 5]
    wins = sum(1 for r in rets if r >= 3.0)
    scans = len({p["scan_kst"] for p in picks})

    by_scan: dict[str, list[dict]] = defaultdict(list)
    for p in picks:
        by_scan[p["scan_kst"]].append(p)
    quality = []
    for scan, chunk in by_scan.items():
        scores = {p["symbol"]: float(p.get("score", 0)) for p in chunk}
        quality.append(ranking_quality_per_scan(chunk, scores))
    rq = aggregate_ranking_quality(quality)

    return {
        "split": split_name,
        "trade_count": len(picks),
        "scan_count": scans,
        "avg_return_2h": round(sum(rets) / len(rets), 4),
        "top2_avg_return_2h": round(sum(top2) / len(top2), 4) if top2 else 0,
        "top5_avg_return_2h": round(sum(top5) / len(top5), 4) if top5 else 0,
        "win_rate": round(wins / len(rets) * 100, 2),
        "sharpe": sharpe(rets),
        "sortino": sortino(rets),
        "mdd": equity_mdd(rets),
        "profit_factor": profit_factor(rets),
        **{f"rank_{k}": v for k, v in rq.items()},
    }


def train_short_ranker(
    train_rows: list[dict],
    feat_names: list[str],
    spec: ShortLabelSpec,
) -> RankingModelBundle:
    labeled = apply_short_label(train_rows, spec)
    return train_model(MODEL, labeled, feat_names)


def evaluate_short_label(
    train_rows: list[dict],
    blind_rows: list[dict],
    feat_names: list[str],
    spec: ShortLabelSpec,
) -> tuple[dict, list[dict], RankingModelBundle | None]:
    labeled_train = apply_short_label(train_rows, spec)
    labeled_blind = apply_short_label(blind_rows, spec)
    try:
        bundle = train_model(MODEL, labeled_train, feat_names)
    except Exception as exc:
        return {"label_id": spec.label_id, "error": str(exc)}, [], None

    def score_fn(row, peers, b=bundle):
        return float(predict_scores(b, [row])[0])

    eval_blind = _eval_rows(labeled_blind)
    picks, base_metrics, _ = evaluate_strategy_on_blind(eval_blind, score_fn)
    metrics = extended_metrics(picks, "blind")
    metrics["label_id"] = spec.label_id
    metrics["label_name"] = spec.name
    metrics["category"] = spec.category

    gen: list[dict] = [{**metrics, "validation_type": "blind_holdout"}]
    for val in split_validation_rows(eval_blind, feat_names, MODEL, bundle):
        gen.append({"label_id": spec.label_id, **val})
    for val in walk_forward_validation(labeled_train + labeled_blind, feat_names, MODEL):
        gen.append({"label_id": spec.label_id, **val})

    return metrics, gen, bundle


def leak_check_rows(rows: list[dict]) -> dict:
    forbidden = 0
    for r in rows:
        for k in r.get("x") or {}:
            if k.startswith(("return_2h", "label_", "outcome_", "short_return")):
                forbidden += 1
    return {"rows_checked": len(rows), "forbidden_keys": forbidden, "passed": forbidden == 0}
