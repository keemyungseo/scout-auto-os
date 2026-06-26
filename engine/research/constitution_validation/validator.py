"""Frozen constitution train/eval — extended metrics."""

from __future__ import annotations

import statistics
from collections import defaultdict

from scout_auto_os.engine.research.ranking_engine.metrics import (
    aggregate_ranking_quality,
    equity_mdd,
    evaluate_strategy_on_blind,
    ranking_quality_per_scan,
    sharpe,
    sortino,
)
from scout_auto_os.engine.research.ranking_engine.models import RankingModelBundle, predict_scores, train_model
from scout_auto_os.engine.research.target_discovery.candidate_generator import generate_label_candidates
from scout_auto_os.engine.research.target_discovery.label_builder import apply_label


def _label_spec():
    return next(s for s in generate_label_candidates() if s.label_id == "return_minus_dd")


def train_frozen_constitution(
    train_rows: list[dict],
    feat_names: list[str],
    model_name: str = "catboost_ranker",
) -> RankingModelBundle:
    spec = _label_spec()
    labeled = apply_label(train_rows, spec)
    return train_model(model_name, labeled, feat_names)


def profit_factor(returns: list[float]) -> float:
    gains = sum(r for r in returns if r > 0)
    losses = abs(sum(r for r in returns if r < 0))
    if losses < 1e-9:
        return round(gains, 4) if gains > 0 else 0.0
    return round(gains / losses, 4)


def rank_correlation_per_scan(rows: list[dict], scores: dict[str, float]) -> float:
    syms = [r["symbol"] for r in rows]
    if len(syms) < 3:
        return 0.0
    rets = [float(r.get("return_2h", 0)) for r in rows]
    sc = [scores.get(s, 0.0) for s in syms]
    rm = statistics.mean(rets)
    sm = statistics.mean(sc)
    num = sum((r - rm) * (s - sm) for r, s in zip(rets, sc))
    den_r = sum((r - rm) ** 2 for r in rets) ** 0.5
    den_s = sum((s - sm) ** 2 for s in sc) ** 0.5
    if den_r < 1e-9 or den_s < 1e-9:
        return 0.0
    return round(num / (den_r * den_s), 4)


def calibration_rows(picks: list[dict], n_bins: int = 10) -> list[dict]:
    if not picks:
        return []
    sorted_p = sorted(picks, key=lambda p: -float(p.get("score", 0)))
    bin_size = max(1, len(sorted_p) // n_bins)
    out: list[dict] = []
    for b in range(n_bins):
        chunk = sorted_p[b * bin_size:(b + 1) * bin_size]
        if not chunk:
            continue
        scores = [float(p.get("score", 0)) for p in chunk]
        success = [1 if float(p.get("return_2h", 0)) >= 3.0 else 0 for p in chunk]
        out.append({
            "bin": b + 1,
            "count": len(chunk),
            "mean_score": round(statistics.mean(scores), 4),
            "success_rate": round(statistics.mean(success) * 100, 2),
            "avg_return_2h": round(statistics.mean([float(p.get("return_2h", 0)) for p in chunk]), 4),
        })
    return out


def evaluate_constitution(
    rows: list[dict],
    bundle: RankingModelBundle,
    split_name: str = "blind",
) -> tuple[list[dict], dict, list[dict]]:
    spec = _label_spec()
    labeled = apply_label(rows, spec)

    def score_fn(row, peers, b=bundle):
        return float(predict_scores(b, [row])[0])

    picks, metrics, quality_scans = evaluate_strategy_on_blind(labeled, score_fn)

    rets = [float(p.get("return_2h", 0)) for p in picks]
    metrics["profit_factor"] = profit_factor(rets)
    metrics["sortino"] = sortino(rets)
    metrics["mdd"] = equity_mdd(rets)
    metrics["split"] = split_name

    corr_vals: list[float] = []
    by_scan: dict[str, list[dict]] = defaultdict(list)
    for r in labeled:
        by_scan[r["scan_kst"]].append(r)
    for scan, chunk in by_scan.items():
        scores = {r["symbol"]: score_fn(r, chunk) for r in chunk}
        corr_vals.append(rank_correlation_per_scan(chunk, scores))
    metrics["rank_correlation"] = round(statistics.mean(corr_vals), 4) if corr_vals else 0.0

    cal = calibration_rows(picks)
    metrics["calibration_gap"] = round(
        statistics.mean(abs(c["mean_score"] / max(abs(c["mean_score"]), 1) - c["success_rate"] / 100)
                        for c in cal), 4,
    ) if cal else 0.0

    return picks, metrics, cal


def metrics_from_picks(picks: list[dict], split_name: str) -> dict:
    if not picks:
        return {"split": split_name, "trade_count": 0}

    rets = [float(p.get("return_2h", 0)) for p in picks]
    wins = sum(1 for r in rets if r >= 3.0)
    top2 = [float(p.get("return_2h", 0)) for p in picks if int(p.get("rank", 99)) <= 2]
    top5 = [float(p.get("return_2h", 0)) for p in picks if int(p.get("rank", 99)) <= 5]
    scans = len({p["scan_kst"] for p in picks})

    by_scan: dict[str, list[dict]] = defaultdict(list)
    for p in picks:
        by_scan[p["scan_kst"]].append(p)

    quality: list[dict] = []
    for scan, chunk in by_scan.items():
        scores = {p["symbol"]: float(p.get("score", 0)) for p in chunk}
        all_rows = chunk
        quality.append(ranking_quality_per_scan(all_rows, scores))
    rq = aggregate_ranking_quality(quality)

    return {
        "split": split_name,
        "trade_count": len(picks),
        "scan_count": scans,
        "avg_return_2h": round(statistics.mean(rets), 4),
        "top2_avg_return_2h": round(statistics.mean(top2), 4) if top2 else 0,
        "top5_avg_return_2h": round(statistics.mean(top5), 4) if top5 else 0,
        "win_rate": round(wins / len(rets) * 100, 2),
        "sharpe": sharpe(rets),
        "sortino": sortino(rets),
        "mdd": equity_mdd(rets),
        "profit_factor": profit_factor(rets),
        "rank_correlation": 0.0,
        **{f"rank_{k}": v for k, v in rq.items()},
    }
