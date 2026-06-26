"""Ranking quality and trading metrics."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict

from scout_auto_os.engine.research.ranking_engine.constants import SUCCESS_RETURN_PCT, TOP_K


def _dcg(rels: list[float], k: int) -> float:
    rels = rels[:k]
    return sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(rels))


def ndcg_at_k(pred_ranks: list[int], relevances: list[int], k: int = 5) -> float:
    if not relevances:
        return 0.0
    ideal = sorted(relevances, reverse=True)
    idcg = _dcg([float(x) for x in ideal], k)
    if idcg <= 0:
        return 0.0
    ordered = [relevances[i] for i in pred_ranks[:k] if i < len(relevances)]
    return _dcg([float(x) for x in ordered], k) / idcg


def average_precision(relevant: set[int], ranked: list[int]) -> float:
    if not relevant:
        return 0.0
    hits = 0
    prec_sum = 0.0
    for i, idx in enumerate(ranked, 1):
        if idx in relevant:
            hits += 1
            prec_sum += hits / i
    return prec_sum / len(relevant) if relevant else 0.0


def precision_at_k(ranked_symbols: list[str], winners: set[str], k: int) -> float:
    picks = ranked_symbols[:k]
    if not picks:
        return 0.0
    return sum(1 for s in picks if s in winners) / k


def recall_at_k(ranked_symbols: list[str], winners: set[str], k: int) -> float:
    if not winners:
        return 0.0
    picks = set(ranked_symbols[:k])
    return len(picks & winners) / len(winners)


def mrr(ranked_symbols: list[str], winners: set[str]) -> float:
    for i, s in enumerate(ranked_symbols, 1):
        if s in winners:
            return 1.0 / i
    return 0.0


def ranking_quality_per_scan(
    rows: list[dict],
    scores: dict[str, float],
    top_n: int = 3,
) -> dict:
    ranked = sorted(rows, key=lambda r: -scores.get(r["symbol"], 0))
    ranked_syms = [r["symbol"] for r in ranked]
    winners = {r["symbol"] for r in rows if int(r.get("outcome_rank", 99)) <= top_n}
    rels = [int(r.get("relevance", 0)) for r in ranked]
    rank_idx = list(range(len(ranked)))
    return {
        "ndcg5": ndcg_at_k(rank_idx, rels, 5),
        "map": average_precision(
            {i for i, r in enumerate(rows) if r["symbol"] in winners},
            rank_idx,
        ),
        "p1": precision_at_k(ranked_syms, winners, 1),
        "p2": precision_at_k(ranked_syms, winners, 2),
        "p5": precision_at_k(ranked_syms, winners, 5),
        "r5": recall_at_k(ranked_syms, winners, 5),
        "mrr": mrr(ranked_syms, winners),
        "hit_top3": 1.0 if any(s in winners for s in ranked_syms[:TOP_K]) else 0.0,
    }


def aggregate_ranking_quality(scan_metrics: list[dict]) -> dict:
    if not scan_metrics:
        return {}
    keys = scan_metrics[0].keys()
    return {k: round(statistics.mean(float(m[k]) for m in scan_metrics), 4) for k in keys}


def sharpe(returns: list[float]) -> float:
    if len(returns) < 2:
        return 0.0
    mu = statistics.mean(returns)
    sd = statistics.pstdev(returns)
    return round(mu / sd * math.sqrt(len(returns)), 4) if sd > 1e-9 else 0.0


def sortino(returns: list[float]) -> float:
    if len(returns) < 2:
        return 0.0
    mu = statistics.mean(returns)
    downs = [min(r, 0) for r in returns]
    dd = statistics.pstdev(downs)
    return round(mu / dd * math.sqrt(len(returns)), 4) if dd > 1e-9 else 0.0


def equity_mdd(returns: list[float]) -> float:
    cum = 0.0
    peak = 0.0
    mdd = 0.0
    for r in returns:
        cum += r
        peak = max(peak, cum)
        mdd = min(mdd, cum - peak)
    return round(mdd, 4)


def trading_metrics_from_picks(picks: list[dict]) -> dict:
    if not picks:
        return {"trade_count": 0}
    rets = [float(p.get("return_2h", 0)) for p in picks]
    wins = sum(1 for r in rets if r >= SUCCESS_RETURN_PCT)
    scans = len({p["scan_kst"] for p in picks})
    top2_rets = [float(p.get("return_2h", 0)) for p in picks if int(p.get("rank", 99)) <= 2]
    top5_rets = [float(p.get("return_2h", 0)) for p in picks if int(p.get("rank", 99)) <= 5]
    return {
        "trade_count": len(picks),
        "scan_count": scans,
        "avg_return_2h": round(statistics.mean(rets), 4),
        "top2_avg_return_2h": round(statistics.mean(top2_rets), 4) if top2_rets else 0,
        "top5_avg_return_2h": round(statistics.mean(top5_rets), 4) if top5_rets else 0,
        "win_rate": round(wins / len(rets) * 100, 2),
        "precision_top3": round(
            sum(1 for p in picks if int(p.get("outcome_rank", 99)) <= 3) / len(picks) * 100, 2,
        ),
        "recall_top3_proxy": round(wins / len(rets) * 100, 2),
        "sharpe": sharpe(rets),
        "sortino": sortino(rets),
        "mdd": equity_mdd(rets),
        "coverage_pct": round(scans / max(scans, 1) * 100, 2),
        "return_per_trade": round(sum(rets) / len(rets), 4),
        "return_per_scan": round(sum(rets) / max(scans, 1), 4),
    }


def evaluate_strategy_on_blind(
    blind_rows: list[dict],
    score_fn,
    top_k: int = 5,
) -> tuple[list[dict], dict, list[dict]]:
    by_scan: dict[str, list[dict]] = defaultdict(list)
    for r in blind_rows:
        by_scan[r["scan_kst"]].append(r)

    picks: list[dict] = []
    quality_scans: list[dict] = []
    for scan, rows in sorted(by_scan.items()):
        scores = {r["symbol"]: score_fn(r, rows) for r in rows}
        ranked = sorted(rows, key=lambda r: -scores[r["symbol"]])
        quality_scans.append(ranking_quality_per_scan(rows, scores))
        for rank, r in enumerate(ranked[:top_k], 1):
            picks.append({**r, "rank": rank, "score": scores[r["symbol"]]})

    trading = trading_metrics_from_picks(picks)
    ranking_q = aggregate_ranking_quality(quality_scans)
    trading.update({f"rank_{k}": v for k, v in ranking_q.items()})
    return picks, trading, quality_scans
