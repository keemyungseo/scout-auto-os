"""Scan-time cluster probability and expected return scoring."""

from __future__ import annotations

import math
import statistics

from scout_auto_os.engine.research.directional.dna.formulas import ClusterFormula

DEFAULT_HOLDING_MIN = 120.0
SCORE_BASE = 35.0
SCORE_RETURN_WEIGHT = 7.0
SCORE_PROB_WEIGHT = 22.0


def _softmax(scores: dict[str, float], temperature: float) -> dict[str, float]:
    if not scores:
        return {}
    tau = max(temperature, 0.25)
    max_s = max(scores.values())
    exps = {k: math.exp((v - max_s) / tau) for k, v in scores.items()}
    total = sum(exps.values()) or 1.0
    return {k: round(v / total * 100.0, 2) for k, v in exps.items()}


def _temperature(scores: dict[str, float]) -> float:
    vals = list(scores.values())
    if len(vals) < 2:
        return 1.0
    return max(0.5, statistics.pstdev(vals))


def _direction_score(expected_return: float, top_prob_pct: float) -> float:
    raw = SCORE_BASE + expected_return * SCORE_RETURN_WEIGHT + (top_prob_pct / 100.0) * SCORE_PROB_WEIGHT
    return round(min(100.0, max(0.0, raw)), 1)


def predict_symbol(
    features: dict,
    long_formulas: list[ClusterFormula],
    short_formulas: list[ClusterFormula],
    expected_returns: dict[str, dict],
) -> dict:
    long_raw = {f.name: f.score(features) for f in long_formulas}
    short_raw = {f.name: f.score(features) for f in short_formulas}

    long_probs = _softmax(long_raw, _temperature(long_raw))
    short_probs = _softmax(short_raw, _temperature(short_raw))

    def _weighted_exp(probs: dict[str, float]) -> float:
        total_p = sum(probs.values()) or 1.0
        acc = 0.0
        for name, p in probs.items():
            er = float(expected_returns.get(name, {}).get("avg_return_2h", 0))
            acc += (p / total_p) * er
        return round(acc, 4)

    long_exp = _weighted_exp(long_probs)
    short_exp = _weighted_exp(short_probs)

    top_long = max(long_probs, key=long_probs.get) if long_probs else ""
    top_short = max(short_probs, key=short_probs.get) if short_probs else ""
    top_long_p = long_probs.get(top_long, 0.0)
    top_short_p = short_probs.get(top_short, 0.0)

    long_score = _direction_score(long_exp, top_long_p)
    short_score = _direction_score(short_exp, top_short_p)

    if long_score >= short_score:
        recommended = "LONG"
        recommended_cluster = top_long
    else:
        recommended = "SHORT"
        recommended_cluster = top_short

    denom = long_score + short_score + 1e-6
    confidence = round(100.0 * max(long_score, short_score) / denom, 1)

    contributions_long = [
        {
            "cluster": name,
            "probability_pct": p,
            "expected_return_2h": float(expected_returns.get(name, {}).get("avg_return_2h", 0)),
            "contribution": round(p / 100.0 * float(expected_returns.get(name, {}).get("avg_return_2h", 0)), 4),
        }
        for name, p in sorted(long_probs.items(), key=lambda x: -x[1])
    ]
    contributions_short = [
        {
            "cluster": name,
            "probability_pct": p,
            "expected_return_2h": float(expected_returns.get(name, {}).get("avg_return_2h", 0)),
            "contribution": round(p / 100.0 * float(expected_returns.get(name, {}).get("avg_return_2h", 0)), 4),
        }
        for name, p in sorted(short_probs.items(), key=lambda x: -x[1])
    ]

    return {
        "long_probability_pct": top_long_p,
        "short_probability_pct": top_short_p,
        "long_expected_return_2h": long_exp,
        "short_expected_return_2h": short_exp,
        "long_score": long_score,
        "short_score": short_score,
        "recommended_direction": recommended,
        "recommended_cluster": recommended_cluster,
        "confidence_pct": confidence,
        "expected_holding_min": DEFAULT_HOLDING_MIN,
        "top_long_cluster": top_long,
        "top_short_cluster": top_short,
        "long_cluster_probs": long_probs,
        "short_cluster_probs": short_probs,
        "contributions_long": contributions_long,
        "contributions_short": contributions_short,
    }
