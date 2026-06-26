"""Winner vs Loser feature DNA — existing features only."""

from __future__ import annotations

import math
import statistics

from scout_auto_os.engine.research.directional.dna.analyzer import welch_p_approx
from scout_auto_os.engine.research.directional.dna.collector import numeric_feature_keys
from scout_auto_os.engine.research.directional.entry_filter.constants import (
    LOSER_QUANTILE,
    WINNER_QUANTILE,
)


def _mean(vals: list[float]) -> float:
    return statistics.mean(vals) if vals else 0.0


def _std(vals: list[float]) -> float:
    return statistics.pstdev(vals) if len(vals) > 1 else 0.0


def split_winner_loser(
    signals: list[dict],
    rank_key: str = "return_2h",
    winner_q: float = WINNER_QUANTILE,
    loser_q: float = LOSER_QUANTILE,
) -> tuple[list[dict], list[dict], dict]:
    """Top/bottom quantile by rank_key (higher = better for long and short)."""
    if not signals:
        return [], [], {"winner_count": 0, "loser_count": 0, "middle_count": 0}

    ranked = sorted(signals, key=lambda s: float(s.get(rank_key, 0)), reverse=True)
    n = len(ranked)
    w_n = max(1, int(n * winner_q))
    l_n = max(1, int(n * loser_q))

    winners = ranked[:w_n]
    losers = ranked[-l_n:]
    middle = ranked[w_n:-l_n] if w_n + l_n < n else []

    values = [float(s[rank_key]) for s in ranked]
    return winners, losers, {
        "total_signals": n,
        "winner_count": len(winners),
        "loser_count": len(losers),
        "middle_count": len(middle),
        "winner_threshold": round(float(winners[-1][rank_key]), 4) if winners else None,
        "loser_threshold": round(float(losers[0][rank_key]), 4) if losers else None,
        "median_return_2h": round(statistics.median(values), 4),
        "mean_return_2h": round(_mean(values), 4),
    }


def compare_winner_loser_features(
    winners: list[dict],
    losers: list[dict],
    feature_keys: list[str],
    direction: str,
) -> list[dict]:
    if not winners or not losers:
        return []

    rows: list[dict] = []
    for key in feature_keys:
        wv = [float(s["features"].get(key, 0)) for s in winners]
        lv = [float(s["features"].get(key, 0)) for s in losers]
        w_mean = _mean(wv)
        l_mean = _mean(lv)
        delta = w_mean - l_mean
        pooled = math.sqrt((_std(wv) ** 2 + _std(lv) ** 2) / 2) or 1e-9
        effect = delta / pooled
        p = welch_p_approx(wv, lv)
        rows.append({
            "direction": direction,
            "feature": key,
            "winner_mean": round(w_mean, 4),
            "loser_mean": round(l_mean, 4),
            "delta": round(delta, 4),
            "effect_size": round(effect, 4),
            "p_approx": p,
            "significant": p < 0.05,
            "winner_favors_higher": delta > 0,
        })

    rows.sort(key=lambda x: abs(x["effect_size"]), reverse=True)
    for i, row in enumerate(rows, 1):
        row["importance_rank"] = i
    return rows


def summarize_dna_profile(importance: list[dict], top_n: int = 15) -> dict:
    top = importance[:top_n]
    sig = [r for r in importance if r.get("significant")]
    higher = [r["feature"] for r in top if r.get("winner_favors_higher")]
    lower = [r["feature"] for r in top if not r.get("winner_favors_higher")]
    return {
        "top_features": [r["feature"] for r in top],
        "significant_count": len(sig),
        "winner_higher_features": higher,
        "winner_lower_features": lower,
        "top_effect_sizes": {r["feature"]: r["effect_size"] for r in top[:5]},
    }


def find_common_dna(
    long_importance: list[dict],
    short_importance: list[dict],
    top_n: int = 20,
) -> list[dict]:
    """Features important in both directions — same winner/loser ordering."""
    long_top = {r["feature"]: r for r in long_importance[:top_n]}
    short_top = {r["feature"]: r for r in short_importance[:top_n]}
    common_keys = set(long_top) & set(short_top)

    rows: list[dict] = []
    for feat in common_keys:
        lr = long_top[feat]
        sr = short_top[feat]
        same_sign = (lr["delta"] > 0) == (sr["delta"] > 0)
        rows.append({
            "feature": feat,
            "long_effect_size": lr["effect_size"],
            "short_effect_size": sr["effect_size"],
            "long_delta": lr["delta"],
            "short_delta": sr["delta"],
            "long_significant": lr.get("significant"),
            "short_significant": sr.get("significant"),
            "same_winner_direction": same_sign,
            "combined_effect": round((abs(lr["effect_size"]) + abs(sr["effect_size"])) / 2, 4),
        })

    rows.sort(key=lambda x: x["combined_effect"], reverse=True)
    for i, row in enumerate(rows, 1):
        row["common_rank"] = i
    return rows
