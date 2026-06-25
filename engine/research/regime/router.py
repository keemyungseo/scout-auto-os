"""Regime × Engine performance matrix and champion router (research only)."""

from __future__ import annotations

import statistics
from collections import defaultdict

from scout_auto_os.engine.research.zero_base.ranking import aggregate_candidate_metrics

ROUTER_ENGINES = (
    "MOMENTUM",
    "BREAKOUT",
    "FORMULA_LEAGUE",
    "FEATURE_LEAGUE",
    "STATE_LEAGUE",
    "REVERSAL",
    "COMPRESSION",
)

SKIP_REGIMES = frozenset({"Bear"})

MIN_REGIME_SAMPLES = 25


def router_score(agg: dict) -> float:
    return (
        float(agg.get("avg_return_2h") or 0)
        + float(agg.get("win_rate") or 0) * 0.1
        + float(agg.get("profit_factor") or 0)
        - float(agg.get("trap_rate") or 0) * 0.5
        - abs(float(agg.get("max_drawdown_avg") or 0)) * 0.3
    )


def build_regime_engine_matrix(
    samples: list[dict],
) -> list[dict]:
    """samples: each has regime, engine, return_2h, label_trap, etc."""
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for s in samples:
        buckets[(s.get("regime", "Unknown"), s.get("engine", ""))].append(s)

    rows: list[dict] = []
    for (regime, engine), items in sorted(buckets.items()):
        if not engine:
            continue
        agg = aggregate_candidate_metrics(items)
        rows.append({
            "regime": regime,
            "engine": engine,
            "sample_count": agg.get("sample_count", 0),
            "avg_return_2h": agg.get("avg_return_2h", 0),
            "win_rate": agg.get("win_rate", 0),
            "profit_factor": agg.get("profit_factor", 0),
            "trap_rate": agg.get("trap_rate", 0),
            "max_drawdown_avg": agg.get("max_drawdown_avg", 0),
            "big_winner_capture_rate": agg.get("big_winner_capture_rate", 0),
            "router_score": round(router_score(agg), 2),
        })
    return rows


def select_regime_champions(matrix: list[dict]) -> list[dict]:
    by_regime: dict[str, list[dict]] = defaultdict(list)
    for row in matrix:
        by_regime[row["regime"]].append(row)

    champions: list[dict] = []
    for regime in sorted(by_regime):
        rows = by_regime[regime]
        if regime in SKIP_REGIMES:
            champions.append({
                "regime": regime,
                "champion_engine": "SKIP",
                "confidence": "high",
                "reason": "validation: bear regime — router policy SKIP (no new entry rules)",
                "sample_count": sum(r["sample_count"] for r in rows),
            })
            continue

        eligible = [r for r in rows if int(r.get("sample_count") or 0) >= MIN_REGIME_SAMPLES]
        if not eligible:
            champions.append({
                "regime": regime,
                "champion_engine": "UNKNOWN",
                "confidence": "insufficient_sample",
                "reason": f"n < {MIN_REGIME_SAMPLES}",
                "sample_count": sum(r["sample_count"] for r in rows),
            })
            continue

        best = max(eligible, key=lambda r: r.get("router_score", 0))
        runners = sorted(eligible, key=lambda r: r.get("router_score", 0), reverse=True)[1:3]
        champions.append({
            "regime": regime,
            "champion_engine": best["engine"],
            "avg_return_2h": best.get("avg_return_2h"),
            "win_rate": best.get("win_rate"),
            "profit_factor": best.get("profit_factor"),
            "trap_rate": best.get("trap_rate"),
            "max_drawdown_avg": best.get("max_drawdown_avg"),
            "router_score": best.get("router_score"),
            "sample_count": best.get("sample_count"),
            "runners_up": [r["engine"] for r in runners],
            "confidence": "medium" if int(best["sample_count"]) >= 80 else "hypothesis",
        })
    return champions


def detect_transitions(scan_regimes: list[tuple[str, str]]) -> list[dict]:
    """scan_regimes: sorted (scan_kst, regime) pairs."""
    transitions: dict[tuple[str, str], int] = defaultdict(int)
    for i in range(1, len(scan_regimes)):
        prev = scan_regimes[i - 1][1]
        cur = scan_regimes[i][1]
        if prev != cur:
            transitions[(prev, cur)] += 1
    out = [
        {"from_regime": a, "to_regime": b, "count": c}
        for (a, b), c in sorted(transitions.items(), key=lambda x: x[1], reverse=True)
    ]
    return out
