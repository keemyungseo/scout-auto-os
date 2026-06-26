"""Blind forward validation — compare champion baselines vs prediction engine."""

from __future__ import annotations

from collections.abc import Callable

from scout_auto_os.engine.research.directional.dna.validator import evaluate_picks_on_scans
from scout_auto_os.engine.research.directional.evaluation import aggregate_directional


def evaluate_method(
    label: str,
    direction: str,
    pick_fn: Callable[[list[dict], int], list[str]],
    by_scan: dict,
    fwd: dict,
    scans: list[str],
    top_k: int = 5,
) -> dict:
    def wrapped(rows: list[dict]) -> list[str]:
        return pick_fn(rows, top_k)

    samples = evaluate_picks_on_scans(by_scan, fwd, scans, wrapped, direction)
    agg = aggregate_directional(samples, direction)
    return {
        "method": label,
        "direction": direction,
        "top_k": top_k,
        "sample_count": agg.get("sample_count", 0),
        "avg_return_2h": agg.get("avg_return_2h", 0),
        "win_rate": agg.get("win_rate", 0),
        "profit_factor": agg.get("profit_factor", 0),
        "trap_rate": agg.get("trap_rate", 0),
        "max_drawdown_avg": agg.get("max_drawdown_avg", 0),
        "score": agg.get("score", 0),
    }


def run_comparison(
    methods: list[tuple[str, str, Callable]],
    by_scan: dict,
    fwd: dict,
    scans: list[str],
    top_k: int = 5,
) -> list[dict]:
    rows: list[dict] = []
    for label, direction, pick_fn in methods:
        rows.append(evaluate_method(label, direction, pick_fn, by_scan, fwd, scans, top_k))
    return rows
