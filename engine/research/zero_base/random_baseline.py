"""Random TOP5 baseline — repeated draws for mean/variance."""

from __future__ import annotations

import random
import statistics
from typing import Callable

from scout_auto_os.engine.research.zero_base.candidates import rank_engine


def generate_random_draws(
    symbols: list[str],
    top_k: int,
    n_draws: int,
    seed: int = 42,
) -> list[list[str]]:
    rng = random.Random(seed)
    if len(symbols) <= top_k:
        return [symbols[:top_k] for _ in range(n_draws)]
    return [rng.sample(symbols, top_k) for _ in range(n_draws)]


def aggregate_random_metrics(
    draw_metrics: list[list[dict]],
) -> dict:
    if not draw_metrics:
        return {"draw_count": 0}
    flat = [m for draw in draw_metrics for m in draw]
    if not flat:
        return {"draw_count": len(draw_metrics)}
    r2h = [float(m.get("return_2h", 0)) for m in flat]
    traps = sum(1 for m in flat if m.get("label_trap"))
    big = sum(1 for m in flat if m.get("label_big_winner"))
    mdd = [float(m.get("max_drawdown_2h", 0)) for m in flat]
    return {
        "draw_count": len(draw_metrics),
        "sample_count": len(flat),
        "avg_return_2h": round(statistics.mean(r2h), 4),
        "return_2h_std": round(statistics.pstdev(r2h), 4) if len(r2h) > 1 else 0,
        "median_return_2h": round(statistics.median(r2h), 4),
        "win_rate": round(sum(1 for x in r2h if x >= 3) / len(r2h) * 100, 2),
        "trap_rate": round(traps / len(flat) * 100, 2),
        "big_winner_capture_rate": round(big / len(flat) * 100, 2),
        "max_drawdown_avg": round(statistics.mean(mdd), 4),
    }


def random_baseline_for_scan(
    symbols: list[str],
    top_k: int,
    n_draws: int,
    metric_fn: Callable[[str], dict | None],
    seed: int = 42,
) -> tuple[dict, list[dict]]:
    """Returns (aggregate stats, per-draw summary rows)."""
    draws = generate_random_draws(symbols, top_k, n_draws, seed)
    draw_metrics: list[list[dict]] = []
    draw_rows: list[dict] = []
    for i, pick in enumerate(draws):
        metrics = []
        for sym in pick:
            m = metric_fn(sym)
            if m:
                metrics.append(m)
        draw_metrics.append(metrics)
        if metrics:
            r2h = [float(m["return_2h"]) for m in metrics]
            draw_rows.append({
                "draw_id": i,
                "symbols": "|".join(pick),
                "avg_return_2h": round(sum(r2h) / len(r2h), 4),
                "sample_count": len(metrics),
            })
    return aggregate_random_metrics(draw_metrics), draw_rows
