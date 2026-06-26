"""Blind validation — Top2 execution vs Top5 / entry-score Top2."""

from __future__ import annotations

import statistics

from scout_auto_os.engine.research.execution_research.constants import SUCCESS_RETURN_PCT


def _agg(returns: list[float], label: str, direction: str, split: str) -> dict:
    n = len(returns)
    wins = sum(1 for r in returns if r >= SUCCESS_RETURN_PCT)
    return {
        "strategy": label,
        "direction": direction,
        "split": split,
        "trade_count": n,
        "avg_return_2h": round(statistics.mean(returns), 4) if returns else 0.0,
        "median_return_2h": round(statistics.median(returns), 4) if returns else 0.0,
        "win_rate_pct": round(wins / n * 100, 2) if n else 0.0,
        "total_return_2h": round(sum(returns), 4),
    }


def compare_strategies(
    top5_returns: list[float],
    entry_top2_returns: list[float],
    exec_top2_returns: list[float],
    direction: str,
    split: str,
) -> list[dict]:
    rows = [
        _agg(top5_returns, "top5_all", direction, split),
        _agg(entry_top2_returns, "top2_entry_score", direction, split),
        _agg(exec_top2_returns, "top2_execution", direction, split),
    ]
    base5 = rows[0]["avg_return_2h"] or 0.01
    base2 = rows[1]["avg_return_2h"] or 0.01
    exec_avg = rows[2]["avg_return_2h"]
    rows[2]["lift_vs_top5_pct"] = round((exec_avg - base5) / abs(base5) * 100, 2)
    rows[2]["lift_vs_entry_top2_pct"] = round((exec_avg - base2) / abs(base2) * 100, 2)
    return rows


def tune_weights_on_train(
    train_rows: list[dict],
    weight_grid: list[dict[str, float]] | None = None,
) -> dict[str, float]:
    """Pick weights maximizing avg 2h return of execution top2 on train."""
    from scout_auto_os.engine.research.execution_research.observation import DEFAULT_WEIGHTS, execution_score

    if not train_rows:
        return DEFAULT_WEIGHTS

    grid = weight_grid or [
        DEFAULT_WEIGHTS,
        {**DEFAULT_WEIGHTS, "obs_return": 0.45, "false_penalty": -1.5},
        {**DEFAULT_WEIGHTS, "breakout": 0.25, "momentum": 0.20},
        {**DEFAULT_WEIGHTS, "obs_return": 0.25, "volume": 0.25, "vwap": 0.15},
    ]

    by_scan: dict[tuple[str, str], list[dict]] = {}
    for r in train_rows:
        key = (r["scan_time_kst"], r["direction"])
        by_scan.setdefault(key, []).append(r)

    best_w = DEFAULT_WEIGHTS
    best_avg = float("-inf")
    for w in grid:
        rets: list[float] = []
        for rows in by_scan.values():
            if len(rows) < 2:
                continue
            for row in rows:
                row["_ws"] = execution_score(
                    {k: row[k] for k in row if k.startswith("obs_") or k in (
                        "volume_surge", "volume_ratio_scan", "vwap_deviation_pct",
                        "atr_increase_ratio", "new_high_breakout", "prior_high_break",
                        "false_breakout_flag", "momentum_persist",
                    )},
                    w,
                )
            top2 = sorted(rows, key=lambda x: x["_ws"], reverse=True)[:2]
            rets.extend(float(x["return_2h"]) for x in top2)
        if rets and statistics.mean(rets) > best_avg:
            best_avg = statistics.mean(rets)
            best_w = w
    return best_w
