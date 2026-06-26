"""Baseline Top2 pickers for blind comparison."""

from __future__ import annotations

from scout_auto_os.engine.portfolio.diversification import diversify_select
from scout_auto_os.engine.research.execution_rule_discovery.constants import TOP2_SIZE


def pick_top2_entry_score(group: list[dict]) -> list[dict]:
    pool = [{**r, "entry_score": r["features"]["entry_score"]} for r in group]
    pool.sort(key=lambda x: x["entry_score"], reverse=True)
    return diversify_select(pool, TOP2_SIZE)


def pick_top2_execution_score(group: list[dict]) -> list[dict]:
    pool = [{**r, "entry_score": float(r["features"].get("execution_score", 0))} for r in group]
    pool.sort(key=lambda x: x["entry_score"], reverse=True)
    return diversify_select(pool, TOP2_SIZE)
