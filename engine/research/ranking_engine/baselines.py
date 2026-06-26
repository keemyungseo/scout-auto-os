"""Baseline scorers for comparison — frozen stacks."""

from __future__ import annotations

from scout_auto_os.engine.research.ranking_engine.constants import BASELINE_NAMES


def score_a6(row: dict, peers: list[dict]) -> float:
    return float(row["x"].get("a6_formula_score", row.get("base_score", 0)))


def score_entry(row: dict, _peers: list[dict]) -> float:
    return float(row["x"].get("entry_score", 0))


def score_formula_v2(row: dict, peers: list[dict]) -> float:
    base = float(row.get("base_score", 0))
    rank_bonus = float(row["x"].get("formula_v2_rank_pct", 0))
    passed = 1.0 if rank_bonus >= 0.9 else 0.0
    return base + passed * 50.0 + rank_bonus * 10.0


def score_execution_proxy(row: dict, _peers: list[dict]) -> float:
  # observation-bar weighted proxy from execution research
    obs_keys = (
        "exec_obs_obs_return_pct",
        "exec_obs_volume_surge",
        "exec_obs_momentum_persist",
        "exec_obs_new_high_breakout",
    )
    return sum(float(row["x"].get(k, 0)) for k in obs_keys) + float(row["x"].get("entry_score", 0)) * 0.1


BASELINE_SCORERS = {
    "current_search_a6": score_a6,
    "entry_score_top5": score_entry,
    "formula_league_v2": score_formula_v2,
    "execution_score_proxy": score_execution_proxy,
}
