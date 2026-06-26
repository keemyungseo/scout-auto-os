"""Diversification — limit correlated picks in same movement group."""

from __future__ import annotations


def movement_group(candidate: dict) -> str:
    f = candidate.get("features") or {}
    pattern = candidate.get("live_pattern", "UNLABELED")
    ret_bucket = round(float(f.get("1h_current_return_pct", 0)), 0)
    range_bucket = round(float(f.get("1h_current_range_pct", 0)), 0)
    return f"{pattern}|r{ret_bucket}|rng{range_bucket}"


def diversify_select(
    ranked: list[dict],
    max_slots: int,
    max_per_group: int = 1,
) -> list[dict]:
    chosen: list[dict] = []
    group_counts: dict[str, int] = {}
    for c in ranked:
        g = movement_group(c)
        if group_counts.get(g, 0) >= max_per_group:
            continue
        chosen.append(c)
        group_counts[g] = group_counts.get(g, 0) + 1
        if len(chosen) >= max_slots:
            break
    return chosen
