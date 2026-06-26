"""Regime-router simulation — specialized rule vs universal baseline."""

from __future__ import annotations

import statistics
from collections import defaultdict

from scout_auto_os.engine.research.execution_rule_discovery.baselines import pick_top2_execution_score
from scout_auto_os.engine.research.execution_rule_discovery.constants import TOP2_SIZE
from scout_auto_os.engine.research.execution_rule_discovery.generator import pick_top2_by_rule
from scout_auto_os.engine.research.rule_portfolio.collectors import PortfolioRule
from scout_auto_os.engine.research.rule_portfolio.constants import BASELINE_RULE_ID, MIN_REGIME_TRADES


def _rule_lookup(profiles: list[dict], portfolio: list[PortfolioRule]) -> dict[str, PortfolioRule]:
    by_id = {p.rule_id: p for p in portfolio}
    return by_id


def build_regime_router_table(profiles: list[dict], min_trades: int = MIN_REGIME_TRADES) -> dict[tuple[str, str], str]:
    """Map (direction, regime) -> best rule_id by avg return in that regime."""
    table: dict[tuple[str, str], str] = {}
    by_dir_regime: dict[tuple[str, str], list[dict]] = defaultdict(list)

    for p in profiles:
        if p["rule_id"] == BASELINE_RULE_ID:
            continue
        regime_avg = p.get("regime_avg_json") or {}
        for regime, avg in regime_avg.items():
            if int(p.get("trade_count", 0)) < min_trades:
                continue
            by_dir_regime[(p["direction"], regime)].append({**p, "_regime_avg": avg})

    for key, rows in by_dir_regime.items():
        best = max(rows, key=lambda r: float(r["_regime_avg"]))
        table[key] = best["rule_id"]
    return table


def simulate_regime_router(
    portfolio: list[PortfolioRule],
    profiles: list[dict],
    groups: list[list[dict]],
) -> dict:
    router = build_regime_router_table(profiles)
    by_id = _rule_lookup(profiles, portfolio)

    router_returns: list[float] = []
    baseline_returns: list[float] = []
    routed_rules: list[str] = []

    for g in groups:
        if len(g) < TOP2_SIZE:
            continue
        direction = g[0]["direction"]
        regime = g[0].get("regime", "unknown")
        rule_id = router.get((direction, regime))
        pr = by_id.get(rule_id) if rule_id else None

        base_picks = pick_top2_execution_score(g)
        baseline_returns.extend(float(p["return_2h"]) for p in base_picks)

        if pr and pr.rule is not None:
            picks = pick_top2_by_rule(g, pr.rule)
            routed_rules.append(rule_id)
        else:
            picks = base_picks
            routed_rules.append(BASELINE_RULE_ID)
        router_returns.extend(float(p["return_2h"]) for p in picks)

    router_avg = round(statistics.mean(router_returns), 4) if router_returns else 0.0
    base_avg = round(statistics.mean(baseline_returns), 4) if baseline_returns else 0.0
    lift = round((router_avg - base_avg) / abs(base_avg or 0.01) * 100, 2)

    return {
        "router_avg_return_2h": router_avg,
        "baseline_avg_return_2h": base_avg,
        "lift_pct": lift,
        "router_beats_baseline": router_avg >= base_avg,
        "router_trade_count": len(router_returns),
        "routes_used": len(set(routed_rules)),
        "router_table_size": len(router),
    }
