"""Performance metrics for generalization folds."""

from __future__ import annotations

import math
import statistics

from scout_auto_os.engine.research.execution_generalization.constants import SUCCESS_RETURN_PCT
from scout_auto_os.engine.research.execution_rule_discovery.baselines import pick_top2_execution_score
from scout_auto_os.engine.research.execution_rule_discovery.generator import avg_top2_return, pick_top2_by_rule
from scout_auto_os.engine.research.rule_discovery.discovered_rule import DiscoveredRule


def collect_trade_returns(
    groups: list[list[dict]],
    rule: DiscoveredRule | None,
    baseline_fn=None,
    direction_filter: str | None = "long",
) -> list[float]:
    rets: list[float] = []
    for g in groups:
        if direction_filter and g[0].get("direction") != direction_filter:
            continue
        if len(g) < 2:
            continue
        if rule is not None:
            picks = pick_top2_by_rule(g, rule)
        else:
            picks = baseline_fn(g) if baseline_fn else g[:2]
        rets.extend(float(p["return_2h"]) for p in picks)
    return rets


def equity_max_drawdown(returns: list[float]) -> float:
    if not returns:
        return 0.0
    cum = 0.0
    peak = 0.0
    mdd = 0.0
    for r in returns:
        cum += r
        peak = max(peak, cum)
        mdd = min(mdd, cum - peak)
    return round(mdd, 4)


def sharpe_approx(returns: list[float]) -> float:
    if len(returns) < 2:
        return 0.0
    mu = statistics.mean(returns)
    sd = statistics.pstdev(returns)
    if sd < 1e-9:
        return 0.0
    return round(mu / sd * math.sqrt(len(returns)), 4)


def evaluate_fold(
    groups: list[list[dict]],
    rule: DiscoveredRule,
    fold_id: str,
    split_type: str,
) -> dict:
    rule_groups = [g for g in groups if g[0].get("direction") == rule.direction]
    base_groups = rule_groups

    rule_rets = collect_trade_returns(rule_groups, rule)
    base_rets = collect_trade_returns(base_groups, None, pick_top2_execution_score, rule.direction)

    rule_m = avg_top2_return(rule_groups, rule)
    base_m = avg_top2_return(base_groups, None, pick_top2_execution_score)

    days = len({g[0]["scan_time_kst"][:10] for g in rule_groups}) or 1
    n_rule = len(rule_rets)
    n_base = len(base_rets)

    return {
        "fold_id": fold_id,
        "split_type": split_type,
        "direction": rule.direction,
        "scan_groups": len(rule_groups),
        "rule_trade_count": n_rule,
        "baseline_trade_count": n_base,
        "rule_avg_return_2h": rule_m["avg_return_2h"],
        "baseline_avg_return_2h": base_m["avg_return_2h"],
        "rule_win_rate_pct": rule_m["win_rate_pct"],
        "baseline_win_rate_pct": base_m["win_rate_pct"],
        "rule_return_per_day": round(sum(rule_rets) / days, 4) if rule_rets else 0.0,
        "baseline_return_per_day": round(sum(base_rets) / days, 4) if base_rets else 0.0,
        "rule_mdd_pct": equity_max_drawdown(rule_rets),
        "baseline_mdd_pct": equity_max_drawdown(base_rets),
        "rule_sharpe": sharpe_approx(rule_rets),
        "baseline_sharpe": sharpe_approx(base_rets),
        "rule_beats_baseline": rule_m["avg_return_2h"] >= base_m["avg_return_2h"],
        "coverage_pct": round(len(rule_groups) / max(len(groups), 1) * 100, 2),
    }


def monthly_returns(returns_with_dates: list[tuple[str, float]]) -> dict[str, float]:
    buckets: dict[str, list[float]] = {}
    for d, r in returns_with_dates:
        key = d[:7]
        buckets.setdefault(key, []).append(r)
    return {k: round(statistics.mean(v), 4) for k, v in sorted(buckets.items())}


def return_stability(monthly_avgs: list[float]) -> float:
    if len(monthly_avgs) < 2:
        return 0.0
    return round(statistics.pstdev(monthly_avgs), 4)
