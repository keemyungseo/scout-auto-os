"""Blind temporal validation for discovered rules."""

from __future__ import annotations

import statistics

from scout_auto_os.engine.research.directional.entry_filter.rule_tree import RuleNode
from scout_auto_os.engine.research.rule_discovery.constants import SUCCESS_RETURN_PCT
from scout_auto_os.engine.research.rule_discovery.discovered_rule import DiscoveredRule


def _mean(vals: list[float]) -> float:
    return round(statistics.mean(vals), 4) if vals else 0.0


def evaluate_discovered_rule(
    signals: list[dict],
    rule: DiscoveredRule,
    baseline_precision: float,
    v2_pass_count: int = 0,
) -> dict:
    winners = [s for s in signals if s.get("cohort") == "winner"]
    n_w = len(winners)
    baseline = n_w / len(signals) if signals else 0.0

    pass_s = [s for s in signals if rule.evaluate(s["features"], s.get("ctx"))]
    fail_s = [s for s in signals if s not in pass_s]
    pass_n = len(pass_s)

    tp = sum(1 for s in pass_s if s.get("cohort") == "winner")
    fp = sum(1 for s in pass_s if s.get("cohort") == "loser")
    fn = sum(1 for s in fail_s if s.get("cohort") == "winner")

    precision = tp / pass_n if pass_n else 0.0
    recall = tp / n_w if n_w else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    lift = precision / baseline if baseline else 0.0

    scans = {s["scan_time_kst"] for s in signals}
    days = len({s[:10] for s in scans}) or 1

    avg2 = _mean([float(s["return_2h"]) for s in pass_s])
    avg4 = _mean([float(s["return_4h"]) for s in pass_s])
    win_rate = sum(1 for s in pass_s if float(s["return_2h"]) >= SUCCESS_RETURN_PCT) / pass_n * 100 if pass_n else 0.0

    coverage = pass_n / len(signals) * 100 if signals else 0.0
    pass_per_day = round(pass_n / days, 4)

    return {
        "rule_id": rule.rule_id,
        "rule_expr": rule.rule_expr,
        "direction": rule.direction,
        "pass_count": pass_n,
        "fail_count": len(fail_s),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "lift": round(lift, 4),
        "avg_return_2h": avg2,
        "avg_return_4h": avg4,
        "win_rate_pct": round(win_rate, 2),
        "coverage_pct": round(coverage, 2),
        "pass_per_day": pass_per_day,
        "baseline_precision": round(baseline_precision, 4),
        "precision_floor": round(baseline_precision - 0.02, 4),
        "meets_precision_floor": precision >= baseline_precision - 0.02,
        "coverage_lift_vs_v2": round((pass_n - v2_pass_count) / max(v2_pass_count, 1) * 100, 2),
    }


def evaluate_v2_tree(
    signals: list[dict],
    tree: RuleNode,
    rule_id: str,
    direction: str,
) -> dict:
    winners = sum(1 for s in signals if s.get("cohort") == "winner")
    baseline = winners / len(signals) if signals else 0.0
    pass_s = [s for s in signals if tree.evaluate(s["features"])]
    pass_n = len(pass_s)
    tp = sum(1 for s in pass_s if s.get("cohort") == "winner")
    precision = tp / pass_n if pass_n else 0.0
    recall = tp / winners if winners else 0.0
    scans = {s["scan_time_kst"] for s in signals}
    days = len({s[:10] for s in scans}) or 1
    return {
        "rule_id": rule_id,
        "rule_expr": tree.describe(),
        "direction": direction,
        "pass_count": pass_n,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "lift": round(precision / baseline, 4) if baseline else 0,
        "avg_return_2h": _mean([float(s["return_2h"]) for s in pass_s]),
        "avg_return_4h": _mean([float(s["return_4h"]) for s in pass_s]),
        "coverage_pct": round(pass_n / len(signals) * 100, 2) if signals else 0,
        "pass_per_day": round(pass_n / days, 4),
    }


def evaluate_hybrid(
    signals: list[dict],
    v2_tree: RuleNode,
    candidate: DiscoveredRule,
    direction: str,
) -> dict:
    """Hybrid = pass if V2 OR candidate (coverage union, precision on blind)."""
    pass_s = [
        s for s in signals
        if v2_tree.evaluate(s["features"]) or candidate.evaluate(s["features"], s.get("ctx"))
    ]
    pass_n = len(pass_s)
    winners = sum(1 for s in signals if s.get("cohort") == "winner")
    baseline = winners / len(signals) if signals else 0.0
    tp = sum(1 for s in pass_s if s.get("cohort") == "winner")
    precision = tp / pass_n if pass_n else 0.0
    scans = {s["scan_time_kst"] for s in signals}
    days = len({s[:10] for s in scans}) or 1
    return {
        "direction": direction,
        "rule_expr": f"(V2 OR {candidate.rule_expr})",
        "pass_count": pass_n,
        "precision": round(precision, 4),
        "recall": round(tp / winners, 4) if winners else 0,
        "lift": round(precision / baseline, 4) if baseline else 0,
        "avg_return_2h": _mean([float(s["return_2h"]) for s in pass_s]),
        "avg_return_4h": _mean([float(s["return_4h"]) for s in pass_s]),
        "coverage_pct": round(pass_n / len(signals) * 100, 2) if signals else 0,
        "pass_per_day": round(pass_n / days, 4),
    }
