"""Rule-level reject audit — condition failures and margins."""

from __future__ import annotations

from scout_auto_os.engine.research.directional.entry_filter.pattern_labels import live_pattern
from scout_auto_os.engine.research.directional.entry_filter.rule_tree import Condition, RuleNode
from scout_auto_os.engine.portfolio.scoring import _conditions_for_tree, _freshness_score


def _feature_label(feature: str) -> str:
    if "body_pct" in feature:
        return "Body"
    if "range_pct" in feature:
        return "Range"
    if "return_pct" in feature or "momentum" in feature:
        return "Momentum"
    if "ma20_distance" in feature:
        return "MA20_Distance"
    if "close_position" in feature:
        return "Close_Position"
    if "volume" in feature:
        return "Volume"
    if "compression" in feature:
        return "Compression"
    return feature


def _condition_result(c: Condition, features: dict) -> dict:
    v = float(features.get(c.feature, 0))
    thr = c.threshold
    if c.operator == "gte":
        passed = v >= thr
        gap = thr - v if not passed else 0.0
    else:
        passed = v <= thr
        gap = v - thr if not passed else 0.0
    gap_pct = round(gap / (abs(thr) + 1e-6) * 100, 4) if not passed else 0.0
    return {
        "letter": c.letter,
        "feature": c.feature,
        "feature_label": _feature_label(c.feature),
        "operator": c.operator,
        "threshold": thr,
        "value": round(v, 6),
        "passed": passed,
        "gap": round(gap, 6),
        "gap_pct": gap_pct,
    }


def _best_or_branch_results(features: dict, tree: RuleNode) -> list[dict]:
    """For failed OR trees, pick branch with fewest failures."""
    if tree.node_type != "or":
        conds = _conditions_for_tree(tree)
        return [_condition_result(c, features) for c in conds]

    best: list[dict] = []
    best_fails = 999
    for child in tree.children:
        conds = _conditions_for_tree(child)
        results = [_condition_result(c, features) for c in conds]
        fails = sum(1 for r in results if not r["passed"])
        if fails < best_fails:
            best_fails = fails
            best = results
    return best


def classify_reject_tier(failed: list[dict]) -> str:
    if not failed:
        return "pass"
    max_gap = max(r["gap_pct"] for r in failed)
    n_fail = len(failed)
    if n_fail <= 1 and max_gap <= 10.0:
        return "near_pass"
    if n_fail <= 2 and max_gap <= 40.0:
        return "medium_reject"
    return "impossible_reject"


def audit_rule(
    features: dict,
    tree: RuleNode,
    direction: str,
) -> dict:
    pattern = live_pattern(features)
    rule_pass = tree.evaluate(features)
    cond_results = _best_or_branch_results(features, tree)
    failed = [r for r in cond_results if not r["passed"]]
    freshness = _freshness_score(features, direction)

    primary_reason = "PASS"
    if pattern == "UNLABELED":
        primary_reason = "Pattern_Reject"
    elif not rule_pass:
        primary_reason = "Rule_Reject"
    elif freshness < 0.25:
        primary_reason = "Freshness_Reject"

    return {
        "live_pattern": pattern,
        "rule_pass": rule_pass,
        "freshness_score": round(freshness, 4),
        "primary_reason": primary_reason,
        "reject_tier": classify_reject_tier(failed) if not rule_pass else "pass",
        "failed_condition_count": len(failed),
        "condition_results": cond_results,
        "failed_conditions": failed,
    }
