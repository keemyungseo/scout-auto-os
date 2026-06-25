"""Regime detection rule candidates — observable signals only (research)."""

from __future__ import annotations

import statistics
from collections import defaultdict

from scout_auto_os.engine.research.regime.classifier import MarketSnapshot, REGIME_STATES


RULE_CANDIDATES: list[dict] = [
    {"rule_id": "R_BTC_TREND_BULL", "signal": "btc_1h", "op": ">=", "threshold": 0.5, "predicts": "Bull"},
    {"rule_id": "R_BTC_TREND_BEAR", "signal": "btc_1h", "op": "<", "threshold": -0.5, "predicts": "Bear"},
    {"rule_id": "R_BREADTH_WIDE", "signal": "breadth_positive_1h", "op": ">=", "threshold": 0.55, "predicts": "Bull"},
    {"rule_id": "R_BREADTH_NARROW", "signal": "breadth_positive_1h", "op": "<", "threshold": 0.35, "predicts": "Bear"},
    {"rule_id": "R_TOP20_RISING", "signal": "top20_positive_pct", "op": ">=", "threshold": 0.60, "predicts": "Strong_Bull"},
    {"rule_id": "R_ATR_PROXY_HIGH", "signal": "median_range_1h", "op": ">=", "threshold": 3.0, "predicts": "Breakout"},
    {"rule_id": "R_VOL_EXPANSION", "signal": "volume_expansion", "op": ">=", "threshold": 0.8, "predicts": "Breakout"},
    {"rule_id": "R_VOL_RATIO_HIGH", "signal": "median_volume_ratio", "op": ">=", "threshold": 1.3, "predicts": "Breakout"},
    {"rule_id": "R_COMPRESSION_HIGH", "signal": "median_compression", "op": ">=", "threshold": 2.0, "predicts": "Bottom"},
    {"rule_id": "R_RELEASE_ACTIVE", "signal": "median_release", "op": ">=", "threshold": 0.12, "predicts": "Breakout"},
    {"rule_id": "R_SIDEWAY_FLAT", "signal": "median_1h", "op": "abs_lt", "threshold": 0.4, "predicts": "Sideway"},
    {"rule_id": "R_CAPITULATION", "signal": "median_1h", "op": "<=", "threshold": -1.5, "predicts": "Capitulation"},
]


def _eval_rule(snap: MarketSnapshot, rule: dict) -> bool:
    val = getattr(snap, rule["signal"], 0)
    op = rule["op"]
    th = rule["threshold"]
    if op == ">=":
        return val >= th
    if op == "<":
        return val < th
    if op == "<=":
        return val <= th
    if op == "abs_lt":
        return abs(val) < th
    return False


def score_rules(
    labeled: list[tuple[MarketSnapshot, str]],
) -> list[dict]:
    """Score each rule by precision/recall vs empirical regime labels."""
    results: list[dict] = []
    for rule in RULE_CANDIDATES:
        target = rule["predicts"]
        tp = fp = fn = 0
        for snap, actual in labeled:
            fired = _eval_rule(snap, rule)
            if fired and actual == target:
                tp += 1
            elif fired and actual != target:
                fp += 1
            elif not fired and actual == target:
                fn += 1
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        results.append({
            **rule,
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
        })
    results.sort(key=lambda x: x["f1"], reverse=True)
    return results


def composite_router_rules(
    labeled: list[tuple[MarketSnapshot, str]],
    top_n: int = 5,
) -> list[dict]:
    """Suggest LIVE-observable composite rules from top single signals."""
    scored = score_rules(labeled)
    top = scored[:top_n]
    composites = [
        {
            "rule_id": "COMPOSITE_BULL_ROUTER",
            "conditions": "btc_1h>=0.5 AND breadth>=0.50 AND median_1h>=0.35",
            "route_engine": "MOMENTUM",
            "regime_target": "Bull",
        },
        {
            "rule_id": "COMPOSITE_BREAKOUT_ROUTER",
            "conditions": "median_release>=0.12 AND median_volume_ratio>=1.1 AND median_range_1h>=2.5",
            "route_engine": "BREAKOUT",
            "regime_target": "Breakout",
        },
        {
            "rule_id": "COMPOSITE_BOTTOM_ROUTER",
            "conditions": "median_compression>=1.5 AND median_1h<0.2 AND breadth<0.48",
            "route_engine": "REVERSAL",
            "regime_target": "Bottom",
        },
        {
            "rule_id": "COMPOSITE_SIDEWAY_ROUTER",
            "conditions": "abs(median_1h)<0.45 AND breadth in [0.38,0.58]",
            "route_engine": "COMPRESSION",
            "regime_target": "Sideway",
        },
        {
            "rule_id": "COMPOSITE_BEAR_SKIP",
            "conditions": "median_1h<=-0.55 AND breadth<0.38",
            "route_engine": "SKIP",
            "regime_target": "Bear",
        },
    ]
    for c in composites:
        c["top_supporting_signals"] = [r["rule_id"] for r in top[:3]]
    return composites
