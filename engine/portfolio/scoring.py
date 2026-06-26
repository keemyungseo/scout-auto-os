"""Entry scoring for portfolio ranking."""

from __future__ import annotations

import math
from datetime import datetime

from scout_auto_os.engine.portfolio.constants import SCORE_WEIGHTS
from scout_auto_os.engine.portfolio.rule_loader import PortfolioRules
from scout_auto_os.engine.research.directional.engines import rank_long, rank_short
from scout_auto_os.engine.research.directional.entry_filter.constants import (
    LONG_DIRECTION_CHAMPION,
    SHORT_DIRECTION_CHAMPION,
)
from scout_auto_os.engine.research.directional.entry_filter.pattern_labels import live_pattern
from scout_auto_os.engine.research.directional.entry_filter.rule_tree import Condition, RuleNode


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _direction_rank_score(rows: list[dict], symbol: str, direction: str) -> float:
    if direction == "long":
        ranked = rank_long(rows, LONG_DIRECTION_CHAMPION, top_k=len(rows))
    else:
        ranked = rank_short(rows, SHORT_DIRECTION_CHAMPION, top_k=len(rows))
    if symbol not in ranked:
        return 0.0
    rank = ranked.index(symbol)
    n = max(len(ranked), 1)
    return _clamp01(1.0 - rank / n)


def _feature_match_and_margin(
    features: dict,
    conditions: list[Condition],
) -> tuple[float, float]:
    if not conditions:
        return 0.0, 0.0
    hits = 0
    margins: list[float] = []
    for c in conditions:
        v = float(features.get(c.feature, 0))
        thr = c.threshold
        if c.operator == "gte":
            ok = v >= thr
            margin = (v - thr) / (abs(thr) + 1e-6) if ok else 0.0
        else:
            ok = v <= thr
            margin = (thr - v) / (abs(thr) + 1e-6) if ok else 0.0
        if ok:
            hits += 1
        margins.append(max(0.0, margin))
    match_ratio = hits / len(conditions)
    margin_score = _clamp01(sum(margins) / len(margins) / 2.0)
    return match_ratio, margin_score


def _conditions_for_tree(tree: RuleNode) -> list[Condition]:
    out: list[Condition] = []

    def walk(node: RuleNode) -> None:
        if node.node_type == "cond" and node.condition:
            out.append(node.condition)
        for ch in node.children:
            walk(ch)

    walk(tree)
    return out


def _recency_score(scan_time_kst: str, latest_scan: str) -> float:
    try:
        t0 = datetime.strptime(scan_time_kst, "%Y-%m-%d %H:%M:%S")
        t1 = datetime.strptime(latest_scan, "%Y-%m-%d %H:%M:%S")
        hours = max(0.0, (t1 - t0).total_seconds() / 3600.0)
        return _clamp01(1.0 - hours / 24.0)
    except ValueError:
        return 0.5


def _freshness_score(features: dict, direction: str) -> float:
    mom = abs(float(features.get("5m_momentum", 0)))
    vol = float(features.get("15m_current_volume_ratio", 1.0))
    body = abs(float(features.get("15m_current_body_pct", 0)))
    raw = mom * 0.4 + min(vol, 3.0) / 3.0 * 0.3 + min(body, 5.0) / 5.0 * 0.3
    if direction == "short":
        ret = float(features.get("1h_current_return_pct", 0))
        raw += _clamp01(abs(min(0.0, ret)) / 10.0) * 0.2
    else:
        ret = float(features.get("1h_current_return_pct", 0))
        raw += _clamp01(max(0.0, ret) / 10.0) * 0.2
    return _clamp01(raw / 1.2)


def score_candidate(
    row: dict,
    direction: str,
    rules: PortfolioRules,
    all_rows: list[dict],
    scan_time_kst: str,
    latest_scan: str,
) -> dict | None:
    features = row.get("features") or {}
    symbol = row["symbol"]
    pattern = live_pattern(features)

    if direction == "long":
        tree = rules.pattern_trees.get(pattern, rules.long_tree)
        meta = rules.pattern_meta.get(pattern, rules.long_meta)
    else:
        tree = rules.pattern_trees.get(pattern, rules.short_tree)
        meta = rules.pattern_meta.get(pattern, rules.short_meta)

    if not tree.evaluate(features):
        return None

    conds = _conditions_for_tree(tree)
    match_ratio, margin_score = _feature_match_and_margin(features, conds)
    direction_conf = _direction_rank_score(all_rows, symbol, direction)
    pattern_conf = _clamp01(float(meta.get("live_score", meta.get("precision", 0.5)) or 0.5))
    rule_conf = _clamp01(float(meta.get("precision", 0.5)))

    recency = _recency_score(scan_time_kst, latest_scan)
    freshness = _freshness_score(features, direction)

    w = SCORE_WEIGHTS
    entry_score = round(
        100.0 * (
            direction_conf * w["direction_confidence"]
            + pattern_conf * w["pattern_confidence"]
            + rule_conf * w["rule_confidence"]
            + match_ratio * w["feature_match_ratio"]
            + margin_score * w["rule_margin"]
            + recency * w["recency"]
            + freshness * w["signal_freshness"]
        ),
        2,
    )

    return {
        "symbol": symbol,
        "direction": direction,
        "live_pattern": pattern,
        "entry_score": entry_score,
        "direction_confidence": round(direction_conf, 4),
        "pattern_confidence": round(pattern_conf, 4),
        "rule_confidence": round(rule_conf, 4),
        "feature_match_ratio": round(match_ratio, 4),
        "rule_margin": round(margin_score, 4),
        "recency": round(recency, 4),
        "signal_freshness": round(freshness, 4),
        "rule_expr": meta.get("rule_expr", tree.describe()),
        "features": features,
        "scan_time_kst": scan_time_kst,
    }


def build_pass_candidates(
    rows: list[dict],
    scan_time_kst: str,
    rules: PortfolioRules,
    latest_scan: str,
) -> tuple[list[dict], list[dict]]:
    long_c: list[dict] = []
    short_c: list[dict] = []
    for row in rows:
        for direction, bucket in (("long", long_c), ("short", short_c)):
            scored = score_candidate(row, direction, rules, rows, scan_time_kst, latest_scan)
            if scored:
                bucket.append(scored)
    long_c.sort(key=lambda x: x["entry_score"], reverse=True)
    short_c.sort(key=lambda x: x["entry_score"], reverse=True)
    return long_c, short_c
