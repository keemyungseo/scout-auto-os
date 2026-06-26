"""Entry-time feature vector for lifecycle classification."""

from __future__ import annotations

import math
from datetime import datetime

from scout_auto_os.engine.portfolio.rule_loader import PortfolioRules
from scout_auto_os.engine.portfolio.scoring import (
    _clamp01,
    _conditions_for_tree,
    _direction_rank_score,
    _feature_match_and_margin,
    _freshness_score,
    _recency_score,
)
from scout_auto_os.engine.research.directional.dna.collector import numeric_feature_keys
from scout_auto_os.engine.research.directional.dna.formulas import ClusterFormula
from scout_auto_os.engine.research.directional.entry_filter.pattern_labels import live_pattern
from scout_auto_os.engine.research.directional.patterns import label_direction_pattern
from scout_auto_os.engine.research.directional.prediction.engine import _softmax, _temperature


def _cluster_features(
    features: dict,
    engine: str,
    direction: str,
    formulas: list[ClusterFormula],
) -> dict[str, float]:
    pool = [f for f in formulas if f.direction == direction and f.engine == engine]
    if not pool:
        pool = [f for f in formulas if f.direction == direction]
    if not pool:
        return {
            "cluster_top_score": 0.0,
            "cluster_score_spread": 0.0,
            "cluster_top_prob": 0.0,
        }

    raw = {f.name: f.score(features) for f in pool}
    probs = _softmax(raw, _temperature(raw))
    top_name = max(raw, key=raw.get)
    vals = sorted(raw.values(), reverse=True)
    spread = vals[0] - vals[1] if len(vals) > 1 else 0.0
    top_cluster = top_name.split("_")[-1] if "_" in top_name else top_name
    cluster_ord = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}.get(top_cluster, 0)
    return {
        "cluster_top_score": round(raw[top_name], 6),
        "cluster_score_spread": round(spread, 6),
        "cluster_top_prob": round(probs.get(top_name, 0.0) / 100.0, 4),
        "cluster_letter_ord": float(cluster_ord),
    }


def _portfolio_scan_features(
    row: dict,
    direction: str,
    rules: PortfolioRules,
    all_rows: list[dict],
    scan_time_kst: str,
    latest_scan: str,
) -> dict[str, float]:
    features = row.get("features") or {}
    symbol = row["symbol"]
    pattern = live_pattern(features)

    if direction == "long":
        tree = rules.pattern_trees.get(pattern, rules.long_tree)
        meta = rules.pattern_meta.get(pattern, rules.long_meta)
    else:
        tree = rules.pattern_trees.get(pattern, rules.short_tree)
        meta = rules.pattern_meta.get(pattern, rules.short_meta)

    rule_pass = 1.0 if tree.evaluate(features) else 0.0
    conds = _conditions_for_tree(tree)
    match_ratio, margin_score = _feature_match_and_margin(features, conds)
    direction_conf = _direction_rank_score(all_rows, symbol, direction)
    pattern_conf = _clamp01(float(meta.get("live_score", meta.get("precision", 0.5)) or 0.5))
    rule_conf = _clamp01(float(meta.get("precision", 0.5)))
    recency = _recency_score(scan_time_kst, latest_scan)
    freshness = _freshness_score(features, direction)

    from scout_auto_os.engine.portfolio.constants import SCORE_WEIGHTS

    w = SCORE_WEIGHTS
    entry_score = 100.0 * (
        direction_conf * w["direction_confidence"]
        + pattern_conf * w["pattern_confidence"]
        + rule_conf * w["rule_confidence"]
        + match_ratio * w["feature_match_ratio"]
        + margin_score * w["rule_margin"]
        + recency * w["recency"]
        + freshness * w["signal_freshness"]
    )

    raw_pat = label_direction_pattern(features)
    out = {
        "entry_score": round(entry_score, 4),
        "rule_pass": rule_pass,
        "direction_confidence": round(direction_conf, 4),
        "pattern_confidence": round(pattern_conf, 4),
        "rule_confidence": round(rule_conf, 4),
        "feature_match_ratio": round(match_ratio, 4),
        "rule_margin": round(margin_score, 4),
        "recency": round(recency, 4),
        "signal_freshness": round(freshness, 4),
    }
    for pat in (
        "LONG_CONTINUATION", "LONG_REVERSAL", "LONG_BREAKOUT",
        "SHORT_CONTINUATION", "SHORT_REVERSAL", "SHORT_BREAKDOWN",
    ):
        out[f"pat_{pat}"] = 1.0 if pattern == pat else 0.0
    for rp in (
        "UP_CONTINUATION", "DOWN_UP", "DOWN_BASE_UP", "UP_ACCELERATION",
        "DOWN_CONTINUATION", "UP_DOWN", "UP_BASE_DOWN", "DOWN_ACCELERATION",
    ):
        out[f"rawpat_{rp}"] = 1.0 if raw_pat == rp else 0.0
    return out


def build_entry_feature_row(
    row: dict,
    direction: str,
    engine: str,
    rules: PortfolioRules,
    formulas: list[ClusterFormula],
    all_rows: list[dict],
    scan_time_kst: str,
    latest_scan: str,
    feature_keys: list[str] | None = None,
) -> dict[str, float]:
    """Scan-time only — no forward fields."""
    features = row.get("features") or {}
    keys = feature_keys or numeric_feature_keys(features)
    vec: dict[str, float] = {f"feat_{k}": float(features.get(k, 0)) for k in keys}
    vec.update(_portfolio_scan_features(row, direction, rules, all_rows, scan_time_kst, latest_scan))
    vec.update(_cluster_features(features, engine, direction, formulas))
    vec["direction_long"] = 1.0 if direction == "long" else 0.0
    vec["direction_short"] = 1.0 if direction == "short" else 0.0
    return vec


def feature_matrix(rows: list[dict], key: str = "x") -> tuple[list[str], list[list[float]]]:
    if not rows:
        return [], []
    names = sorted(rows[0][key].keys())
    mat = [[float(r[key].get(n, 0.0)) for n in names] for r in rows]
    return names, mat
