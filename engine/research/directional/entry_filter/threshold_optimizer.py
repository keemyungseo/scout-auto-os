"""Grid search threshold optimizer — no ML, rule-based only."""

from __future__ import annotations

import math
import statistics
from typing import Literal

Operator = Literal["gte", "lte"]

MAX_GRID_STEPS = 50
MIN_PASS_SAMPLES = 15
MIN_LIFT_FOR_USE = 1.15
MIN_F1_FOR_USE = 0.12


def _mean(vals: list[float]) -> float:
    return statistics.mean(vals) if vals else 0.0


def _grid_thresholds(values: list[float], max_steps: int = MAX_GRID_STEPS) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if math.isclose(lo, hi, rel_tol=1e-9):
        return [round(lo, 6)]
    n_steps = min(max_steps, max(10, int(len(set(round(v, 4) for v in values)) * 0.5)))
    step = (hi - lo) / n_steps
    thresholds = [round(lo + step * i, 6) for i in range(n_steps + 1)]
    return sorted(set(thresholds))


def _passes(value: float, threshold: float, operator: Operator) -> bool:
    if operator == "gte":
        return value >= threshold
    return value <= threshold


def _metrics_at_threshold(
    signals: list[dict],
    feature: str,
    threshold: float,
    operator: Operator,
) -> dict:
    winners = [s for s in signals if s.get("cohort") == "winner"]
    losers = [s for s in signals if s.get("cohort") == "loser"]
    n_w, n_l = len(winners), len(losers)
    baseline_winner_rate = n_w / len(signals) if signals else 0.0

    pass_s = [s for s in signals if _passes(float(s["features"].get(feature, 0)), threshold, operator)]
    fail_s = [s for s in signals if s not in pass_s]

    pass_w = [s for s in pass_s if s.get("cohort") == "winner"]
    pass_l = [s for s in pass_s if s.get("cohort") == "loser"]
    fail_w = [s for s in fail_s if s.get("cohort") == "winner"]

    pass_n = len(pass_s)
    fail_n = len(fail_s)
    tp = len(pass_w)
    fp = len(pass_l)

    precision = tp / pass_n if pass_n else 0.0
    recall = tp / n_w if n_w else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    lift = precision / baseline_winner_rate if baseline_winner_rate > 0 else 0.0

    # odds ratio: (tp/fp) / (fn/tn) using winner/loser among pass/fail
    fail_l = fail_n - len(fail_w)
    tp_odds = tp / fp if fp else (tp if tp else 0.01)
    fn_odds = len(fail_w) / fail_l if fail_l else (len(fail_w) if len(fail_w) else 0.01)
    odds_ratio = tp_odds / fn_odds if fn_odds else 0.0

    winner_ratio_pass = tp / pass_n * 100 if pass_n else 0.0
    loser_ratio_pass = fp / pass_n * 100 if pass_n else 0.0
    winner_ratio_fail = len(fail_w) / fail_n * 100 if fail_n else 0.0
    loser_ratio_fail = fail_l / fail_n * 100 if fail_n else 0.0

    ret2_pass = _mean([float(s["return_2h"]) for s in pass_s])
    ret2_fail = _mean([float(s["return_2h"]) for s in fail_s])
    ret4_pass = _mean([float(s["return_4h"]) for s in pass_s])
    ret4_fail = _mean([float(s["return_4h"]) for s in fail_s])

    win_rate_pass = sum(1 for s in pass_s if float(s["return_2h"]) >= 3.0) / pass_n * 100 if pass_n else 0.0

    return {
        "feature": feature,
        "threshold": round(threshold, 6),
        "operator": operator,
        "pass_count": pass_n,
        "fail_count": fail_n,
        "winner_ratio_pass_pct": round(winner_ratio_pass, 2),
        "loser_ratio_pass_pct": round(loser_ratio_pass, 2),
        "winner_ratio_fail_pct": round(winner_ratio_fail, 2),
        "loser_ratio_fail_pct": round(loser_ratio_fail, 2),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "lift": round(lift, 4),
        "odds_ratio": round(odds_ratio, 4),
        "avg_return_2h_pass": round(ret2_pass, 4),
        "avg_return_2h_fail": round(ret2_fail, 4),
        "avg_return_4h_pass": round(ret4_pass, 4),
        "avg_return_4h_fail": round(ret4_fail, 4),
        "win_rate_pass_pct": round(win_rate_pass, 2),
        "expected_return_lift_2h": round(ret2_pass - ret2_fail, 4),
    }


def grid_search_feature(
    signals: list[dict],
    feature: str,
    direction: str,
) -> tuple[dict, list[dict]]:
    values = [float(s["features"].get(feature, 0)) for s in signals]
    grid = _grid_thresholds(values)
    curve: list[dict] = []

    best: dict | None = None
    for op in ("gte", "lte"):
        for thr in grid:
            m = _metrics_at_threshold(signals, feature, thr, op)
            m["direction"] = direction
            curve.append(m)
            if m["pass_count"] < MIN_PASS_SAMPLES:
                continue
            score = m["f1"] * 0.5 + m["lift"] * 0.3 + m["expected_return_lift_2h"] * 0.02
            if best is None or score > best.get("_score", -1):
                best = {**m, "_score": score}

    if best is None and curve:
        best = max(curve, key=lambda x: x["f1"])

    if best:
        best = {k: v for k, v in best.items() if k != "_score"}
        use = (
            best.get("pass_count", 0) >= MIN_PASS_SAMPLES
            and best.get("lift", 0) >= MIN_LIFT_FOR_USE
            and best.get("f1", 0) >= MIN_F1_FOR_USE
            and best.get("avg_return_2h_pass", 0) > best.get("avg_return_2h_fail", 0)
        )
        best["use_in_filter"] = use
        best["filter_direction"] = direction

    return best or {}, curve


def grid_search_all(
    signals: list[dict],
    feature_keys: list[str],
    direction: str,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Returns best_per_feature, threshold_curves, precision_curves, return_curves."""
    best_rows: list[dict] = []
    threshold_curves: list[dict] = []
    precision_curves: list[dict] = []
    return_curves: list[dict] = []

    for feat in feature_keys:
        best, curve = grid_search_feature(signals, feat, direction)
        if best:
            best_rows.append(best)
        for c in curve:
            threshold_curves.append(c)
            precision_curves.append({
                "direction": direction,
                "feature": c["feature"],
                "threshold": c["threshold"],
                "operator": c["operator"],
                "precision": c["precision"],
                "recall": c["recall"],
                "f1": c["f1"],
                "lift": c["lift"],
            })
            return_curves.append({
                "direction": direction,
                "feature": c["feature"],
                "threshold": c["threshold"],
                "operator": c["operator"],
                "avg_return_2h_pass": c["avg_return_2h_pass"],
                "avg_return_2h_fail": c["avg_return_2h_fail"],
                "avg_return_4h_pass": c["avg_return_4h_pass"],
                "avg_return_4h_fail": c["avg_return_4h_fail"],
            })

    best_rows.sort(key=lambda x: x.get("f1", 0), reverse=True)
    return best_rows, threshold_curves, precision_curves, return_curves


def evaluate_combined_rule(
    signals: list[dict],
    rules: list[dict],
) -> dict:
    """AND combination of feature thresholds."""
    if not rules:
        return {"pass_count": 0, "rules_applied": 0}

    def passes_all(s: dict) -> bool:
        for r in rules:
            v = float(s["features"].get(r["feature"], 0))
            if not _passes(v, r["threshold"], r["operator"]):
                return False
        return True

    pass_s = [s for s in signals if passes_all(s)]
    fail_s = [s for s in signals if not passes_all(s)]
    tp = sum(1 for s in pass_s if s.get("cohort") == "winner")
    fp = sum(1 for s in pass_s if s.get("cohort") == "loser")
    pass_n = len(pass_s)
    n_w = sum(1 for s in signals if s.get("cohort") == "winner")
    precision = tp / pass_n if pass_n else 0.0
    recall = tp / n_w if n_w else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "pass_count": pass_n,
        "fail_count": len(fail_s),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "avg_return_2h_pass": round(_mean([float(s["return_2h"]) for s in pass_s]), 4),
        "avg_return_4h_pass": round(_mean([float(s["return_4h"]) for s in pass_s]), 4),
        "win_rate_pass_pct": round(
            sum(1 for s in pass_s if float(s["return_2h"]) >= 3.0) / pass_n * 100, 2,
        ) if pass_n else 0.0,
        "rules_applied": len(rules),
    }
