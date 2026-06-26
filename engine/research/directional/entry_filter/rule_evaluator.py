"""Evaluate rule trees on labeled signals."""

from __future__ import annotations

import statistics

from scout_auto_os.engine.research.directional.entry_filter.rule_tree import RuleNode


def evaluate_rule_tree(
    signals: list[dict],
    tree: RuleNode,
    rule_id: str,
    scope: str = "all",
) -> dict:
    winners = [s for s in signals if s.get("cohort") == "winner"]
    n_w = len(winners)
    baseline = n_w / len(signals) if signals else 0.0

    pass_s = [s for s in signals if tree.evaluate(s["features"])]
    fail_s = [s for s in signals if s not in pass_s]
    pass_n = len(pass_s)
    fail_n = len(fail_s)

    tp = sum(1 for s in pass_s if s.get("cohort") == "winner")
    fp = sum(1 for s in pass_s if s.get("cohort") == "loser")
    fn = sum(1 for s in fail_s if s.get("cohort") == "winner")

    precision = tp / pass_n if pass_n else 0.0
    recall = tp / n_w if n_w else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    lift = precision / baseline if baseline > 0 else 0.0

    fail_l = fail_n - fn
    tp_odds = tp / fp if fp else (tp if tp else 0.01)
    fn_odds = fn / fail_l if fail_l else (fn if fn else 0.01)
    odds_ratio = tp_odds / fn_odds if fn_odds else 0.0

    scans = {s["scan_time_kst"] for s in signals}
    pass_scans = {s["scan_time_kst"] for s in pass_s}
    days = len({s[:10] for s in scans}) or 1

    avg2 = _mean([float(s["return_2h"]) for s in pass_s])
    avg4 = _mean([float(s["return_4h"]) for s in pass_s])
    win_rate = sum(1 for s in pass_s if float(s["return_2h"]) >= 3.0) / pass_n * 100 if pass_n else 0.0

    return {
        "rule_id": rule_id,
        "rule_expr": tree.describe(),
        "scope": scope,
        "signal_count": len(signals),
        "pass_count": pass_n,
        "fail_count": fail_n,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "lift": round(lift, 4),
        "odds_ratio": round(odds_ratio, 4),
        "avg_return_2h": round(avg2, 4),
        "avg_return_4h": round(avg4, 4),
        "win_rate_pct": round(win_rate, 2),
        "pass_scan_count": len(pass_scans),
        "total_scan_count": len(scans),
        "pass_per_scan": round(pass_n / len(scans), 4) if scans else 0.0,
        "pass_per_day": round(pass_n / days, 4),
        "trade_frequency_pct": round(pass_n / len(signals) * 100, 2) if signals else 0.0,
    }


def _mean(vals: list[float]) -> float:
    return statistics.mean(vals) if vals else 0.0


def live_selection_score(
    row: dict,
    baseline_precision: float,
    target_pass_per_day: float = 3.0,
) -> float:
    """Higher = better for LIVE: recall up, precision maintained, enough frequency."""
    prec = float(row.get("precision", 0))
    recall = float(row.get("recall", 0))
    pass_day = float(row.get("pass_per_day", 0))
    if prec < baseline_precision * 0.90:
        return -1.0
    freq_score = min(pass_day / target_pass_per_day, 1.0) if target_pass_per_day > 0 else 0.0
    prec_score = min(prec / baseline_precision, 1.0) if baseline_precision > 0 else prec
    return recall * 0.45 + freq_score * 0.30 + prec_score * 0.25


def select_best_live_rule(
    rows: list[dict],
    baseline_precision: float,
    min_pass_count: int = 5,
) -> dict | None:
    eligible = [
        r for r in rows
        if r.get("pass_count", 0) >= min_pass_count
        and float(r.get("precision", 0)) >= baseline_precision * 0.90
    ]
    if not eligible:
        eligible = sorted(rows, key=lambda x: x.get("f1", 0), reverse=True)
        return eligible[0] if eligible else None

    for r in eligible:
        r["live_score"] = round(live_selection_score(r, baseline_precision), 4)
    eligible.sort(key=lambda x: (x.get("live_score", 0), x.get("recall", 0)), reverse=True)
    return eligible[0]
