"""
A6 score formula core (frozen A6 = Base A + A2 + A5 bonuses).
Original sources:
  - scout_phase22_search_formula_evolution.py
  - scout_phase23_search_formula_league.py (formula_scores_a6)
"""

from __future__ import annotations

import math
import statistics

from export_a6_core_for_hong.a6_common import g
from export_a6_core_for_hong.a6_state_core import Thresholds, winner_loser_sets

EXPANSION_METRICS = (
    "1h_current_return_pct",
    "1h_current_range_pct",
    "2h_current_return_pct",
    "2h_current_range_pct",
    "2h_current_ma20_distance_pct",
    "30m_current_return_pct",
    "15m_current_volume_ratio",
    "5m_range_energy",
)


def within_scan_pct(row: dict, peers: list[dict], key: str) -> float:
    # Original: scout_phase22_search_formula_evolution.within_scan_pct
    v = g(row["features"], key)
    vals = [g(p["features"], key) for p in peers]
    if not vals:
        return 0.5
    return sum(1 for x in vals if x <= v) / len(vals)


def bonus_a3_raw(row: dict, peers: list[dict], th: Thresholds, stats: dict) -> float:
    # Original: scout_phase22_search_formula_evolution.bonus_a3_raw
    f = row["features"]
    release = g(f, "5m_release") > 0 or row.get("states", {}).get("5m") == "Release"
    vol_pct = within_scan_pct(row, peers, "15m_current_volume_ratio")
    vol_state = row.get("states", {}).get("15m") in ("VolumeSupport", "Expansion")
    vol_ok = vol_state or g(f, "15m_current_volume_ratio") >= stats.get("vol_support_p50", 1.0)
    if release and vol_ok:
        return 0.5 + 0.5 * vol_pct
    return 0.25 * vol_pct if release or vol_ok else 0.0


def bonus_a4_raw(row: dict, stats: dict) -> float:
    # Original: scout_phase22_search_formula_evolution.bonus_a4_raw
    v = g(row["features"], "5m_compression")
    wm, ws = stats["winner_comp_mean"], stats["winner_comp_std"]
    z = abs(v - wm) / ws
    return math.exp(-0.5 * z * z)


def bonus_a5_raw(row: dict, peers: list[dict], stats: dict) -> float:
    # Original: scout_phase22_search_formula_evolution.bonus_a5_raw
    mw = stats["expansion_metric_w"]
    score = 0.0
    for m, w in mw.items():
        score += w * within_scan_pct(row, peers, m)
    return score


def train_feature_ig(train: list[dict], by_scan_train: dict[str, list[dict]], fn) -> float:
    # Original: scout_phase22_search_formula_evolution.train_feature_ig
    top2 = set()
    for rows in by_scan_train.values():
        if len(rows) >= 2:
            top2.add((rows[0]["scan_kst"], rows[0]["symbol"]))
            top2.add((rows[1]["scan_kst"], rows[1]["symbol"]))
    pos, neg = [], []
    for scan, rows in by_scan_train.items():
        if len(rows) < 4:
            continue
        for r in rows:
            val = fn(r, rows)
            if (r["scan_kst"], r["symbol"]) in top2:
                pos.append(val)
            else:
                neg.append(val)
    if not pos or not neg:
        return 0.0
    pooled = statistics.pstdev(pos + neg) or 1.0
    return max((statistics.mean(pos) - statistics.mean(neg)) / pooled, 0.0)


def build_train_stats(train: list[dict], by_scan_train: dict[str, list[dict]], th: Thresholds) -> dict:
    # Original: scout_phase22_search_formula_evolution.build_train_stats
    w_train, _ = winner_loser_sets(by_scan_train)
    w_feats = [r["features"] for r in w_train]
    comp_vals = [g(f, "5m_compression") for f in w_feats]
    wm = statistics.mean(comp_vals) if comp_vals else 0.0
    ws = statistics.pstdev(comp_vals) if len(comp_vals) > 1 else 1.0

    metric_ig: dict[str, float] = {}
    top2 = set()
    for rows in by_scan_train.values():
        if len(rows) >= 2:
            top2.add((rows[0]["scan_kst"], rows[0]["symbol"]))
            top2.add((rows[1]["scan_kst"], rows[1]["symbol"]))
    for m in EXPANSION_METRICS:
        pos = [g(r["features"], m) for r in train if (r["scan_kst"], r["symbol"]) in top2]
        neg = [g(r["features"], m) for r in train if (r["scan_kst"], r["symbol"]) not in top2]
        if pos and neg:
            sd = statistics.pstdev([g(r["features"], m) for r in train]) or 1.0
            metric_ig[m] = max(abs(statistics.mean(pos) - statistics.mean(neg)) / sd, 0.0)
        else:
            metric_ig[m] = 0.0
    ig_sum = sum(metric_ig.values()) or 1.0
    metric_w = {k: v / ig_sum for k, v in metric_ig.items()}

    from export_a6_core_for_hong.a6_common import percentile

    return {
        "winner_comp_mean": wm,
        "winner_comp_std": ws,
        "vol_support_p50": percentile([g(f, "15m_current_volume_ratio") for f in w_feats], 0.5) if w_feats else 1.0,
        "expansion_metric_w": metric_w,
        "ig_a1": train_feature_ig(train, by_scan_train, lambda r, p: within_scan_pct(r, p, "2h_current_ma20_distance_pct")),
        "ig_a2": train_feature_ig(train, by_scan_train, lambda r, p: within_scan_pct(r, p, "1h_current_range_pct")),
        "ig_a3": train_feature_ig(train, by_scan_train, lambda r, p: bonus_a3_raw(r, p, th, {"vol_support_p50": percentile([g(x["features"], "15m_current_volume_ratio") for x in w_train], 0.5) if w_train else 1.0})),
        "ig_a4": train_feature_ig(train, by_scan_train, lambda r, p: bonus_a4_raw(r, {"winner_comp_mean": wm, "winner_comp_std": ws})),
        "ig_a5": train_feature_ig(train, by_scan_train, lambda r, p: bonus_a5_raw(r, p, {"expansion_metric_w": metric_w})),
    }


def formula_scores(row: dict, peers: list[dict], base: float, th: Thresholds, stats: dict) -> dict[str, float]:
    # Original: scout_phase22_search_formula_evolution.formula_scores
    b1 = within_scan_pct(row, peers, "2h_current_ma20_distance_pct")
    b2 = within_scan_pct(row, peers, "1h_current_range_pct")
    b3 = bonus_a3_raw(row, peers, th, stats)
    b4 = bonus_a4_raw(row, stats)
    b5 = bonus_a5_raw(row, peers, stats)
    return {
        "A": base,
        "A1": base + stats["ig_a1"] * b1,
        "A2": base + stats["ig_a2"] * b2,
        "A3": base + stats["ig_a3"] * b3,
        "A4": base + stats["ig_a4"] * b4,
        "A5": base + stats["ig_a5"] * b5,
    }


def formula_scores_a6(row: dict, peers: list[dict], base: float, th: Thresholds, stats: dict) -> dict[str, float]:
    # Original: scout_phase23_search_formula_league.formula_scores_a6
    scores = formula_scores(row, peers, base, th, stats)
    b2 = within_scan_pct(row, peers, "1h_current_range_pct")
    b5 = bonus_a5_raw(row, peers, stats)
    scores["A6"] = base + stats["ig_a2"] * b2 + stats["ig_a5"] * b5
    return scores


def score_candidate_a6(row: dict, peers: list[dict], profile: dict, th: Thresholds, stats: dict) -> float:
    """Convenience: return frozen A6 score for one candidate."""
    from export_a6_core_for_hong.a6_state_core import state_match_score

    base = state_match_score(row["states"], row["transitions"], profile)
    return formula_scores_a6(row, peers, base, th, stats)["A6"]
