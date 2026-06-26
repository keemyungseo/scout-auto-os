"""Scan-level condition classification from existing features only."""

from __future__ import annotations

import statistics
from collections import defaultdict

from scout_auto_os.engine.research.adaptive_feature_weight.constants import CONDITION_DEFS


def _scan_aggregate(rows: list[dict], field: str, agg: str) -> float:
    vals = [float(r["x"].get(field, 0)) for r in rows]
    if not vals:
        return 0.0
    if agg == "mean":
        return statistics.mean(vals)
    return statistics.median(vals)


def _train_thresholds(train_rows: list[dict]) -> dict[str, float]:
    by_scan: dict[str, list[dict]] = defaultdict(list)
    for r in train_rows:
        by_scan[r["scan_kst"]].append(r)

    field_vals: dict[str, list[float]] = defaultdict(list)
    for rows in by_scan.values():
        for cid, spec in CONDITION_DEFS.items():
            if spec.get("threshold") is not None:
                continue
            if spec.get("extra") == "reversal":
                ret = _scan_aggregate(rows, spec["field"], spec["agg"])
                prev = _scan_aggregate(rows, "dna_1h_previous_return_pct", "median")
                field_vals["reversal"].append(1.0 if ret < 0 and prev > 0 else 0.0)
                continue
            field_vals[cid].append(_scan_aggregate(rows, spec["field"], spec["agg"]))

    thresholds: dict[str, float] = {}
    for key, vals in field_vals.items():
        if vals:
            thresholds[key] = statistics.median(vals)
    return thresholds


def classify_scan(rows: list[dict], thresholds: dict[str, float]) -> list[str]:
    tags: list[str] = []
    for cid, spec in CONDITION_DEFS.items():
        if spec.get("extra") == "reversal":
            ret = _scan_aggregate(rows, spec["field"], spec["agg"])
            prev = _scan_aggregate(rows, "dna_1h_previous_return_pct", "median")
            if ret < 0 and prev > 0:
                tags.append("reversal")
            continue

        val = _scan_aggregate(rows, spec["field"], spec["agg"])
        thr = spec.get("threshold")
        if thr is None:
            thr = thresholds.get(cid, val)

        if spec["op"] == "gte" and val >= thr:
            tags.append(cid)
        elif spec["op"] == "lte" and val <= thr:
            tags.append(cid)

    return tags or ["unclassified"]


def build_scan_condition_map(
    rows: list[dict],
    thresholds: dict[str, float],
) -> dict[str, list[str]]:
    by_scan: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_scan[r["scan_kst"]].append(r)
    return {scan: classify_scan(scan_rows, thresholds) for scan, scan_rows in by_scan.items()}


def primary_condition(tags: list[str]) -> str:
    priority = (
        "breakout", "compression", "high_volatility", "low_volatility",
        "volume_surge", "momentum", "strong_trend", "bull_leader",
        "bear_leader", "sideway", "reversal", "weak_trend", "volume_decline",
        "range_expansion", "unclassified",
    )
    for p in priority:
        if p in tags:
            return p
    return tags[0] if tags else "unclassified"
