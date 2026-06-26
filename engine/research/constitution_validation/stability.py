"""Performance drift and feature-importance stability."""

from __future__ import annotations

import statistics
from collections import defaultdict

from scout_auto_os.engine.research.ranking_engine.importance import gain_importance, merge_importance, shap_rows
from scout_auto_os.engine.research.ranking_engine.models import RankingModelBundle
from scout_auto_os.engine.research.target_discovery.candidate_generator import generate_label_candidates
from scout_auto_os.engine.research.target_discovery.label_builder import apply_label
from scout_auto_os.engine.research.constitution_validation.validator import train_frozen_constitution


def _label_spec():
    return next(s for s in generate_label_candidates() if s.label_id == "return_minus_dd")


def performance_drift(picks: list[dict]) -> list[dict]:
    """Weekly avg 2h trend on blind picks."""
    buckets: dict[str, list[float]] = defaultdict(list)
    for p in picks:
        buckets[p["scan_kst"][:10]].append(float(p.get("return_2h", 0)))

    weeks = sorted(buckets)
    out: list[dict] = []
    prev_avg = None
    for w in weeks:
        rets = buckets[w]
        avg = round(statistics.mean(rets), 4) if rets else 0.0
        drift = round(avg - prev_avg, 4) if prev_avg is not None else 0.0
        out.append({
            "week": w,
            "trade_count": len(rets),
            "avg_return_2h": avg,
            "week_over_week_drift": drift,
        })
        prev_avg = avg
    return out


def drift_summary(drift_rows: list[dict]) -> dict:
    if len(drift_rows) < 2:
        return {"drift_detected": False, "reason": "insufficient_weeks"}
    drifts = [r["week_over_week_drift"] for r in drift_rows[1:]]
    avgs = [r["avg_return_2h"] for r in drift_rows]
    trend = avgs[-1] - avgs[0]
    volatile = statistics.pstdev(avgs) if len(avgs) > 1 else 0.0
    return {
        "drift_detected": abs(trend) > 1.0 or volatile > 2.0,
        "calendar_trend_pct": round(trend / abs(avgs[0] or 0.01) * 100, 2),
        "weekly_volatility": round(volatile, 4),
        "mean_wow_drift": round(statistics.mean(drifts), 4),
        "weeks": len(drift_rows),
    }


def importance_stability(
    train_rows: list[dict],
    feat_names: list[str],
) -> tuple[list[dict], dict]:
    """Compare top features: first half vs second half of train calendar."""
    scans = sorted({r["scan_kst"] for r in train_rows})
    mid = len(scans) // 2
    mid_scan = scans[mid]
    first = [r for r in train_rows if r["scan_kst"] <= mid_scan]
    second = [r for r in train_rows if r["scan_kst"] > mid_scan]
    if len(first) < 50 or len(second) < 50:
        return [], {"stable": False, "reason": "insufficient_train_split"}

    b1 = train_frozen_constitution(first, feat_names)
    b2 = train_frozen_constitution(second, feat_names)
    spec = _label_spec()
    imp1 = merge_importance(gain_importance(b1), [], shap_rows(b1, apply_label(first[:300], spec)))
    imp2 = merge_importance(gain_importance(b2), [], shap_rows(b2, apply_label(second[:300], spec)))

    top1 = {r["feature"] for r in sorted(imp1, key=lambda x: -x.get("combined_score", 0))[:15]}
    top2 = {r["feature"] for r in sorted(imp2, key=lambda x: -x.get("combined_score", 0))[:15]}
    overlap = len(top1 & top2)

    rows: list[dict] = []
    fmap1 = {r["feature"]: float(r.get("combined_score", 0)) for r in imp1}
    fmap2 = {r["feature"]: float(r.get("combined_score", 0)) for r in imp2}
    for f in top1 | top2:
        rows.append({
            "feature": f,
            "first_half_importance": round(fmap1.get(f, 0), 6),
            "second_half_importance": round(fmap2.get(f, 0), 6),
            "delta": round(fmap2.get(f, 0) - fmap1.get(f, 0), 6),
        })
    rows.sort(key=lambda x: -abs(x["delta"]))

    summary = {
        "top15_overlap": overlap,
        "top15_overlap_pct": round(overlap / 15 * 100, 2),
        "stable": overlap >= 8,
        "first_half_scans": mid,
        "second_half_scans": len(scans) - mid,
    }
    return rows, summary
