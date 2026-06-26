"""Cluster rules by activation features and empirical regime affinity."""

from __future__ import annotations

from collections import Counter

from scout_auto_os.engine.research.rule_portfolio.constants import CLUSTER_LABELS, FEATURE_CLUSTER_HINTS


def _expr_clusters(rule_expr: str) -> list[str]:
    hits: list[str] = []
    for feat, label in FEATURE_CLUSTER_HINTS.items():
        if feat in rule_expr:
            hits.append(label)
    return hits or ["mixed"]


def _empirical_cluster(profile: dict) -> str:
    regime_avg = profile.get("regime_avg_json") or {}
    if not regime_avg:
        return "mixed"
    best = max(regime_avg, key=regime_avg.get)
    mapping = {
        "bull": "bull_trend",
        "bear": "reversal",
        "sideway": "sideways",
        "recovery": "recovery",
        "crash": "reversal",
    }
    for prefix, label in mapping.items():
        if best.startswith(prefix):
            return label
    return "mixed"


def assign_cluster(profile: dict) -> dict:
    expr_labels = _expr_clusters(profile.get("rule_expr", ""))
    empirical = _empirical_cluster(profile)
    primary = Counter(expr_labels).most_common(1)[0][0] if expr_labels else "mixed"

    if empirical != "mixed" and empirical in expr_labels:
        cluster_id = empirical
    elif primary in CLUSTER_LABELS:
        cluster_id = primary
    else:
        cluster_id = empirical if empirical != "mixed" else primary

    members = sorted(set(expr_labels + [empirical]))
    return {
        "rule_id": profile["rule_id"],
        "rule_expr": profile["rule_expr"],
        "direction": profile["direction"],
        "cluster_id": cluster_id,
        "expr_clusters": "|".join(expr_labels),
        "empirical_cluster": empirical,
        "cluster_members": "|".join(m for m in members if m in CLUSTER_LABELS),
        "best_regime": profile.get("best_regime", "unknown"),
        "volatility_preference": profile.get("volatility_preference", "unknown"),
        "avg_return_2h": profile.get("avg_return_2h", 0),
        "trade_count": profile.get("trade_count", 0),
        "status_tags": profile.get("status_tags", ""),
    }


def summarize_clusters(cluster_rows: list[dict]) -> list[dict]:
    buckets: dict[str, list[dict]] = {}
    for row in cluster_rows:
        buckets.setdefault(row["cluster_id"], []).append(row)

    summary: list[dict] = []
    for cid, rows in sorted(buckets.items()):
        returns = [float(r["avg_return_2h"]) for r in rows if int(r.get("trade_count", 0)) > 0]
        summary.append({
            "cluster_id": cid,
            "rule_count": len(rows),
            "avg_return_mean": round(sum(returns) / len(returns), 4) if returns else 0.0,
            "top_rule_id": max(rows, key=lambda r: float(r.get("avg_return_2h", 0)))["rule_id"],
            "regime_affinity": Counter(r.get("best_regime", "unknown") for r in rows).most_common(1)[0][0],
        })
    return summary
