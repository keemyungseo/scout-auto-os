"""Entry-time predictability of Trade Type."""

from __future__ import annotations

import statistics

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

ENTRY_FEATURE_KEYS = (
    "entry_score",
    "direction_long",
    "h4_score",
    "5m_compression",
    "15m_current_return_pct",
    "1h_current_return_pct",
    "15m_current_volume_ratio",
    "5m_momentum",
)


def _entry_vector(rec) -> list[float]:
    f = rec.features or {}
    return [
        float(rec.entry_score),
        1.0 if rec.direction == "long" else 0.0,
        float(f.get("h4_score", 0)),
        float(f.get("5m_compression", 0)),
        float(f.get("15m_current_return_pct", 0)),
        float(f.get("1h_current_return_pct", 0)),
        float(f.get("15m_current_volume_ratio", 1)),
        float(f.get("5m_momentum", 0)),
    ]


def evaluate_entry_predictability(
    records: list,
    cluster_labels: list[int],
) -> dict:
    if len(records) < 10 or len(set(cluster_labels)) < 2:
        return {
            "predictable": False,
            "cv_accuracy": 0.0,
            "random_baseline": round(1.0 / max(len(set(cluster_labels)), 1), 4),
            "n_samples": len(records),
            "method": "insufficient_sample",
        }

    X = np.array([_entry_vector(r) for r in records], dtype=float)
    y = np.array(cluster_labels, dtype=int)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    n_classes = len(set(cluster_labels))
    random_baseline = 1.0 / n_classes

    model = LogisticRegression(max_iter=500, random_state=42)
    try:
        scores = cross_val_score(model, Xs, y, cv=min(5, len(records) // 3), scoring="accuracy")
        cv_acc = float(statistics.mean(scores))
    except Exception:
        cv_acc = random_baseline

    pattern_map: dict[str, dict[int, int]] = {}
    for rec, label in zip(records, cluster_labels):
        pat = rec.live_pattern or "unknown"
        pattern_map.setdefault(pat, {})
        pattern_map[pat][label] = pattern_map[pat].get(label, 0) + 1

    pattern_rows = []
    for pat, counts in pattern_map.items():
        total = sum(counts.values())
        dominant = max(counts, key=counts.get)
        pattern_rows.append({
            "live_pattern": pat,
            "dominant_type": f"TYPE_{dominant}",
            "concentration_pct": round(counts[dominant] / total * 100, 1),
            "sample_count": total,
        })

    return {
        "predictable": cv_acc > random_baseline + 0.1,
        "cv_accuracy": round(cv_acc, 4),
        "random_baseline": round(random_baseline, 4),
        "lift_vs_random": round(cv_acc - random_baseline, 4),
        "n_samples": len(records),
        "n_types": n_classes,
        "method": "logistic_regression_cv",
        "pattern_dominance": pattern_rows,
    }
