"""Probability calibration for top3 classifier."""

from __future__ import annotations

import statistics
from collections import defaultdict


def calibration_bins(rows: list[dict], prob_key: str = "pred_prob", label_key: str = "label_top3", n_bins: int = 10) -> list[dict]:
    if not rows:
        return []
    sorted_rows = sorted(rows, key=lambda r: -float(r.get(prob_key, 0)))
    bin_size = max(1, len(sorted_rows) // n_bins)
    out: list[dict] = []
    for b in range(n_bins):
        chunk = sorted_rows[b * bin_size:(b + 1) * bin_size]
        if not chunk:
            continue
        probs = [float(r[prob_key]) for r in chunk]
        actuals = [int(r[label_key]) for r in chunk]
        out.append({
            "bin": b + 1,
            "count": len(chunk),
            "predicted_probability": round(statistics.mean(probs), 4),
            "actual_probability": round(statistics.mean(actuals), 4),
            "gap": round(statistics.mean(probs) - statistics.mean(actuals), 4),
        })
    return out
