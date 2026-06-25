"""Feature importance and success/fail DNA analysis."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict


def _mean(vals: list[float]) -> float:
    return statistics.mean(vals) if vals else 0.0


def _std(vals: list[float]) -> float:
    return statistics.pstdev(vals) if len(vals) > 1 else 0.0


def welch_p_approx(a: list[float], b: list[float]) -> float:
    if len(a) < 5 or len(b) < 5:
        return 1.0
    ma, mb = _mean(a), _mean(b)
    sa, sb = _std(a), _std(b)
    na, nb = len(a), len(b)
    se = math.sqrt((sa * sa / na) + (sb * sb / nb)) or 1e-9
    z = abs(ma - mb) / se
    return round(2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2)))), 4)


def analyze_feature_importance(
    samples: list[dict],
    feature_keys: list[str],
) -> list[dict]:
    success = [s for s in samples if s["success"]]
    fail = [s for s in samples if not s["success"]]
    if not success or not fail:
        return []

    rows: list[dict] = []
    for key in feature_keys:
        sv = [float(s["features"].get(key, 0)) for s in success]
        fv = [float(s["features"].get(key, 0)) for s in fail]
        delta = _mean(sv) - _mean(fv)
        pooled = math.sqrt((_std(sv) ** 2 + _std(fv) ** 2) / 2) or 1e-9
        effect = delta / pooled
        p = welch_p_approx(sv, fv)
        rows.append({
            "feature": key,
            "success_mean": round(_mean(sv), 4),
            "fail_mean": round(_mean(fv), 4),
            "delta": round(delta, 4),
            "effect_size": round(effect, 4),
            "p_approx": p,
            "significant": p < 0.05,
        })
    rows.sort(key=lambda x: abs(x["effect_size"]), reverse=True)
    for i, row in enumerate(rows, 1):
        row["importance_rank"] = i
    return rows


def pattern_dna_summary(engine: str, samples: list[dict], importance: list[dict]) -> dict:
    success = sum(1 for s in samples if s["success"])
    return {
        "engine": engine,
        "sample_count": len(samples),
        "success_count": success,
        "fail_count": len(samples) - success,
        "success_rate": round(success / len(samples) * 100, 2) if samples else 0,
        "top_features": [r["feature"] for r in importance[:10]],
        "significant_features": sum(1 for r in importance if r.get("significant")),
    }
