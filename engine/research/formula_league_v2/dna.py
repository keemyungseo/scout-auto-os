"""Search Formula DNA analysis — common features in survivors."""

from __future__ import annotations

import re
from collections import Counter

from scout_auto_os.engine.research.formula_league_v2.constants import BASELINE_FORMULA_ID


def _extract_features(expr: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z0-9_]+", expr)
    stop = {
        "AND", "OR", "NOT", "linear", "rank", "gte", "lte", "base", "A6", "ig_a2", "ig_a5",
        "SF", "ATOM", "AND3", "ORAND", "LIN", "LIN2", "SR", "SD", "SW", "RNK",
    }
    return [
        t for t in tokens
        if t not in stop and not t.replace(".", "").replace("m", "").isdigit()
        and ("_" in t or t.startswith("derived"))
    ]


def _extract_thresholds(expr: str) -> list[str]:
    return re.findall(r">=?\s*([-\d.]+)|<=\s*([-\d.]+)", expr)


def analyze_formula_dna(
    survivor_rows: list[dict],
    formula_exprs: dict[str, str],
) -> tuple[list[dict], list[dict]]:
    survivors = [r for r in survivor_rows if r.get("survived") and r["formula_id"] != BASELINE_FORMULA_ID]
    feature_counter: Counter = Counter()
    combo_counter: Counter = Counter()
    threshold_counter: Counter = Counter()

    for row in survivors:
        expr = formula_exprs.get(row["formula_id"], "")
        feats = _extract_features(expr)
        feature_counter.update(feats)
        if " AND " in expr:
            combo_counter["AND"] += 1
        if " OR " in expr:
            combo_counter["OR"] += 1
        if "NOT" in expr:
            combo_counter["NOT"] += 1
        if expr.startswith("linear"):
            combo_counter["LINEAR"] += 1
        for thr in _extract_thresholds(expr):
            for t in thr:
                if t:
                    threshold_counter[t] += 1

    dna_rows: list[dict] = []
    total = len(survivors) or 1
    for feat, cnt in feature_counter.most_common(40):
        dna_rows.append({
            "dna_type": "feature",
            "token": feat,
            "count": cnt,
            "pct_of_survivors": round(cnt / total * 100, 2),
        })
    for combo, cnt in combo_counter.most_common():
        dna_rows.append({
            "dna_type": "combo",
            "token": combo,
            "count": cnt,
            "pct_of_survivors": round(cnt / total * 100, 2),
        })

    importance_rows: list[dict] = []
    for feat, cnt in feature_counter.most_common(30):
        importance_rows.append({
            "feature": feat,
            "survivor_appearances": cnt,
            "importance_rank": len(importance_rows) + 1,
            "importance_score": round(cnt / total, 4),
        })

    return dna_rows, importance_rows
