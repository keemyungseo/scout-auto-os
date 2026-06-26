"""Score candidates within a scan using a SearchFormula."""

from __future__ import annotations

import scout_phase22_search_formula_evolution as p22
import scout_phase23_search_formula_league as p23

from scout_auto_os.engine.research.formula_league_v2.constants import BASELINE_FORMULA_ID
from scout_auto_os.engine.research.formula_league_v2.formula import SearchFormula


def score_a6_baseline(row: dict, peers: list[dict], base: float, th, stats: dict) -> float:
    return p23.formula_scores_a6(row, peers, base, th, stats)["A6"]


def score_row(
    row: dict,
    peers: list[dict],
    formula: SearchFormula,
    base: float,
    th=None,
    stats: dict | None = None,
) -> float:
    if formula.formula_id == BASELINE_FORMULA_ID or formula.kind == "baseline":
        return score_a6_baseline(row, peers, base, th, stats or {})

    if formula.kind == "linear":
        feat_score = sum(
            float(row["features"].get(feat, 0)) * w
            for feat, w in formula.linear_terms
        )
        pct_bonus = sum(
            p22.within_scan_pct(row, peers, feat) * w
            for feat, w in formula.linear_terms
            if feat in row["features"]
        )
        return base * 0.3 + feat_score + pct_bonus

    if formula.root is None:
        return base

    passed = formula.root.evaluate(row["features"], row.get("ctx"))
    tie = p22.within_scan_pct(row, peers, "1h_current_return_pct")
    return base + (formula.pass_bonus if passed else 0.0) + tie * 5.0


def rank_scan(
    rows: list[dict],
    formula: SearchFormula,
    th,
    stats: dict,
    top_k: int = 5,
) -> list[dict]:
    scored = []
    for r in rows:
        base = float(r.get("base_score", 0))
        s = score_row(r, rows, formula, base, th, stats)
        scored.append({**r, "formula_score": s})
    scored.sort(key=lambda x: (-x["formula_score"], -float(x["features"].get("1h_current_return_pct", 0))))
    return scored[:top_k]
