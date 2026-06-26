"""Blind evaluation of SearchFormula picks with forward outcomes."""

from __future__ import annotations

from scout_auto_os.engine.research.formula_league_v2.constants import TOP_K
from scout_auto_os.engine.research.formula_league_v2.formula import SearchFormula
from scout_auto_os.engine.research.formula_league_v2.metrics import aggregate_formula_metrics, enrich_forward_metrics
from scout_auto_os.engine.research.formula_league_v2.scorer import rank_scan
from scout_auto_os.engine.research.zero_base.forward_eval import compute_forward_metrics


def evaluate_formula_on_scans(
    formula: SearchFormula,
    annotated: dict[str, list[dict]],
    scans: list[str],
    fwd: dict[tuple[str, str], list],
    th,
    stats: dict,
    top_k: int = TOP_K,
) -> tuple[list[dict], dict]:
    samples: list[dict] = []
    scans_used: set[str] = set()

    for scan in scans:
        rows = annotated.get(scan, [])
        if len(rows) < top_k:
            continue
        winners = {r["symbol"] for r in sorted(rows, key=lambda x: -float(x.get("max_up_4h") or 0))[:3]}
        picks = rank_scan(rows, formula, th, stats, top_k)
        scans_used.add(scan)
        for p in picks:
            key = (scan, p["symbol"])
            klines = fwd.get(key, [])
            metrics = enrich_forward_metrics(compute_forward_metrics(klines), klines)
            if not metrics:
                continue
            samples.append({
                "formula_id": formula.formula_id,
                "scan_kst": scan,
                "symbol": p["symbol"],
                "formula_score": p.get("formula_score"),
                "outcome_rank": p.get("outcome_rank"),
                "in_top3_outcome": p["symbol"] in winners,
                **metrics,
            })

    agg = aggregate_formula_metrics(samples, scans_used, len(scans))
    agg["formula_id"] = formula.formula_id
    agg["formula_expr"] = formula.formula_expr
    agg["kind"] = formula.kind
    agg["source"] = formula.source
    return samples, agg
