"""Ranking dataset — labels from forward only."""

from __future__ import annotations

from scout_auto_os.engine.portfolio.rule_loader import PortfolioRules
from scout_auto_os.engine.research.directional.dna.formulas import ClusterFormula
from scout_auto_os.engine.research.directional.dna.validator import split_scans
from scout_auto_os.engine.research.execution_research.observation import compute_observation_features
from scout_auto_os.engine.research.formula_league_v2.annotate import (
    attach_scan_rank_context,
    label_winner_cohort,
)
from scout_auto_os.engine.research.formula_league_v2.constants import RANK_FEATURES
from scout_auto_os.engine.research.formula_league_v2.features import enrich_derived_features
from scout_auto_os.engine.research.formula_league_v2.metrics import enrich_forward_metrics, return_12h_from_klines
from scout_auto_os.engine.research.ranking_engine.constants import RELEVANCE_GRADES, SUCCESS_RETURN_PCT
from scout_auto_os.engine.research.ranking_engine.features import build_ranking_feature_row
from scout_auto_os.engine.research.zero_base.forward_eval import compute_forward_metrics

import scout_phase22_search_formula_evolution as p22


def prepare_annotated(by_scan: dict[str, list[dict]]):
    from scout_auto_os.engine.research.formula_league_v2.annotate import (
        annotate_universe,
        attach_base_scores,
        attach_scan_rank_context,
    )
    annotated, th, profile = annotate_universe(by_scan)
    attach_base_scores(annotated, profile)
    attach_scan_rank_context(annotated, list(RANK_FEATURES))
    train_flat = [r for rows in annotated.values() for r in rows]
    stats = p22.build_train_stats(train_flat, annotated, th)
    return annotated, th, stats


def _outcome_rank(rows: list[dict], symbol: str) -> int:
    ranked = sorted(rows, key=lambda r: -float(r.get("max_up_4h") or 0))
    for i, r in enumerate(ranked, 1):
        if r["symbol"] == symbol:
            return i
    return 99


def collect_ranking_dataset(
    annotated: dict[str, list[dict]],
    fwd: dict[tuple[str, str], list],
    rules: PortfolioRules,
    formulas: list[ClusterFormula],
    th,
    stats: dict,
    direction: str = "long",
) -> list[dict]:
    latest_scan = max(annotated.keys())

    out: list[dict] = []
    for scan, rows in annotated.items():
        label_winner_cohort(rows)
        for row in rows:
            enrich_derived_features(row)
            klines = fwd.get((scan, row["symbol"]), [])
            if klines:
                obs = compute_observation_features(klines, direction, row["features"])
                row["obs_features"] = obs
            else:
                row["obs_features"] = {}

            x = build_ranking_feature_row(
                row, rows, rules, formulas, scan, latest_scan, th, stats, direction,
            )
            metrics = {}
            if klines:
                entry = float(klines[0][1])
                metrics = enrich_forward_metrics(compute_forward_metrics(klines, entry), klines)
                metrics["return_12h"] = return_12h_from_klines(klines, entry)

            outcome_rank = _outcome_rank(rows, row["symbol"])
            rel = RELEVANCE_GRADES.get(outcome_rank, 0)
            ret2 = float(metrics.get("return_2h", 0)) if metrics else 0.0

            out.append({
                "scan_kst": scan,
                "symbol": row["symbol"],
                "direction": direction,
                "x": x,
                "outcome_rank": outcome_rank,
                "relevance": rel,
                "label_top1": 1 if outcome_rank == 1 else 0,
                "label_top2": 1 if outcome_rank <= 2 else 0,
                "label_top3": 1 if outcome_rank <= 3 else 0,
                "label_top5": 1 if outcome_rank <= 5 else 0,
                "label_success_2h": 1 if ret2 >= SUCCESS_RETURN_PCT else 0,
                "return_2h": ret2,
                "return_4h": float(metrics.get("return_4h", 0)) if metrics else 0.0,
                "return_6h": float(metrics.get("return_6h", 0)) if metrics else 0.0,
                "return_12h": float(metrics.get("return_12h", 0)) if metrics else 0.0,
                "max_drawdown_2h": float(metrics.get("max_drawdown_2h", 0)) if metrics else 0.0,
            })
    return out


def split_by_scans(rows: list[dict], train_ratio: float = 0.7) -> tuple[list[dict], list[dict]]:
    scans = sorted({r["scan_kst"] for r in rows})
    train_scans, blind_scans = split_scans(scans, train_ratio)
    train_set, blind_set = set(train_scans), set(blind_scans)
    train = [r for r in rows if r["scan_kst"] in train_set]
    blind = [r for r in rows if r["scan_kst"] in blind_set]
    return train, blind


def build_group_sizes(rows: list[dict]) -> list[int]:
    from collections import defaultdict
    counts: dict[str, int] = defaultdict(int)
    order: list[str] = []
    for r in rows:
        if r["scan_kst"] not in counts:
            order.append(r["scan_kst"])
        counts[r["scan_kst"]] += 1
    return [counts[s] for s in order]
