"""Short constitution dataset — direction=short features, short labels."""

from __future__ import annotations

from scout_auto_os.engine.portfolio.rule_loader import PortfolioRules
from scout_auto_os.engine.research.directional.dna.formulas import ClusterFormula
from scout_auto_os.engine.research.directional.dna.validator import split_scans
from scout_auto_os.engine.research.execution_research.observation import compute_observation_features
from scout_auto_os.engine.research.formula_league_v2.annotate import label_winner_cohort
from scout_auto_os.engine.research.formula_league_v2.constants import RANK_FEATURES
from scout_auto_os.engine.research.formula_league_v2.features import enrich_derived_features
from scout_auto_os.engine.research.ranking_engine.features import build_ranking_feature_row
from scout_auto_os.engine.research.short_constitution.constants import DIRECTION, RELEVANCE_GRADES
from scout_auto_os.engine.research.short_constitution.label_builder import compute_short_label_metrics


def prepare_annotated(by_scan: dict[str, list[dict]]):
    from scout_auto_os.engine.research.formula_league_v2.annotate import (
        annotate_universe,
        attach_base_scores,
        attach_scan_rank_context,
    )
    import scout_phase22_search_formula_evolution as p22

    annotated, th, profile = annotate_universe(by_scan)
    attach_base_scores(annotated, profile)
    attach_scan_rank_context(annotated, list(RANK_FEATURES))
    train_flat = [r for rows in annotated.values() for r in rows]
    stats = p22.build_train_stats(train_flat, annotated, th)
    return annotated, th, stats


def _short_outcome_rank(rows: list[dict], symbol: str) -> int:
    ranked = sorted(
        rows,
        key=lambda r: (-float((r.get("short_label_metrics") or {}).get("max_down_2h", 0)), r["symbol"]),
    )
    for i, r in enumerate(ranked, 1):
        if r["symbol"] == symbol:
            return i
    return 99


def collect_short_dataset(
    annotated: dict[str, list[dict]],
    fwd: dict[tuple[str, str], list],
    rules: PortfolioRules,
    formulas: list[ClusterFormula],
    th,
    stats: dict,
) -> list[dict]:
    latest_scan = max(annotated.keys())
    out: list[dict] = []

    for scan, rows in annotated.items():
        label_winner_cohort(rows)
        metrics_by_sym: dict[str, dict] = {}
        for row in rows:
            enrich_derived_features(row)
            klines = fwd.get((scan, row["symbol"]), [])
            if klines:
                obs = compute_observation_features(klines, DIRECTION, row["features"])
                row["obs_features"] = obs
                entry = float(klines[0][1])
                metrics_by_sym[row["symbol"]] = compute_short_label_metrics(klines, entry)
            else:
                row["obs_features"] = {}
                metrics_by_sym[row["symbol"]] = {}

        for row in rows:
            lm = metrics_by_sym.get(row["symbol"], {})
            x = build_ranking_feature_row(
                row, rows, rules, formulas, scan, latest_scan, th, stats, DIRECTION,
            )
            sr2 = float(lm.get("short_return_2h", 0))
            rank = _short_outcome_rank(
                [{**row, "short_label_metrics": lm} for row in rows],
                row["symbol"],
            )
            rel = RELEVANCE_GRADES.get(rank, 0)

            out.append({
                "scan_kst": scan,
                "symbol": row["symbol"],
                "direction": DIRECTION,
                "x": x,
                "max_up_4h": float(row.get("max_up_4h") or 0),
                "short_label_metrics": lm,
                "baseline_outcome_rank": rank,
                "baseline_relevance": rel,
                "outcome_rank": rank,
                "relevance": rel,
                "label_top3": 1 if rank <= 3 else 0,
                "return_2h": sr2,
                "short_return_2h": sr2,
            })
    return out


def split_by_scans(rows: list[dict], train_ratio: float = 0.7) -> tuple[list[dict], list[dict]]:
    scans = sorted({r["scan_kst"] for r in rows})
    train_scans, blind_scans = split_scans(scans, train_ratio)
    train_set, blind_set = set(train_scans), set(blind_scans)
    return [r for r in rows if r["scan_kst"] in train_set], [r for r in rows if r["scan_kst"] in blind_set]
