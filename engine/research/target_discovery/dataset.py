"""Target discovery dataset — same features as Ranking V1, extended label metrics."""

from __future__ import annotations

from scout_auto_os.engine.portfolio.rule_loader import PortfolioRules
from scout_auto_os.engine.research.directional.dna.formulas import ClusterFormula
from scout_auto_os.engine.research.directional.dna.validator import split_scans
from scout_auto_os.engine.research.execution_research.observation import compute_observation_features
from scout_auto_os.engine.research.formula_league_v2.annotate import label_winner_cohort
from scout_auto_os.engine.research.formula_league_v2.constants import RANK_FEATURES
from scout_auto_os.engine.research.formula_league_v2.features import enrich_derived_features
from scout_auto_os.engine.research.ranking_engine.dataset import prepare_annotated
from scout_auto_os.engine.research.ranking_engine.features import build_ranking_feature_row
from scout_auto_os.engine.research.target_discovery.constants import RELEVANCE_GRADES
from scout_auto_os.engine.research.target_discovery.label_builder import compute_label_metrics


def _baseline_outcome_rank(rows: list[dict], symbol: str) -> int:
    ranked = sorted(rows, key=lambda r: -float(r.get("max_up_4h") or 0))
    for i, r in enumerate(ranked, 1):
        if r["symbol"] == symbol:
            return i
    return 99


def collect_target_discovery_dataset(
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
            label_metrics: dict = {}
            if klines:
                entry = float(klines[0][1])
                label_metrics = compute_label_metrics(klines, entry)

            baseline_rank = _baseline_outcome_rank(rows, row["symbol"])
            baseline_rel = RELEVANCE_GRADES.get(baseline_rank, 0)
            ret2 = float(label_metrics.get("return_2h", 0))

            out.append({
                "scan_kst": scan,
                "symbol": row["symbol"],
                "direction": direction,
                "x": x,
                "max_up_4h": float(row.get("max_up_4h") or 0),
                "label_metrics": label_metrics,
                "baseline_outcome_rank": baseline_rank,
                "baseline_relevance": baseline_rel,
                "outcome_rank": baseline_rank,
                "relevance": baseline_rel,
                "label_top1": 1 if baseline_rank == 1 else 0,
                "label_top2": 1 if baseline_rank <= 2 else 0,
                "label_top3": 1 if baseline_rank <= 3 else 0,
                "label_top5": 1 if baseline_rank <= 5 else 0,
                "return_2h": ret2,
                "return_4h": float(label_metrics.get("return_4h", 0)),
                "return_6h": float(label_metrics.get("return_6h", 0)),
                "return_12h": float(label_metrics.get("return_12h", 0)),
                "max_drawdown_2h": float(label_metrics.get("max_drawdown_2h", 0)),
            })
    return out


def split_by_scans(rows: list[dict], train_ratio: float = 0.7) -> tuple[list[dict], list[dict]]:
    scans = sorted({r["scan_kst"] for r in rows})
    train_scans, blind_scans = split_scans(scans, train_ratio)
    train_set, blind_set = set(train_scans), set(blind_scans)
    train = [r for r in rows if r["scan_kst"] in train_set]
    blind = [r for r in rows if r["scan_kst"] in blind_set]
    return train, blind
