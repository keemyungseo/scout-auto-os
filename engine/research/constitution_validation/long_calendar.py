"""Longest available calendar loader and metadata."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from scout_auto_os.engine.portfolio.rule_loader import load_portfolio_rules
from scout_auto_os.engine.research.directional.prediction.loader import load_formulas, resolve_formulas_path
from scout_auto_os.engine.research.ranking_engine.features import feature_matrix
from scout_auto_os.engine.research.target_discovery.dataset import collect_target_discovery_dataset
from scout_auto_os.engine.research.ranking_engine.dataset import prepare_annotated
from scout_auto_os.engine.research.zero_base.runner import load_candidates_jsonl, load_forward_klines


def _parse_scan(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def load_constitution_dataset(
    candidates_path: Path,
    forward_path: Path,
    data_dir: Path,
    pkg_root: Path,
) -> tuple[list[dict], dict, list[str]]:
    by_scan = load_candidates_jsonl(candidates_path)
    fwd = load_forward_klines(forward_path)
    rules = load_portfolio_rules(data_dir, pkg_root)
    formulas = load_formulas(resolve_formulas_path(data_dir, pkg_root))

    annotated, th, stats = prepare_annotated(by_scan)
    dataset = collect_target_discovery_dataset(annotated, fwd, rules, formulas, th, stats)
    feat_names, _ = feature_matrix(dataset)

    scans = sorted({r["scan_kst"] for r in dataset})
    if not scans:
        meta = {"calendar_days": 0, "scan_count": 0, "sample_count": 0}
        return dataset, meta, feat_names

    start = _parse_scan(scans[0])
    end = _parse_scan(scans[-1])
    calendar_days = (end.date() - start.date()).days + 1

    meta = {
        "start_date": scans[0][:10],
        "end_date": scans[-1][:10],
        "calendar_days": calendar_days,
        "scan_count": len(scans),
        "sample_count": len(dataset),
        "feature_count": len(feat_names),
        "symbols_per_scan_avg": round(len(dataset) / max(len(scans), 1), 2),
        "target_calendar_days": 90,
        "meets_3month_target": calendar_days >= 90,
        "data_source": str(candidates_path),
        "constitution": {
            "features": "ranking_engine_v1",
            "model": "catboost_ranker",
            "label": "return_minus_dd",
        },
    }
    return dataset, meta, feat_names


def split_chronological(rows: list[dict], train_ratio: float) -> tuple[list[dict], list[dict]]:
    scans = sorted({r["scan_kst"] for r in rows})
    cut = max(1, int(len(scans) * train_ratio))
    train_set = set(scans[:cut])
    blind_set = set(scans[cut:])
    train = [r for r in rows if r["scan_kst"] in train_set]
    blind = [r for r in rows if r["scan_kst"] in blind_set]
    return train, blind
