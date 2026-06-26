"""Build history database from seed bundle + incremental append."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from scout_auto_os.engine.portfolio.rule_loader import load_portfolio_rules
from scout_auto_os.engine.research.constitution_validation.long_calendar import (
    load_constitution_dataset,
    split_chronological,
)
from scout_auto_os.engine.research.constitution_validation.regime_validator import tag_scan_regimes
from scout_auto_os.engine.research.infrastructure.dataset_manager import HistoryDatabase
from scout_auto_os.engine.research.infrastructure.forward_labeler import build_forward_labels
from scout_auto_os.engine.research.infrastructure.scan_archiver import build_archive_from_dataset
from scout_auto_os.engine.research.ranking_engine.features import feature_matrix
from scout_auto_os.engine.research.zero_base.runner import load_candidates_jsonl, load_forward_klines


def import_seed_bundle(
    db: HistoryDatabase,
    candidates_path: Path,
    forward_path: Path,
    data_dir: Path,
    pkg_root: Path,
) -> dict:
    by_scan = load_candidates_jsonl(candidates_path)
    fwd = load_forward_klines(forward_path)
    dataset, calendar, feat_names = load_constitution_dataset(
        candidates_path, forward_path, data_dir, pkg_root,
    )
    train_rows, _ = split_chronological(dataset, 0.7)

    existing = db.all_scan_keys()
    new_dataset = [r for r in dataset if r["scan_kst"] not in existing]

    if new_dataset:
        archive_info = build_archive_from_dataset(
            new_dataset, feat_names, train_rows, by_scan, db,
        )
    else:
        archive_info = {"archived_candidates": 0, "scan_count": 0, "skipped": "all_present"}

    label_stats = apply_forward_labels(db, by_scan, fwd)
    return {
        "calendar": calendar,
        "feature_count": len(feat_names),
        "archive": archive_info,
        "labels": label_stats,
        "total_scans": db.scan_count(),
        "total_samples": db.sample_count(),
    }


def apply_forward_labels(
    db: HistoryDatabase,
    by_scan: dict[str, list[dict]],
    fwd: dict[tuple[str, str], list],
) -> dict:
    labeled = 0
    scans_done = 0
    for scan_kst, rows in by_scan.items():
        sym_labels = 0
        for row in rows:
            sym = row["symbol"]
            klines = fwd.get((scan_kst, sym), [])
            labels = build_forward_labels(klines)
            if labels:
                db.upsert_label(scan_kst, sym, labels)
                sym_labels += 1
                labeled += 1
        if sym_labels > 0:
            scans_done += 1
            raw = by_scan.get(scan_kst, [])
            regimes = tag_scan_regimes(raw, scan_kst) if raw else {}
            db.upsert_scan(
                scan_kst=scan_kst,
                scan_date=scan_kst[:10],
                symbol_count=len(rows),
                regimes=regimes,
                label_ready=sym_labels == len(rows),
            )
    return {"labeled_rows": labeled, "labeled_scans": scans_done}


def append_scan(
    db: HistoryDatabase,
    scan_kst: str,
    candidates: list[dict],
    features_by_symbol: dict[str, dict],
    scores_by_symbol: dict[str, float],
    regimes: dict[str, str],
) -> int:
    """Incremental API for future LIVE scan archival (research-only hook)."""
    ranked = sorted(scores_by_symbol.items(), key=lambda x: -x[1])
    rank_map = {sym: i + 1 for i, (sym, _) in enumerate(ranked)}
    db.upsert_scan(
        scan_kst=scan_kst,
        scan_date=scan_kst[:10],
        symbol_count=len(candidates),
        regimes=regimes,
        label_ready=False,
    )
    n = 0
    for row in candidates:
        sym = row["symbol"]
        db.upsert_candidate(
            scan_kst=scan_kst,
            symbol=sym,
            rank_pred=rank_map.get(sym, 99),
            pred_score=float(scores_by_symbol.get(sym, 0)),
            in_top50=rank_map.get(sym, 99) <= 50,
            features=features_by_symbol.get(sym, row.get("features", {})),
            max_up_4h=float(row.get("max_up_4h") or 0),
        )
        n += 1
    return n
