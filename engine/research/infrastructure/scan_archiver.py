"""Archive scan snapshots with frozen constitution scores."""

from __future__ import annotations

from collections import defaultdict

from scout_auto_os.engine.research.constitution_validation.regime_validator import tag_scan_regimes
from scout_auto_os.engine.research.constitution_validation.validator import train_frozen_constitution
from scout_auto_os.engine.research.infrastructure.constants import TOP_K_ARCHIVE
from scout_auto_os.engine.research.infrastructure.dataset_manager import HistoryDatabase
from scout_auto_os.engine.research.ranking_engine.models import predict_scores


def score_rows(rows: list[dict], feat_names: list[str], bundle) -> list[dict]:
    scored: list[dict] = []
    for r in rows:
        score = float(predict_scores(bundle, [r])[0])
        scored.append({**r, "pred_score": score})
    return scored


def rank_within_scans(rows: list[dict]) -> list[dict]:
    by_scan: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_scan[r["scan_kst"]].append(r)

    out: list[dict] = []
    for scan in sorted(by_scan):
        ranked = sorted(by_scan[scan], key=lambda x: -float(x.get("pred_score", 0)))
        for i, row in enumerate(ranked, 1):
            out.append({
                **row,
                "rank_pred": i,
                "in_top50": i <= TOP_K_ARCHIVE,
            })
    return out


def archive_scored_rows(
    db: HistoryDatabase,
    rows: list[dict],
    raw_by_scan: dict[str, list[dict]],
) -> int:
    archived = 0
    by_scan: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_scan[r["scan_kst"]].append(r)

    for scan_kst, chunk in by_scan.items():
        raw = raw_by_scan.get(scan_kst, [])
        regimes = tag_scan_regimes(raw, scan_kst) if raw else {}
        db.upsert_scan(
            scan_kst=scan_kst,
            scan_date=scan_kst[:10],
            symbol_count=len(chunk),
            regimes=regimes,
            label_ready=False,
        )
        for row in chunk:
            features = row.get("x") or {}
            db.upsert_candidate(
                scan_kst=scan_kst,
                symbol=row["symbol"],
                rank_pred=int(row.get("rank_pred", 99)),
                pred_score=float(row.get("pred_score", 0)),
                in_top50=bool(row.get("in_top50")),
                features=features,
                max_up_4h=float(row.get("max_up_4h") or 0),
            )
            archived += 1
    return archived


def build_archive_from_dataset(
    dataset: list[dict],
    feat_names: list[str],
    train_rows: list[dict],
    raw_by_scan: dict[str, list[dict]],
    db: HistoryDatabase,
) -> dict:
    bundle = train_frozen_constitution(train_rows, feat_names)
    scored = score_rows(dataset, feat_names, bundle)
    ranked = rank_within_scans(scored)
    n = archive_scored_rows(db, ranked, raw_by_scan)
    return {
        "archived_candidates": n,
        "scan_count": len({r["scan_kst"] for r in ranked}),
        "model": "catboost_ranker_frozen",
        "label": "return_minus_dd",
    }
