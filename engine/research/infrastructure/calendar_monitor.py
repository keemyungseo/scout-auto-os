"""Calendar coverage and validation readiness monitor."""

from __future__ import annotations

from datetime import datetime, timedelta

from scout_auto_os.engine.research.infrastructure.constants import (
    CALENDAR_WINDOWS,
    LIVE_VALIDATION_TARGET_DAYS,
    MIN_REGIME_SCANS,
    MIN_SAMPLES_PER_VALIDATION,
    VALIDATION_TARGET_DAYS,
)
from scout_auto_os.engine.research.infrastructure.dataset_manager import HistoryDatabase


def calendar_status(db: HistoryDatabase) -> list[dict]:
    days_total = db.calendar_days()
    start, end = db.date_range()
    rows: list[dict] = []

    for window in CALENDAR_WINDOWS:
        scans = db.scans_in_window(window)
        ready = days_total >= window and scans >= 20
        rows.append({
            "window_days": window,
            "calendar_days_available": min(days_total, window),
            "scan_count": scans,
            "validation_ready": ready,
            "coverage_pct": round(min(days_total, window) / window * 100, 2),
            "start_date": start,
            "end_date": end,
        })
    return rows


def dataset_status(db: HistoryDatabase, version: str) -> dict:
    start, end = db.date_range()
    days = db.calendar_days()
    return {
        "dataset_version": version,
        "start_date": start,
        "end_date": end,
        "calendar_days": days,
        "scan_count": db.scan_count(),
        "sample_count": db.sample_count(),
        "labeled_count": db.labeled_count(),
        "label_coverage_pct": round(db.labeled_count() / max(db.sample_count(), 1) * 100, 2),
        "validation_90d_ready": days >= VALIDATION_TARGET_DAYS,
        "live_validation_ready": days >= LIVE_VALIDATION_TARGET_DAYS and db.labeled_count() >= MIN_SAMPLES_PER_VALIDATION,
        "days_to_90d": max(0, VALIDATION_TARGET_DAYS - days),
        "days_to_live_validation": max(0, LIVE_VALIDATION_TARGET_DAYS - days),
    }


def coverage_report(db: HistoryDatabase) -> list[dict]:
    rows: list[dict] = []
    regime_maps = {
        "bull_bear_sideways": db.regime_counts("regime_market"),
        "volatility": db.regime_counts("regime_volatility"),
        "structure": db.regime_counts("regime_structure"),
        "dynamics": db.regime_counts("regime_dynamics"),
        "ecology": db.regime_counts("regime_ecology"),
    }
    for axis, counts in regime_maps.items():
        for regime, n in sorted(counts.items()):
            rows.append({
                "regime_axis": axis,
                "regime": regime,
                "scan_count": n,
                "sufficient": n >= MIN_REGIME_SCANS,
                "gap_scans_needed": max(0, MIN_REGIME_SCANS - n),
            })
    return rows


def regime_gaps(db: HistoryDatabase) -> list[dict]:
    return [r for r in coverage_report(db) if not r["sufficient"]]


def validation_readiness(db: HistoryDatabase) -> dict:
    status = dataset_status(db, "scout_constitution_v1")
    gaps = regime_gaps(db)
    cal = calendar_status(db)
    w90 = next((c for c in cal if c["window_days"] == 90), {})
    return {
        "can_validate_90d": status["validation_90d_ready"],
        "can_live_validate": status["live_validation_ready"],
        "days_remaining_90d": status["days_to_90d"],
        "days_remaining_live": status["days_to_live_validation"],
        "regime_gaps_count": len(gaps),
        "weakest_regimes": gaps[:5],
        "window_90_scans": w90.get("scan_count", 0),
    }
