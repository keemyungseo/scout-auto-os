"""Integrity checks — leak, duplicate, missing, label errors."""

from __future__ import annotations

import json
from collections import defaultdict

from scout_auto_os.engine.research.infrastructure.dataset_manager import HistoryDatabase

FORBIDDEN_FEATURE_PREFIXES = ("return_2h", "label_", "outcome_")


def check_duplicates(db: HistoryDatabase) -> list[dict]:
    issues: list[dict] = []
    with db._connect() as conn:
        rows = conn.execute(
            """
            SELECT scan_kst, symbol, COUNT(*) AS n
            FROM candidates GROUP BY scan_kst, symbol HAVING n > 1
            """,
        ).fetchall()
        for r in rows:
            issues.append({
                "check": "duplicate",
                "scan_kst": r["scan_kst"],
                "symbol": r["symbol"],
                "count": r["n"],
            })
    return issues


def check_missing_labels(db: HistoryDatabase) -> list[dict]:
    issues: list[dict] = []
    with db._connect() as conn:
        rows = conn.execute(
            """
            SELECT c.scan_kst, COUNT(*) AS missing
            FROM candidates c
            LEFT JOIN forward_labels f ON c.scan_kst=f.scan_kst AND c.symbol=f.symbol
            WHERE f.symbol IS NULL
            GROUP BY c.scan_kst
            """,
        ).fetchall()
        for r in rows:
            issues.append({
                "check": "missing_label",
                "scan_kst": r["scan_kst"],
                "missing_count": r["missing"],
            })
    return issues


def check_leak_in_features(db: HistoryDatabase, sample_limit: int = 500) -> list[dict]:
    issues: list[dict] = []
    with db._connect() as conn:
        rows = conn.execute(
            "SELECT scan_kst, symbol, features_json FROM candidates LIMIT ?",
            (sample_limit,),
        ).fetchall()
    for r in rows:
        try:
            feats = json.loads(r["features_json"])
        except json.JSONDecodeError:
            issues.append({"check": "json_error", "scan_kst": r["scan_kst"], "symbol": r["symbol"]})
            continue
        for k in feats:
            for bad in FORBIDDEN_FEATURE_PREFIXES:
                if k.startswith(bad):
                    issues.append({
                        "check": "leak_feature",
                        "scan_kst": r["scan_kst"],
                        "symbol": r["symbol"],
                        "feature": k,
                    })
                    break
    return issues


def check_label_consistency(db: HistoryDatabase) -> list[dict]:
    issues: list[dict] = []
    with db._connect() as conn:
        rows = conn.execute(
            """
            SELECT scan_kst, symbol, return_2h, return_minus_dd, max_drawdown_2h
            FROM forward_labels
            """,
        ).fetchall()
    for r in rows:
        expected = round(float(r["return_2h"]) + float(r["max_drawdown_2h"]), 4)
        actual = round(float(r["return_minus_dd"]), 4)
        if abs(expected - actual) > 0.02:
            issues.append({
                "check": "label_error",
                "scan_kst": r["scan_kst"],
                "symbol": r["symbol"],
                "expected_minus_dd": expected,
                "actual_minus_dd": actual,
            })
    return issues


def check_future_data(db: HistoryDatabase) -> list[dict]:
    """Ensure labels exist only when scan is in DB (no orphan labels)."""
    issues: list[dict] = []
    with db._connect() as conn:
        rows = conn.execute(
            """
            SELECT f.scan_kst, f.symbol FROM forward_labels f
            LEFT JOIN scans s ON f.scan_kst=s.scan_kst
            WHERE s.scan_kst IS NULL
            """,
        ).fetchall()
        for r in rows:
            issues.append({
                "check": "orphan_label",
                "scan_kst": r["scan_kst"],
                "symbol": r["symbol"],
            })
    return issues


def run_quality_checks(db: HistoryDatabase) -> dict:
    dup = check_duplicates(db)
    missing = check_missing_labels(db)
    leak = check_leak_in_features(db)
    label_err = check_label_consistency(db)
    orphan = check_future_data(db)

    all_issues = dup + missing + leak + label_err + orphan
    passed = len(all_issues) == 0

    return {
        "passed": passed,
        "duplicate_count": len(dup),
        "missing_label_scans": len(missing),
        "leak_feature_count": len(leak),
        "label_error_count": len(label_err),
        "orphan_label_count": len(orphan),
        "issues": all_issues[:100],
        "summary": (
            "PASS - all integrity checks passed"
            if passed
            else f"FAIL - {len(all_issues)} issues found"
        ),
    }
