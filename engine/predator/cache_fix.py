"""Cache key fix validation and post-fix re-evaluation."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

from scout_auto_os.engine.predator.labeled_reevaluation import (
    band_calibration,
    check_policy_b_decision_consistency,
    check_trade_key_policy_mismatch,
    compute_labeled_metrics,
    false_accept_detail,
    false_skip_detail,
    run_labeled_reevaluation,
)
from scout_auto_os.engine.predator.prediction_key import prediction_key_from_row
from scout_auto_os.storage.db import now_kst

SHADOW_BACKUP = "value_gate_runtime_shadow.cache_key_bug_backup.csv"
LABELED_BACKUP = "value_gate_runtime_shadow_labeled.cache_key_bug_backup.csv"

POLICY_TEST_ENTER = 74
POLICY_TEST_SKIP = 83
POLICY_TEST_FALSE_SKIP = 11


def backup_cache_bug_files(shadow_dir: Path) -> dict[str, str]:
    backed: dict[str, str] = {}
    pairs = (
        ("value_gate_runtime_shadow.csv", SHADOW_BACKUP),
        ("value_gate_runtime_shadow_labeled.csv", LABELED_BACKUP),
    )
    for src_name, dst_name in pairs:
        src = shadow_dir / src_name
        if src.exists():
            dst = shadow_dir / dst_name
            shutil.copy2(src, dst)
            backed[src_name] = str(dst)
    return backed


def validate_cache_fix(data_dir: Path) -> dict:
    shadow_dir = data_dir / "runtime_shadow"
    labeled_path = shadow_dir / "value_gate_runtime_shadow_labeled.csv"
    rows = list(csv.DictReader(labeled_path.open(encoding="utf-8"))) if labeled_path.exists() else []

    rule_mm = check_policy_b_decision_consistency(rows)
    tk_mm = check_trade_key_policy_mismatch(rows, data_dir)

    enter = sum(1 for r in rows if r.get("policy_b_decision") == "ENTER")
    skip = sum(1 for r in rows if r.get("policy_b_decision") == "SKIP")
    false_skip = sum(1 for r in rows if r.get("false_skip") == "1")

    ok = (
        len(rule_mm) == 0
        and len(tk_mm) == 0
        and enter == POLICY_TEST_ENTER
        and skip == POLICY_TEST_SKIP
    )

    mismatches = rule_mm + tk_mm
    if mismatches:
        _write_csv(shadow_dir / "policy_b_cache_fix_mismatch.csv", mismatches)

    return {
        "ok": ok,
        "rule_mismatch_count": len(rule_mm),
        "trade_key_mismatch_count": len(tk_mm),
        "enter_count": enter,
        "skip_count": skip,
        "false_skip_count": false_skip,
        "policy_test_enter": POLICY_TEST_ENTER,
        "policy_test_skip": POLICY_TEST_SKIP,
        "enter_match": enter == POLICY_TEST_ENTER,
        "skip_match": skip == POLICY_TEST_SKIP,
        "validated_at": now_kst(),
    }


def run_cache_fix_reevaluation(data_dir: Path) -> dict:
    shadow_dir = data_dir / "runtime_shadow"
    labeled_path = shadow_dir / "value_gate_runtime_shadow_labeled.csv"
    rows = list(csv.DictReader(labeled_path.open(encoding="utf-8"))) if labeled_path.exists() else []

    validation = validate_cache_fix(data_dir)
    metrics = compute_labeled_metrics(rows)
    bands = band_calibration(rows)
    fs_detail = false_skip_detail(rows)
    fa_detail = false_accept_detail(rows)

    _write_csv(shadow_dir / "value_gate_cache_fix_reevaluation.csv", [metrics])
    _write_csv(shadow_dir / "value_gate_cache_fix_band_calibration.csv", bands)
    _write_csv(shadow_dir / "value_gate_cache_fix_false_skip.csv", fs_detail)
    _write_csv(shadow_dir / "value_gate_cache_fix_false_accept.csv", fa_detail)

    verdict = _decide_verdict(validation, metrics)
    _write_report(shadow_dir / "value_gate_cache_fix_report.md", validation, metrics, verdict)

    return {
        "ok": True,
        "verdict": verdict,
        "validation": validation,
        "metrics": metrics,
    }


def _decide_verdict(validation: dict, metrics: dict) -> str:
    if not validation.get("enter_match") or not validation.get("skip_match"):
        if validation.get("trade_key_mismatch_count", 1) > 0:
            return "CACHE_FIX_FAILED"
        return "NEEDS_MORE_DATA"
    if validation.get("ok"):
        fs = metrics.get("false_skip_count", 99)
        if fs <= POLICY_TEST_FALSE_SKIP + 5:
            return "CACHE_FIX_SUCCESS_KEEP_POLICY_B_SHADOW"
        return "CACHE_FIX_SUCCESS_REVIEW_POLICY"
    return "CACHE_FIX_FAILED"


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _write_report(path: Path, validation: dict, metrics: dict, verdict: str) -> None:
    lines = [
        "# Value Gate Cache Key Fix V1 — Report",
        "",
        f"**Verdict:** `{verdict}`",
        f"**Validated at:** {validation.get('validated_at', '')}",
        "",
        "## Validation",
        "",
        f"- Rule mismatch: {validation.get('rule_mismatch_count', 0)}",
        f"- Trade key mismatch: {validation.get('trade_key_mismatch_count', 0)}",
        f"- ENTER: {validation.get('enter_count')} (policy test {POLICY_TEST_ENTER})",
        f"- SKIP: {validation.get('skip_count')} (policy test {POLICY_TEST_SKIP})",
        f"- false_skip: {validation.get('false_skip_count')} (was 44, policy test {POLICY_TEST_FALSE_SKIP})",
        "",
        "## Post-fix metrics",
        "",
        f"- Accepted avg ROI: {metrics.get('accepted_avg_roi')}%",
        f"- Skipped avg ROI: {metrics.get('skipped_avg_roi')}%",
        f"- Weighted ROI: {metrics.get('weighted_roi')}%",
        "",
        "## Final answers",
        "",
        "1. symbol|side cache was in ShadowPredictionCache.lookup() fallback and DNA linear scan.",
        "2. prediction_key now trade_key / scan_id only.",
        f"3. Mismatch after fix: rule={validation.get('rule_mismatch_count')} tk={validation.get('trade_key_mismatch_count')}.",
        f"4. ENTER/SKIP match policy test: {validation.get('enter_match')} / {validation.get('skip_match')}.",
        f"5. false_skip reduced: {validation.get('false_skip_count')} vs 44.",
        "6. Policy B: KEEP SHADOW if validation ok; REVIEW if false_skip still elevated.",
        "7. LIVE: remain on hold until SHADOW observation window completes.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
