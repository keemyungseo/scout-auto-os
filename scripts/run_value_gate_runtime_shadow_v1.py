#!/usr/bin/env python3
"""Backfill Policy B runtime shadow log from replay bundle (157 trades)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))

from scout_auto_os.engine.predator.cache_fix import backup_cache_bug_files
from scout_auto_os.engine.predator.inference import load_replay_bundle
from scout_auto_os.engine.predator.runtime_shadow import ValueGateRuntimeShadow
from scout_auto_os.engine.predator.shadow_labeler import load_labeler_config, run_shadow_labeler
from scout_auto_os.engine.predator.timestamp_fix import (
    BACKUP_NAME,
    backup_shadow_csv,
    run_timestamp_validation,
    validate_shadow_timestamps,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Value Gate Runtime Shadow backfill")
    parser.add_argument("--mode", choices=("live", "replay"), default="replay")
    parser.add_argument("--force", action="store_true", help="Backup cache-bug CSVs before replay")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--skip-labeler", action="store_true")
    args = parser.parse_args()

    data_dir = PKG / "data"
    shadow_dir = data_dir / "runtime_shadow"

    if args.force:
        backed = backup_cache_bug_files(shadow_dir)
        for src, dst in backed.items():
            print(f"[RUNTIME SHADOW] cache-key backup {src} -> {dst}")

    backup_path = backup_shadow_csv(shadow_dir)
    if backup_path:
        print(f"[RUNTIME SHADOW] backup: {backup_path}")

    shadow = ValueGateRuntimeShadow(data_dir, enabled=True, mode=args.mode)

    if args.mode == "replay":
        rows = load_replay_bundle(data_dir / "trade_dna")
        print(f"[RUNTIME SHADOW] replay candidates={len(rows)}")
        logged, skipped = shadow.replay_backfill(rows)
        print(f"[RUNTIME SHADOW] logged={len(logged)} skipped={len(skipped)}")
        if skipped:
            print(f"[RUNTIME SHADOW] MISSING_ORIGINAL_TIMESTAMP={len(skipped)}")
    else:
        print("[RUNTIME SHADOW] live mode: use main loop on_scan — no batch backfill")
        return

    report = shadow.write_report()
    print(f"[RUNTIME SHADOW] report: {report}")

    validation_result = run_timestamp_validation(
        shadow_dir,
        mode=args.mode,
        skipped=skipped if args.mode == "replay" else [],
        backup_path=backup_path,
    )
    validation = validation_result["validation"]
    print(
        f"[TIMESTAMP FIX] unique_ts={validation['unique_timestamp_count']} "
        f"future={validation['future_timestamp_count']} ok={validation['ok']}"
    )
    if not args.skip_validation and not validation["ok"]:
        print("[TIMESTAMP FIX] VALIDATION FAILED — see timestamp_fix_report.md")
        sys.exit(1)

    if not args.skip_labeler:
        cfg = load_labeler_config()
        label_result = run_shadow_labeler(
            data_dir,
            cfg=cfg,
            mode="replay",
            force=True,
            pkg_root=PKG,
        )
        s = label_result["summary"]
        print(
            f"[SHADOW LABELER] labeled_4h={s['labeled_4h_rows']} "
            f"false_skip={s['false_skip_count']} false_accept={s['false_accept_count']}"
        )


if __name__ == "__main__":
    main()
