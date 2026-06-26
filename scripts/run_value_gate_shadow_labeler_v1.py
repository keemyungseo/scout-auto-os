#!/usr/bin/env python3
"""Run Value Gate Shadow Labeler V1 — attach forward outcomes to shadow log."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))

from scout_auto_os.engine.predator.shadow_labeler import (  # noqa: E402
    load_labeler_config,
    run_shadow_labeler,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Value Gate Shadow Labeler V1")
    parser.add_argument("--hours", type=int, choices=(2, 4), default=4)
    parser.add_argument("--mode", choices=("live", "replay"), default="live")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--diagnose-only", action="store_true")
    args = parser.parse_args()

    config_path = PKG / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    data_dir = PKG / "data"
    cfg = load_labeler_config(config)

    result = run_shadow_labeler(
        data_dir,
        cfg=cfg,
        hours=args.hours,
        force=args.force,
        dry_run=args.dry_run,
        mode=args.mode,
        diagnose_only=args.diagnose_only,
        pkg_root=PKG,
    )
    if args.diagnose_only:
        s = result["summary"]
        print(f"[DIAGNOSTICS] rows={s['total_rows']} counts={s['diagnosis_counts']}")
        print(f"[DIAGNOSTICS] replay_join_ok={s.get('replay_join_ok')} mismatch={s.get('timestamp_mismatch_rows')}")
        print(f"[DIAGNOSTICS] csv: {result['diagnostics_csv']}")
        return

    s = result["summary"]
    stats = result["stats"]
    print(
        f"[SHADOW LABELER] mode={args.mode} rows={s['total_rows']} waiting={s['waiting_rows']} "
        f"2h={s['labeled_2h_rows']} 4h={s['labeled_4h_rows']} "
        f"false_skip={s['false_skip_count']} false_accept={s['false_accept_count']}"
    )
    print(
        f"[SHADOW LABELER] new_full={stats['new_full']} new_partial={stats['new_partial']} "
        f"skipped={stats['skipped']} dry_run={args.dry_run}"
    )
    if not args.dry_run:
        print(f"[SHADOW LABELER] labeled: {result['labeled_path']}")


if __name__ == "__main__":
    main()
