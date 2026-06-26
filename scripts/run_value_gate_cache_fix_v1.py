#!/usr/bin/env python3
"""Value Gate cache key fix V1 — backup, regenerate shadow, validate, re-evaluate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))

from scout_auto_os.engine.predator.cache_fix import (  # noqa: E402
    backup_cache_bug_files,
    run_cache_fix_reevaluation,
    validate_cache_fix,
)
from scout_auto_os.engine.predator.inference import load_replay_bundle  # noqa: E402
from scout_auto_os.engine.predator.runtime_shadow import ValueGateRuntimeShadow  # noqa: E402
from scout_auto_os.engine.predator.shadow_labeler import load_labeler_config, run_shadow_labeler  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Value Gate Cache Key Fix V1")
    parser.add_argument("--skip-shadow", action="store_true")
    parser.add_argument("--skip-labeler", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    data_dir = PKG / "data"
    shadow_dir = data_dir / "runtime_shadow"

    backed = backup_cache_bug_files(shadow_dir)
    for src, dst in backed.items():
        print(f"[CACHE FIX] backup {src} -> {dst}")

    if not args.validate_only and not args.skip_shadow:
        shadow = ValueGateRuntimeShadow(data_dir, enabled=True, mode="replay")
        rows = load_replay_bundle(data_dir / "trade_dna")
        logged, skipped = shadow.replay_backfill(rows)
        print(f"[CACHE FIX] shadow replay logged={len(logged)} skipped={len(skipped)}")
        shadow.write_report()

    if not args.validate_only and not args.skip_labeler:
        cfg = load_labeler_config()
        label_result = run_shadow_labeler(
            data_dir, cfg=cfg, mode="replay", force=True, pkg_root=PKG,
        )
        s = label_result["summary"]
        print(
            f"[CACHE FIX] labeler 4h={s['labeled_4h_rows']} "
            f"false_skip={s['false_skip_count']} false_accept={s['false_accept_count']}"
        )

    validation = validate_cache_fix(data_dir)
    print(
        f"[CACHE FIX] rule_mm={validation['rule_mismatch_count']} "
        f"tk_mm={validation['trade_key_mismatch_count']} "
        f"ENTER={validation['enter_count']}/{validation['policy_test_enter']} "
        f"SKIP={validation['skip_count']}/{validation['policy_test_skip']} "
        f"false_skip={validation['false_skip_count']} ok={validation['ok']}"
    )

    result = run_cache_fix_reevaluation(data_dir)
    print(f"[CACHE FIX] verdict={result['verdict']}")
    print(f"[CACHE FIX] report: {shadow_dir / 'value_gate_cache_fix_report.md'}")


if __name__ == "__main__":
    main()
