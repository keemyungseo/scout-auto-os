#!/usr/bin/env python3
"""Run Entry Rule Optimizer V2 — LIVE rule combination search."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PKG / "research_bundle"))

from scout_auto_os.engine.research.directional.entry_filter.rule_optimizer_v2 import EntryRuleOptimizerV2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(PKG / "data"))
    p.add_argument("--target-pass-per-day", type=float, default=3.0)
    args = p.parse_args()

    runner = EntryRuleOptimizerV2(
        Path(args.data_dir),
        pkg_root=PKG,
        candidates_path=PKG / "research_bundle" / "seed" / "candidates.jsonl",
        forward_path=PKG / "research_bundle" / "forward" / "forward_klines_15m.jsonl",
    )
    result = runner.run(target_pass_per_day=args.target_pass_per_day)
    bl = result.get("best_long") or {}
    bs = result.get("best_short") or {}
    print(f"Report: {result['report_path']}")
    print(f"Long V2: {bl.get('rule_expr')} | pass={bl.get('pass_count')} prec={bl.get('precision')} recall={bl.get('recall')}")
    print(f"Short V2: {bs.get('rule_expr')} | pass={bs.get('pass_count')} prec={bs.get('precision')} recall={bs.get('recall')}")
    print(f"Pattern rules: {len(result.get('pattern_best', []))}")


if __name__ == "__main__":
    main()
