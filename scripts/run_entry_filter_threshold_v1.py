#!/usr/bin/env python3
"""Run Entry Filter Threshold Optimizer V1."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PKG / "research_bundle"))

from scout_auto_os.engine.research.directional.entry_filter.threshold_runner import EntryFilterThresholdRunner


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(PKG / "data"))
    p.add_argument("--max-rules", type=int, default=4)
    args = p.parse_args()

    runner = EntryFilterThresholdRunner(
        Path(args.data_dir),
        pkg_root=PKG,
        candidates_path=PKG / "research_bundle" / "seed" / "candidates.jsonl",
        forward_path=PKG / "research_bundle" / "forward" / "forward_klines_15m.jsonl",
    )
    result = runner.run(max_rule_features=args.max_rules)
    print(f"Report: {result['report_path']}")
    print("Long rules:", [r["feature"] for r in result["long_rules"]])
    print("Short rules:", [r["feature"] for r in result["short_rules"]])
    print("Long combined f1:", result["long_stats"].get("f1"))
    print("Short combined f1:", result["short_stats"].get("f1"))


if __name__ == "__main__":
    main()
