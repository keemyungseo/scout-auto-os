#!/usr/bin/env python3
"""Run Execution Engine V1 research."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PKG / "research_bundle"))

from scout_auto_os.engine.research.execution_research.runner import ExecutionResearchRunner


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(PKG / "data"))
    args = p.parse_args()

    runner = ExecutionResearchRunner(
        Path(args.data_dir),
        pkg_root=PKG,
        candidates_path=PKG / "research_bundle" / "seed" / "candidates.jsonl",
        forward_path=PKG / "research_bundle" / "forward" / "forward_klines_15m.jsonl",
    )
    result = runner.run()
    if not result:
        sys.exit(1)
    c = result["combined"]
    print(f"Report: {result['report_path']}")
    print(f"Blind exec top2 avg={c['exec_top2_avg']} lift vs entry top2={c['lift_vs_entry_top2_pct']}%")


if __name__ == "__main__":
    main()
