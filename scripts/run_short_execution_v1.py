#!/usr/bin/env python3
"""Run Short Execution Research V1."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PKG / "research_bundle"))

from scout_auto_os.engine.research.short_execution.runner import ShortExecutionRunner


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(PKG / "data"))
    args = p.parse_args()

    runner = ShortExecutionRunner(
        Path(args.data_dir),
        pkg_root=PKG,
        workspace_root=ROOT,
        candidates_path=PKG / "research_bundle" / "seed" / "candidates.jsonl",
        forward_path=PKG / "research_bundle" / "forward" / "forward_klines_15m.jsonl",
    )
    result = runner.run()
    if result is None:
        sys.exit(1)
    d = result["decision"]
    print(f"best_exit={d['q3_exit_constitution_recommendation']['primary']}")
    print(f"rec_hold={d['q2_recommended_holding_minutes']}m")
    sys.exit(0)


if __name__ == "__main__":
    main()
