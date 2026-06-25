#!/usr/bin/env python3
"""Run full Zero-Base Discovery on research bundle (Lab Stream)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PKG / "research_bundle"))

from scout_auto_os.engine.research.zero_base.runner import ZeroBaseRunner


def main():
    p = argparse.ArgumentParser(description="SCOUT Zero-Base Discovery")
    p.add_argument("--data-dir", default=str(PKG / "data"), help="Output data directory")
    p.add_argument("--max-scans", type=int, default=None, help="Limit scans (default: all)")
    p.add_argument("--random-draws", type=int, default=100)
    p.add_argument("--top-k", type=int, default=5)
    args = p.parse_args()

    runner = ZeroBaseRunner(
        Path(args.data_dir),
        random_draws=args.random_draws,
        top_k=args.top_k,
    )
    result = runner.run(max_scans=args.max_scans)
    print(f"Report: {Path(args.data_dir) / 'zero_base' / 'zero_base_report.md'}")
    print(f"Champion board top: {result.get('champion_board', [])[:3]}")
    better = result.get("better_than_a6", [])
    print(f"Beats A6: {[c.get('engine') for c in better]}")


if __name__ == "__main__":
    main()
