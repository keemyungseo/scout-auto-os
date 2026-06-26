#!/usr/bin/env python3
"""Run Scan Cadence Optimizer V1."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PKG / "research_bundle"))

from scout_auto_os.engine.research.cadence.runner import ScanCadenceOptimizerRunner


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(PKG / "data"))
    p.add_argument("--lookback-days", type=int, default=180)
    args = p.parse_args()

    runner = ScanCadenceOptimizerRunner(
        Path(args.data_dir),
        pkg_root=PKG,
        candidates_path=PKG / "research_bundle" / "seed" / "candidates.jsonl",
        forward_path=PKG / "research_bundle" / "forward" / "forward_klines_15m.jsonl",
        lookback_days=args.lookback_days,
    )
    result = runner.run()
    if not result:
        print("Cadence optimizer failed")
        sys.exit(1)
    rec = result["meta"]["recommendation"]
    print(f"Report: {result['report_path']}")
    print(
        f"LIVE: primary={rec.get('primary_scan_interval_min')}m "
        f"refresh={rec.get('candidate_refresh_interval_min')}m "
        f"rebalance={rec.get('portfolio_rebalance_interval_min')}m",
    )


if __name__ == "__main__":
    main()
