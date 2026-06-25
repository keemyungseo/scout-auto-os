#!/usr/bin/env python3
"""Run SCOUT Zero-Base Validation V1 — blind June validation on historical bundle."""

from __future__ import annotations

import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PKG / "research_bundle"))

from scout_auto_os.engine.research.zero_base.validation import ZeroBaseValidationRunner


def main():
    runner = ZeroBaseValidationRunner(
        data_dir=PKG / "data",
        candidates_path=PKG / "research_bundle" / "seed" / "candidates.jsonl",
        forward_path=PKG / "research_bundle" / "forward" / "forward_klines_15m.jsonl",
        random_draws=100,
        train_cutoff="2026-06-01",
    )
    result = runner.run()
    print(f"Report: {result['report_path']}")
    print(f"Validation scans: {result['meta']['validation_scans']}")
    print(f"Champion candidates: {sum(1 for r in result['board'] if r.get('champion_eligible'))}")
    top = result["board"][:5]
    for r in top:
        print(
            f"  #{r.get('board_rank')} {r.get('engine')} avg2h={r.get('avg_return_2h')}% "
            f"sig={r.get('statistically_significant')}"
        )


if __name__ == "__main__":
    main()
