#!/usr/bin/env python3
"""Run Directional DNA Discovery V1."""

from __future__ import annotations

import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PKG / "research_bundle"))

from scout_auto_os.engine.research.directional.dna.runner import DirectionalDnaRunner


def main():
    runner = DirectionalDnaRunner(
        PKG / "data",
        candidates_path=PKG / "research_bundle" / "seed" / "candidates.jsonl",
        forward_path=PKG / "research_bundle" / "forward" / "forward_klines_15m.jsonl",
    )
    result = runner.run()
    print(f"Report: {result['report_path']}")
    print(f"LIVE candidates: {len(result['live_candidates'])}")
    for lc in result["live_candidates"][:5]:
        print(f"  {lc['formula_name']} blind_avg2h={lc['blind_avg2h']}")


if __name__ == "__main__":
    main()
