#!/usr/bin/env python3
"""Run SCOUT Regime Engine research — situation → engine router."""

from __future__ import annotations

import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PKG / "research_bundle"))

from scout_auto_os.engine.research.regime.runner import RegimeEngineRunner


def main():
    runner = RegimeEngineRunner(
        data_dir=PKG / "data",
        candidates_path=PKG / "research_bundle" / "seed" / "candidates.jsonl",
        forward_path=PKG / "research_bundle" / "forward" / "forward_klines_15m.jsonl",
    )
    result = runner.run()
    print(f"Report: {result['report_path']}")
    print("Regime distribution:")
    for k, v in sorted(result["regime_counts"].items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")
    print("Champion router:")
    for c in result["champions"]:
        print(f"  {c['regime']} -> {c.get('champion_engine')}")


if __name__ == "__main__":
    main()
