#!/usr/bin/env python3
"""Run Trade DNA Predictor V1."""

from __future__ import annotations

import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PKG / "research_bundle"))

from scout_auto_os.engine.research.trade_dna.predictor.runner import TradeDNAPredictorRunner


def main() -> None:
    runner = TradeDNAPredictorRunner(
        Path(PKG / "data"),
        pkg_root=PKG,
        candidates_path=PKG / "research_bundle" / "seed" / "candidates.jsonl",
    )
    meta = runner.run()
    print(f"Accuracy: {meta['accuracy']} | Search rank: #{meta['search_rank']}")
    print(f"Report: {meta['report_path']}")


if __name__ == "__main__":
    main()
