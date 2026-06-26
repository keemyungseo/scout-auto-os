#!/usr/bin/env python3
"""Run Trade Value Estimator V1 (Season3)."""

from __future__ import annotations

import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PKG / "research_bundle"))

from scout_auto_os.engine.research.trade_dna.value_estimator.runner import TradeValueEstimatorRunner


def main() -> None:
    runner = TradeValueEstimatorRunner(
        Path(PKG / "data"),
        pkg_root=PKG,
        candidates_path=PKG / "research_bundle" / "seed" / "candidates.jsonl",
    )
    meta = runner.run()
    print(f"ROI R2={meta['roi_r2']} MAE={meta['roi_mae']}% | Sharpe delta={meta['sharpe_delta']}")
    print(f"Report: {meta['report_path']}")


if __name__ == "__main__":
    main()
