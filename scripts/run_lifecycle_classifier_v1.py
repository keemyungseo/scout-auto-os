#!/usr/bin/env python3
"""Run Lifecycle Classifier V1."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PKG / "research_bundle"))

from scout_auto_os.engine.research.lifecycle_classifier.runner import LifecycleClassifierRunner


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(PKG / "data"))
    p.add_argument("--lookback-months", type=int, default=6)
    args = p.parse_args()

    runner = LifecycleClassifierRunner(
        Path(args.data_dir),
        pkg_root=PKG,
        candidates_path=PKG / "research_bundle" / "seed" / "candidates.jsonl",
        forward_path=PKG / "research_bundle" / "forward" / "forward_klines_15m.jsonl",
        lookback_months=args.lookback_months,
    )
    result = runner.run()
    lm = result["long_metrics"]["aggregate"]
    sm = result["short_metrics"]["aggregate"]
    print(f"Report: {result['report_path']}")
    print(f"Long  val macro_f1={lm['macro_f1']} acc={result['long_metrics']['accuracy']}")
    print(f"Short val macro_f1={sm['macro_f1']} acc={result['short_metrics']['accuracy']}")


if __name__ == "__main__":
    main()
