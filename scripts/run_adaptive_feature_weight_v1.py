#!/usr/bin/env python3
"""Run Adaptive Feature Weight Engine V1."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PKG / "research_bundle"))

from scout_auto_os.engine.research.adaptive_feature_weight.runner import AdaptiveFeatureWeightRunner


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(PKG / "data"))
    args = p.parse_args()

    runner = AdaptiveFeatureWeightRunner(
        Path(args.data_dir),
        pkg_root=PKG,
        candidates_path=PKG / "research_bundle" / "seed" / "candidates.jsonl",
        forward_path=PKG / "research_bundle" / "forward" / "forward_klines_15m.jsonl",
    )
    result = runner.run()
    if not result:
        sys.exit(1)
    d = result["decision"]
    print(d["summary"])


if __name__ == "__main__":
    main()
