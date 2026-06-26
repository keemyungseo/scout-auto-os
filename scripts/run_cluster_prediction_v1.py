#!/usr/bin/env python3
"""Run SCOUT Cluster Prediction Engine V1."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PKG / "research_bundle"))

from scout_auto_os.engine.research.directional.prediction.runner import ClusterPredictionRunner


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(PKG / "data"))
    p.add_argument("--max-scans", type=int, default=None)
    p.add_argument("--top-k", type=int, default=5)
    args = p.parse_args()

    runner = ClusterPredictionRunner(
        Path(args.data_dir),
        pkg_root=PKG,
        candidates_path=PKG / "research_bundle" / "seed" / "candidates.jsonl",
        forward_path=PKG / "research_bundle" / "forward" / "forward_klines_15m.jsonl",
    )
    result = runner.run(max_scans=args.max_scans, top_k=args.top_k)
    print(f"Report: {result['report_path']}")
    for row in result["comparison"]:
        if row["method"] == "PREDICTION_ENGINE":
            print(f"Prediction {row['direction']}: avg2h={row['avg_return_2h']}% n={row['sample_count']}")
    print("Slot combined avg2h:", result["slot_summary"].get("combined_avg_return_2h"))


if __name__ == "__main__":
    main()
