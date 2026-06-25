#!/usr/bin/env python3
"""Run SCOUT Directional Zero-Base V1 — Long/Short validation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PKG / "research_bundle"))

from scout_auto_os.engine.research.directional.runner import DirectionalZeroBaseRunner


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(PKG / "data"))
    p.add_argument("--max-scans", type=int, default=None)
    p.add_argument("--random-draws", type=int, default=100)
    args = p.parse_args()

    runner = DirectionalZeroBaseRunner(
        Path(args.data_dir),
        candidates_path=PKG / "research_bundle" / "seed" / "candidates.jsonl",
        forward_path=PKG / "research_bundle" / "forward" / "forward_klines_15m.jsonl",
        random_draws=args.random_draws,
    )
    result = runner.run(max_scans=args.max_scans)
    print(f"Report: {result['report_path']}")
    print("Long TOP3:", [r["engine"] for r in result["long_board"][:3]])
    print("Short TOP3:", [r["engine"] for r in result["short_board"][:3]])
    print("Slots:", result["slot_sim"])


if __name__ == "__main__":
    main()
