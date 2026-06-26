#!/usr/bin/env python3
"""Run Direction Champion Winner/Loser Entry DNA analysis (V1)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PKG / "research_bundle"))

from scout_auto_os.engine.research.directional.entry_filter.runner import DirectionChampionEntryDnaRunner


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(PKG / "data"))
    p.add_argument("--lookback-months", type=int, default=6)
    p.add_argument("--top-k", type=int, default=5)
    args = p.parse_args()

    runner = DirectionChampionEntryDnaRunner(
        Path(args.data_dir),
        pkg_root=PKG,
        candidates_path=PKG / "research_bundle" / "seed" / "candidates.jsonl",
        forward_path=PKG / "research_bundle" / "forward" / "forward_klines_15m.jsonl",
        lookback_months=args.lookback_months,
    )
    result = runner.run(top_k=args.top_k)
    print(f"Report: {result['report_path']}")
    print(f"Signals: long={result['meta']['long_signal_count']} short={result['meta']['short_signal_count']}")
    print(f"Long top DNA: {result['long_profile'].get('top_features', [])[:5]}")
    print(f"Short top DNA: {result['short_profile'].get('top_features', [])[:5]}")
    print(f"Common DNA features: {result['common_dna_count']}")


if __name__ == "__main__":
    main()
