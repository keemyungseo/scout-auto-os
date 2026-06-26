#!/usr/bin/env python3
"""Run Trade DNA Engine V1."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PKG / "research_bundle"))

from scout_auto_os.engine.research.trade_dna.runner import TradeDNARunner


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(PKG / "data"))
    p.add_argument("--replay-days", type=int, default=15)
    args = p.parse_args()

    runner = TradeDNARunner(
        Path(args.data_dir),
        pkg_root=PKG,
        candidates_path=PKG / "research_bundle" / "seed" / "candidates.jsonl",
        forward_path=PKG / "research_bundle" / "forward" / "forward_klines_15m.jsonl",
    )
    meta = runner.run(replay_days=args.replay_days)
    print(f"Types: {meta['n_types']} | Trades: {meta['n_trades']}")
    print(f"Expected lift: {meta['lift'].get('expected_lift_pp')}%p")


if __name__ == "__main__":
    main()
