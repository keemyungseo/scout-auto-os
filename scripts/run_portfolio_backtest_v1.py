#!/usr/bin/env python3
"""Run Portfolio Engine V1 backtest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PKG / "research_bundle"))

from scout_auto_os.engine.portfolio.backtest import PortfolioBacktestRunner


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(PKG / "data"))
    p.add_argument("--lookback-days", type=int, default=180)
    args = p.parse_args()

    runner = PortfolioBacktestRunner(
        Path(args.data_dir),
        pkg_root=PKG,
        candidates_path=PKG / "research_bundle" / "seed" / "candidates.jsonl",
        forward_path=PKG / "research_bundle" / "forward" / "forward_klines_15m.jsonl",
    )
    result = runner.run(lookback_days=args.lookback_days)
    s = result["stats"]
    print(f"Report: {result['report_path']}")
    print(f"Trades: {s.get('total_trades')} cumulative={s.get('cumulative_return_2h')}% MDD={s.get('max_drawdown')}%")
    print(f"Win rate: {s.get('win_rate_pct')}% replacements: {s.get('replacement_count')}")


if __name__ == "__main__":
    main()
