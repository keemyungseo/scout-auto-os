#!/usr/bin/env python3
"""Run Signal Lifecycle Engine V1."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PKG / "research_bundle"))

from scout_auto_os.engine.research.signal_lifecycle.runner import SignalLifecycleRunner


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(PKG / "data"))
    p.add_argument("--lookback-months", type=int, default=6)
    args = p.parse_args()

    runner = SignalLifecycleRunner(
        Path(args.data_dir),
        pkg_root=PKG,
        candidates_path=PKG / "research_bundle" / "seed" / "candidates.jsonl",
        forward_path=PKG / "research_bundle" / "forward" / "forward_klines_15m.jsonl",
        lookback_months=args.lookback_months,
    )
    result = runner.run()
    m = result["meta"]
    print(f"Report: {result['report_path']}")
    print(
        f"Signals: long={m['long_signal_count']} short={m['short_signal_count']} "
        f"timeline_rows={m['timeline_row_count']} clusters={result['cluster_count']}",
    )


if __name__ == "__main__":
    main()
