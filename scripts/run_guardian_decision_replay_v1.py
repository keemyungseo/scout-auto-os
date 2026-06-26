#!/usr/bin/env python3
"""Run Guardian Decision Engine V1 on 157-trade replay bundle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))

from scout_auto_os.engine.guardian.decision_report import run_replay_decisions  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Guardian Decision replay V1")
    parser.add_argument("--elapsed", type=int, default=240, help="Position snapshot minutes")
    args = parser.parse_args()

    data_dir = PKG / "data"
    result = run_replay_decisions(data_dir, elapsed_minutes=args.elapsed)
    print(f"[GUARDIAN] trades={result['trade_count']} actions={result['action_counts']}")
    print(f"[GUARDIAN] csv: {result['decision_csv']}")
    print(f"[GUARDIAN] log: {result['log_csv']}")
    print(f"[GUARDIAN] report: {result['report_md']}")
    met = result["scenarios"]["met"]
    hei = result["scenarios"]["hei"]
    print(f"[GUARDIAN] MET blocks_extended_hold={met['blocks_extended_hold']} action={met['action']}")
    print(f"[GUARDIAN] HEI switches_to_trail={hei['switches_to_trail']} action={hei['action']}")


if __name__ == "__main__":
    main()
