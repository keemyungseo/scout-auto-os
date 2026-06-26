#!/usr/bin/env python3
"""Run Predator Value Gate V1 (Season3)."""

from __future__ import annotations

import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PKG / "research_bundle"))

from scout_auto_os.engine.predator.runner import PredatorValueGateRunner


def main() -> None:
    runner = PredatorValueGateRunner(Path(PKG / "data"))
    meta = runner.run()
    print(f"Verdict: {meta['verdict']} | Sharpe {meta['baseline_sharpe']} -> {meta['gated_sharpe']}")
    print(f"False skip={meta['false_skip_count']} false accept={meta['false_accept_count']}")
    print(f"Report: {meta['report_path']}")


if __name__ == "__main__":
    main()
