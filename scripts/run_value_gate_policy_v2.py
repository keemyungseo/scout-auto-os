#!/usr/bin/env python3
"""Run Predator Value Gate Policy Test V2."""

from __future__ import annotations

import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PKG / "research_bundle"))

from scout_auto_os.engine.predator.policy_test_runner import ValueGatePolicyTestRunner


def main() -> None:
    runner = ValueGatePolicyTestRunner(Path(PKG / "data"))
    meta = runner.run()
    print(f"Verdict: {meta['verdict']} | Recommended: Policy {meta['recommended_policy']}")
    print(f"Report: {meta['report_path']}")


if __name__ == "__main__":
    main()
