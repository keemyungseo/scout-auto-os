#!/usr/bin/env python3
"""Run Rule Discovery Engine V1."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PKG / "research_bundle"))

from scout_auto_os.engine.research.rule_discovery.runner import RuleDiscoveryRunner


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(PKG / "data"))
    args = p.parse_args()

    runner = RuleDiscoveryRunner(
        Path(args.data_dir),
        pkg_root=PKG,
        candidates_path=PKG / "research_bundle" / "seed" / "candidates.jsonl",
        forward_path=PKG / "research_bundle" / "forward" / "forward_klines_15m.jsonl",
    )
    result = runner.run()
    if not result:
        print("Rule discovery failed — see logs")
        sys.exit(1)
    rec = result["recommendation"]
    print(f"Report: {result['report_path']}")
    print(f"Decision: {rec.get('decision')} - {rec.get('reason')}")


if __name__ == "__main__":
    main()
