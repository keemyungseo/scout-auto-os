#!/usr/bin/env python3
"""Run Reject Analysis Engine V1."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PKG / "research_bundle"))

from scout_auto_os.engine.portfolio.reject_analysis.runner import RejectAnalysisRunner


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(PKG / "data"))
    p.add_argument("--lookback-days", type=int, default=180)
    args = p.parse_args()

    runner = RejectAnalysisRunner(
        Path(args.data_dir),
        pkg_root=PKG,
        candidates_path=PKG / "research_bundle" / "seed" / "candidates.jsonl",
        lookback_days=args.lookback_days,
    )
    result = runner.run()
    m = result["meta"]
    print(f"Report: {result['report_path']}")
    print(f"Funnel: champion={m['funnel']['direction_champion']} rule_pass={m['funnel']['rule_pass_rate_pct']}% portfolio={m['funnel']['portfolio_pass_rate_pct']}%")
    print(f"Bottleneck: {m['bottleneck'].get('stage')} - {m['bottleneck'].get('top_feature_blocker') or m['bottleneck'].get('top_blocker')}")


if __name__ == "__main__":
    main()
