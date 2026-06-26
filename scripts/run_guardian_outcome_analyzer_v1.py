#!/usr/bin/env python3
"""Run Guardian Outcome Analyzer V1 — evaluate Guardian decisions after trade close."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))

from scout_auto_os.engine.guardian.outcome_replay import run_outcome_analysis  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Guardian Outcome Analyzer V1")
    parser.parse_args()

    result = run_outcome_analysis(PKG / "data")
    a = result["analysis"]
    print(f"[OUTCOME] trades={result['trade_count']} avg_score={result['avg_guardian_score']}")
    print(f"[OUTCOME] grades={result['grade_distribution']}")
    print(f"[OUTCOME] score_vs_roi={a['correlation'].get('guardian_score_vs_roi')}")
    print(f"[OUTCOME] csv: {result['outcome_csv']}")
    print(f"[OUTCOME] summary: {result['summary_json']}")


if __name__ == "__main__":
    main()
