#!/usr/bin/env python3
"""Run Guardian Progress Engine V1 on 157-trade replay."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))

from scout_auto_os.engine.guardian.progress_replay import run_progress_replay  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Guardian Progress replay V1")
    parser.add_argument("--elapsed", type=int, default=240)
    args = parser.parse_args()

    config_path = PKG / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}

    data_dir = PKG / "data"
    result = run_progress_replay(data_dir, elapsed_minutes=args.elapsed, config=config)
    s = result["summary"]
    print(f"[GUARDIAN PROGRESS] trades={result['trade_count']} avg_score={s.get('avg_guardian_score')}")
    print(f"[GUARDIAN PROGRESS] states={s.get('state_counts')}")
    print(f"[GUARDIAN PROGRESS] MET thesis_failed={result['met_thesis_failed']}")
    print(f"[GUARDIAN PROGRESS] csv: {result['progress_csv']}")
    print(f"[GUARDIAN PROGRESS] json: {result['summary_json']}")


if __name__ == "__main__":
    main()
