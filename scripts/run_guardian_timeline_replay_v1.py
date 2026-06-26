#!/usr/bin/env python3
"""Run Guardian Timeline Replay V1 on 157 trades."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))

from scout_auto_os.engine.guardian.timeline_replay import run_timeline_replay  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Guardian Timeline Replay V1")
    parser.add_argument("--max-minutes", type=int, default=240)
    args = parser.parse_args()

    config_path = PKG / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}

    result = run_timeline_replay(
        PKG / "data", PKG, max_minutes=args.max_minutes, config=config,
    )
    a = result["analysis"]
    print(f"[TIMELINE] trades={result['trade_count']} points={result['timeline_points']}")
    print(f"[TIMELINE] avg_exit_rec={a.get('avg_exit_rec_minutes')}m avg_trail={a.get('avg_trail_start_minutes')}m")
    print(f"[TIMELINE] csv: {result['timeline_csv']}")
    print(f"[TIMELINE] summary: {result['summary_json']}")


if __name__ == "__main__":
    main()
