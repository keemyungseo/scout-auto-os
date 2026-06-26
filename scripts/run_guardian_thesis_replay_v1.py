#!/usr/bin/env python3
"""Run Guardian Trade Thesis replay V1 — 157 trades with thesis context."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))

from scout_auto_os.engine.guardian.thesis_replay import run_thesis_replay  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Guardian Trade Thesis replay V1")
    parser.add_argument("--elapsed", type=int, default=240)
    args = parser.parse_args()

    config_path = PKG / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}

    result = run_thesis_replay(PKG / "data", elapsed_minutes=args.elapsed, config=config)
    print(f"[GUARDIAN THESIS] trades={result['trade_count']}")
    print(f"[GUARDIAN THESIS] jsonl: {result['thesis_jsonl']}")
    print(f"[GUARDIAN THESIS] log: {result['thesis_log_csv']}")
    print(f"[GUARDIAN THESIS] analysis: {result['analysis_json']}")
    for action, stats in sorted(result["analysis"].items()):
        print(f"  {action}: n={stats['count']} dna={stats['predicted_dna']}")


if __name__ == "__main__":
    main()
