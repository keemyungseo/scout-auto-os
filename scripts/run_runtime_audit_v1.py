#!/usr/bin/env python3
"""Run Performance First Runtime Gate V1 audit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PKG / "research_bundle"))

from scout_auto_os.engine.runtime_audit.report import RuntimeAuditReport


def main() -> None:
    p = argparse.ArgumentParser(description="SCOUT Runtime Audit — Performance First Gate V1")
    p.add_argument("--data-dir", default=str(PKG / "data"))
    p.add_argument("--config", default=str(PKG / "config.yaml"))
    p.add_argument("--lookback-scans", type=int, default=55)
    p.add_argument("--candidates", default="")
    p.add_argument("--forward", default="")
    args = p.parse_args()

    config_path = Path(args.config)
    with config_path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_dir = Path(args.data_dir)
    candidates = Path(args.candidates) if args.candidates else PKG / "research_bundle" / "seed" / "candidates.jsonl"
    forward = Path(args.forward) if args.forward else PKG / "research_bundle" / "forward" / "forward_klines_15m.jsonl"

    report = RuntimeAuditReport(data_dir, PKG, config, candidates, forward)
    meta = report.run(lookback_scans=args.lookback_scans)
    print(f"Gate summary: {meta.get('gate_summary')}")
    print(f"Report: {meta.get('report_path')}")


if __name__ == "__main__":
    main()
