#!/usr/bin/env python3
"""Run Full System Replay Validation V1 — 15-day Long3+Short3."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PKG / "research_bundle"))

from scout_auto_os.engine.research.system_replay.runner import SystemReplayRunner


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(PKG / "data"))
    p.add_argument("--replay-days", type=int, default=15)
    args = p.parse_args()

    runner = SystemReplayRunner(
        Path(args.data_dir),
        pkg_root=PKG,
        candidates_path=PKG / "research_bundle" / "seed" / "candidates.jsonl",
        forward_path=PKG / "research_bundle" / "forward" / "forward_klines_15m.jsonl",
    )
    meta = runner.run(replay_days=args.replay_days)
    print(f"Report: {meta['report_dir']}/system_replay_validation.md")
    print(f"Season2: {meta['report_dir']}/season2_final_assessment.md")
    for k, v in meta.get("answers", {}).items():
        safe = str(v).replace("\u2014", "-").replace("\u2192", "->")
        print(f"  {k}: {safe}")


if __name__ == "__main__":
    main()
