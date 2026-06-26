#!/usr/bin/env python3
"""Run Constitution Validation V1 — final blind validation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PKG / "research_bundle"))

from scout_auto_os.engine.research.constitution_validation.runner import ConstitutionValidationRunner


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(PKG / "data"))
    args = p.parse_args()

    runner = ConstitutionValidationRunner(
        Path(args.data_dir),
        pkg_root=PKG,
        candidates_path=PKG / "research_bundle" / "seed" / "candidates.jsonl",
        forward_path=PKG / "research_bundle" / "forward" / "forward_klines_15m.jsonl",
    )
    result = runner.run()
    if result is None:
        sys.exit(1)
    msg = result["decision"]["q3_core_engine_ready"].replace("\u2014", "-")
    print(msg)
    sys.exit(0)


if __name__ == "__main__":
    main()
