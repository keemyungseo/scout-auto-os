#!/usr/bin/env python3
"""Write Position Evaluation V1 report."""

from __future__ import annotations

import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))

import yaml
from scout_auto_os.engine.position_evaluation.runner import PositionEvaluationRunner

cfg = yaml.safe_load((PKG / "config.yaml").read_text(encoding="utf-8"))
runner = PositionEvaluationRunner(cfg, PKG / "data")
print(runner.write_report())
