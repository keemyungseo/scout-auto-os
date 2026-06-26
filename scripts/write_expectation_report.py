#!/usr/bin/env python3
"""Write Expectation Engine V1 report."""

from __future__ import annotations

import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))

import yaml
from scout_auto_os.engine.expectation.runner import ExpectationRunner
from scout_auto_os.engine.position_evaluation.thesis import ThesisStore

cfg = yaml.safe_load((PKG / "config.yaml").read_text(encoding="utf-8"))
store = ThesisStore(PKG / "data")
runner = ExpectationRunner(cfg, PKG / "data", thesis_store=store)
print(runner.write_report())
