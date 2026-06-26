#!/usr/bin/env python3
"""Run Portfolio Decision Engine V1 replay on 157 trades."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))

from scout_auto_os.engine.portfolio.decision.decision_replay import run_portfolio_decision_replay  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Portfolio Decision Engine V1")
    parser.parse_args()

    config_path = PKG / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}

    result = run_portfolio_decision_replay(PKG / "data", config=config)
    a = result["analysis"]
    print(f"[PORTFOLIO] trades={result['trade_count']} decisions={result['decision_rows']}")
    print(f"[PORTFOLIO] util={a.get('avg_slot_utilization')} replacements={a.get('replacement_count')}")
    print(f"[PORTFOLIO] missed={a.get('missed_trades')} repl_success={a.get('replacement_success_rate')}")
    print(f"[PORTFOLIO] csv: {result['decision_csv']}")


if __name__ == "__main__":
    main()
