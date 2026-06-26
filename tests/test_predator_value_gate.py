"""Predator Value Gate V1 tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(ROOT))

from scout_auto_os.engine.predator.predator_output import enrich_predator_candidate, validate_predator_output
from scout_auto_os.engine.predator.trade_contract import build_trade_contract
from scout_auto_os.engine.predator.value_gate import (
    GateAction,
    evaluate_gate,
    is_manual_protected,
    recommended_size,
)


class ValueGateRuleTests(unittest.TestCase):
    def test_size_tiers(self) -> None:
        self.assertEqual(recommended_size(30), 0.0)
        self.assertEqual(recommended_size(55), 0.1)
        self.assertEqual(recommended_size(65), 0.3)
        self.assertEqual(recommended_size(75), 0.6)
        self.assertEqual(recommended_size(85), 1.0)

    def test_skip_below_50(self) -> None:
        g = evaluate_gate(40, dna_type="TYPE_0", runner_probability=0.9)
        self.assertEqual(g["action"], GateAction.SKIP.value)

    def test_type1_forced_skip(self) -> None:
        g = evaluate_gate(90, dna_type="TYPE_1", runner_probability=0.1)
        self.assertEqual(g["action"], GateAction.SKIP.value)
        self.assertTrue(g["shadow_only"])

    def test_manual_no_action(self) -> None:
        pos = {"source": "MANUAL", "manual_lock": 1, "auto_manage": 0}
        self.assertTrue(is_manual_protected(pos))
        g = evaluate_gate(90, dna_type="TYPE_0", runner_probability=0.9, is_manual_protected=True)
        self.assertEqual(g["action"], GateAction.NO_ACTION.value)


class PredatorOutputTests(unittest.TestCase):
    def test_enriched_output_and_contract(self) -> None:
        cand = {"symbol": "WLDUSDT", "side": "long", "entry_score": 70}
        preds = {
            "value_score": 72,
            "predicted_roi": 15.0,
            "predicted_peak_roi": 20.0,
            "predicted_drawdown": 5.0,
            "predicted_win_prob": 0.85,
            "predicted_sharpe": 1.2,
            "predicted_dna_type": "TYPE_0",
            "runner_probability": 0.92,
        }
        out = enrich_predator_candidate(cand, preds)
        self.assertEqual(out["recommended_size"], 0.6)
        self.assertNotIn("expected_hold_time", out["trade_contract"])
        errs = validate_predator_output(out)
        self.assertEqual(errs, [])
        c = build_trade_contract(out)
        self.assertEqual(c["exit_profile"], "runner")
        self.assertFalse(c["early_exit_allowed"])


if __name__ == "__main__":
    unittest.main()
