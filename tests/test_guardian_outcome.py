"""Guardian Outcome Analyzer V1 tests."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(ROOT))

from scout_auto_os.engine.guardian.outcome_metrics import extract_trade_facts
from scout_auto_os.engine.guardian.outcome_replay import pearson, run_outcome_analysis
from scout_auto_os.engine.guardian.outcome_scores import evaluate_trade_outcome
from scout_auto_os.engine.guardian.trade_thesis import GuardianTradeThesis
from scout_auto_os.engine.control.guardian_outcome_status import build_guardian_outcome_status


def _thesis(**kw) -> GuardianTradeThesis:
    base = dict(
        thesis_id="th_test",
        contract_id="2026-06-01 10:00:00|BTCUSDT|long",
        prediction_key="2026-06-01 10:00:00|BTCUSDT|long",
        symbol="BTCUSDT",
        side="long",
        predator_version="season3_predator_v1",
        formula_name="policy_b_soft_50s",
        expected_roi=10.0,
        expected_peak_roi=12.0,
        expected_horizon=120,
        expected_drawdown=8.0,
        expected_win_prob=0.8,
        value_score=70.0,
        predicted_dna="TYPE_0",
        entry_reason="test entry",
        confidence=75.0,
    )
    base.update(kw)
    return GuardianTradeThesis(**base)


class OutcomeMetricsTests(unittest.TestCase):
    def test_extract_facts(self) -> None:
        points = [
            {"timestamp": "t1", "elapsed_minutes": 15, "current_roi": 5.0,
             "recommendation": "HOLD", "guardian_state": "ON_TRACK", "guardian_score": 60},
            {"timestamp": "t2", "elapsed_minutes": 30, "current_roi": 12.0,
             "recommendation": "TRAIL", "guardian_state": "AHEAD", "guardian_score": 75},
            {"timestamp": "t3", "elapsed_minutes": 45, "current_roi": 9.0,
             "recommendation": "EXIT", "guardian_state": "COMPLETED", "guardian_score": 70},
        ]
        facts = extract_trade_facts("2026-06-01 10:00:00|BTCUSDT|long", points, _thesis())
        self.assertIsNotNone(facts)
        assert facts is not None
        self.assertEqual(facts.peak_roi, 12.0)
        self.assertEqual(facts.final_roi, 9.0)
        self.assertEqual(facts.hold_count, 1)
        self.assertEqual(facts.trail_start_minutes, 30)
        self.assertEqual(facts.exit_minutes, 45)
        self.assertAlmostEqual(facts.max_drawdown, 3.0, places=2)


class OutcomeScoreTests(unittest.TestCase):
    def test_high_score_on_good_trade(self) -> None:
        points = [
            {"elapsed_minutes": 15, "current_roi": 8.0, "recommendation": "HOLD",
             "guardian_state": "ON_TRACK", "guardian_score": 65},
            {"elapsed_minutes": 30, "current_roi": 11.0, "recommendation": "TRAIL",
             "guardian_state": "AHEAD", "guardian_score": 80},
            {"elapsed_minutes": 45, "current_roi": 10.0, "recommendation": "EXIT",
             "guardian_state": "COMPLETED", "guardian_score": 78},
        ]
        for p in points:
            p.setdefault("timestamp", "t")
        facts = extract_trade_facts("2026-06-01 10:00:00|BTCUSDT|long", points, _thesis())
        assert facts is not None
        ev = evaluate_trade_outcome(facts, points)
        self.assertGreaterEqual(ev.overall_guardian_score, 70)
        self.assertIn(ev.outcome_grade, ("EXCELLENT", "GOOD"))
        self.assertIn("grade=", ev.explanation)

    def test_pearson(self) -> None:
        self.assertAlmostEqual(pearson([1, 2, 3], [2, 4, 6]) or 0, 1.0, places=4)


class OutcomeReplayTests(unittest.TestCase):
    def test_replay_157(self) -> None:
        data_dir = ROOT / "scout_auto_os" / "data"
        if not (data_dir / "guardian" / "guardian_timeline.csv").exists():
            self.skipTest("timeline missing")
        result = run_outcome_analysis(data_dir)
        self.assertEqual(result["trade_count"], 157)
        self.assertGreater(result["avg_guardian_score"], 0)
        path = data_dir / "guardian" / "guardian_outcome_summary.json"
        self.assertTrue(path.exists())
        summary = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(len(summary.get("best_10", [])), 10)
        self.assertEqual(len(summary.get("worst_10", [])), 10)

    def test_api_payload(self) -> None:
        data_dir = ROOT / "scout_auto_os" / "data"
        if not (data_dir / "guardian" / "guardian_timeline.csv").exists():
            self.skipTest("timeline missing")
        if not (data_dir / "guardian" / "guardian_outcome_summary.json").exists():
            run_outcome_analysis(data_dir)
        payload = build_guardian_outcome_status(data_dir)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["mode"], "OUTCOME_ANALYSIS")
        self.assertIn("avg_guardian_score", payload["summary"])


if __name__ == "__main__":
    unittest.main()
