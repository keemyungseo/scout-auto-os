"""Guardian Decision Engine V1 tests."""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(ROOT))

from scout_auto_os.engine.guardian.decision_engine import decide
from scout_auto_os.engine.guardian.decision_rules import compute_metrics
from scout_auto_os.engine.guardian.decision_report import run_replay_decisions
from scout_auto_os.engine.guardian.guardian_actions import GuardianAction
from scout_auto_os.engine.guardian.guardian_decision_log import GuardianDecisionLog


def _contract(**kw) -> dict:
    base = {
        "symbol": "TESTUSDT",
        "side": "long",
        "expected_roi": 10.0,
        "expected_peak_roi": 15.0,
        "expected_drawdown": 8.0,
        "expected_win_prob": 0.8,
        "value_score": 60.0,
        "dna_type": "TYPE_0",
        "exit_profile": "runner",
        "early_exit_allowed": False,
        "trail_priority": True,
        "gate_action": "ENTER",
    }
    base.update(kw)
    return base


def _position(**kw) -> dict:
    base = {
        "current_roi": 4.0,
        "elapsed_minutes": 60,
        "peak_roi": 5.0,
        "drawdown_from_peak": 1.0,
        "expected_horizon": 120,
    }
    base.update(kw)
    return base


class MetricsTests(unittest.TestCase):
    def test_progress_ratio(self) -> None:
        m = compute_metrics(_contract(expected_roi=10.0), _position(current_roi=5.0))
        self.assertAlmostEqual(m.progress_ratio, 0.5)

    def test_time_progress(self) -> None:
        m = compute_metrics(_contract(), _position(elapsed_minutes=60, expected_horizon=120))
        self.assertAlmostEqual(m.time_progress, 0.5)

    def test_drawdown_pressure(self) -> None:
        m = compute_metrics(
            _contract(expected_drawdown=4.0),
            _position(drawdown_from_peak=5.0),
        )
        self.assertAlmostEqual(m.drawdown_pressure, 1.25)

    def test_overperformance(self) -> None:
        m = compute_metrics(
            _contract(expected_peak_roi=10.0),
            _position(current_roi=12.0, peak_roi=12.0, drawdown_from_peak=0),
        )
        self.assertTrue(m.overperformance)


class DecisionRuleTests(unittest.TestCase):
    def test_early_hold(self) -> None:
        d = decide(_contract(expected_roi=20.0), _position(current_roi=5.0, elapsed_minutes=30))
        self.assertEqual(d.action, GuardianAction.HOLD.value)
        self.assertIn("early", d.reason.lower())

    def test_drawdown_exit(self) -> None:
        d = decide(
            _contract(expected_drawdown=5.0),
            _position(drawdown_from_peak=6.0, current_roi=8.0, peak_roi=14.0),
        )
        self.assertEqual(d.action, GuardianAction.EXIT.value)
        self.assertIn("drawdown_pressure", d.reason)

    def test_overperformance_trail(self) -> None:
        d = decide(
            _contract(expected_roi=10.0, expected_peak_roi=12.0),
            _position(current_roi=15.0, peak_roi=15.0, drawdown_from_peak=0, elapsed_minutes=60),
        )
        self.assertEqual(d.action, GuardianAction.TRAIL.value)
        self.assertIn("TRAIL", d.reason)

    def test_at_target_trail(self) -> None:
        d = decide(
            _contract(expected_roi=10.0, expected_peak_roi=15.0),
            _position(current_roi=10.0, peak_roi=10.0, drawdown_from_peak=0, elapsed_minutes=60),
        )
        self.assertEqual(d.action, GuardianAction.TRAIL.value)

    def test_target_band_weakening_trail(self) -> None:
        d = decide(
            _contract(expected_roi=10.0, expected_drawdown=10.0),
            _position(current_roi=9.0, peak_roi=12.0, drawdown_from_peak=7.0, elapsed_minutes=90),
        )
        self.assertEqual(d.action, GuardianAction.TRAIL.value)
        self.assertIn("weakening", d.reason.lower())

    def test_met_extended_hold_exit(self) -> None:
        d = decide(
            _contract(symbol="METUSDT", expected_roi=3.0, expected_drawdown=8.0, exit_profile="early_exit"),
            _position(
                current_roi=1.0,
                elapsed_minutes=2880,
                peak_roi=7.0,
                drawdown_from_peak=6.0,
                expected_horizon=90,
            ),
            contract_id="met_test",
        )
        self.assertEqual(d.action, GuardianAction.EXIT.value)
        self.assertIn("MET", d.reason)

    def test_hei_trail(self) -> None:
        d = decide(
            _contract(
                symbol="HEIUSDT",
                expected_roi=30.0,
                expected_peak_roi=53.0,
                expected_drawdown=32.0,
                exit_profile="runner",
            ),
            _position(
                current_roi=35.0,
                elapsed_minutes=60,
                peak_roi=53.0,
                drawdown_from_peak=0,
                expected_horizon=90,
            ),
            contract_id="hei_test",
        )
        self.assertEqual(d.action, GuardianAction.TRAIL.value)

    def test_manual_no_action(self) -> None:
        d = decide(_contract(), _position(), manual=True)
        self.assertEqual(d.action, GuardianAction.NO_ACTION.value)

    def test_emergency_exit(self) -> None:
        d = decide(_contract(), _position(current_roi=-20.0, peak_roi=-5.0, drawdown_from_peak=15.0))
        self.assertEqual(d.action, GuardianAction.EMERGENCY_EXIT.value)

    def test_reason_is_human_readable(self) -> None:
        d = decide(_contract(), _position())
        self.assertIn("contract expected_roi", d.reason)
        self.assertIn("position current_roi", d.reason)
        self.assertIn("progress_ratio", d.reason)
        self.assertGreater(len(d.reason), 40)


class GuardianLogTests(unittest.TestCase):
    def test_csv_fields(self) -> None:
        d = decide(_contract(), _position(), contract_id="tk1")
        row = d.to_row()
        for field in ("symbol", "action", "reason", "progress_ratio", "contract_id"):
            self.assertIn(field, row)

    def test_log_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            logger = GuardianDecisionLog(out)
            rows = [decide(_contract(), _position(), contract_id="a").to_row()]
            logger.write_decisions(rows)
            self.assertTrue((out / "guardian_decision.csv").exists())
            self.assertTrue((out / "guardian_decision_log.csv").exists())
            with (out / "guardian_decision.csv").open(encoding="utf-8") as f:
                reader = csv.DictReader(f)
                row = next(reader)
                self.assertEqual(row["action"], GuardianAction.HOLD.value)


class ReplayTests(unittest.TestCase):
    def test_replay_157_if_data_present(self) -> None:
        data_dir = ROOT / "scout_auto_os" / "data"
        if not (data_dir / "trade_dna" / "value_prediction.csv").exists():
            self.skipTest("replay bundle missing")
        result = run_replay_decisions(data_dir)
        self.assertEqual(result["trade_count"], 157)
        self.assertTrue(Path(result["decision_csv"]).exists())
        self.assertTrue(result["scenarios"]["met"]["blocks_extended_hold"])
        self.assertTrue(result["scenarios"]["hei"]["switches_to_trail"])


if __name__ == "__main__":
    unittest.main()
