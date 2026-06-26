"""Guardian Progress Engine V1 tests."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(ROOT))

from scout_auto_os.engine.guardian.progress_config import load_progress_weights
from scout_auto_os.engine.guardian.progress_engine import evaluate_progress
from scout_auto_os.engine.guardian.progress_metrics import compute_progress_metrics
from scout_auto_os.engine.guardian.progress_replay import run_progress_replay
from scout_auto_os.engine.guardian.progress_score import compute_guardian_score
from scout_auto_os.engine.guardian.progress_state import classify_guardian_state


def _contract(**kw) -> dict:
    base = {
        "contract_id": "test|long",
        "symbol": "TESTUSDT",
        "side": "long",
        "expected_roi": 10.0,
        "expected_peak_roi": 15.0,
        "expected_drawdown": 8.0,
        "expected_win_prob": 0.8,
        "expected_horizon": 120,
        "value_score": 65.0,
        "dna_type": "TYPE_0",
        "exit_profile": "runner",
        "gate_action": "ENTER",
    }
    base.update(kw)
    return base


def _position(**kw) -> dict:
    base = {
        "current_roi": 5.0,
        "peak_roi": 6.0,
        "drawdown_from_peak": 1.0,
        "elapsed_minutes": 60,
    }
    base.update(kw)
    return base


class ProgressMetricsTests(unittest.TestCase):
    def test_ratios(self) -> None:
        m = compute_progress_metrics(_contract(expected_roi=10.0, expected_peak_roi=15.0), _position())
        self.assertAlmostEqual(m.progress_ratio, 0.5)
        self.assertAlmostEqual(m.time_progress, 0.5)
        self.assertAlmostEqual(m.peak_progress, 0.4)
        self.assertAlmostEqual(m.drawdown_pressure, 0.125)


class ScoreTests(unittest.TestCase):
    def test_score_bounded(self) -> None:
        m = compute_progress_metrics(_contract(), _position())
        score = compute_guardian_score(m, _contract())
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)


class StateTests(unittest.TestCase):
    def test_building(self) -> None:
        m = compute_progress_metrics(_contract(), _position(current_roi=2.0, elapsed_minutes=30))
        state, _ = classify_guardian_state(m, _contract())
        self.assertEqual(state, "BUILDING")

    def test_met_thesis_failed(self) -> None:
        m = compute_progress_metrics(
            _contract(symbol="METUSDT", expected_roi=3.0, expected_horizon=90),
            _position(current_roi=1.0, elapsed_minutes=2880, peak_roi=7.0, drawdown_from_peak=6.0),
        )
        state, lines = classify_guardian_state(m, _contract(symbol="METUSDT"))
        self.assertEqual(state, "THESIS_FAILED")
        self.assertTrue(any("time_progress" in ln for ln in lines))

    def test_ahead(self) -> None:
        m = compute_progress_metrics(
            _contract(expected_roi=10.0, expected_drawdown=8.0),
            _position(current_roi=12.0, peak_roi=17.0, drawdown_from_peak=5.0),
        )
        state, _ = classify_guardian_state(m, _contract())
        self.assertEqual(state, "AHEAD")


class EvaluateProgressTests(unittest.TestCase):
    def test_human_readable_reason(self) -> None:
        r = evaluate_progress(_contract(), _position())
        self.assertIn("metrics progress=", r.reason)
        self.assertIn("guardian_state=", r.reason)
        self.assertIn(r.guardian_state, r.reason)

    def test_recommendation_hold(self) -> None:
        r = evaluate_progress(_contract(), _position(current_roi=5.0, elapsed_minutes=60))
        self.assertIn(r.recommendation, ("HOLD", "BUILDING", "ON_TRACK"))

    def test_met_recommendation_exit(self) -> None:
        r = evaluate_progress(
            _contract(symbol="METUSDT", expected_roi=3.0, expected_horizon=90),
            _position(current_roi=1.0, elapsed_minutes=2880, peak_roi=7.0, drawdown_from_peak=6.0),
            contract_id="met",
        )
        self.assertEqual(r.guardian_state, "THESIS_FAILED")
        self.assertEqual(r.recommendation, "EXIT")

    def test_to_row_fields(self) -> None:
        row = evaluate_progress(_contract(), _position()).to_row()
        for f in (
            "contract_id", "symbol", "progress_ratio", "time_progress",
            "peak_progress", "drawdown_pressure", "guardian_score",
            "guardian_state", "recommendation", "reason",
        ):
            self.assertIn(f, row)


class ConfigTests(unittest.TestCase):
    def test_weights_normalize(self) -> None:
        w = load_progress_weights({
            "guardian": {"progress": {"weights": {
                "roi_progress": 1, "time_alignment": 1, "drawdown_health": 1,
                "value_score": 1, "win_probability": 1,
            }}},
        })
        total = w.roi_progress + w.time_alignment + w.drawdown_health + w.value_score + w.win_probability
        self.assertAlmostEqual(total, 1.0)


class ReplayTests(unittest.TestCase):
    def test_replay_157(self) -> None:
        data_dir = ROOT / "scout_auto_os" / "data"
        if not (data_dir / "trade_dna" / "value_prediction.csv").exists():
            self.skipTest("replay bundle missing")
        result = run_progress_replay(data_dir)
        self.assertEqual(result["trade_count"], 157)
        self.assertTrue(result["met_thesis_failed"])
        self.assertTrue(Path(result["progress_csv"]).exists())
        summary = json.loads(Path(result["summary_json"]).read_text(encoding="utf-8"))
        self.assertEqual(summary["total_rows"], 157)
        self.assertIn("state_counts", summary)


class CommandCenterStatusTests(unittest.TestCase):
    def test_build_status_empty(self) -> None:
        from scout_auto_os.engine.control.guardian_progress_status import build_guardian_progress_status
        with tempfile.TemporaryDirectory() as tmp:
            payload = build_guardian_progress_status(Path(tmp))
            self.assertTrue(payload["ok"])
            self.assertFalse(payload["has_data"])


if __name__ == "__main__":
    unittest.main()
