"""Guardian Trade Thesis Engine V1 tests."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(ROOT))

from scout_auto_os.engine.guardian.guardian_thesis_log import merge_thesis_progress
from scout_auto_os.engine.guardian.progress_engine import evaluate_progress
from scout_auto_os.engine.guardian.thesis_replay import analyze_by_action, run_thesis_replay
from scout_auto_os.engine.guardian.thesis_store import GuardianThesisStore
from scout_auto_os.engine.guardian.trade_thesis import (
    build_entry_reason,
    build_thesis_from_replay_row,
    compute_confidence,
)


def _replay_row() -> dict:
    return {
        "trade_key": "2026-06-01 10:00:00|BTCUSDT|long",
        "scan_kst": "2026-06-01 10:00:00",
        "symbol": "BTCUSDT",
        "direction": "long",
        "predicted_roi": 15.0,
        "predicted_peak_roi": 20.0,
        "predicted_drawdown": 8.0,
        "predicted_win_prob": 0.85,
        "value_score": 65.0,
        "predicted_dna_type": "TYPE_0",
        "runner_probability": 0.92,
        "actual_roi": 10.0,
        "actual_peak_roi": 12.0,
        "actual_drawdown": 2.0,
    }


class TradeThesisTests(unittest.TestCase):
    def test_required_fields(self) -> None:
        t = build_thesis_from_replay_row(_replay_row())
        self.assertTrue(t.thesis_id.startswith("th_"))
        self.assertEqual(t.contract_id, _replay_row()["trade_key"])
        self.assertEqual(t.symbol, "BTCUSDT")
        self.assertIn("dna=TYPE_0", t.entry_reason)
        self.assertGreater(t.confidence, 0)

    def test_entry_reason_human_readable(self) -> None:
        reason = build_entry_reason(
            gate_reason="value_score_60_69",
            predicted_dna="TYPE_0",
            value_score=65,
            side="long",
            runner_probability=0.9,
        )
        self.assertIn("gate=value_score_60_69", reason)
        self.assertIn("score_band=60_69", reason)

    def test_confidence_bounded(self) -> None:
        c = compute_confidence(value_score=80, expected_win_prob=0.9, runner_probability=0.95)
        self.assertGreaterEqual(c, 0)
        self.assertLessEqual(c, 100)


class ThesisLogTests(unittest.TestCase):
    def test_merge_does_not_change_action(self) -> None:
        row = _replay_row()
        thesis = build_thesis_from_replay_row(row)
        contract = {
            "symbol": "BTCUSDT",
            "side": "long",
            "expected_roi": 15.0,
            "expected_peak_roi": 20.0,
            "expected_drawdown": 8.0,
            "expected_win_prob": 0.85,
            "value_score": 65.0,
            "dna_type": "TYPE_0",
            "exit_profile": "runner",
            "expected_horizon": 120,
            "contract_id": thesis.contract_id,
        }
        progress = evaluate_progress(
            contract,
            {"current_roi": 10, "elapsed_minutes": 60, "peak_roi": 12, "drawdown_from_peak": 2},
            contract_id=thesis.contract_id,
        )
        merged = merge_thesis_progress(thesis, progress)
        self.assertEqual(merged["action"], progress.recommendation)
        self.assertEqual(merged["guardian_state"], progress.guardian_state)
        self.assertEqual(merged["entry_reason"], thesis.entry_reason)
        self.assertIn("entry_reason", merged)
        self.assertIn("predicted_dna", merged)
        self.assertIn("confidence", merged)

    def test_log_fields(self) -> None:
        thesis = build_thesis_from_replay_row(_replay_row())
        contract = {
            "symbol": "BTCUSDT", "side": "long", "expected_roi": 15, "expected_peak_roi": 20,
            "expected_drawdown": 8, "expected_win_prob": 0.85, "value_score": 65,
            "dna_type": "TYPE_0", "exit_profile": "runner", "expected_horizon": 120,
            "contract_id": thesis.contract_id,
        }
        progress = evaluate_progress(contract, {"current_roi": 5, "elapsed_minutes": 30, "peak_roi": 6, "drawdown_from_peak": 1})
        row = merge_thesis_progress(thesis, progress)
        for f in (
            "symbol", "action", "reason", "entry_reason", "predicted_dna",
            "expected_roi", "expected_horizon", "confidence", "guardian_state",
        ):
            self.assertIn(f, row)


class ThesisStoreTests(unittest.TestCase):
    def test_save_and_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            store = GuardianThesisStore(data_dir)
            t = build_thesis_from_replay_row(_replay_row())
            store.save_batch([t])
            loaded = GuardianThesisStore(data_dir).get(t.contract_id)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.thesis_id, t.thesis_id)


class AnalysisTests(unittest.TestCase):
    def test_by_action_distribution(self) -> None:
        rows = [
            {"action": "EXIT", "entry_reason": "a", "predicted_dna": "TYPE_1", "confidence": 40},
            {"action": "EXIT", "entry_reason": "b", "predicted_dna": "TYPE_0", "confidence": 50},
            {"action": "HOLD", "entry_reason": "c", "predicted_dna": "TYPE_0", "confidence": 70},
        ]
        analysis = analyze_by_action(rows)
        self.assertEqual(analysis["EXIT"]["count"], 2)
        self.assertIn("TYPE_1", analysis["EXIT"]["predicted_dna"])


class ThesisReplayTests(unittest.TestCase):
    def test_replay_157(self) -> None:
        data_dir = ROOT / "scout_auto_os" / "data"
        if not (data_dir / "trade_dna" / "value_prediction.csv").exists():
            self.skipTest("replay bundle missing")
        result = run_thesis_replay(data_dir)
        self.assertEqual(result["trade_count"], 157)
        self.assertTrue(Path(result["thesis_log_csv"]).exists())
        analysis = json.loads(Path(result["analysis_json"]).read_text(encoding="utf-8"))
        self.assertTrue(len(analysis) > 0)
        for action, stats in analysis.items():
            self.assertIn("predicted_dna", stats)
            self.assertIn("confidence", stats)


class CommandCenterThesisTests(unittest.TestCase):
    def test_status_includes_thesis(self) -> None:
        from scout_auto_os.engine.control.guardian_progress_status import build_guardian_progress_status
        data_dir = ROOT / "scout_auto_os" / "data"
        if not (data_dir / "guardian" / "guardian_thesis_log.csv").exists():
            self.skipTest("thesis log not generated")
        payload = build_guardian_progress_status(data_dir)
        self.assertIn("thesis", payload)
        self.assertTrue(payload["data_sources"].get("thesis_log_csv"))


if __name__ == "__main__":
    unittest.main()
