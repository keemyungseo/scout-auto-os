"""Guardian Timeline Replay V1 tests."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(ROOT))

from scout_auto_os.engine.guardian.timeline_analysis import analyze_timelines
from scout_auto_os.engine.guardian.timeline_curve import build_trade_snapshots, snapshots_from_klines
from scout_auto_os.engine.guardian.timeline_engine import evaluate_trade_timeline
from scout_auto_os.engine.guardian.timeline_replay import HEI_TRADE_ID, run_timeline_replay
from scout_auto_os.engine.control.guardian_timeline_status import timeline_for_trade


class TimelineCurveTests(unittest.TestCase):
    def test_kline_snapshots(self) -> None:
        klines = [
            [0, 100, 101, 99, 100.5, 1000],
            [0, 100.5, 102, 100, 101.5, 1100],
            [0, 101.5, 103, 101, 102.0, 1200],
        ]
        snaps = snapshots_from_klines(klines, "long", "2026-06-01 10:00:00")
        self.assertEqual(len(snaps), 3)
        self.assertEqual(snaps[0]["elapsed_minutes"], 15)
        self.assertGreater(snaps[-1]["current_roi"], 0)


class TimelineEngineTests(unittest.TestCase):
    def _row(self) -> dict:
        return {
            "trade_key": "2026-06-01 10:00:00|BTCUSDT|long",
            "scan_kst": "2026-06-01 10:00:00",
            "symbol": "BTCUSDT",
            "direction": "long",
            "predicted_roi": 10.0,
            "predicted_peak_roi": 15.0,
            "predicted_drawdown": 8.0,
            "predicted_win_prob": 0.8,
            "value_score": 65.0,
            "predicted_dna_type": "TYPE_0",
            "runner_probability": 0.9,
            "actual_roi": 8.0,
            "actual_peak_roi": 12.0,
        }

    def test_transitions_recorded(self) -> None:
        klines = [
            [0, 100, 100, 99, 100, 1000],
            [0, 100, 105, 99, 104, 1000],
            [0, 104, 106, 103, 105, 1000],
            [0, 105, 105, 100, 101, 1000],
        ]
        snaps = snapshots_from_klines(klines, "long", "2026-06-01 10:00:00")
        tl = evaluate_trade_timeline(self._row(), snaps)
        self.assertGreater(len(tl.points), 0)
        self.assertIn("guardian_state", tl.points[0])
        self.assertIn("reason", tl.points[0])

    def test_point_fields(self) -> None:
        snaps = [{"timestamp": "t", "elapsed_minutes": 15, "current_roi": 2.0,
                  "peak_roi": 3.0, "drawdown_from_peak": 1.0}]
        p = evaluate_trade_timeline(self._row(), snaps).points[0]
        for f in ("trade_id", "progress_ratio", "guardian_score", "recommendation"):
            self.assertIn(f, p)


class TimelineReplayTests(unittest.TestCase):
    def test_replay_157(self) -> None:
        data_dir = ROOT / "scout_auto_os" / "data"
        pkg = ROOT / "scout_auto_os"
        if not (data_dir / "trade_dna" / "value_prediction.csv").exists():
            self.skipTest("replay bundle missing")
        result = run_timeline_replay(data_dir, pkg)
        self.assertEqual(result["trade_count"], 157)
        self.assertGreater(result["timeline_points"], 500)
        self.assertTrue(Path(result["timeline_csv"]).exists())
        summary = json.loads(Path(result["summary_json"]).read_text(encoding="utf-8"))
        self.assertEqual(summary["trade_count"], 157)
        self.assertTrue(summary["featured"]["met_thesis_failed"])

    def test_hei_in_timeline(self) -> None:
        data_dir = ROOT / "scout_auto_os" / "data"
        if not (data_dir / "guardian" / "guardian_timeline.csv").exists():
            self.skipTest("run replay first")
        rows = timeline_for_trade(data_dir, HEI_TRADE_ID)
        self.assertGreater(len(rows), 0)

    def test_timeline_api_structure(self) -> None:
        from scout_auto_os.engine.control.guardian_timeline_status import build_guardian_timeline_status
        data_dir = ROOT / "scout_auto_os" / "data"
        if not (data_dir / "guardian" / "guardian_timeline_summary.json").exists():
            self.skipTest("summary missing")
        payload = build_guardian_timeline_status(data_dir)
        self.assertIn("trades_index", payload)
        self.assertIn("summary", payload)


if __name__ == "__main__":
    unittest.main()
