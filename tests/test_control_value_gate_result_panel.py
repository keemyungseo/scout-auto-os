"""Value Gate Result Panel tests."""

from __future__ import annotations

import csv
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(ROOT))

from scout_auto_os.engine.control.control_api import ControlService, create_control_app
from scout_auto_os.engine.control.dashboard import load_template
from scout_auto_os.engine.control.value_gate_result_status import build_value_gate_result_status

fastapi = unittest.skipUnless(
    __import__("importlib").util.find_spec("fastapi") is not None,
    "fastapi not installed",
)

REAL_DATA = ROOT / "scout_auto_os" / "data" / "runtime_shadow"


def _write_csv(path: Path, header: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


class ValueGateResultStatusTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_empty_when_files_missing(self) -> None:
        payload = build_value_gate_result_status(self.data_dir)
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["dry_run"])
        self.assertEqual(payload["summary"]["labeled_rows"], 0)
        self.assertEqual(payload["band_calibration"], [])
        self.assertEqual(payload["false_skips"], [])
        self.assertEqual(payload["false_accepts"], [])
        self.assertFalse(payload["has_data"])

    def test_summary_load(self) -> None:
        d = self.data_dir / "runtime_shadow"
        d.mkdir(parents=True)
        _write_csv(
            d / "value_gate_cache_fix_reevaluation.csv",
            ["policy", "policy_name", "trade_count", "enter_count", "skip_count",
             "weighted_roi", "sharpe", "mdd", "win_rate", "false_skip_count", "false_accept_count",
             "long_enter_count", "short_enter_count", "long_skip_count", "short_skip_count",
             "long_false_skip", "short_false_skip", "long_false_accept", "short_false_accept"],
            [{
                "policy": "B", "policy_name": "Soft 50s", "trade_count": "157",
                "enter_count": "74", "skip_count": "83",
                "weighted_roi": "409.4", "sharpe": "10.77", "mdd": "-6.08", "win_rate": "93.24",
                "false_skip_count": "11", "false_accept_count": "7",
                "long_enter_count": "29", "short_enter_count": "45",
                "long_skip_count": "46", "short_skip_count": "37",
                "long_false_skip": "7", "short_false_skip": "4",
                "long_false_accept": "2", "short_false_accept": "5",
            }],
        )
        (d / "value_gate_cache_fix_report.md").write_text(
            "**Verdict:** `CACHE_FIX_SUCCESS_KEEP_POLICY_B_SHADOW`\n", encoding="utf-8",
        )
        payload = build_value_gate_result_status(self.data_dir)
        s = payload["summary"]
        self.assertEqual(s["enter_count"], 74)
        self.assertEqual(s["skip_count"], 83)
        self.assertEqual(s["false_skip_count"], 11)
        self.assertEqual(s["verdict"], "CACHE_FIX_SUCCESS_KEEP_POLICY_B_SHADOW")

    def test_band_calibration_load(self) -> None:
        d = self.data_dir / "runtime_shadow"
        d.mkdir(parents=True)
        _write_csv(
            d / "value_gate_cache_fix_band_calibration.csv",
            ["score_band", "count", "avg_roi_2h", "avg_roi_4h", "avg_peak_roi",
             "win_rate", "mdd", "false_skip_rate", "false_accept_rate"],
            [{"score_band": "50-59", "count": "18", "avg_roi_2h": "25.1",
              "avg_roi_4h": "25.9", "avg_peak_roi": "36.4", "win_rate": "94.12",
              "mdd": "-3.87", "false_skip_rate": "5.56", "false_accept_rate": "11.11"}],
        )
        bands = build_value_gate_result_status(self.data_dir)["band_calibration"]
        self.assertEqual(len(bands), 1)
        self.assertEqual(bands[0]["band"], "50-59")
        self.assertEqual(bands[0]["avg_roi_2h"], "25.1")

    def test_false_skip_limit_20(self) -> None:
        d = self.data_dir / "runtime_shadow"
        d.mkdir(parents=True)
        rows = [{
            "symbol": f"S{i}", "side": "LONG", "value_score": "40",
            "runner_prob": "0.9", "predicted_dna_type": "TYPE_0",
            "actual_roi_2h": str(i), "actual_roi_4h": "0",
            "actual_peak_roi": str(i * 2), "actual_drawdown": "-5",
            "false_skip_reason": "peak",
        } for i in range(30)]
        _write_csv(
            d / "value_gate_cache_fix_false_skip.csv",
            list(rows[0].keys()),
            rows,
        )
        out = build_value_gate_result_status(self.data_dir, false_skip_limit=20)["false_skips"]
        self.assertEqual(len(out), 20)
        self.assertEqual(out[0]["symbol"], "S29")

    def test_false_accept_limit_20(self) -> None:
        d = self.data_dir / "runtime_shadow"
        d.mkdir(parents=True)
        rows = [{
            "symbol": f"A{i}", "side": "SHORT" if i % 2 else "LONG",
            "value_score": "60", "runner_prob": "0.8", "predicted_dna_type": "TYPE_0",
            "predicted_drawdown": "10", "actual_roi_2h": "-5", "actual_roi_4h": "-3",
            "actual_drawdown": "-12", "false_accept_reason": "dd",
            "is_short_false_accept": "1" if i % 2 else "0",
        } for i in range(25)]
        _write_csv(
            d / "value_gate_cache_fix_false_accept.csv",
            list(rows[0].keys()),
            rows,
        )
        out = build_value_gate_result_status(self.data_dir, false_accept_limit=20)["false_accepts"]
        self.assertEqual(len(out), 20)

    def test_mismatch_zero_from_health(self) -> None:
        if not REAL_DATA.exists():
            self.skipTest("real runtime_shadow data not present")
        payload = build_value_gate_result_status(ROOT / "scout_auto_os" / "data")
        h = payload["data_health"]
        self.assertEqual(h["policy_rule_mismatch_count"], 0)
        self.assertEqual(h["trade_key_mismatch_count"], 0)
        self.assertEqual(h["future_timestamp_count"], 0)
        self.assertEqual(h["labeled_4h_rows"], 157)


@fastapi
class ValueGateResultAPITests(unittest.TestCase):
    def setUp(self) -> None:
        from fastapi.testclient import TestClient

        self._tmp = tempfile.TemporaryDirectory()
        self.control_dir = Path(self._tmp.name) / "control"
        self.data_dir = Path(self._tmp.name) / "data"
        self.control_dir.mkdir()
        self.data_dir.mkdir()
        self.svc = ControlService(
            self.control_dir,
            data_dir=self.data_dir,
        )
        self.client = TestClient(create_control_app(self.svc))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_value_gate_results_api_empty(self) -> None:
        r = self.client.get("/control/value-gate-results")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(data["ok"])
        self.assertTrue(data["dry_run"])
        self.assertIn("summary", data)
        self.assertIn("band_calibration", data)
        self.assertIn("data_health", data)

    def test_html_has_value_gate_panel(self) -> None:
        html = load_template()
        self.assertIn("Value Gate Result Panel", html)
        self.assertIn("value-gate-result-panel", html)
        self.assertIn("/control/value-gate-results", html)

    def test_dashboard_unaffected_when_value_gate_empty(self) -> None:
        r = self.client.get("/control/dashboard-status")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["api_health"], "ok")
        r2 = self.client.get("/control/predator-shadow")
        self.assertEqual(r2.status_code, 200)
        self.assertTrue(r2.json()["ok"])

    def test_value_gate_api_failure_isolated(self) -> None:
        with patch.object(
            self.svc,
            "value_gate_results",
            side_effect=RuntimeError("boom"),
        ):
            with self.assertRaises(RuntimeError):
                self.svc.value_gate_results()
        r = self.client.get("/control/dashboard-status")
        self.assertEqual(r.status_code, 200)


if __name__ == "__main__":
    unittest.main()
