"""Predator Shadow Panel tests."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(ROOT))

from scout_auto_os.engine.control.control_api import ControlService, create_control_app
from scout_auto_os.engine.control.dashboard import load_template
from scout_auto_os.engine.control.manual_close import DryRunCloseExecutor
from scout_auto_os.engine.control.predator_shadow_status import build_predator_shadow_status

fastapi = unittest.skipUnless(
    __import__("importlib").util.find_spec("fastapi") is not None,
    "fastapi not installed",
)

SHADOW_HEADER = [
    "timestamp", "scan_id", "symbol", "side",
    "baseline_decision", "baseline_size", "policy_b_decision", "policy_b_size",
    "value_score", "runner_prob", "predicted_dna_type",
    "predicted_roi", "predicted_peak_roi", "predicted_drawdown", "predicted_win_prob",
    "reason", "manual_lock", "source", "auto_manage",
]


def _write_shadow_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SHADOW_HEADER, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


class PredatorShadowStatusTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_empty_when_files_missing(self) -> None:
        payload = build_predator_shadow_status(self.data_dir)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["recent_candidates"], [])
        self.assertEqual(payload["short_watch"], [])
        self.assertEqual(payload["total_candidates_today"], 0)

    def test_summary_json_load(self) -> None:
        d = self.data_dir / "runtime_shadow"
        d.mkdir(parents=True)
        (d / "value_gate_shadow_summary.json").write_text(json.dumps({
            "policy_name": "Soft 50s",
            "total_candidates_today": 5,
            "policy_enter_count": 2,
            "policy_skip_count": 3,
            "avg_value_score": 55.5,
        }), encoding="utf-8")
        payload = build_predator_shadow_status(self.data_dir)
        self.assertEqual(payload["policy_name"], "Soft 50s")
        self.assertEqual(payload["total_candidates_today"], 5)

    def test_recent_limit_30(self) -> None:
        rows = [{
            "timestamp": f"2026-06-01 10:{i:02d}:00",
            "symbol": f"SYM{i}",
            "side": "LONG",
            "baseline_decision": "ENTER",
            "baseline_size": "1.0",
            "policy_b_decision": "SKIP",
            "policy_b_size": "0",
            "value_score": "40",
            "runner_prob": "0.9",
            "predicted_dna_type": "TYPE_0",
            "predicted_roi": "1",
            "predicted_drawdown": "2",
            "predicted_win_prob": "0.5",
            "reason": "t",
            "manual_lock": "0",
            "source": "BOT",
            "auto_manage": "1",
        } for i in range(40)]
        _write_shadow_csv(self.data_dir / "runtime_shadow" / "value_gate_runtime_shadow.csv", rows)
        payload = build_predator_shadow_status(self.data_dir, recent_limit=30)
        self.assertEqual(len(payload["recent_candidates"]), 30)
        self.assertEqual(payload["recent_candidates"][0]["symbol"], "SYM39")

    def test_watch_limit_20(self) -> None:
        watch_path = self.data_dir / "runtime_shadow" / "short_false_accept_watch.csv"
        watch_path.parent.mkdir(parents=True, exist_ok=True)
        fields = ["timestamp", "scan_id", "symbol", "side", "value_score", "runner_prob",
                  "predicted_dna_type", "predicted_drawdown", "predicted_win_prob", "reason"]
        with watch_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for i in range(25):
                w.writerow({
                    "timestamp": f"t{i}", "symbol": f"W{i}", "value_score": "65",
                    "runner_prob": "0.55", "predicted_dna_type": "TYPE_0",
                    "predicted_drawdown": "10", "predicted_win_prob": "0.6", "reason": "r",
                })
        payload = build_predator_shadow_status(self.data_dir, watch_limit=20)
        self.assertEqual(len(payload["short_watch"]), 20)

    def test_manual_lock_candidate(self) -> None:
        _write_shadow_csv(self.data_dir / "runtime_shadow" / "value_gate_runtime_shadow.csv", [{
            "timestamp": "2026-06-01 10:00:00",
            "symbol": "WLDUSDT",
            "side": "LONG",
            "baseline_decision": "NO_ACTION",
            "baseline_size": "0",
            "policy_b_decision": "NO_ACTION",
            "policy_b_size": "0",
            "value_score": "90",
            "runner_prob": "0.95",
            "predicted_dna_type": "TYPE_0",
            "predicted_roi": "10",
            "predicted_drawdown": "5",
            "predicted_win_prob": "0.9",
            "reason": "manual",
            "manual_lock": "1",
            "source": "MANUAL",
            "auto_manage": "0",
        }])
        row = build_predator_shadow_status(self.data_dir)["recent_candidates"][0]
        self.assertEqual(row["manual_lock"], "1")
        self.assertEqual(row["policy_b_decision"], "NO_ACTION")


@fastapi
class PredatorShadowAPITests(unittest.TestCase):
    def setUp(self) -> None:
        from fastapi.testclient import TestClient

        self._tmp = tempfile.TemporaryDirectory()
        self.control_dir = Path(self._tmp.name) / "control"
        self.data_dir = Path(self._tmp.name) / "data"
        self.control_dir.mkdir()
        self.data_dir.mkdir()
        self.svc = ControlService(
            self.control_dir,
            executor=DryRunCloseExecutor(self.control_dir),
            data_dir=self.data_dir,
        )
        self.client = TestClient(create_control_app(self.svc))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_predator_shadow_api(self) -> None:
        r = self.client.get("/control/predator-shadow")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(data["ok"])
        self.assertTrue(data["dry_run"])
        self.assertIn("recent_candidates", data)

    def test_html_has_predator_panel(self) -> None:
        html = load_template()
        self.assertIn("Predator Shadow Panel", html)
        self.assertIn("predator-shadow-panel", html)
        self.assertIn("/control/predator-shadow", html)

    def test_dashboard_unaffected_when_shadow_empty(self) -> None:
        r = self.client.get("/control/dashboard-status")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["api_health"], "ok")


if __name__ == "__main__":
    unittest.main()
