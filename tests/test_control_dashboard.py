"""SCOUT Command Center dashboard tests."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(ROOT))

from scout_auto_os.engine.control.control_api import ControlService, create_control_app
from scout_auto_os.engine.control.dashboard import dashboard_status, load_template
from scout_auto_os.engine.control.manual_close import DryRunCloseExecutor

fastapi = unittest.skipUnless(
    __import__("importlib").util.find_spec("fastapi") is not None,
    "fastapi not installed",
)


@fastapi
class CommandCenterDashboardTests(unittest.TestCase):
    def setUp(self) -> None:
        from fastapi.testclient import TestClient

        self._tmp = tempfile.TemporaryDirectory()
        self.control_dir = Path(self._tmp.name)
        self.svc = ControlService(
            self.control_dir,
            executor=DryRunCloseExecutor(self.control_dir),
        )
        self.client = TestClient(create_control_app(self.svc))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_command_center_page_loads(self) -> None:
        r = self.client.get("/command-center")
        self.assertEqual(r.status_code, 200)
        self.assertIn("SCOUT Command Center", r.text)
        self.assertIn("DRY RUN MODE", r.text)
        self.assertIn("DryRunCloseExecutor", r.text)

    def test_dashboard_status_endpoint(self) -> None:
        r = self.client.get("/control/dashboard-status")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("paused", data)
        self.assertIn("recent_logs", data)
        self.assertEqual(data["executor"], "DryRunCloseExecutor")

    def test_pause_resume_via_api(self) -> None:
        self.client.post("/control/bot/pause", json={"requested_by": "test"})
        st = self.client.get("/control/dashboard-status").json()
        self.assertTrue(st["paused"])
        self.client.post("/control/bot/resume", json={"requested_by": "test"})
        st = self.client.get("/control/dashboard-status").json()
        self.assertFalse(st["paused"])

    def test_emergency_requires_clear_before_resume(self) -> None:
        self.client.post("/control/bot/emergency-stop", json={"requested_by": "test"})
        r = self.client.post("/control/bot/resume", json={"requested_by": "test"})
        self.assertEqual(r.status_code, 403)
        self.client.post("/control/bot/clear-emergency", json={"requested_by": "test"})
        self.client.post("/control/bot/resume", json={"requested_by": "test"})
        st = self.client.get("/control/dashboard-status").json()
        self.assertFalse(st["emergency"])

    def test_wld_manual_lock_in_status(self) -> None:
        self.client.post(
            "/control/position/WLDUSDT/manual-lock",
            json={"requested_by": "dashboard"},
        )
        st = self.client.get("/control/dashboard-status").json()
        self.assertIn("WLDUSDT", st["manual_locks"])
        self.assertGreaterEqual(st["manual_locks_count"], 1)

    def test_manual_close_two_step_dry_run(self) -> None:
        r = self.client.post(
            "/control/position/BTCUSDT/close",
            json={"requested_by": "dashboard"},
        )
        self.assertEqual(r.status_code, 200)
        confirm_id = r.json()["confirm_id"]
        self.assertTrue(confirm_id)
        st = self.client.get("/control/dashboard-status").json()
        self.assertGreaterEqual(st["pending_close_count"], 1)
        r2 = self.client.post(
            "/control/position/BTCUSDT/confirm-close",
            json={"confirm_id": confirm_id, "requested_by": "dashboard"},
        )
        self.assertEqual(r2.status_code, 200)
        self.assertTrue(r2.json().get("ok"))

    def test_control_log_in_dashboard_status(self) -> None:
        self.client.post("/control/bot/pause", json={"requested_by": "dash"})
        st = self.client.get("/control/dashboard-status").json()
        actions = [row["action"] for row in st["recent_logs"]]
        self.assertIn("bot_pause", actions)


class DashboardModuleTests(unittest.TestCase):
    def test_template_loads(self) -> None:
        html = load_template()
        self.assertIn("EMERGENCY STOP", html)
        self.assertNotIn("cdn.jsdelivr", html.lower())

    def test_dashboard_status_shape(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            svc = ControlService(Path(td), executor=DryRunCloseExecutor(Path(td)))
            svc.pause("t")
            d = dashboard_status(svc)
            self.assertTrue(d["paused"])
            self.assertEqual(d["executor"], "DryRunCloseExecutor")


if __name__ == "__main__":
    unittest.main()
