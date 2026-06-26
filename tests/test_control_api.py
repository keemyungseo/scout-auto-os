"""SCOUT Command Center V1 — Safety Control API tests."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(ROOT))

from scout_auto_os.engine.control.control_api import ControlService, create_control_app
from scout_auto_os.engine.control.manual_close import DryRunCloseExecutor
from scout_auto_os.engine.control.safety_guard import SafetyGuard

fastapi = unittest.skipUnless(
    __import__("importlib").util.find_spec("fastapi") is not None,
    "fastapi not installed",
)


class SafetyGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.control_dir = Path(self._tmp.name)
        self.guard = SafetyGuard(self.control_dir)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_pause_blocks_new_entry_allows_eval(self) -> None:
        svc = ControlService(self.control_dir)
        svc.pause("tester", "unit_test")
        ok, reason = self.guard.can_enter_new("BTCUSDT")
        self.assertFalse(ok)
        self.assertEqual(reason, "bot_paused")
        ok_eval, _ = self.guard.can_evaluate_positions()
        self.assertTrue(ok_eval)
        ok_exit, _ = self.guard.can_auto_exit("BTCUSDT")
        self.assertTrue(ok_exit)

    def test_emergency_blocks_all_auto_actions(self) -> None:
        svc = ControlService(self.control_dir)
        svc.emergency_stop("tester", "unit_test")
        self.assertFalse(self.guard.can_enter_new()[0])
        self.assertFalse(self.guard.can_auto_order()[0])
        self.assertFalse(self.guard.can_auto_exit()[0])
        self.assertTrue(self.guard.can_evaluate_positions()[0])

    def test_manual_lock_wld(self) -> None:
        svc = ControlService(self.control_dir)
        svc.manual_lock("WLDUSDT", "tester", "protect manual position")
        self.assertTrue(self.guard.is_manual_locked("WLDUSDT"))
        ok, reason = self.guard.can_enter_new("WLDUSDT")
        self.assertFalse(ok)
        self.assertEqual(reason, "manual_lock")
        ok_exit, reason_exit = self.guard.can_auto_exit("WLDUSDT")
        self.assertFalse(ok_exit)
        self.assertEqual(reason_exit, "manual_lock")
        ok_mod, _ = self.guard.can_modify_position("WLDUSDT")
        self.assertFalse(ok_mod)


class ControlServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.control_dir = Path(self._tmp.name)
        self.executor = DryRunCloseExecutor(self.control_dir)
        self.svc = ControlService(self.control_dir, executor=self.executor)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_manual_close_two_step(self) -> None:
        req = self.svc.request_close("ETHUSDT", "tester", "take profit")
        self.assertEqual(req["status"], "confirmation_required")
        confirm_id = req["confirm_id"]
        dry_path = self.control_dir / "dry_run_closes.jsonl"
        self.assertFalse(dry_path.exists())
        bad = self.svc.confirm_close("ETHUSDT", "wrong_id", "tester")
        self.assertFalse(bad["ok"])
        self.assertEqual(bad["error"], "invalid_confirm_id")
        self.assertFalse(dry_path.exists())
        ok = self.svc.confirm_close("ETHUSDT", confirm_id, "tester")
        self.assertTrue(ok["ok"])
        self.assertTrue(dry_path.exists())

    def test_resume_blocked_during_emergency(self) -> None:
        self.svc.emergency_stop("tester")
        with self.assertRaises(PermissionError):
            self.svc.resume("tester")
        self.svc.clear_emergency("supervisor", "acknowledged")
        result = self.svc.resume("supervisor")
        self.assertTrue(result["ok"])

    def test_all_actions_logged(self) -> None:
        self.svc.pause("alice", "maintenance")
        self.svc.manual_lock("WLDUSDT", "alice", "manual pos")
        req = self.svc.request_close("BTCUSDT", "alice")
        self.svc.confirm_close("BTCUSDT", req["confirm_id"], "alice")
        self.svc.manual_unlock("WLDUSDT", "alice")
        self.svc.resume("alice")
        rows = self.svc.logger.rows()
        actions = {r["action"] for r in rows}
        self.assertIn("bot_pause", actions)
        self.assertIn("manual_lock", actions)
        self.assertIn("manual_close_request", actions)
        self.assertIn("manual_close_confirm", actions)
        self.assertIn("manual_unlock", actions)
        self.assertIn("bot_resume", actions)
        for field in ("timestamp", "action", "requested_by", "status"):
            for r in rows:
                self.assertIn(field, r)
                self.assertTrue(r[field])


@fastapi
class ControlAPITests(unittest.TestCase):
    def setUp(self) -> None:
        from fastapi.testclient import TestClient

        self._tmp = tempfile.TemporaryDirectory()
        self.control_dir = Path(self._tmp.name)
        self.svc = ControlService(self.control_dir, executor=DryRunCloseExecutor(self.control_dir))
        self.client = TestClient(create_control_app(self.svc))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_pause_endpoint(self) -> None:
        r = self.client.post("/control/bot/pause", json={"requested_by": "api", "reason": "test"})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["ok"])
        self.assertTrue(r.json()["state"]["paused"])

    def test_confirm_close_bad_id(self) -> None:
        self.client.post("/control/position/BTCUSDT/close", json={"requested_by": "api"})
        r = self.client.post(
            "/control/position/BTCUSDT/confirm-close",
            json={"confirm_id": "bad", "requested_by": "api"},
        )
        self.assertEqual(r.status_code, 400)


if __name__ == "__main__":
    unittest.main()
