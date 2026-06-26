"""SCOUT Security Layer V1 tests."""

from __future__ import annotations

import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(ROOT))

from scout_auto_os.engine.control.control_api import ControlService, create_control_app
from scout_auto_os.engine.control.manual_close import DryRunCloseExecutor
from scout_auto_os.engine.control.security.auth import AuthManager, verify_password, hash_password
from scout_auto_os.engine.control.security.session_store import SessionStore
from scout_auto_os.tests.control_auth_helper import TEST_PASSWORD, login_client, test_password_hash

fastapi = unittest.skipUnless(
    __import__("importlib").util.find_spec("fastapi") is not None,
    "fastapi not installed",
)
bcrypt = unittest.skipUnless(
    __import__("importlib").util.find_spec("bcrypt") is not None,
    "bcrypt not installed",
)


@bcrypt
class PasswordTests(unittest.TestCase):
    def test_hash_and_verify(self) -> None:
        h = hash_password("secret")
        self.assertTrue(verify_password("secret", h))
        self.assertFalse(verify_password("wrong", h))


@fastapi
class SecurityLayerTests(unittest.TestCase):
    def setUp(self) -> None:
        from fastapi.testclient import TestClient

        self._tmp = tempfile.TemporaryDirectory()
        self.control_dir = Path(self._tmp.name)
        self.pw_hash = test_password_hash()
        self.svc = ControlService(self.control_dir, executor=DryRunCloseExecutor(self.control_dir))
        self.client = TestClient(create_control_app(self.svc, admin_password_hash=self.pw_hash))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_1_login_success(self) -> None:
        r = self.client.post("/auth/login", json={"password": TEST_PASSWORD})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["ok"])
        self.assertIn("scout_session", r.cookies)

    def test_2_login_failure(self) -> None:
        r = self.client.post("/auth/login", json={"password": "wrong"})
        self.assertEqual(r.status_code, 401)

    def test_3_session_maintained(self) -> None:
        login_client(self.client)
        r = self.client.get("/control/status")
        self.assertEqual(r.status_code, 200)

    def test_4_session_expired(self) -> None:
        store = SessionStore(ttl_seconds=1)
        auth = AuthManager(self.control_dir, password_hash=self.pw_hash, session_store=store)
        from fastapi.testclient import TestClient
        from scout_auto_os.engine.control.security.install import register_security
        from fastapi import FastAPI

        app = FastAPI()
        register_security(app, auth)

        @app.get("/control/status")
        def status() -> dict:
            return {"ok": True}

        client = TestClient(app)
        login_client(client)
        self.assertEqual(client.get("/control/status").status_code, 200)
        time.sleep(1.1)
        r = client.get("/control/status")
        self.assertEqual(r.status_code, 401)

    def test_5_logout(self) -> None:
        login_client(self.client)
        r = self.client.post("/auth/logout")
        self.assertEqual(r.status_code, 200)
        r2 = self.client.get("/control/status")
        self.assertEqual(r2.status_code, 401)

    def test_6_api_auth_failure(self) -> None:
        r = self.client.get("/control/status")
        self.assertEqual(r.status_code, 401)

    def test_7_api_auth_success(self) -> None:
        login_client(self.client)
        r = self.client.get("/control/dashboard-status")
        self.assertEqual(r.status_code, 200)

    def test_8_login_lockout_after_5_failures(self) -> None:
        for _ in range(5):
            self.client.post("/auth/login", json={"password": "bad"})
        r = self.client.post("/auth/login", json={"password": TEST_PASSWORD})
        self.assertEqual(r.status_code, 429)

    def test_command_center_shows_login_without_session(self) -> None:
        r = self.client.get("/command-center")
        self.assertEqual(r.status_code, 200)
        self.assertIn("LOGIN", r.text)

    def test_command_center_dashboard_after_login(self) -> None:
        login_client(self.client)
        r = self.client.get("/command-center")
        self.assertIn("SCOUT Command Center", r.text)
        self.assertIn("btn-logout", r.text)

    def test_security_log_written(self) -> None:
        self.client.post("/auth/login", json={"password": "bad"})
        login_client(self.client)
        self.client.post("/auth/logout")
        log_path = self.control_dir / "security.log"
        self.assertTrue(log_path.exists())
        text = log_path.read_text(encoding="utf-8")
        self.assertIn("LOGIN FAILED", text)
        self.assertIn("LOGIN SUCCESS", text)
        self.assertIn("LOGOUT", text)


if __name__ == "__main__":
    unittest.main()
