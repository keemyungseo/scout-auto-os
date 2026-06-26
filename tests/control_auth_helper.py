"""Shared auth helpers for Command Center tests."""

from __future__ import annotations

import bcrypt

TEST_PASSWORD = "testpass"


def test_password_hash(password: str = TEST_PASSWORD) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def login_client(client, password: str = TEST_PASSWORD) -> None:
    r = client.post("/auth/login", json={"password": password})
    if r.status_code != 200:
        raise AssertionError(f"login failed: {r.status_code} {r.text}")
