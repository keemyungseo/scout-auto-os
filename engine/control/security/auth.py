"""Password verification and auth manager."""

from __future__ import annotations

import os

import bcrypt

from scout_auto_os.engine.control.security.security_log import SecurityLogger
from scout_auto_os.engine.control.security.session_store import SessionStore


def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, password_hash: str) -> bool:
    if not plain or not password_hash:
        return False
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def load_password_hash(explicit: str | None = None) -> str:
    value = (explicit or os.environ.get("SCOUT_ADMIN_PASSWORD_HASH", "")).strip()
    if not value:
        raise RuntimeError(
            "SCOUT_ADMIN_PASSWORD_HASH is required for Command Center security"
        )
    return value


class AuthManager:
    def __init__(
        self,
        control_dir,
        *,
        password_hash: str,
        session_store: SessionStore | None = None,
        security_logger: SecurityLogger | None = None,
        cookie_secure: bool | None = None,
    ) -> None:
        from pathlib import Path

        self.control_dir = Path(control_dir)
        self.password_hash = password_hash
        self.sessions = session_store or SessionStore()
        self.logger = security_logger or SecurityLogger(self.control_dir)
        if cookie_secure is None:
            cookie_secure = os.environ.get("SCOUT_COOKIE_SECURE", "").lower() in (
                "1",
                "true",
                "yes",
            )
        self.cookie_secure = cookie_secure

    def login(self, password: str, ip: str) -> tuple[bool, str, str | None]:
        locked, remain = self.sessions.is_locked(ip)
        if locked:
            self.logger.log("LOGIN FAILED", ip, f"locked remaining={remain}s")
            return False, "locked", None

        if verify_password(password, self.password_hash):
            self.sessions.record_success(ip)
            sid = self.sessions.create_session(ip)
            self.logger.log("LOGIN SUCCESS", ip)
            return True, "ok", sid

        now_locked = self.sessions.record_failure(ip)
        detail = "locked_15m" if now_locked else "bad_password"
        self.logger.log("LOGIN FAILED", ip, detail)
        return False, detail, None

    def logout(self, session_id: str | None, ip: str) -> None:
        self.sessions.destroy(session_id)
        self.logger.log("LOGOUT", ip)

    def validate_session(self, session_id: str | None, ip: str) -> tuple[bool, str]:
        ok, reason = self.sessions.validate(session_id, ip)
        if not ok and reason == "expired" and session_id:
            self.logger.log("SESSION EXPIRED", ip)
        return ok, reason
