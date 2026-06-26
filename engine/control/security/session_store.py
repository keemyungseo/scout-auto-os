"""In-memory session store with sliding expiry."""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field

from scout_auto_os.engine.control.security.config import (
    LOCKOUT_MINUTES,
    MAX_LOGIN_FAILURES,
    SESSION_TTL_MINUTES,
)


@dataclass
class SessionStore:
    ttl_seconds: int = SESSION_TTL_MINUTES * 60
    max_failures: int = MAX_LOGIN_FAILURES
    lockout_seconds: int = LOCKOUT_MINUTES * 60
    _sessions: dict[str, dict] = field(default_factory=dict)
    _failures: dict[str, int] = field(default_factory=dict)
    _blocked_until: dict[str, float] = field(default_factory=dict)

    def create_session(self, ip: str) -> str:
        sid = secrets.token_urlsafe(32)
        now = time.time()
        self._sessions[sid] = {
            "ip": ip,
            "created_at": now,
            "expires_at": now + self.ttl_seconds,
            "last_activity": now,
        }
        return sid

    def validate(self, session_id: str | None, ip: str) -> tuple[bool, str]:
        if not session_id or session_id not in self._sessions:
            return False, "missing"
        session = self._sessions[session_id]
        now = time.time()
        if now > session["expires_at"]:
            del self._sessions[session_id]
            return False, "expired"
        session["expires_at"] = now + self.ttl_seconds
        session["last_activity"] = now
        session["ip"] = ip
        return True, "ok"

    def destroy(self, session_id: str | None) -> None:
        if session_id:
            self._sessions.pop(session_id, None)

    def is_locked(self, ip: str) -> tuple[bool, int]:
        until = self._blocked_until.get(ip, 0.0)
        now = time.time()
        if until > now:
            return True, int(until - now)
        if until:
            self._blocked_until.pop(ip, None)
            self._failures.pop(ip, None)
        return False, 0

    def record_failure(self, ip: str) -> bool:
        count = self._failures.get(ip, 0) + 1
        self._failures[ip] = count
        if count >= self.max_failures:
            self._blocked_until[ip] = time.time() + self.lockout_seconds
            return True
        return False

    def record_success(self, ip: str) -> None:
        self._failures.pop(ip, None)
        self._blocked_until.pop(ip, None)

    def expire_session(self, session_id: str) -> None:
        self.destroy(session_id)
