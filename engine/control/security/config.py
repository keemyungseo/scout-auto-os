"""Security configuration constants."""

from __future__ import annotations

SESSION_COOKIE_NAME = "scout_session"
SESSION_TTL_MINUTES = 30
MAX_LOGIN_FAILURES = 5
LOCKOUT_MINUTES = 15
CONFIRM_HEADER = "X-SCOUT-Confirm"
