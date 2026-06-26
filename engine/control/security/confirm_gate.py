"""Future double-confirm gate for destructive control actions (foundation only)."""

from __future__ import annotations

from scout_auto_os.engine.control.security.config import CONFIRM_HEADER

# Destructive actions that will require X-SCOUT-Confirm in a future release.
DESTRUCTIVE_ACTIONS = frozenset({
    "emergency_stop",
    "bot_pause",
    "bot_resume",
    "manual_close",
    "portfolio_override",
})


def requires_confirm_header(action: str) -> bool:
    """Return True when action will need confirm dialog (not enforced yet)."""
    return action in DESTRUCTIVE_ACTIONS


def confirm_header_present(headers: dict) -> bool:
    """Check optional confirm header — reserved for V2."""
    return bool(headers.get(CONFIRM_HEADER) or headers.get(CONFIRM_HEADER.lower()))
