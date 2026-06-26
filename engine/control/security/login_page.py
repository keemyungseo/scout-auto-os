"""Login page template for Command Center."""

from __future__ import annotations

from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parents[3]
LOGIN_TEMPLATE = _PKG_ROOT / "templates" / "login.html"


def load_login_template(path: Path | None = None) -> str:
    tpl = path or LOGIN_TEMPLATE
    if not tpl.exists():
        raise FileNotFoundError(f"Login template not found: {tpl}")
    return tpl.read_text(encoding="utf-8")
