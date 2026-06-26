"""Command Center dashboard — HTML page + enriched status."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scout_auto_os.engine.control.manual_close import DryRunCloseExecutor

_PKG_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEMPLATE = _PKG_ROOT / "templates" / "command_center.html"


def dashboard_status(service) -> dict:
    """Status payload for the command center UI."""
    base = service.status()
    bot = base.get("bot_state", {})
    logs = service.logger.rows()
    last = logs[-1] if logs else {}
    pending = [
        a for a in base.get("pending_actions", [])
        if a.get("status") == "confirmation_required"
    ]
    is_dry = isinstance(service.executor, DryRunCloseExecutor)
    return {
        **base,
        "paused": bool(bot.get("paused")),
        "emergency": bool(bot.get("emergency")),
        "manual_locks_count": len(base.get("manual_locks", [])),
        "pending_close_count": len(pending),
        "last_control_action": last.get("action", "—"),
        "last_control_timestamp": last.get("timestamp", "—"),
        "api_health": "ok",
        "executor": "DryRunCloseExecutor" if is_dry else type(service.executor).__name__,
        "dry_run": is_dry,
        "recent_logs": list(reversed(logs[-20:])),
    }


def load_template(path: Path | None = None) -> str:
    tpl = path or DEFAULT_TEMPLATE
    if not tpl.exists():
        raise FileNotFoundError(f"Dashboard template not found: {tpl}")
    return tpl.read_text(encoding="utf-8")


def register_dashboard_routes(app: Any, service, template_path: Path | None = None) -> None:
    from fastapi.responses import HTMLResponse

    html = load_template(template_path)

    @app.get("/command-center", response_class=HTMLResponse)
    def command_center() -> HTMLResponse:
        return HTMLResponse(content=html)

    @app.get("/control/dashboard-status")
    def get_dashboard_status() -> dict:
        return dashboard_status(service)

    @app.get("/control/positions")
    def get_positions() -> dict:
        return service.positions()

    @app.get("/control/predator-shadow")
    def get_predator_shadow() -> dict:
        return service.predator_shadow()

    @app.get("/control/value-gate-results")
    def get_value_gate_results() -> dict:
        return service.value_gate_results()

    @app.get("/control/guardian-progress")
    def get_guardian_progress() -> dict:
        return service.guardian_progress()

    @app.get("/control/guardian-timeline")
    def get_guardian_timeline() -> dict:
        return service.guardian_timeline()

    @app.get("/control/guardian-outcome")
    def get_guardian_outcome() -> dict:
        return service.guardian_outcome()

    @app.get("/control/portfolio-decision")
    def get_portfolio_decision() -> dict:
        return service.portfolio_decision()
