"""SCOUT Command Center — Safety Control API (FastAPI)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scout_auto_os.engine.control.bot_state import BotStateStore
from scout_auto_os.engine.control.control_logger import ControlLogger
from scout_auto_os.engine.control.manual_close import (
    CloseExecutor,
    DryRunCloseExecutor,
    PendingActionStore,
)
from scout_auto_os.engine.control.manual_lock import ManualLockStore
from scout_auto_os.engine.control.dashboard import register_dashboard_routes
from scout_auto_os.engine.control.safety_guard import SafetyGuard
from scout_auto_os.engine.control.security.auth import AuthManager, load_password_hash
from scout_auto_os.engine.control.security.install import register_security

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except ImportError:
    FastAPI = None  # type: ignore[misc, assignment]
    HTTPException = Exception  # type: ignore[misc, assignment]
    BaseModel = object  # type: ignore[misc, assignment]
    Field = lambda *a, **k: None  # type: ignore[misc, assignment]


class ControlRequest(BaseModel):
    requested_by: str = "operator"
    reason: str = ""


class ConfirmCloseRequest(BaseModel):
    confirm_id: str
    requested_by: str = "operator"


class ControlService:
    """Core control logic — usable without HTTP."""

    def __init__(
        self,
        control_dir: Path,
        *,
        executor: CloseExecutor | None = None,
        bot_control_path: Path | None = None,
        manual_override=None,
        data_dir: Path | None = None,
    ) -> None:
        self.control_dir = control_dir
        self.control_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir = data_dir or control_dir.parent
        self.logger = ControlLogger(control_dir)
        self.bot_state = BotStateStore(control_dir)
        self.manual_locks = ManualLockStore(control_dir)
        self.pending = PendingActionStore(control_dir)
        self.guard = SafetyGuard(control_dir, bot_control_path)
        self.executor = executor or DryRunCloseExecutor(control_dir)
        self._manual_override = manual_override

    def pause(self, requested_by: str = "operator", reason: str = "") -> dict:
        state = self.bot_state.pause(by=requested_by, reason=reason)
        self.guard.sync_to_bot_control()
        self.logger.log("bot_pause", requested_by=requested_by, status="ok", reason=reason)
        return {"ok": True, "state": state}

    def resume(self, requested_by: str = "operator", reason: str = "") -> dict:
        try:
            state = self.bot_state.resume(by=requested_by)
        except PermissionError as exc:
            self.logger.log("bot_resume", requested_by=requested_by, status="denied", reason=str(exc))
            raise
        self.guard.sync_to_bot_control()
        self.logger.log("bot_resume", requested_by=requested_by, status="ok", reason=reason)
        return {"ok": True, "state": state}

    def emergency_stop(self, requested_by: str = "operator", reason: str = "") -> dict:
        state = self.bot_state.emergency_stop(by=requested_by, reason=reason)
        self.guard.sync_to_bot_control()
        self.logger.log("emergency_stop", requested_by=requested_by, status="ok", reason=reason)
        return {"ok": True, "state": state}

    def clear_emergency(self, requested_by: str = "operator", reason: str = "") -> dict:
        state = self.bot_state.clear_emergency(by=requested_by, reason=reason)
        self.guard.sync_to_bot_control()
        self.logger.log("clear_emergency", requested_by=requested_by, status="ok", reason=reason)
        return {"ok": True, "state": state, "note": "still_paused_until_resume"}

    def request_close(self, symbol: str, requested_by: str = "operator", reason: str = "") -> dict:
        sym = symbol.upper()
        if self.manual_locks.is_locked(sym):
            self.logger.log("manual_close_request", symbol=sym, requested_by=requested_by,
                            status="denied", reason="manual_lock")
            raise PermissionError("manual_lock_active")
        entry = self.pending.create_close_request(sym, by=requested_by, reason=reason)
        self.logger.log(
            "manual_close_request", symbol=sym, requested_by=requested_by,
            status="confirmation_required", reason=reason, confirm_id=entry["confirm_id"],
        )
        return {
            "ok": True,
            "symbol": sym,
            "status": "confirmation_required",
            "confirm_id": entry["confirm_id"],
            "action_id": entry["id"],
        }

    def confirm_close(self, symbol: str, confirm_id: str, requested_by: str = "operator") -> dict:
        sym = symbol.upper()
        result = self.pending.confirm_close(sym, confirm_id, self.executor)
        if not result.get("ok") and result.get("error") == "invalid_confirm_id":
            self.logger.log(
                "manual_close_confirm", symbol=sym, requested_by=requested_by,
                status="denied", reason="invalid_confirm_id", confirm_id=confirm_id,
            )
            return result
        status = "executed" if result.get("ok") else "failed"
        self.logger.log(
            "manual_close_confirm", symbol=sym, requested_by=requested_by,
            status=status, confirm_id=confirm_id,
        )
        return result

    def manual_lock(self, symbol: str, requested_by: str = "operator", reason: str = "") -> dict:
        sym = symbol.upper()
        meta = self.manual_locks.lock(sym, by=requested_by, reason=reason)
        self._queue_override("manual_lock", sym, reason)
        self.logger.log("manual_lock", symbol=sym, requested_by=requested_by, status="ok", reason=reason)
        return {"ok": True, "symbol": sym, "lock": meta}

    def manual_unlock(self, symbol: str, requested_by: str = "operator", reason: str = "") -> dict:
        sym = symbol.upper()
        ok = self.manual_locks.unlock(sym, by=requested_by)
        if ok:
            self._queue_override("unlock", sym, reason)
        self.logger.log(
            "manual_unlock", symbol=sym, requested_by=requested_by,
            status="ok" if ok else "not_found", reason=reason,
        )
        return {"ok": ok, "symbol": sym}

    def status(self) -> dict:
        return {
            "bot_state": self.bot_state.load(),
            "manual_locks": sorted(self.manual_locks.locked_symbols()),
            "pending_actions": self.pending._read().get("actions", []),
        }

    def positions(self) -> dict:
        from scout_auto_os.engine.control.position_status import build_guardian_positions
        payload = build_guardian_positions(self.data_dir, self.control_dir)
        bot = self.bot_state.load()
        payload["bot_paused"] = bool(bot.get("paused"))
        payload["bot_emergency"] = bool(bot.get("emergency"))
        return payload

    def predator_shadow(self) -> dict:
        from scout_auto_os.engine.control.predator_shadow_status import build_predator_shadow_status
        bot = self.bot_state.load()
        return build_predator_shadow_status(
            self.data_dir,
            bot_paused=bool(bot.get("paused")),
            bot_emergency=bool(bot.get("emergency")),
        )

    def value_gate_results(self) -> dict:
        from scout_auto_os.engine.control.value_gate_result_status import build_value_gate_result_status
        return build_value_gate_result_status(self.data_dir)

    def guardian_progress(self) -> dict:
        from scout_auto_os.engine.control.guardian_progress_status import build_guardian_progress_status
        return build_guardian_progress_status(self.data_dir)

    def guardian_timeline(self) -> dict:
        from scout_auto_os.engine.control.guardian_timeline_status import build_guardian_timeline_status
        return build_guardian_timeline_status(self.data_dir)

    def guardian_outcome(self) -> dict:
        from scout_auto_os.engine.control.guardian_outcome_status import build_guardian_outcome_status
        return build_guardian_outcome_status(self.data_dir)

    def portfolio_decision(self) -> dict:
        from scout_auto_os.engine.control.portfolio_decision_status import build_portfolio_decision_status
        return build_portfolio_decision_status(self.data_dir)

    def _queue_override(self, action: str, symbol: str, details: str) -> None:
        if not self._manual_override:
            return
        data = self._manual_override.load()
        data.setdefault("events", []).append({
            "symbol": symbol, "action": action, "details": details,
        })
        self._manual_override.save(data)


def create_control_app(
    service: ControlService,
    *,
    admin_password_hash: str | None = None,
    cookie_secure: bool | None = None,
) -> Any:
    if FastAPI is None:
        raise ImportError("fastapi is required for create_control_app — pip install fastapi")

    app = FastAPI(title="SCOUT Command Center", version="1.0", docs_url=None, redoc_url=None)

    auth = AuthManager(
        service.control_dir,
        password_hash=load_password_hash(admin_password_hash),
        cookie_secure=cookie_secure,
    )
    register_security(app, auth)

    @app.get("/control/status")
    def get_status() -> dict:
        return service.status()

    @app.post("/control/bot/pause")
    def post_pause(body: ControlRequest) -> dict:
        return service.pause(body.requested_by, body.reason)

    @app.post("/control/bot/resume")
    def post_resume(body: ControlRequest) -> dict:
        try:
            return service.resume(body.requested_by, body.reason)
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc

    @app.post("/control/bot/emergency-stop")
    def post_emergency(body: ControlRequest) -> dict:
        return service.emergency_stop(body.requested_by, body.reason)

    @app.post("/control/bot/clear-emergency")
    def post_clear_emergency(body: ControlRequest) -> dict:
        return service.clear_emergency(body.requested_by, body.reason)

    @app.post("/control/position/{symbol}/close")
    def post_close(symbol: str, body: ControlRequest) -> dict:
        try:
            return service.request_close(symbol, body.requested_by, body.reason)
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc

    @app.post("/control/position/{symbol}/confirm-close")
    def post_confirm_close(symbol: str, body: ConfirmCloseRequest) -> dict:
        result = service.confirm_close(symbol, body.confirm_id, body.requested_by)
        if not result.get("ok") and result.get("error") == "invalid_confirm_id":
            raise HTTPException(status_code=400, detail="invalid_confirm_id")
        if not result.get("ok") and result.get("error") == "no_pending_close":
            raise HTTPException(status_code=404, detail="no_pending_close")
        return result

    @app.post("/control/position/{symbol}/manual-lock")
    def post_manual_lock(symbol: str, body: ControlRequest) -> dict:
        return service.manual_lock(symbol, body.requested_by, body.reason)

    @app.post("/control/position/{symbol}/manual-unlock")
    def post_manual_unlock(symbol: str, body: ControlRequest) -> dict:
        return service.manual_unlock(symbol, body.requested_by, body.reason)

    register_dashboard_routes(app, service)

    return app
