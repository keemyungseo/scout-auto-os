"""Two-step manual close with confirmation."""

from __future__ import annotations

import json
import secrets
import uuid
from pathlib import Path
from typing import Protocol

from scout_auto_os.storage.db import now_kst


class CloseExecutor(Protocol):
    def close_position(self, symbol: str, *, source: str = "MANUAL_CLOSE") -> dict:
        ...


class DryRunCloseExecutor:
    """No live orders — records intent only."""

    def __init__(self, log_dir: Path | None = None) -> None:
        self.log_dir = log_dir
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def close_position(self, symbol: str, *, source: str = "MANUAL_CLOSE") -> dict:
        result = {
            "ok": True,
            "dry_run": True,
            "symbol": symbol.upper(),
            "source": source,
            "timestamp": now_kst(),
        }
        if self.log_dir:
            path = self.log_dir / "dry_run_closes.jsonl"
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(result) + "\n")
        return result


class ExecutionEngineCloseAdapter:
    """Wraps ExecutionEngine — only used after confirmation."""

    def __init__(self, execution, position_mgr) -> None:
        self.execution = execution
        self.position_mgr = position_mgr

    def close_position(self, symbol: str, *, source: str = "MANUAL_CLOSE") -> dict:
        sym = symbol.upper()
        closed = []
        for p in self.position_mgr.open_positions():
            if p["symbol"].upper() != sym:
                continue
            self.position_mgr.close_by_user(p["position_id"], self.execution, alert_mgr=None)
            closed.append(p["position_id"])
        if not closed:
            return {"ok": False, "symbol": sym, "source": source, "error": "no_open_position"}
        return {"ok": True, "symbol": sym, "source": source, "position_ids": closed, "dry_run": False}


class PendingActionStore:
    def __init__(self, control_dir: Path) -> None:
        self.path = control_dir / "pending_actions.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write({"actions": []})

    def _read(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, FileNotFoundError):
            return {"actions": []}

    def _write(self, data: dict) -> None:
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def create_close_request(self, symbol: str, *, by: str = "operator", reason: str = "") -> dict:
        sym = symbol.upper()
        action_id = str(uuid.uuid4())
        confirm_id = secrets.token_hex(8)
        entry = {
            "id": action_id,
            "action": "manual_close",
            "symbol": sym,
            "status": "confirmation_required",
            "confirm_id": confirm_id,
            "requested_by": by,
            "reason": reason,
            "created_at": now_kst(),
        }
        data = self._read()
        data["actions"] = [a for a in data.get("actions", []) if a.get("symbol") != sym]
        data["actions"].append(entry)
        self._write(data)
        return entry

    def confirm_close(self, symbol: str, confirm_id: str, executor: CloseExecutor) -> dict:
        sym = symbol.upper()
        data = self._read()
        pending = None
        remaining = []
        for a in data.get("actions", []):
            if a.get("symbol") == sym and a.get("status") == "confirmation_required":
                pending = a
            else:
                remaining.append(a)
        if not pending:
            return {"ok": False, "error": "no_pending_close", "symbol": sym}
        if pending.get("confirm_id") != confirm_id:
            return {"ok": False, "error": "invalid_confirm_id", "symbol": sym}
        result = executor.close_position(sym, source="MANUAL_CLOSE")
        pending["status"] = "executed" if result.get("ok") else "failed"
        pending["executed_at"] = now_kst()
        pending["result"] = result
        data["actions"] = remaining
        self._write(data)
        return {"ok": result.get("ok", False), "symbol": sym, "confirm_id": confirm_id, "result": result}
