"""Bot pause / resume / emergency state."""

from __future__ import annotations

import json
from pathlib import Path

from scout_auto_os.storage.db import now_kst

DEFAULT = {
    "paused": False,
    "emergency": False,
    "new_entries_allowed": True,
    "auto_orders_allowed": True,
    "auto_exits_allowed": True,
    "position_eval_allowed": True,
    "updated_at": "",
    "updated_by": "system",
}


class BotStateStore:
    def __init__(self, control_dir: Path) -> None:
        self.path = control_dir / "bot_state.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.save(dict(DEFAULT))

    def load(self) -> dict:
        if not self.path.exists():
            return dict(DEFAULT)
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return dict(DEFAULT)
        return {**DEFAULT, **data}

    def save(self, data: dict, *, by: str = "system") -> dict:
        data = {**DEFAULT, **data}
        data["updated_at"] = now_kst()
        data["updated_by"] = by
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return data

    def pause(self, by: str = "operator", reason: str = "") -> dict:
        return self.save({
            "paused": True,
            "new_entries_allowed": False,
            "auto_orders_allowed": False,
            "auto_exits_allowed": True,
            "position_eval_allowed": True,
        }, by=by)

    def resume(self, by: str = "operator") -> dict:
        state = self.load()
        if state.get("emergency"):
            raise PermissionError("emergency_active_manual_ack_required")
        return self.save({
            "paused": False,
            "new_entries_allowed": True,
            "auto_orders_allowed": True,
            "auto_exits_allowed": True,
            "position_eval_allowed": True,
        }, by=by)

    def emergency_stop(self, by: str = "operator", reason: str = "") -> dict:
        return self.save({
            "paused": True,
            "emergency": True,
            "new_entries_allowed": False,
            "auto_orders_allowed": False,
            "auto_exits_allowed": False,
            "position_eval_allowed": True,
        }, by=by)

    def clear_emergency(self, by: str = "operator", reason: str = "") -> dict:
        state = self.load()
        if not state.get("emergency"):
            return state
        return self.save({
            "emergency": False,
            "paused": True,
            "new_entries_allowed": False,
            "auto_orders_allowed": False,
            "auto_exits_allowed": False,
            "position_eval_allowed": True,
        }, by=by)
