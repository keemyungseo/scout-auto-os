"""Dashboard / operator control file (read each tick by main loop)."""

from __future__ import annotations

import json
from pathlib import Path

from scout_auto_os.storage.db import now_kst

DEFAULT = {
    "kill_switch": False,
    "bot_stop_requested": False,
    "new_entries_allowed": True,
    "updated_at": "",
    "updated_by": "system",
}


class BotControl:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.save(DEFAULT)

    def load(self) -> dict:
        if not self.path.exists():
            return dict(DEFAULT)
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return dict(DEFAULT)
        return {**DEFAULT, **data}

    def save(self, data: dict) -> None:
        data["updated_at"] = now_kst()
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def set_kill_switch(self, active: bool, by: str = "dashboard") -> None:
        data = self.load()
        data["kill_switch"] = active
        data["new_entries_allowed"] = not active
        data["updated_by"] = by
        self.save(data)

    def request_bot_stop(self, by: str = "dashboard") -> None:
        data = self.load()
        data["bot_stop_requested"] = True
        data["kill_switch"] = True
        data["new_entries_allowed"] = False
        data["updated_by"] = by
        self.save(data)

    def clear_bot_stop(self) -> None:
        data = self.load()
        data["bot_stop_requested"] = False
        data["updated_by"] = "system"
        self.save(data)

    def bot_stop_requested(self) -> bool:
        return bool(self.load().get("bot_stop_requested", False))

    def kill_switch_active(self) -> bool:
        data = self.load()
        return bool(data.get("kill_switch", False))

    def new_entries_allowed(self) -> bool:
        data = self.load()
        if data.get("kill_switch"):
            return False
        return bool(data.get("new_entries_allowed", True))
