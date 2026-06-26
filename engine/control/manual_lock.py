"""Manual lock registry — symbols the bot must never touch."""

from __future__ import annotations

import json
from pathlib import Path

from scout_auto_os.storage.db import now_kst


class ManualLockStore:
    def __init__(self, control_dir: Path) -> None:
        self.path = control_dir / "manual_locks.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write({"locks": {}})

    def _read(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, FileNotFoundError):
            return {"locks": {}}

    def _write(self, data: dict) -> None:
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def locked_symbols(self) -> set[str]:
        data = self._read()
        locks = data.get("locks", {})
        return {sym for sym, meta in locks.items() if meta.get("active", True)}

    def is_locked(self, symbol: str) -> bool:
        return symbol.upper() in {s.upper() for s in self.locked_symbols()}

    def lock(self, symbol: str, *, by: str = "operator", reason: str = "") -> dict:
        sym = symbol.upper()
        data = self._read()
        locks = data.setdefault("locks", {})
        locks[sym] = {
            "active": True,
            "locked_at": now_kst(),
            "by": by,
            "reason": reason,
        }
        self._write(data)
        return locks[sym]

    def unlock(self, symbol: str, *, by: str = "operator") -> bool:
        sym = symbol.upper()
        data = self._read()
        locks = data.get("locks", {})
        if sym not in locks:
            return False
        del locks[sym]
        self._write(data)
        return True
