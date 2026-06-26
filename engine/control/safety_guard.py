"""Central safety gate for bot actions."""

from __future__ import annotations

from pathlib import Path

from scout_auto_os.engine.bot_control import BotControl
from scout_auto_os.engine.control.bot_state import BotStateStore
from scout_auto_os.engine.control.manual_lock import ManualLockStore


class SafetyGuard:
    def __init__(self, control_dir: Path, bot_control_path: Path | None = None) -> None:
        self.control_dir = control_dir
        self.control_dir.mkdir(parents=True, exist_ok=True)
        self.bot_state = BotStateStore(control_dir)
        self.manual_locks = ManualLockStore(control_dir)
        self._bot_control = BotControl(bot_control_path) if bot_control_path else None

    def state(self) -> dict:
        return self.bot_state.load()

    def locked_symbols(self) -> set[str]:
        return self.manual_locks.locked_symbols()

    def is_manual_locked(self, symbol: str) -> bool:
        return self.manual_locks.is_locked(symbol)

    def can_enter_new(self, symbol: str = "") -> tuple[bool, str]:
        st = self.state()
        if st.get("emergency"):
            return False, "emergency_stop"
        if st.get("paused") or not st.get("new_entries_allowed", True):
            return False, "bot_paused"
        if symbol and self.is_manual_locked(symbol):
            return False, "manual_lock"
        return True, "ok"

    def can_evaluate_positions(self) -> tuple[bool, str]:
        st = self.state()
        if not st.get("position_eval_allowed", True):
            return False, "evaluation_disabled"
        return True, "ok"

    def can_auto_order(self) -> tuple[bool, str]:
        st = self.state()
        if st.get("emergency"):
            return False, "emergency_stop"
        if st.get("paused") or not st.get("auto_orders_allowed", True):
            return False, "bot_paused"
        return True, "ok"

    def can_auto_exit(self, symbol: str = "") -> tuple[bool, str]:
        st = self.state()
        if st.get("emergency"):
            return False, "emergency_stop"
        if not st.get("auto_exits_allowed", True):
            return False, "auto_exits_blocked"
        if symbol and self.is_manual_locked(symbol):
            return False, "manual_lock"
        return True, "ok"

    def can_modify_position(self, symbol: str) -> tuple[bool, str]:
        if self.is_manual_locked(symbol):
            return False, "manual_lock"
        st = self.state()
        if st.get("emergency"):
            return False, "emergency_stop"
        return True, "ok"

    def sync_to_bot_control(self) -> None:
        """Mirror control state into legacy bot_control.json for main loop."""
        if not self._bot_control:
            return
        st = self.state()
        data = self._bot_control.load()
        paused_or_emergency = bool(st.get("paused") or st.get("emergency"))
        data["kill_switch"] = paused_or_emergency or not st.get("new_entries_allowed", True)
        data["new_entries_allowed"] = bool(st.get("new_entries_allowed", True)) and not st.get("emergency")
        if st.get("emergency"):
            data["bot_stop_requested"] = False
        self._bot_control.save(data)

    def apply_legacy_control(self, control: dict) -> dict:
        """Merge bot_control.json changes into control bot_state (one-way read)."""
        st = self.bot_state.load()
        if control.get("kill_switch") and not st.get("emergency"):
            st = self.bot_state.pause(by=control.get("updated_by", "legacy"))
        return st
