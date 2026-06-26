"""Risk gates for paper and future live."""

from __future__ import annotations

from scout_auto_os.storage.db import Database, now_kst


class RiskManager:
    def __init__(self, config: dict, db: Database) -> None:
        self.config = config
        self.db = db
        self.risk = config["risk"]
        self.position_cfg = config["position"]
        self.initial_capital = float(config.get("initial_capital", 10000))
        self._kill_switch_override: bool | None = None
        self._new_entries_allowed = True

    def apply_control(self, control: dict) -> None:
        """Apply runtime control from bot_control.json."""
        if "kill_switch" in control:
            self._kill_switch_override = bool(control["kill_switch"])
        if "new_entries_allowed" in control:
            self._new_entries_allowed = bool(control["new_entries_allowed"])

    @property
    def kill_switch(self) -> bool:
        if self._kill_switch_override is not None:
            return self._kill_switch_override
        return bool(self.risk.get("kill_switch", False))

    def can_enter_long(self, slots_used: int) -> tuple[bool, str]:
        if self.kill_switch or not self._new_entries_allowed:
            return False, "kill_switch_active"
        if slots_used >= int(self.position_cfg["max_long_slots"]):
            return False, "max_long_slots"
        if slots_used >= int(self.position_cfg["max_total_slots"]):
            return False, "max_total_slots"
        if self._daily_loss_breached():
            return False, "daily_loss_limit"
        return True, "ok"

    def can_enter_short(self, slots_used: int) -> tuple[bool, str]:
        if self.kill_switch or not self._new_entries_allowed:
            return False, "kill_switch_active"
        max_short = int(self.position_cfg.get("max_short_slots", 0))
        if max_short <= 0:
            return False, "short_disabled"
        if slots_used >= max_short:
            return False, "max_short_slots"
        total_used = slots_used  # caller should pass combined if needed
        if total_used >= int(self.position_cfg["max_total_slots"]):
            return False, "max_total_slots"
        if self._daily_loss_breached():
            return False, "daily_loss_limit"
        return True, "ok"

    def _daily_loss_breached(self) -> bool:
        limit = float(self.risk.get("daily_loss_limit_pct", -10))
        today = now_kst()[:10]
        rows = self.db.fetchall(
            "SELECT pnl_pct FROM trades WHERE action='EXIT' AND timestamp LIKE ?",
            (f"{today}%",),
        )
        if not rows:
            return False
        total = sum(r["pnl_pct"] or 0 for r in rows)
        return total <= limit

    def check_single_trade(self, pnl_pct: float) -> bool:
        limit = float(self.risk.get("max_single_trade_loss_pct", -12))
        return pnl_pct >= limit
