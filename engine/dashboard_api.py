"""Dashboard API — real-time execution state JSON (R015)."""

from __future__ import annotations

import json
from pathlib import Path

from scout_auto_os.storage.db import Database, now_kst


def build_slot_states(open_positions: list[dict], max_slots: int = 2) -> list[dict]:
    """Slot1/Slot2 = A6_LONG positions by entry_time order."""
    auto_long = [
        p for p in open_positions
        if p.get("engine") == "A6_LONG" and p.get("status") == "OPEN"
    ]
    auto_long.sort(key=lambda x: x.get("entry_time", ""))
    slots: list[dict] = []
    for i in range(max_slots):
        n = i + 1
        if i < len(auto_long):
            p = auto_long[i]
            slots.append({
                "slot": n,
                "state": "OCCUPIED",
                "position_id": p["position_id"],
                "symbol": p["symbol"],
                "side": p["side"],
                "source": p["source"],
                "entry_time": p["entry_time"],
                "entry_price": p["entry_price"],
                "current_price": p.get("current_price"),
                "unrealized_pnl_pct": p.get("unrealized_pnl_pct"),
                "expected_ev": p.get("expected_ev"),
                "exit_plan": p.get("exit_plan"),
                "exit_reason": p.get("exit_reason") or "",
                "manual_lock": bool(p.get("manual_lock")),
            })
        else:
            slots.append({"slot": n, "state": "EMPTY"})
    return slots


class DashboardAPI:
    def __init__(self, config: dict, db: Database, data_dir: Path) -> None:
        self.config = config
        self.db = db
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.data_dir / "execution_api.json"

    def write(
        self,
        position_mgr,
        sync_summary: dict | None = None,
        live_health: dict | None = None,
        bot_control: dict | None = None,
        engine_state: str = "unknown",
    ) -> None:
        open_pos = self.db.fetchall(
            "SELECT * FROM positions WHERE status='OPEN' ORDER BY entry_time"
        )
        max_slots = int(self.config.get("position", {}).get("max_long_slots", 2))
        slots_used = len([p for p in open_pos if p.get("engine") == "A6_LONG"])
        manual_pos = [p for p in open_pos if p.get("source") == "MANUAL" or p.get("manual_lock")]

        payload = {
            "timestamp": now_kst(),
            "bot": {
                "state": engine_state,
                "paper_mode": bool(self.config.get("paper_mode", True)),
                "kill_switch": (bot_control or {}).get("kill_switch", False),
                "new_entries_allowed": (bot_control or {}).get("new_entries_allowed", True),
            },
            "order_size_usdt": float(self.config.get("execution", {}).get("order_size_usdt", 5)),
            "slots": {
                "used": slots_used,
                "max": max_slots,
                "available": max(0, max_slots - slots_used),
            },
            "slot1": next((s for s in build_slot_states(open_pos, max_slots) if s["slot"] == 1), {"slot": 1, "state": "EMPTY"}),
            "slot2": next((s for s in build_slot_states(open_pos, max_slots) if s["slot"] == 2), {"slot": 2, "state": "EMPTY"}),
            "slot_detail": build_slot_states(open_pos, max_slots),
            "positions": [
                {
                    "position_id": p["position_id"],
                    "symbol": p["symbol"],
                    "side": p["side"],
                    "source": p["source"],
                    "engine": p.get("engine"),
                    "entry_time": p["entry_time"],
                    "entry_price": p["entry_price"],
                    "current_price": p.get("current_price"),
                    "unrealized_pnl_pct": p.get("unrealized_pnl_pct"),
                    "expected_ev": p.get("expected_ev"),
                    "manual_lock": bool(p.get("manual_lock")),
                    "exit_plan": p.get("exit_plan"),
                    "exit_reason": p.get("exit_reason") or "",
                }
                for p in open_pos
            ],
            "manual_positions": [
                {
                    "symbol": p["symbol"],
                    "entry_price": p["entry_price"],
                    "manual_lock": bool(p.get("manual_lock")),
                }
                for p in manual_pos
            ],
            "sync": sync_summary or {},
            "live": live_health or {},
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
