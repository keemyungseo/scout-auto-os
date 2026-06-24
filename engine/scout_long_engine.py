"""A6 Long engine — scan → select → paper entry."""

from __future__ import annotations

from scout_auto_os.storage.db import Database


class ScoutLongEngine:
    def __init__(self, config: dict, db: Database) -> None:
        self.config = config
        self.db = db
        self.enabled = bool(config["long_engine"].get("enabled", True))

    def select_candidate(
        self,
        top5: list[dict],
        occupied: set[str],
        locked: set[str],
    ) -> dict | None:
        if not self.enabled:
            return None
        for row in top5:
            sym = row["symbol"]
            if sym in occupied or sym in locked:
                continue
            return row
        return None

    def try_entry(
        self,
        candidate: dict,
        position_mgr,
        execution,
        alert_mgr,
        risk_mgr,
    ) -> str | None:
        if not candidate:
            return None
        ok, _reason = risk_mgr.can_enter_long(position_mgr.long_slots_used())
        if not ok:
            return None
        if not position_mgr.has_slot():
            return None
        sym = candidate["symbol"]
        price = float(candidate["entry_price"])
        pid = position_mgr.create_position(
            sym, "LONG", price, "AUTO", "A6_LONG",
            a6_score=candidate["a6_score"],
            expected_ev=candidate["expected_ev"],
        )
        execution.paper_entry(
            pid, sym, "LONG", price,
            candidate.get("reason", "A6_top_candidate"),
            "A6_LONG",
        )
        alert_mgr.entry_alert(
            sym, "LONG", price,
            candidate.get("reason", ""),
            float(candidate.get("expected_ev", 0)),
        )
        self.db.log_event("scout_long_engine", "entry", {"symbol": sym, "position_id": pid})
        return pid

    def try_fill_slots(
        self,
        top5: list[dict],
        occupied: set[str],
        locked: set[str],
        position_mgr,
        execution,
        alert_mgr,
        risk_mgr,
    ) -> list[str]:
        """Fill up to max_long_slots independently (no meta rotation)."""
        entered: list[str] = []
        while position_mgr.has_slot():
            cand = self.select_candidate(top5, occupied, locked)
            if not cand:
                break
            pid = self.try_entry(cand, position_mgr, execution, alert_mgr, risk_mgr)
            if not pid:
                break
            entered.append(pid)
            occupied.add(cand["symbol"])
        return entered
