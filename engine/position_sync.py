"""Sync Binance exchange positions with local AUTO/USER positions (R015)."""

from __future__ import annotations

from scout_auto_os.engine.binance_client import BinanceClient, ExchangePosition
from scout_auto_os.storage.db import Database, now_kst


class PositionSync:
    def __init__(self, config: dict, db: Database, client: BinanceClient) -> None:
        self.config = config
        self.db = db
        self.client = client
        self.paper_mode = bool(config.get("paper_mode", True))
        self.max_long = int(config.get("position", {}).get("max_long_slots", 2))

    def sync(self, position_mgr, execution, alert_mgr) -> dict:
        """Reconcile exchange ↔ DB. Returns summary for dashboard."""
        summary = {
            "exchange_positions": 0,
            "db_open": 0,
            "synced": 0,
            "closed_by_user": 0,
            "imported_manual": 0,
            "mismatches": [],
            "slots_used": 0,
            "slots_max": self.max_long,
        }
        if self.paper_mode or not self.client.configured:
            open_rows = position_mgr.open_positions()
            summary["db_open"] = len(open_rows)
            summary["slots_used"] = position_mgr.long_slots_used()
            return summary

        try:
            exchange = {p.symbol: p for p in self.client.get_positions()}
        except Exception as exc:
            alert_mgr.error_alert("position_sync", str(exc))
            self.db.log_event("position_sync", "fetch_failed", {"error": str(exc)})
            return summary

        summary["exchange_positions"] = len(exchange)
        db_open = position_mgr.open_positions()
        summary["db_open"] = len(db_open)
        summary["slots_used"] = position_mgr.long_slots_used()
        db_by_symbol = {p["symbol"]: p for p in db_open}

        for sym, pos in db_by_symbol.items():
            ex = exchange.get(sym)
            if not ex:
                if pos["source"] == "AUTO" and pos["auto_manage"]:
                    self._mark_closed_by_user(pos, position_mgr, execution, alert_mgr)
                    summary["closed_by_user"] += 1
                continue
            summary["synced"] += 1
            entry = ex.entry_price
            px = ex.entry_price
            upnl_pct = 0.0
            if entry > 0:
                mark = entry + ex.unrealized_pnl / ex.qty if ex.qty else entry
                upnl_pct = (mark - entry) / entry * 100 if ex.side == "LONG" else (entry - mark) / entry * 100
            self.db.execute(
                """UPDATE positions SET current_price=?, unrealized_pnl_pct=?,
                   last_update_time=? WHERE position_id=? AND status='OPEN'""",
                (px, upnl_pct, now_kst(), pos["position_id"]),
            )

        for sym, ex in exchange.items():
            if sym in db_by_symbol:
                continue
            self._import_user_position(sym, ex, position_mgr)
            summary["imported_manual"] += 1
            alert_mgr.manual_alert(sym, "imported_from_exchange")

        for sym, ex in exchange.items():
            dbp = db_by_symbol.get(sym)
            if not dbp:
                continue
            if dbp["side"] != ex.side:
                summary["mismatches"].append(f"{sym}: side db={dbp['side']} ex={ex.side}")

        self.db.log_event("position_sync", "sync_complete", summary)
        return summary

    def _mark_closed_by_user(self, pos, position_mgr, execution, alert_mgr) -> None:
        px = pos.get("current_price") or pos["entry_price"]
        position_mgr.close_by_user(pos["position_id"], execution, alert_mgr, exit_price=px)

    def _import_user_position(self, symbol: str, ex: ExchangePosition, position_mgr) -> None:
        if symbol in position_mgr.occupied_symbols():
            return
        position_mgr.create_position(
            symbol, ex.side, ex.entry_price, "MANUAL", "USER",
            manual_lock=True, auto_manage=False,
        )

    def slots_available(self, position_mgr) -> int:
        return max(0, self.max_long - position_mgr.long_slots_used())
