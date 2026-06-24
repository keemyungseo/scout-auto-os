"""R015 Live Execution Engine — paper + real via paper_mode toggle."""

from __future__ import annotations

import csv
import threading
import time
import uuid
from pathlib import Path

from scout_auto_os.engine.binance_client import BinanceClient, OrderResult
from scout_auto_os.storage.db import Database, now_kst


class ExecutionEngine:
    """
    Unified execution interface.
    paper_mode=true  → paper fill (R013 behaviour)
    paper_mode=false → Binance Futures MARKET orders
    """

    def __init__(self, config: dict, db: Database, csv_dir: Path, root: Path) -> None:
        self.config = config
        self.db = db
        self.csv_dir = csv_dir
        self.csv_dir.mkdir(parents=True, exist_ok=True)
        exec_cfg = config.get("execution", {})
        self.paper_mode = bool(config.get("paper_mode", True))
        self.order_size_usdt = float(exec_cfg.get("order_size_usdt", 5))
        self.max_retries = int(exec_cfg.get("max_retries", 3))
        self.retry_delay = float(exec_cfg.get("retry_delay_sec", 2))
        self.leverage = int(exec_cfg.get("leverage", 1))
        self.fee_pct = 0.05
        rest_base = config.get("live_data", {}).get("rest_base", "https://fapi.binance.com")
        self.client = BinanceClient(rest_base=rest_base)
        self._lock = threading.Lock()
        self._pending: set[str] = set()
        orders_dir = root / "logs"
        orders_dir.mkdir(parents=True, exist_ok=True)
        self.orders_path = orders_dir / "orders.csv"

    @property
    def is_live(self) -> bool:
        return not self.paper_mode

    def paper_entry(
        self,
        position_id: str,
        symbol: str,
        side: str,
        price: float,
        reason: str,
        engine: str = "A6_LONG",
    ) -> str:
        if self.paper_mode:
            return self._paper_entry(position_id, symbol, side, price, reason, engine)
        return self._live_entry(position_id, symbol, side, price, reason, engine)

    def paper_exit(
        self,
        position_id: str,
        symbol: str,
        side: str,
        price: float,
        pnl_pct: float,
        reason: str,
        engine: str = "A6_LONG",
    ) -> str:
        if self.paper_mode:
            return self._paper_exit(position_id, symbol, side, price, pnl_pct, reason, engine)
        return self._live_exit(position_id, symbol, side, price, pnl_pct, reason, engine)

    def force_close(self, symbol: str, reason: str = "force_close") -> OrderResult | None:
        """Manual override: REDUCE_ONLY market close."""
        if self.paper_mode:
            return None
        pos = self.client.get_position(symbol)
        if not pos:
            return OrderResult(ok=False, symbol=symbol, error="no_exchange_position")
        close_side = "SELL" if pos.side == "LONG" else "BUY"
        return self._place_with_retry(
            symbol, close_side, pos.qty, reduce_only=True, action="FORCE_CLOSE", reason=reason,
        )

    def _paper_entry(self, position_id, symbol, side, price, reason, engine) -> str:
        trade_id = self.db.new_id("trd")
        ts = now_kst()
        self.db.execute(
            """INSERT INTO trades
            (trade_id, position_id, symbol, side, action, price, quantity_pct, timestamp, reason, engine)
            VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (trade_id, position_id, symbol, side, "ENTRY", price, 100.0, ts, reason, engine),
        )
        self._append_trades_csv(trade_id, position_id, symbol, "ENTRY", price, ts, reason)
        return trade_id

    def _paper_exit(self, position_id, symbol, side, price, pnl_pct, reason, engine) -> str:
        trade_id = self.db.new_id("trd")
        ts = now_kst()
        self.db.execute(
            """INSERT INTO trades
            (trade_id, position_id, symbol, side, action, price, pnl_pct, fee_pct, timestamp, reason, engine)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (trade_id, position_id, symbol, side, "EXIT", price, pnl_pct, self.fee_pct, ts, reason, engine),
        )
        self._append_trades_csv(trade_id, position_id, symbol, "EXIT", price, ts, reason, pnl_pct)
        return trade_id

    def _live_entry(self, position_id, symbol, side, price, reason, engine) -> str:
        if side.upper() != "LONG":
            self._log_order(symbol, "BUY", price, 0, "SKIP", "short_not_supported")
            raise RuntimeError("Only LONG supported in R015")
        key = f"entry:{symbol.upper()}"
        with self._lock:
            if key in self._pending:
                self._log_order(symbol, "BUY", price, 0, "DUPLICATE", "pending_entry")
                raise RuntimeError(f"duplicate entry blocked: {symbol}")
            self._pending.add(key)
        try:
            if self.client.get_position(symbol):
                self._log_order(symbol, "BUY", price, 0, "DUPLICATE", "exchange_position_exists")
                raise RuntimeError(f"exchange already has position: {symbol}")
            self._prepare_symbol(symbol)
            qty = self.client.qty_from_usdt(symbol, self.order_size_usdt, price)
            result = self._place_with_retry(
                symbol, "BUY", qty, reduce_only=False,
                action="ENTRY", reason=reason, position_id=position_id,
            )
            if not result.ok:
                raise RuntimeError(result.error)
            fill_px = result.price if result.price > 0 else price
            self.db.execute(
                "UPDATE positions SET entry_price=?, current_price=? WHERE position_id=?",
                (fill_px, fill_px, position_id),
            )
            trade_id = self.db.new_id("trd")
            ts = now_kst()
            self.db.execute(
                """INSERT INTO trades
                (trade_id, position_id, symbol, side, action, price, quantity_pct, timestamp, reason, engine)
                VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (trade_id, position_id, symbol, side, "ENTRY", fill_px, 100.0, ts, reason, engine),
            )
            self._append_trades_csv(trade_id, position_id, symbol, "ENTRY", fill_px, ts, reason)
            self.db.log_event("execution", "live_entry", {
                "symbol": symbol, "qty": result.qty, "price": fill_px, "order_id": result.order_id,
            })
            return trade_id
        finally:
            with self._lock:
                self._pending.discard(key)

    def _live_exit(self, position_id, symbol, side, price, pnl_pct, reason, engine) -> str:
        key = f"exit:{symbol.upper()}"
        with self._lock:
            if key in self._pending:
                self._log_order(symbol, "SELL", price, 0, "DUPLICATE", "pending_exit")
                return self.db.new_id("trd")
            self._pending.add(key)
        try:
            ex_pos = self.client.get_position(symbol)
            if not ex_pos:
                self._log_order(symbol, "SELL", price, 0, "MISMATCH", "no_exchange_position")
                return self._paper_exit(position_id, symbol, side, price, pnl_pct, reason, engine)
            close_side = "SELL" if ex_pos.side == "LONG" else "BUY"
            result = self._place_with_retry(
                symbol, close_side, ex_pos.qty, reduce_only=True,
                action="EXIT", reason=reason, position_id=position_id,
            )
            fill_px = result.price if result.ok and result.price > 0 else price
            entry_row = self.db.fetchone(
                "SELECT entry_price FROM positions WHERE position_id=?", (position_id,),
            )
            entry_px = float(entry_row["entry_price"]) if entry_row else price
            if entry_px > 0 and fill_px > 0:
                pnl_pct = (fill_px - entry_px) / entry_px * 100 if side == "LONG" else (entry_px - fill_px) / entry_px * 100
            trade_id = self.db.new_id("trd")
            ts = now_kst()
            status = "FILLED" if result.ok else "FAILED"
            self.db.execute(
                """INSERT INTO trades
                (trade_id, position_id, symbol, side, action, price, pnl_pct, fee_pct, timestamp, reason, engine)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (trade_id, position_id, symbol, side, "EXIT", fill_px, pnl_pct, self.fee_pct, ts, reason, engine),
            )
            self._append_trades_csv(trade_id, position_id, symbol, "EXIT", fill_px, ts, reason, pnl_pct)
            if not result.ok:
                self.db.log_event("execution", "live_exit_failed", {"symbol": symbol, "error": result.error})
                raise RuntimeError(f"live exit failed: {result.error}")
            return trade_id
        finally:
            with self._lock:
                self._pending.discard(key)

    def _prepare_symbol(self, symbol: str) -> None:
        try:
            self.client.set_margin_type(symbol, "ISOLATED")
        except Exception:
            pass
        try:
            self.client.set_leverage(symbol, self.leverage)
        except Exception:
            pass

    def _place_with_retry(
        self,
        symbol: str,
        side: str,
        qty: float,
        reduce_only: bool,
        action: str,
        reason: str = "",
        position_id: str = "",
    ) -> OrderResult:
        last = OrderResult(ok=False, symbol=symbol, side=side, qty=qty, error="no_attempt")
        for attempt in range(1, self.max_retries + 1):
            cid = f"scout_{action.lower()}_{uuid.uuid4().hex[:16]}"
            try:
                result = self.client.market_order(
                    symbol, side, qty, reduce_only=reduce_only, client_order_id=cid,
                )
            except Exception as exc:
                result = OrderResult(ok=False, symbol=symbol, side=side, qty=qty, error=str(exc))
            status = "FILLED" if result.ok else f"RETRY_{attempt}"
            self._log_order(
                symbol, f"{side}{'_REDUCE' if reduce_only else ''}",
                result.price, qty, status, result.error or reason,
            )
            if result.ok:
                return result
            last = result
            if attempt < self.max_retries:
                time.sleep(self.retry_delay * attempt)
        self.db.log_event("execution", "order_failed", {
            "symbol": symbol, "side": side, "qty": qty, "action": action,
            "error": last.error, "position_id": position_id,
        })
        return last

    def _log_order(
        self, symbol: str, action: str, price: float, qty: float,
        result: str, error: str = "",
    ) -> None:
        write_header = not self.orders_path.exists()
        with self.orders_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["time", "symbol", "action", "price", "qty", "result", "error"])
            if write_header:
                w.writeheader()
            w.writerow({
                "time": now_kst(),
                "symbol": symbol,
                "action": action,
                "price": price,
                "qty": qty,
                "result": result,
                "error": error,
            })

    def _append_trades_csv(self, trade_id, position_id, symbol, action, price, ts, reason, pnl_pct=None):
        row = {
            "trade_id": trade_id, "position_id": position_id, "symbol": symbol,
            "action": action, "price": price, "timestamp": ts, "reason": reason,
        }
        if pnl_pct is not None:
            row["pnl_pct"] = pnl_pct
        import csv as csvmod
        path = self.csv_dir / "trades.csv"
        write_header = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as f:
            w = csvmod.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                w.writeheader()
            w.writerow(row)
