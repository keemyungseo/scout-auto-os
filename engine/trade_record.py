"""Trade ledger hooks — save OPEN on entry, update on exit (V1.1)."""

from __future__ import annotations

from pathlib import Path

from scout_auto_os.storage.trade_record_db import TradeRecordDB


class TradeRecordService:
    def __init__(self, db_path: Path, trade_size_usdt: float, leverage: int) -> None:
        self.db = TradeRecordDB(db_path)
        self.trade_size_usdt = trade_size_usdt
        self.leverage = leverage

    @staticmethod
    def _pnl_usdt(trade_size_usdt: float, pnl_pct: float) -> float:
        return round(trade_size_usdt * pnl_pct / 100.0, 4)

    def record_open(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        quantity: float,
        entry_reason: str,
    ) -> int:
        row_id = self.db.insert_open(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            quantity=quantity,
            trade_size_usdt=self.trade_size_usdt,
            leverage=self.leverage,
            entry_reason=entry_reason,
        )
        print(f"[DB] trade opened saved symbol={symbol} id={row_id}")
        return row_id

    def record_close(
        self,
        symbol: str,
        exit_price: float,
        exit_reason: str,
        realized_pnl_pct: float,
        status: str = "CLOSED",
    ) -> int | None:
        pnl_usdt = self._pnl_usdt(self.trade_size_usdt, realized_pnl_pct)
        row_id = self.db.close_latest_open(
            symbol=symbol,
            exit_price=exit_price,
            exit_reason=exit_reason,
            realized_pnl_pct=round(realized_pnl_pct, 4),
            realized_pnl_usdt=pnl_usdt,
            status=status,
        )
        if row_id is not None:
            print(f"[DB] trade closed updated symbol={symbol} id={row_id} status={status}")
        return row_id
