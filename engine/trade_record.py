"""Trade ledger hooks + V1.2 review context (V1.1)."""

from __future__ import annotations

from pathlib import Path

from scout_auto_os.storage.trade_record_db import TradeRecordDB


class TradeRecordService:
    def __init__(self, db_path: Path, trade_size_usdt: float, leverage: int) -> None:
        self.db = TradeRecordDB(db_path)
        self.trade_size_usdt = trade_size_usdt
        self.leverage = leverage
        self._pending_context: dict[str, dict] = {}
        self._hold_stats_provider = None

    def set_hold_stats_provider(self, provider) -> None:
        self._hold_stats_provider = provider

    def set_entry_context(self, symbol: str, context: dict) -> None:
        self._pending_context[symbol.upper()] = context

    @staticmethod
    def _pnl_usdt(trade_size_usdt: float, pnl_pct: float) -> float:
        return round(trade_size_usdt * pnl_pct / 100.0, 4)

    @staticmethod
    def _exit_analysis(symbol: str, exit_pnl_pct: float, hold_stats: dict) -> dict:
        max_p = float(hold_stats.get("max", exit_pnl_pct))
        min_p = float(hold_stats.get("min", exit_pnl_pct))
        drawdown = min_p
        if max_p > 0:
            quality = min(100.0, max(0.0, (exit_pnl_pct / max_p) * 100))
        else:
            quality = 50.0 if exit_pnl_pct >= 0 else 20.0
        if exit_pnl_pct >= max_p * 0.85:
            comment = "exit_near_peak"
        elif exit_pnl_pct < 0 and max_p > 3:
            comment = "gave_back_profit"
        elif exit_pnl_pct < min_p + 1:
            comment = "exit_near_trough"
        else:
            comment = "normal_exit"
        return {
            "max_profit_pct_during_hold": round(max_p, 4),
            "max_drawdown_pct_during_hold": round(drawdown, 4),
            "exit_quality_score": round(quality, 2),
            "exit_comment": comment,
            "missed_more_upside_pct_30m_after_exit": None,
            "post_exit_30m_return": None,
            "post_exit_2h_return": None,
        }

    def record_open(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        quantity: float,
        entry_reason: str,
    ) -> int:
        ctx = self._pending_context.pop(symbol.upper(), None)
        row_id = self.db.insert_open(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            quantity=quantity,
            trade_size_usdt=self.trade_size_usdt,
            leverage=self.leverage,
            entry_reason=entry_reason,
            context=ctx,
        )
        print(f"[DB] trade opened saved symbol={symbol} id={row_id}")
        if ctx:
            print(f"[REVIEW] entry context saved symbol={symbol} snapshot={ctx.get('top5_snapshot_id')}")
        return row_id

    def enrich_entry_context(self, symbol: str, context: dict) -> None:
        row_id = self.db.update_entry_context(symbol, context)
        if row_id:
            print(f"[REVIEW] entry context saved symbol={symbol} id={row_id}")

    def record_close(
        self,
        symbol: str,
        exit_price: float,
        exit_reason: str,
        realized_pnl_pct: float,
        status: str = "CLOSED",
    ) -> int | None:
        pnl_usdt = self._pnl_usdt(self.trade_size_usdt, realized_pnl_pct)
        hold_stats = {"max": realized_pnl_pct, "min": realized_pnl_pct}
        if self._hold_stats_provider:
            hold_stats = self._hold_stats_provider(symbol)
        exit_analysis = self._exit_analysis(symbol, realized_pnl_pct, hold_stats)
        row_id = self.db.close_latest_open(
            symbol=symbol,
            exit_price=exit_price,
            exit_reason=exit_reason,
            realized_pnl_pct=round(realized_pnl_pct, 4),
            realized_pnl_usdt=pnl_usdt,
            status=status,
            exit_analysis=exit_analysis,
        )
        if row_id is not None:
            print(f"[DB] trade closed updated symbol={symbol} id={row_id} status={status}")
            print(f"[REVIEW] exit analysis saved symbol={symbol} quality={exit_analysis['exit_quality_score']}")
        return row_id
