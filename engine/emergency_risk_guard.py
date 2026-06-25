"""Emergency Risk Guard — force exit on severe loss (V1.3)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime


@dataclass
class RiskGuardDecision:
    should_exit: bool
    reason: str = ""
    roi_pct: float = 0.0
    pnl_usdt: float = 0.0
    hold_minutes: int = 0


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    return float(raw)


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    return int(raw)


class EmergencyRiskGuard:
    def __init__(self, config: dict) -> None:
        rg = config.get("risk_guard", {})
        self.max_roi_loss = _float_env("MAX_POSITION_ROI_LOSS_PCT", float(rg.get("max_position_roi_loss_pct", -18)))
        self.max_loss_usdt = _float_env("MAX_POSITION_UNREALIZED_LOSS_USDT", float(rg.get("max_position_unrealized_loss_usdt", -0.60)))
        self.max_hold_neg = _int_env("MAX_HOLD_MINUTES_WITH_NEGATIVE_PNL", int(rg.get("max_hold_minutes_with_negative_pnl", 45)))
        self.leverage = int(config.get("execution", {}).get("leverage", 1))
        self.trade_size = float(config.get("execution", {}).get("order_size_usdt", 5))

    @staticmethod
    def hold_minutes(entry_time: str, now_str: str) -> int:
        try:
            t0 = datetime.strptime(entry_time, "%Y-%m-%d %H:%M:%S")
            t1 = datetime.strptime(now_str, "%Y-%m-%d %H:%M:%S")
            return int((t1 - t0).total_seconds() / 60)
        except ValueError:
            return 0

    def evaluate(
        self,
        pos: dict,
        price_pnl_pct: float,
        unrealized_usdt: float | None = None,
        now_str: str | None = None,
    ) -> RiskGuardDecision:
        from scout_auto_os.storage.db import now_kst

        now_str = now_str or now_kst()
        hold = self.hold_minutes(pos.get("entry_time", ""), now_str)
        roi = price_pnl_pct * self.leverage
        if unrealized_usdt is None:
            unrealized_usdt = self.trade_size * price_pnl_pct / 100.0 * self.leverage

        if roi <= self.max_roi_loss:
            return RiskGuardDecision(True, "roi_loss_limit", roi, unrealized_usdt, hold)
        if unrealized_usdt <= self.max_loss_usdt:
            return RiskGuardDecision(True, "unrealized_loss_usdt", roi, unrealized_usdt, hold)
        if hold >= self.max_hold_neg and price_pnl_pct < 0:
            return RiskGuardDecision(True, "hold_timeout_negative", roi, unrealized_usdt, hold)
        return RiskGuardDecision(False, "", roi, unrealized_usdt, hold)
