"""Entry Quality Guard — LIVE entry filter (V1.3). Does not modify A6 scoring."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from scout_auto_os.engine.market_snapshot import build_entry_market_snapshot


@dataclass
class EntryQualityResult:
    passed: bool
    block_reason: str = ""
    entry_quality_score: float = 0.0
    metrics: dict = field(default_factory=dict)


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    return float(raw)


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


class EntryQualityGuard:
    def __init__(self, config: dict) -> None:
        eq = config.get("entry_quality", {})
        self.min_24h_qv = _float_env("MIN_24H_QUOTE_VOLUME_USDT", float(eq.get("min_24h_quote_volume_usdt", 10_000_000)))
        self.min_1h_qv = _float_env("MIN_1H_QUOTE_VOLUME_USDT", float(eq.get("min_1h_quote_volume_usdt", 400_000)))
        self.min_5m_ratio = _float_env("MIN_5M_QUOTE_VOLUME_RATIO", float(eq.get("min_5m_quote_volume_ratio", 1.5)))
        self.max_pullback = _float_env("MAX_PULLBACK_FROM_15M_HIGH_PCT", float(eq.get("max_pullback_from_15m_high_pct", 2.5)))
        self.block_red = _bool_env("BLOCK_LAST_5M_RED_CANDLE", bool(eq.get("block_last_5m_red_candle", True)))
        self.max_slippage = _float_env("MAX_ESTIMATED_SLIPPAGE_PCT", float(eq.get("max_estimated_slippage_pct", 0.3)))
        self.block_negative_ev = _bool_env("BLOCK_NEGATIVE_EV", bool(eq.get("block_negative_ev", True)))
        self.rest_base = config.get("live_data", {}).get("rest_base", "https://fapi.binance.com")

    def evaluate(self, symbol: str, candidate: dict) -> EntryQualityResult:
        price = float(candidate.get("entry_price") or 0)
        expected_ev = candidate.get("expected_ev")
        metrics: dict = {"entry_quality_pass": False}

        try:
            snap = build_entry_market_snapshot(self.rest_base, symbol, price)
            metrics.update(snap)
        except Exception as exc:
            reason = "market_data_unavailable"
            print(f"[ENTRY BLOCKED] symbol={symbol} reason={reason} detail={exc}")
            return EntryQualityResult(False, reason, 0.0, {**metrics, "entry_block_reason": reason})

        score = 100.0
        checks: list[tuple[str, bool, float]] = []

        qv_ok = metrics["quote_volume_24h"] >= self.min_24h_qv or metrics["quote_volume_1h"] >= self.min_1h_qv
        checks.append(("low_quote_volume", qv_ok, 25.0))

        vol_ok = metrics["quote_volume_5m_ratio"] >= self.min_5m_ratio
        checks.append(("weak_5m_volume", vol_ok, 20.0))

        pb_ok = metrics["pullback_from_15m_high_pct"] <= self.max_pullback
        checks.append(("bad_timing_pullback", pb_ok, 20.0))

        red_ok = not (self.block_red and metrics["last_5m_candle_direction"] == "red")
        checks.append(("bad_timing_red_candle", red_ok, 15.0))

        slip_ok = metrics["estimated_slippage_pct"] <= self.max_slippage
        checks.append(("high_slippage", slip_ok, 10.0))

        ev_val = None if expected_ev is None else float(expected_ev)
        if self.block_negative_ev:
            ev_ok = ev_val is not None and ev_val >= 0
        else:
            ev_ok = True
        checks.append(("negative_ev", ev_ok, 10.0))

        for reason_key, ok, weight in checks:
            if not ok:
                metrics["entry_block_reason"] = reason_key
                metrics["entry_quality_pass"] = False
                metrics["entry_quality_score"] = round(max(0.0, score - weight), 2)
                print(f"[ENTRY BLOCKED] symbol={symbol} reason={reason_key}")
                return EntryQualityResult(False, reason_key, metrics["entry_quality_score"], metrics)
            score = min(100.0, score)

        metrics["entry_quality_pass"] = True
        metrics["entry_block_reason"] = ""
        metrics["entry_quality_score"] = round(score, 2)
        metrics["expected_ev"] = ev_val
        return EntryQualityResult(True, "", metrics["entry_quality_score"], metrics)
