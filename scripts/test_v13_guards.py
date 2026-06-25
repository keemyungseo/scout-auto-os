"""SCOUT LIVE V1.3 guard tests — paper/dry-run, no live orders."""

from __future__ import annotations

import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(ROOT))

from scout_auto_os.engine.emergency_risk_guard import EmergencyRiskGuard
from scout_auto_os.engine.entry_block_summary import EntryBlockSummary
from scout_auto_os.engine.entry_quality_guard import EntryQualityGuard, EntryQualityResult
from scout_auto_os.engine.scout_long_engine import ScoutLongEngine
from scout_auto_os.storage.db import Database, now_kst


@dataclass
class MockGuard:
    results: dict[str, EntryQualityResult] = field(default_factory=dict)

    def evaluate(self, symbol: str, candidate: dict) -> EntryQualityResult:
        return self.results.get(symbol, EntryQualityResult(True, "", 100.0, {"entry_quality_pass": True}))


class MockPM:
    def __init__(self) -> None:
        self.slots = 2
        self.created: list[str] = []

    def long_slots_used(self) -> int:
        return len(self.created)

    def has_slot(self) -> bool:
        return len(self.created) < self.slots

    def create_position(self, symbol, side, price, source, engine, **kw) -> str:
        self.created.append(symbol)
        return f"pos_{symbol}"


class MockExec:
    trade_recorder = None

    def paper_entry(self, *a, **k):
        return "trd1"


class MockAlerts:
    entries: list[str] = []

    def entry_alert(self, sym, *a, **k):
        MockAlerts.entries.append(sym)

    @staticmethod
    def entry_block_summary_alert(msg):
        pass


class MockRisk:
    @staticmethod
    def can_enter_long(slots):
        return True, ""


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


class MockDB:
    def log_event(self, *a, **k):
        pass


def test_negative_ev_block():
    guard = MockGuard({
        "BADUSDT": EntryQualityResult(False, "negative_ev", 70, {"entry_quality_pass": False}),
    })
    engine = ScoutLongEngine({"long_engine": {"enabled": True}}, MockDB(), guard)
    pm = MockPM()
    top5 = [{"rank": 1, "symbol": "BADUSDT", "entry_price": 1.0, "a6_score": 5, "expected_ev": -1, "reason": "t"}]
    entered = engine.try_fill_slots(top5, set(), set(), pm, MockExec(), MockAlerts(), MockRisk())
    if entered:
        _fail("negative EV should block entry")
    print("OK: negative_ev block")


def test_red_candle_skip_next():
    guard = MockGuard({
        "REDCUSDT": EntryQualityResult(False, "bad_timing_red_candle", 75, {}),
        "GOODUSDT": EntryQualityResult(True, "", 100, {"entry_quality_pass": True}),
    })
    engine = ScoutLongEngine({"long_engine": {"enabled": True}}, MockDB(), guard)
    pm = MockPM()
    top5 = [
        {"rank": 1, "symbol": "REDCUSDT", "entry_price": 1.0, "a6_score": 5, "expected_ev": 3, "reason": "t"},
        {"rank": 2, "symbol": "GOODUSDT", "entry_price": 1.0, "a6_score": 4, "expected_ev": 2, "reason": "t"},
    ]
    MockAlerts.entries.clear()
    entered = engine.try_fill_slots(top5, set(), set(), pm, MockExec(), MockAlerts(), MockRisk())
    if entered != ["pos_GOODUSDT"] or MockAlerts.entries != ["GOODUSDT"]:
        _fail(f"expected GOODUSDT entry, got {entered}")
    print("OK: red candle skip + next candidate entry")


def test_low_volume_block():
    guard = MockGuard({
        "LOWUSDT": EntryQualityResult(False, "low_quote_volume", 60, {}),
    })
    engine = ScoutLongEngine({"long_engine": {"enabled": True}}, MockDB(), guard)
    pm = MockPM()
    top5 = [{"rank": 1, "symbol": "LOWUSDT", "entry_price": 1.0, "a6_score": 5, "expected_ev": 3, "reason": "t"}]
    if engine.try_fill_slots(top5, set(), set(), pm, MockExec(), MockAlerts(), MockRisk()):
        _fail("low volume should block")
    print("OK: low_quote_volume block")


def test_weak_5m_block():
    guard = MockGuard({
        "WEAKUSDT": EntryQualityResult(False, "weak_5m_volume", 65, {}),
    })
    engine = ScoutLongEngine({"long_engine": {"enabled": True}}, MockDB(), guard)
    pm = MockPM()
    top5 = [{"rank": 1, "symbol": "WEAKUSDT", "entry_price": 1.0, "a6_score": 5, "expected_ev": 3, "reason": "t"}]
    if engine.try_fill_slots(top5, set(), set(), pm, MockExec(), MockAlerts(), MockRisk()):
        _fail("weak 5m should block")
    print("OK: weak_5m_volume block")


def test_risk_guard_roi_exit():
    cfg = {"execution": {"leverage": 3, "order_size_usdt": 7}, "risk_guard": {}}
    rg = EmergencyRiskGuard(cfg)
    pos = {"symbol": "ALICEUSDT", "entry_time": "2026-01-01 00:00:00"}
    d = rg.evaluate(pos, price_pnl_pct=-6.1, now_str="2026-01-01 01:00:00")
    if not d.should_exit or d.reason != "roi_loss_limit":
        _fail(f"expected roi exit at -18% ROI, got {d}")
    print("OK: ROI -18% force exit trigger")


def test_risk_guard_hold_timeout():
    cfg = {"execution": {"leverage": 1, "order_size_usdt": 7}, "risk_guard": {}}
    rg = EmergencyRiskGuard(cfg)
    pos = {"symbol": "METUSDT", "entry_time": "2026-01-01 00:00:00"}
    d = rg.evaluate(pos, price_pnl_pct=-1.0, now_str="2026-01-01 01:00:00")
    if not d.should_exit or d.reason != "hold_timeout_negative":
        _fail(f"expected hold timeout exit, got {d}")
    print("OK: 45m negative hold exit trigger")


def test_block_summary():
    s = EntryBlockSummary(interval_sec=0)
    sent: list[str] = []
    s.record_block("weak_5m_volume")
    s.record_block("weak_5m_volume")
    s.record_pass()
    if not s.maybe_telegram_summary(lambda m: sent.append(m)):
        _fail("summary should send")
    if "blocked_count: 2" not in sent[0] or "weak_5m_volume" not in sent[0]:
        _fail(f"bad summary: {sent[0]}")
    print("OK: entry block summary")


def test_entry_guard_env_defaults():
    g = EntryQualityGuard({"entry_quality": {}, "live_data": {"rest_base": "https://fapi.binance.com"}})
    assert g.min_24h_qv == 10_000_000
    assert g.min_5m_ratio == 1.5
    print("OK: entry guard env defaults")


def main() -> None:
    print("=== SCOUT LIVE V1.3 Guard Tests ===")
    test_negative_ev_block()
    test_red_candle_skip_next()
    test_low_volume_block()
    test_weak_5m_block()
    test_risk_guard_roi_exit()
    test_risk_guard_hold_timeout()
    test_block_summary()
    test_entry_guard_env_defaults()
    print("=== ALL PASSED ===")


if __name__ == "__main__":
    main()
