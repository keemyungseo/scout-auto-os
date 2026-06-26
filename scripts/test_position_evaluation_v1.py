#!/usr/bin/env python3
"""Position Evaluation Engine V1 tests."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1].parent
PKG = ROOT / "scout_auto_os"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PKG / "research_bundle"))

from scout_research_r006_pilot_execution_engine import Bar
from scout_auto_os.engine.position_evaluation.decision import decide
from scout_auto_os.engine.position_evaluation.evaluator import EvaluationMetrics, PositionEvaluator
from scout_auto_os.engine.position_evaluation.manual_guard import can_enter, can_exit, is_protected
from scout_auto_os.engine.position_evaluation.runner import PositionEvaluationRunner
from scout_auto_os.engine.position_evaluation.side_rules import protective_stop_hit, roi_pct
from scout_auto_os.engine.position_evaluation.thesis import TradeThesis, build_thesis_for_entry
from scout_auto_os.engine.state_exit_engine import StateExitEngine
from scout_auto_os.engine.state_engine import compute_alive_score


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def _bars_short_favorable(n: int = 20, start: float = 100.0) -> list[Bar]:
    out: list[Bar] = []
    for i in range(n):
        p = start - i * 0.2
        out.append(Bar(t_ms=i * 300_000, o=p + 0.05, h=p + 0.3, l=p - 0.05, c=p))
    return out


def _bars_long_up(n: int = 20, start: float = 100.0) -> list[Bar]:
    out: list[Bar] = []
    for i in range(n):
        p = start + i * 0.2
        out.append(Bar(t_ms=i * 300_000, o=p, h=p + 0.2, l=p - 0.05, c=p + 0.1))
    return out


def test_bot_position_evaluated():
    with tempfile.TemporaryDirectory() as td:
        cfg = {
            "position_evaluation": {"enabled": True, "min_review_interval_sec": 0},
            "expectation": {"enabled": False},
        }
        runner = PositionEvaluationRunner(cfg, Path(td))
        pos = {
            "position_id": "pos_test1",
            "symbol": "BTCUSDT",
            "side": "LONG",
            "source": "AUTO",
            "auto_manage": 1,
            "manual_lock": 0,
            "entry_time": "2026-06-01 10:00:00",
            "entry_price": 100.0,
            "engine": "PORTFOLIO_LONG",
            "a6_score": 75,
        }
        tid = runner.create_thesis_for_position(
            position_id=pos["position_id"],
            symbol=pos["symbol"],
            side=pos["side"],
            entry_time=pos["entry_time"],
            entry_price=pos["entry_price"],
            engine=pos["engine"],
            entry_score=75,
        )
        bars = _bars_long_up()
        dec, exit_d = runner.evaluate_position(pos, bars, 2.0, None)
        if dec is None:
            _fail("BOT position should be evaluated")
        print(f"OK: bot evaluated action={dec.action} thesis_id={tid.thesis_id}")


def test_manual_never_exit():
    pos = {"source": "MANUAL", "auto_manage": 0, "manual_lock": 1, "symbol": "WLDUSDT"}
    if can_exit(pos):
        _fail("manual must not exit")
    if not is_protected(pos):
        _fail("manual must be protected")
    thesis = build_thesis_for_entry("p1", "WLDUSDT", "LONG", "2026-06-01 10:00:00", 1.0, source="MANUAL", auto_manage=False)
    metrics = EvaluationMetrics(
        30.0, 3000, 1000, 100, 35, 1, 35, 0, True, True, True, False, False, 80, 10, 70, 1.0,
    )
    dec = decide(metrics, thesis, is_manual=True)
    if dec.action != "NO_ACTION_MANUAL_POSITION" or dec.should_exit:
        _fail(f"manual action wrong: {dec}")
    print("OK: manual NO_ACTION_MANUAL_POSITION")


def test_wld_manual_lock():
    pos = {"symbol": "WLDUSDT", "manual_lock": 1, "auto_manage": 0, "source": "MANUAL"}
    if can_enter("WLDUSDT", set(), set(), [pos]):
        _fail("WLD entry must be blocked when manual open")
    print("OK: WLD manual_lock blocks entry")


def test_long_roi():
    r = roi_pct("LONG", 100, 103)
    if r != 3.0:
        _fail(f"long roi expected 3.0 got {r}")
    print("OK: long ROI")


def test_short_roi():
    r = roi_pct("SHORT", 100, 97)
    if r != 3.0:
        _fail(f"short roi expected 3.0 got {r}")
    print("OK: short ROI")


def test_short_high_break_stop():
    bars = [Bar(t_ms=0, o=100, h=101, l=99, c=100)]
    bars.append(Bar(t_ms=300_000, o=100, h=112, l=99, c=111))
    if not protective_stop_hit("SHORT", bars, 100, 10):
        _fail("short should hit protective stop on high break")
    print("OK: short high break protective stop")


def test_horizon_exceeded_weak_roi_exit_pressure():
    thesis = build_thesis_for_entry("p2", "XUSDT", "SHORT", "2026-06-01 10:00:00", 10.0)
    thesis.expected_horizon_min = 60
    thesis.expected_return_pct = 3.0
    ev = PositionEvaluator()
    bars = _bars_short_favorable(5)
    m = ev.evaluate(thesis, "p2", "SHORT", 10.0, 9.97, 60, bars)
    dec = decide(m, thesis)
    if m.exit_pressure_score < 25:
        _fail(f"exit pressure too low: {m.exit_pressure_score}")
    if dec.action not in ("EXIT", "STOP_TIGHTEN"):
        _fail(f"expected EXIT/STOP_TIGHTEN got {dec.action}")
    print(f"OK: horizon exceeded weak roi -> {dec.action} pressure={m.exit_pressure_score}")


def test_target_exceeded_trail():
    thesis = build_thesis_for_entry("p3", "YUSDT", "LONG", "2026-06-01 10:00:00", 100.0)
    thesis.expected_return_pct = 3.0
    metrics = EvaluationMetrics(
        5.0, 45, 166, 75, 6, 1, 5.5, 0.5, True, True, True, True, False, 85, 15, 80, 105.0,
    )
    dec = decide(metrics, thesis)
    if dec.action != "TRAIL":
        _fail(f"expected TRAIL got {dec.action}")
    print("OK: ROI target + momentum -> TRAIL")


def test_max_hold_forced_exit():
    thesis = build_thesis_for_entry("p4", "ZUSDT", "LONG", "2026-06-01 10:00:00", 50.0)
    thesis.max_hold_minutes = 240
    metrics = EvaluationMetrics(
        1.0, 241, 33, 100, 2, 0.5, 2, 1, False, True, True, False, False, 50, 55, 40, 50.5,
    )
    dec = decide(metrics, thesis)
    if not dec.should_exit or dec.action != "EXIT":
        _fail(f"max_hold should EXIT got {dec}")
    print("OK: max_hold forced EXIT")


def test_thesis_id_in_logs():
    with tempfile.TemporaryDirectory() as td:
        data = Path(td)
        cfg = {"position_evaluation": {"enabled": True, "min_review_interval_sec": 0}}
        runner = PositionEvaluationRunner(cfg, data)
        pos = {
            "position_id": "pos_log",
            "symbol": "ETHUSDT",
            "side": "SHORT",
            "source": "AUTO",
            "auto_manage": 1,
            "manual_lock": 0,
            "entry_time": "2026-06-01 12:00:00",
            "entry_price": 2000.0,
            "engine": "PORTFOLIO_SHORT",
            "a6_score": 80,
        }
        thesis = runner.create_thesis_for_position(
            pos["position_id"], pos["symbol"], pos["side"],
            pos["entry_time"], pos["entry_price"], engine=pos["engine"], entry_score=80,
        )
        runner.evaluate_position(pos, _bars_short_favorable(), 4.0, None)
        review = (data / "position_evaluation" / "position_review.csv").read_text(encoding="utf-8")
        if thesis.thesis_id not in review:
            _fail("thesis_id missing from position_review.csv")
        db_thesis = json.loads((data / "position_evaluation" / "trade_thesis.jsonl").read_text(encoding="utf-8").strip())
        if db_thesis.get("thesis_id") != thesis.thesis_id:
            _fail("thesis_id mismatch in jsonl")
        print("OK: thesis_id linked in logs")


def test_short_side_alive_score():
    bars = _bars_short_favorable(30)
    score = compute_alive_score(bars, side="SHORT")
    if not score or score.alive_score <= 0:
        _fail("short alive score should be computable")
    print(f"OK: short alive_score={score.alive_score}")


def main():
    test_bot_position_evaluated()
    test_manual_never_exit()
    test_wld_manual_lock()
    test_long_roi()
    test_short_roi()
    test_short_high_break_stop()
    test_horizon_exceeded_weak_roi_exit_pressure()
    test_target_exceeded_trail()
    test_max_hold_forced_exit()
    test_thesis_id_in_logs()
    test_short_side_alive_score()
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
