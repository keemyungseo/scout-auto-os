"""SCOUT LIVE V1.4 State Engine tests."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research_bundle"))

from scout_research_r006_pilot_execution_engine import Bar
from scout_auto_os.engine.state_engine import compute_alive_score
from scout_auto_os.engine.state_exit_engine import StateExitEngine
from scout_auto_os.engine.position_state_manager import PositionStateManager


def _bars_trending_up(n: int = 30, start: float = 100.0) -> list[Bar]:
    out: list[Bar] = []
    for i in range(n):
        p = start + i * 0.15
        out.append(Bar(t_ms=i * 300_000, o=p, h=p + 0.2, l=p - 0.05, c=p + 0.1))
    return out


def _bars_exhausted(n: int = 40, start: float = 100.0) -> list[Bar]:
    out: list[Bar] = []
    for i in range(n):
        if i < 20:
            p = start + i * 0.2
        else:
            p = start + 4 - (i - 20) * 0.25
        out.append(Bar(t_ms=i * 300_000, o=p, h=p + 0.05, l=p - 0.15, c=p - 0.05))
    return out


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def test_alive_score_high_on_uptrend():
    bars = _bars_trending_up()
    score = compute_alive_score(bars, hold_alive=70, exit_alive=45)
    if not score or score.alive_score < 55:
        _fail(f"uptrend should have decent alive score, got {score}")
    print(f"OK: uptrend alive_score={score.alive_score} rec={score.hold_recommendation}")


def test_state_exit_after_target_low_score():
    cfg = {"state_engine": {"hold_target_minutes": 120, "exit_alive_score": 45, "hold_alive_score": 70}}
    engine = StateExitEngine(cfg)
    bars = _bars_exhausted(50)
    entry = compute_alive_score(bars[:5], hold_alive=70, exit_alive=45)
    current = compute_alive_score(bars, hold_alive=70, exit_alive=45)
    if not entry or not current:
        _fail("score compute failed")
    d = engine.evaluate(bars, bars[0].o, entry, current, hold_minutes=130)
    if not d.should_exit:
        _fail(f"expected state exit after 130m low score, got {d}")
    print(f"OK: state exit reason={d.reason}")


def test_hold_high_alive_past_target():
    cfg = {"state_engine": {"hold_target_minutes": 120, "hold_alive_score": 70}}
    engine = StateExitEngine(cfg)
    bars = _bars_trending_up(50)
    entry = compute_alive_score(bars[:5], hold_alive=70, exit_alive=45)
    current = compute_alive_score(bars, hold_alive=70, exit_alive=45)
    if not entry or not current:
        _fail("score failed")
    if current.alive_score >= 70:
        d = engine.evaluate(bars, bars[0].o, entry, current, hold_minutes=200)
        if d.should_exit:
            _fail(f"high alive should hold past 2h, got {d.reason}")
        print("OK: high alive holds past target")
    else:
        print(f"OK: skip hold test (score={current.alive_score})")


def test_review_store():
    tmp = Path(tempfile.mkdtemp())
    mgr = PositionStateManager({"state_engine": {"review_interval_sec": 0}}, tmp, lambda s, t: _bars_trending_up())
    pos = {
        "position_id": "pos_test",
        "symbol": "TESTUSDT",
        "entry_time": "2026-06-01 10:00:00",
        "entry_price": 100.0,
    }
    bars = _bars_trending_up()
    mgr.register_entry("pos_test", "TESTUSDT", pos["entry_time"], bars)
    decision = mgr.maybe_review(pos, bars, 1.5)
    if not (tmp / "position_review.csv").exists():
        _fail("position_review.csv not created")
    print("OK: position review saved")


def main() -> None:
    print("=== SCOUT LIVE V1.4 State Engine Tests ===")
    test_alive_score_high_on_uptrend()
    test_state_exit_after_target_low_score()
    test_hold_high_alive_past_target()
    test_review_store()
    print("=== ALL PASSED ===")


if __name__ == "__main__":
    main()
