#!/usr/bin/env python3
"""Expectation Engine V1 tests."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1].parent
PKG = ROOT / "scout_auto_os"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PKG / "research_bundle"))

from scout_auto_os.engine.expectation.curve_builder import build_expected_path, load_research_curve
from scout_auto_os.engine.expectation.expectation_score import compute_expectation_score
from scout_auto_os.engine.expectation.progress_tracker import compute_progress
from scout_auto_os.engine.expectation.runner import ExpectationRunner
from scout_auto_os.engine.expectation.thesis_state_machine import compute_thesis_state
from scout_auto_os.engine.expectation.dynamic_thesis import maybe_extension_thesis
from scout_auto_os.engine.position_evaluation.thesis import TradeThesis, build_thesis_for_entry, ThesisStore


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def _data_dir() -> Path:
    return PKG / "data"


def test_outperform():
    side = "SHORT"
    base, _, _ = load_research_curve(side, _data_dir())
    thesis = build_thesis_for_entry("p1", "X", side, "2026-06-01 10:00:00", 10.0)
    path = build_expected_path(thesis.thesis_id, "p1", "X", side, 3.0, 90, 0.6, _data_dir())
    prog = compute_progress(path, 45, 4.5)
    if prog.progress_ratio < 120:
        _fail(f"expected OUTPERFORM ratio got {prog.progress_ratio}")
    score = compute_expectation_score(prog, momentum_alive=True, trend_alive=True, volume_alive=True,
                                      peak_roi=4.5, expected_peak_window=90, elapsed_min=45,
                                      exit_pressure=10, entry_success_prob=0.6)
    st = compute_thesis_state(prog, score, elapsed_min=45, expected_horizon=90, expected_return=3.0,
                              current_roi=4.5, max_hold_min=240)
    if st.state not in ("OUTPERFORM", "THESIS_COMPLETE"):
        _fail(f"expected OUTPERFORM got {st.state}")
    print(f"OK: OUTPERFORM ratio={prog.progress_ratio} state={st.state}")


def test_underperform():
    thesis = build_thesis_for_entry("p2", "Y", "LONG", "2026-06-01 10:00:00", 100.0)
    path = build_expected_path(thesis.thesis_id, "p2", "Y", "LONG", 3.0, 120, 0.6, _data_dir())
    prog = compute_progress(path, 120, 0.5)
    score = compute_expectation_score(prog, momentum_alive=False, trend_alive=False, volume_alive=False,
                                      peak_roi=0.5, expected_peak_window=120, elapsed_min=120,
                                      exit_pressure=40, entry_success_prob=0.6)
    st = compute_thesis_state(prog, score, elapsed_min=120, expected_horizon=120, expected_return=3.0,
                              current_roi=0.5, max_hold_min=240)
    if st.state not in ("UNDERPERFORM", "THESIS_FAILED"):
        _fail(f"expected UNDERPERFORM got {st.state}")
    print(f"OK: UNDERPERFORM state={st.state} ratio={prog.progress_ratio}")


def test_thesis_complete():
    thesis = build_thesis_for_entry("p3", "Z", "SHORT", "2026-06-01 10:00:00", 50.0)
    path = build_expected_path(thesis.thesis_id, "p3", "Z", "SHORT", 3.0, 90, 0.7, _data_dir())
    prog = compute_progress(path, 85, 3.2)
    score = compute_expectation_score(prog, momentum_alive=True, trend_alive=True, volume_alive=True,
                                      peak_roi=3.2, expected_peak_window=90, elapsed_min=85,
                                      exit_pressure=5, entry_success_prob=0.7)
    st = compute_thesis_state(prog, score, elapsed_min=85, expected_horizon=90, expected_return=3.0,
                              current_roi=3.2, max_hold_min=240)
    if st.state not in ("THESIS_COMPLETE", "ON_TRACK", "OUTPERFORM"):
        _fail(f"expected complete-ish state got {st.state}")
    print(f"OK: THESIS_COMPLETE family state={st.state}")


def test_thesis_failed():
    thesis = build_thesis_for_entry("p4", "A", "LONG", "2026-06-01 10:00:00", 100.0)
    path = build_expected_path(thesis.thesis_id, "p4", "A", "LONG", 3.0, 90, 0.5, _data_dir())
    prog = compute_progress(path, 200, 0.2)
    score = compute_expectation_score(prog, momentum_alive=False, trend_alive=False, volume_alive=False,
                                      peak_roi=0.2, expected_peak_window=120, elapsed_min=200,
                                      exit_pressure=65, entry_success_prob=0.5)
    st = compute_thesis_state(prog, score, prior_state="UNDERPERFORM", elapsed_min=200,
                              expected_horizon=90, expected_return=3.0, current_roi=0.2, max_hold_min=240)
    if st.state not in ("THESIS_FAILED", "EXIT_READY"):
        _fail(f"expected THESIS_FAILED got {st.state}")
    print(f"OK: THESIS_FAILED state={st.state} score={score.score}")


def test_hei_extension():
    thesis = build_thesis_for_entry("p_hei", "HEIUSDT", "LONG", "2026-06-01 10:00:00", 1.0)
    thesis.expected_return_pct = 3.0
    thesis.expected_horizon_min = 90
    path = build_expected_path(thesis.thesis_id, "p_hei", "HEIUSDT", "LONG", 3.0, 90, 0.6, _data_dir())
    nt, np, tr = maybe_extension_thesis(
        thesis, path, current_roi=30.0, elapsed_min=60, thesis_state="THESIS_COMPLETE", data_dir=_data_dir(),
    )
    if not nt or not tr or tr.transition_type != "EXTENSION":
        _fail("HEI should create extension thesis")
    print(f"OK: HEI extension new_return={nt.expected_return_pct}% horizon={nt.expected_horizon_min}m")


def test_met_exit_pressure():
    thesis = build_thesis_for_entry("p_met", "METUSDT", "LONG", "2026-06-01 10:00:00", 10.0)
    thesis.expected_horizon_min = 90
    thesis.max_hold_minutes = 240
    path = build_expected_path(thesis.thesis_id, "p_met", "METUSDT", "LONG", 3.0, 90, 0.6, _data_dir())
    elapsed = 2880
    prog = compute_progress(path, elapsed, 1.0)
    score = compute_expectation_score(prog, momentum_alive=False, trend_alive=False, volume_alive=False,
                                      peak_roi=2.0, expected_peak_window=120, elapsed_min=elapsed,
                                      exit_pressure=70, entry_success_prob=0.6)
    st = compute_thesis_state(prog, score, prior_state="UNDERPERFORM", elapsed_min=elapsed,
                              expected_horizon=90, expected_return=3.0, current_roi=1.0, max_hold_min=240)
    if st.state not in ("EXIT_READY", "THESIS_FAILED") or score.score >= 50:
        _fail(f"MET pattern should fail: state={st.state} score={score.score}")
    print(f"OK: MET scenario state={st.state} score={score.score} pressure=70")


def test_long_curve():
    curve, meta, src = load_research_curve("LONG", _data_dir())
    if len(curve) < 4 or not src:
        _fail("long curve missing research sources")
    print(f"OK: long curve points={len(curve)} sources={len(src)}")


def test_short_curve():
    curve, meta, src = load_research_curve("SHORT", _data_dir())
    if len(curve) < 4:
        _fail("short curve empty")
    print(f"OK: short curve points={len(curve)} peak_window={meta['peak_window_min']}")


def test_score_range():
    thesis = build_thesis_for_entry("p5", "B", "SHORT", "2026-06-01 10:00:00", 5.0)
    path = build_expected_path(thesis.thesis_id, "p5", "B", "SHORT", 3.0, 90, 0.5, _data_dir())
    prog = compute_progress(path, 60, 2.0)
    score = compute_expectation_score(prog, momentum_alive=True, trend_alive=True, volume_alive=True,
                                      peak_roi=2.5, expected_peak_window=90, elapsed_min=60,
                                      exit_pressure=20, entry_success_prob=0.5)
    if not 0 <= score.score <= 100:
        _fail(f"score out of range {score.score}")
    print(f"OK: expectation_score={score.score}")


def test_shared_thesis_id():
    with tempfile.TemporaryDirectory() as td:
        data = Path(td)
        research = PKG / "data"
        cfg = {"expectation": {"enabled": True}, "position_evaluation": {"enabled": True}}
        store = ThesisStore(data)
        exp = ExpectationRunner(cfg, data, thesis_store=store)
        thesis = build_thesis_for_entry("pos_x", "ETHUSDT", "SHORT", "2026-06-01 12:00:00", 2000.0)
        store.append(thesis)
        path = exp.create_path_for_thesis(thesis)
        if path.thesis_id != thesis.thesis_id:
            _fail("path thesis_id mismatch")
        review = exp.evaluate(
            thesis, elapsed_min=60, current_roi=2.0, peak_roi=2.5,
            momentum_alive=True, trend_alive=True, volume_alive=True, exit_pressure=15,
        )
        if not review or review.path.thesis_id != thesis.thesis_id:
            _fail("review thesis_id mismatch")
        log = (data / "expectation" / "expected_path.jsonl").read_text(encoding="utf-8")
        if thesis.thesis_id not in log:
            _fail("thesis_id not in expected_path.jsonl")
        print(f"OK: shared thesis_id={thesis.thesis_id}")


def main():
    test_outperform()
    test_underperform()
    test_thesis_complete()
    test_thesis_failed()
    test_hei_extension()
    test_met_exit_pressure()
    test_long_curve()
    test_short_curve()
    test_score_range()
    test_shared_thesis_id()
    print("ALL EXPECTATION TESTS PASSED")


if __name__ == "__main__":
    main()
