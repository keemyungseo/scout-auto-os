"""SCOUT Research V1.5 — State League & Evolution tests."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research_bundle"))

from scout_research_r006_pilot_execution_engine import Bar
from scout_auto_os.engine.state_engine import (
    LIVE_STATE_FORMULA,
    StateFormulaWeights,
    compute_alive_from_snap,
    compute_alive_score,
)
from scout_auto_os.engine.research.state_formula_generator import generate_state_formulas
from scout_auto_os.engine.research.state_league import replay_formula
from scout_auto_os.engine.research.state_evolution import analyze_evolution
from scout_auto_os.engine.research.position_evolution import PositionEvolutionStore
from scout_auto_os.engine.research.storage import ResearchStore


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def _bars_up(n: int = 30) -> list[Bar]:
    out: list[Bar] = []
    for i in range(n):
        p = 100 + i * 0.2
        out.append(Bar(t_ms=i * 300_000, o=p, h=p + 0.3, l=p - 0.05, c=p + 0.15))
    return out


def test_live_weights_unchanged():
    bars = _bars_up()
    live = compute_alive_score(bars)
    explicit = compute_alive_score(bars, weights=LIVE_STATE_FORMULA)
    if not live or not explicit or abs(live.alive_score - explicit.alive_score) > 0.01:
        _fail("LIVE default weights must match LIVE_STATE_FORMULA")
    print(f"OK: LIVE alive_score={live.alive_score}")


def test_formula_generator():
    formulas = generate_state_formulas(32)
    if len(formulas) < 10:
        _fail(f"expected 10+ formulas, got {len(formulas)}")
    if formulas[0].name != "LIVE_V14":
        _fail("first formula must be LIVE_V14")
    names = {f.name for f in formulas}
    if len(names) != len(formulas):
        _fail("duplicate formula names")
    print(f"OK: generated {len(formulas)} state formulas")


def test_replay_formula():
    bars = _bars_up(40)
    alt = StateFormulaWeights("TEST", 35, 20, 30, 10, 5)
    out = replay_formula(bars, alt)
    if not out:
        _fail("replay_formula returned None")
    print(f"OK: replay return={out.return_pct}% held={out.bars_held}bars mfe={out.mfe_pct}")


def test_evolution_analysis():
    rows = []
    for i in range(120):
        ok = i % 3 == 0
        rows.append({
            "symbol": f"SYM{i}",
            "alive_score": 75 if ok else 40,
            "realized_pnl_pct": 5 if ok else -2,
            "trend_alive": 25 if ok else 5,
            "momentum_alive": 25 if ok else 8,
            "volume_alive": 25 if ok else 8,
            "expansion_alive": 15 if ok else 5,
        })
    result = analyze_evolution(rows)
    if result.get("status") != "ok":
        _fail(f"evolution status {result.get('status')}")
    if not result.get("component_contribution"):
        _fail("expected component contribution")
    print(f"OK: evolution n={result['sample_count']} proposals={len(result['proposals'])}")


def test_storage_state_league():
    with tempfile.TemporaryDirectory() as tmp:
        store = ResearchStore(Path(tmp))
        store.write_state_league([{
            "league_rank": 1, "formula_name": "LIVE_V14", "sample_count": 50,
            "win_rate": 55, "profit_factor": 1.2, "league_score": 80, "tier": "hypothesis",
        }])
        rows = store.read_state_league()
        if not rows or rows[0]["formula_name"] != "LIVE_V14":
            _fail("state league csv roundtrip failed")
        store.write_state_proposals({"proposals": [{"tier": "hypothesis", "title": "test"}]})
        if not store.read_state_proposals().get("proposals"):
            _fail("state proposals json failed")
        print("OK: storage state league + proposals")


def test_position_evolution_store():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        evo = PositionEvolutionStore(root)
        rec = {
            "record_time_kst": "2026-01-01 12:00:00",
            "source": "test",
            "position_id": "p1",
            "symbol": "BTCUSDT",
            "checkpoint_min": 30,
            "entry_time_kst": "2026-01-01 11:00:00",
            "exit_time_kst": "",
            "realized_pnl_pct": 2.5,
            "alive_score": 72,
            "alive_delta": 5,
            "trend_alive": 25,
            "momentum_alive": 25,
            "volume_alive": 25,
            "expansion_alive": 15,
            "acceleration": 10,
            "exhaustion": 0,
            "recommendation": "HOLD",
            "review_reason": "test",
            "formula_name": "LIVE_V14",
        }
        if not evo.append(rec):
            _fail("first append failed")
        if evo.append(rec):
            _fail("duplicate should be skipped")
        if len(evo.read_all()) != 1:
            _fail("read_all count")
        print("OK: position evolution store dedupe")


def main():
    test_live_weights_unchanged()
    test_formula_generator()
    test_replay_formula()
    test_evolution_analysis()
    test_storage_state_league()
    test_position_evolution_store()
    print("\nALL V1.5 STATE LEAGUE TESTS PASSED")


if __name__ == "__main__":
    main()
