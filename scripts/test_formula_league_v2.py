"""Formula League V2 tests."""

from __future__ import annotations

import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PKG / "research_bundle"))


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def test_generator():
    from scout_auto_os.engine.research.formula_league_v2.generator import generate_search_formulas

    train = [{
        "scan_kst": "2026-06-01 00:00:00",
        "symbol": "BTCUSDT",
        "features": {"1h_current_body_pct": 5.0, "1h_current_range_pct": 10.0, "1h_current_return_pct": 2.0,
                     "15m_current_volume_ratio": 1.5, "5m_momentum": 1.0, "5m_compression": 0.5},
        "max_up_4h": 10.0,
        "cohort": "winner",
    }] * 20
  # need losers too
    train[10:] = [{
        **train[0],
        "symbol": f"X{i}",
        "cohort": "loser",
        "max_up_4h": 1.0,
    } for i in range(10)]
    formulas = generate_search_formulas(train)
    if len(formulas) < 10:
        _fail(f"expected many formulas got {len(formulas)}")
    print(f"OK: generated {len(formulas)} formulas")


def test_full_run():
    from scout_auto_os.engine.research.formula_league_v2.runner import FormulaLeagueV2Runner

    runner = FormulaLeagueV2Runner(
        PKG / "data",
        PKG,
        PKG / "research_bundle" / "seed" / "candidates.jsonl",
        PKG / "research_bundle" / "forward" / "forward_klines_15m.jsonl",
    )
    result = runner.run()
    if not result:
        _fail("run failed")
    out = PKG / "data" / "formula_league_v2"
    for name in (
        "formula_league.csv",
        "formula_scores.csv",
        "formula_survivors.csv",
        "formula_dna.csv",
        "formula_report.md",
    ):
        if not (out / name).exists():
            _fail(f"missing {name}")
    print(f"OK: {result['meta']['formulas_generated']} formulas")


def main():
    test_generator()
    test_full_run()
    print("\nALL FORMULA LEAGUE V2 TESTS PASSED")


if __name__ == "__main__":
    main()
