"""Portfolio Engine V1 tests."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PKG / "research_bundle"))

from scout_auto_os.engine.portfolio.diversification import diversify_select
from scout_auto_os.engine.portfolio.engine import PortfolioEngine
from scout_auto_os.engine.portfolio.backtest import filter_2h_scans
from scout_auto_os.engine.portfolio.slot_manager import SlotBook, update_slots


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def test_diversify():
    cands = [
        {"symbol": f"S{i}", "features": {"1h_current_return_pct": i}, "live_pattern": "LONG_CONTINUATION", "entry_score": 100 - i}
        for i in range(5)
    ]
    out = diversify_select(cands, 3)
    if len(out) > 3:
        _fail("too many")
    print(f"OK: diversified {len(out)}")


def test_replacement():
    from scout_auto_os.engine.portfolio.slot_manager import SlotHolding
    book = SlotBook()
    book.long_slots.append(SlotHolding("A", "long", 50.0, "t1", "LONG_CONTINUATION", "t2"))
    cands = [{"symbol": "B", "entry_score": 60.0, "direction": "long", "live_pattern": "LONG_BREAKOUT", "features": {"1h_current_return_pct": 5}}]
    book, entries, reps = update_slots(book, cands, [], "scan2", "scan3", replacement_margin=0.08)
    if not entries:
        _fail("expected entry")
    print(f"OK: slot update entries={len(entries)}")


def test_backtest_run():
    with tempfile.TemporaryDirectory() as tmp:
        data = Path(tmp)
        (data / "zero_base").mkdir(parents=True)
        src_rules = PKG / "data" / "zero_base" / "entry_filter_rules_v2.json"
        if not src_rules.exists():
            src_rules = PKG / "research_bundle" / "reports" / "entry_filter_rules_v2.json"
        (data / "zero_base" / "entry_filter_rules_v2.json").write_text(
            src_rules.read_text(encoding="utf-8"), encoding="utf-8",
        )
        from scout_auto_os.engine.portfolio.backtest import PortfolioBacktestRunner
        runner = PortfolioBacktestRunner(
            data,
            pkg_root=PKG,
            candidates_path=PKG / "research_bundle" / "seed" / "candidates.jsonl",
            forward_path=PKG / "research_bundle" / "forward" / "forward_klines_15m.jsonl",
        )
        result = runner.run(lookback_days=180)
        out = data / "portfolio"
        if not (out / "portfolio_report.md").exists():
            _fail("report missing")
        print(f"OK: trades={result['stats'].get('total_trades')}")


def test_filter_2h():
    scans = [f"2026-06-01 {h:02d}:00:00" for h in range(0, 24, 1)]
    kept = filter_2h_scans(scans)
    if len(kept) < 10:
        _fail("2h filter too aggressive")
    print(f"OK: 2h scans {len(kept)}")


def main():
    test_diversify()
    test_replacement()
    test_filter_2h()
    test_backtest_run()
    print("\nALL PORTFOLIO ENGINE V1 TESTS PASSED")


if __name__ == "__main__":
    main()
