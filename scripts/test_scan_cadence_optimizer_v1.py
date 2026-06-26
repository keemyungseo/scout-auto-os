"""Scan Cadence Optimizer V1 tests."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PKG / "research_bundle"))

from scout_auto_os.engine.research.cadence.schedule import build_cadence_schedule


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def test_schedule_sub120():
    base = ["2026-06-01 00:00:00", "2026-06-01 02:00:00", "2026-06-01 04:00:00"]
    s15 = build_cadence_schedule(base, 15)
    if len(s15) < 8:
        _fail(f"expected more 15m ticks, got {len(s15)}")
    if s15[0] != ("2026-06-01 00:00:00", "2026-06-01 00:00:00"):
        _fail("first tick mismatch")
    print(f"OK: 15m schedule len={len(s15)}")


def test_schedule_120():
    base = ["2026-06-01 00:00:00", "2026-06-01 02:00:00"]
    s120 = build_cadence_schedule(base, 120)
    if len(s120) != 2:
        _fail("120m should match base")
    print("OK: 120m schedule")


def test_full_run():
    with tempfile.TemporaryDirectory() as tmp:
        data = Path(tmp)
        (data / "zero_base").mkdir(parents=True)
        for src in (
            PKG / "data" / "zero_base" / "entry_filter_rules_v2.json",
            PKG / "research_bundle" / "reports" / "entry_filter_rules_v2.json",
        ):
            if src.exists():
                (data / "zero_base" / "entry_filter_rules_v2.json").write_text(
                    src.read_text(encoding="utf-8"), encoding="utf-8",
                )
                break
        from scout_auto_os.engine.research.cadence.runner import ScanCadenceOptimizerRunner

        runner = ScanCadenceOptimizerRunner(
            data,
            PKG,
            PKG / "research_bundle" / "seed" / "candidates.jsonl",
            PKG / "research_bundle" / "forward" / "forward_klines_15m.jsonl",
            lookback_days=180,
        )
        result = runner.run()
        if not result:
            _fail("run failed")
        out = data / "cadence"
        for name in (
            "cadence_summary.csv",
            "cadence_report.md",
            "cadence_portfolio_log.csv",
            "cadence_turnover_report.csv",
        ):
            if not (out / name).exists():
                _fail(f"missing {name}")
        print(f"OK: intervals={len(result['summaries'])}")


def main():
    test_schedule_sub120()
    test_schedule_120()
    test_full_run()
    print("\nALL SCAN CADENCE OPTIMIZER V1 TESTS PASSED")


if __name__ == "__main__":
    main()
