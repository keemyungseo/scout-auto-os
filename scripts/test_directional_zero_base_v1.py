"""SCOUT Directional Zero-Base V1 — small sample tests."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PKG / "research_bundle"))

from scout_auto_os.engine.research.directional.engines import rank_long, rank_short, LONG_ENGINES, SHORT_ENGINES
from scout_auto_os.engine.research.directional.evaluation import to_short_metrics, to_long_metrics, aggregate_directional
from scout_auto_os.engine.research.directional.patterns import label_direction_pattern, LONG_PATTERNS, SHORT_PATTERNS
from scout_auto_os.engine.research.directional.runner import DirectionalZeroBaseRunner
from scout_auto_os.engine.research.zero_base.forward_eval import compute_forward_metrics


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def test_pattern_label():
    f = {
        "15m_previous_return_pct": -1.0,
        "15m_current_return_pct": 1.2,
        "1h_previous_return_pct": -0.5,
        "1h_current_return_pct": 0.3,
        "5m_compression": 2,
    }
    pat = label_direction_pattern(f)
    if pat not in LONG_PATTERNS + SHORT_PATTERNS + ("UNLABELED",):
        _fail(f"unknown pattern {pat}")
    print(f"OK: pattern={pat}")


def test_short_return_invert():
    klines = [[0, 100, 105, 95, 102, 1], [1, 102, 103, 98, 97, 1],
              [2, 97, 98, 94, 95, 1], [3, 95, 96, 92, 93, 1]]
    long_m = compute_forward_metrics(klines, 100)
    short_m = to_short_metrics(long_m)
    if short_m["short_return_2h"] != -long_m["return_2h"]:
        _fail("short return invert mismatch")
    print(f"OK: long_r2h={long_m['return_2h']} short_r2h={short_m['short_return_2h']}")


def test_rank_engines():
    rows = []
    for line in open(PKG / "research_bundle" / "seed" / "candidates.jsonl", encoding="utf-8"):
        if line.strip():
            r = __import__("json").loads(line)
            if r["scan_kst"] == "2026-06-01 00:00:00":
                rows.append({"symbol": r["symbol"], "features": r["features"]})
    if len(rows) < 10:
        _fail("not enough rows")
    long_p = rank_long(rows, "LONG_V_REVERSAL", 5)
    short_p = rank_short(rows, "SHORT_V_REVERSAL", 5)
    if len(long_p) != 5 or len(short_p) != 5:
        _fail("rank top5 failed")
    print(f"OK: long={long_p[0]} short={short_p[0]}")


def test_small_run():
    with tempfile.TemporaryDirectory() as tmp:
        runner = DirectionalZeroBaseRunner(
            Path(tmp),
            candidates_path=PKG / "research_bundle" / "seed" / "candidates.jsonl",
            forward_path=PKG / "research_bundle" / "forward" / "forward_klines_15m.jsonl",
            random_draws=10,
        )
        # limit engines via max_scans
        result = runner.run(max_scans=10)
        if not result.get("long_board"):
            _fail("no long board")
        if not (Path(tmp) / "zero_base" / "directional_report.md").exists():
            _fail("report missing")
        print(f"OK: long_top={result['long_board'][0].get('engine')} slots={result['slot_sim']}")


def main():
    test_pattern_label()
    test_short_return_invert()
    test_rank_engines()
    test_small_run()
    print("\nALL DIRECTIONAL ZERO-BASE V1 TESTS PASSED")


if __name__ == "__main__":
    main()
