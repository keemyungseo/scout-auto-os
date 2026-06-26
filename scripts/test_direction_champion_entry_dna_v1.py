"""Direction Champion Entry DNA V1 tests."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PKG / "research_bundle"))

from scout_auto_os.engine.research.directional.entry_filter.analyzer import (
    compare_winner_loser_features,
    split_winner_loser,
)
from scout_auto_os.engine.research.directional.entry_filter.collector import filter_scans_last_months
from scout_auto_os.engine.research.directional.entry_filter.runner import DirectionChampionEntryDnaRunner


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def test_split_quantiles():
    signals = [{"return_2h": float(i), "features": {"5m_momentum": i}} for i in range(100)]
    w, l, meta = split_winner_loser(signals)
    if len(w) != 20 or len(l) != 20:
        _fail(f"expected 20/20 split got {len(w)}/{len(l)}")
    if w[0]["return_2h"] <= w[-1]["return_2h"]:
        _fail("winners not sorted")
    print("OK: winner/loser split 20/20")


def test_feature_compare():
    winners = [{"features": {"5m_momentum": 5.0}} for _ in range(30)]
    losers = [{"features": {"5m_momentum": -1.0}} for _ in range(30)]
    rows = compare_winner_loser_features(winners, losers, ["5m_momentum"], "long")
    if not rows or rows[0]["effect_size"] <= 0:
        _fail("expected positive effect for momentum")
    print(f"OK: effect_size={rows[0]['effect_size']}")


def test_filter_months():
    scans = [f"2026-06-{d:02d} 12:00:00" for d in range(1, 16)]
    out = filter_scans_last_months(scans, 6)
    if len(out) != 15:
        _fail("filter should keep all when within window")
    print("OK: month filter")


def test_full_run():
    with tempfile.TemporaryDirectory() as tmp:
        runner = DirectionChampionEntryDnaRunner(
            Path(tmp),
            pkg_root=PKG,
            candidates_path=PKG / "research_bundle" / "seed" / "candidates.jsonl",
            forward_path=PKG / "research_bundle" / "forward" / "forward_klines_15m.jsonl",
        )
        result = runner.run(top_k=3)
        out = Path(tmp) / "zero_base"
        for fname in (
            "direction_champion_entry_dna_report.md",
            "direction_champion_signals.csv",
            "entry_dna_feature_importance_long.csv",
        ):
            if not (out / fname).exists():
                _fail(f"missing {fname}")
        if result["meta"]["long_signal_count"] < 10:
            _fail("too few long signals")
        print(f"OK: signals={result['meta']['long_signal_count']}")


def main():
    test_split_quantiles()
    test_feature_compare()
    test_filter_months()
    test_full_run()
    print("\nALL ENTRY DNA V1 TESTS PASSED")


if __name__ == "__main__":
    main()
