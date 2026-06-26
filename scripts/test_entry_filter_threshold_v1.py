"""Entry Filter Threshold Optimizer V1 tests."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PKG / "research_bundle"))

from scout_auto_os.engine.research.directional.entry_filter.threshold_optimizer import (
    _metrics_at_threshold,
    grid_search_feature,
)
from scout_auto_os.engine.research.directional.entry_filter.threshold_runner import EntryFilterThresholdRunner


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def test_metrics():
    signals = []
    for i in range(100):
        cohort = "winner" if i < 20 else ("loser" if i >= 80 else "middle")
        signals.append({
            "cohort": cohort,
            "return_2h": float(i),
            "return_4h": float(i) * 0.5,
            "features": {"5m_momentum": float(i)},
        })
    m = _metrics_at_threshold(signals, "5m_momentum", 15.0, "gte")
    if m["pass_count"] < 20:
        _fail("expected pass group")
    if m["precision"] <= 0:
        _fail("precision should be positive for separating threshold")
    print(f"OK: precision={m['precision']} lift={m['lift']}")


def test_grid_search():
    signals = []
    for i in range(80):
        cohort = "winner" if i < 16 else ("loser" if i >= 64 else "middle")
        signals.append({
            "cohort": cohort,
            "return_2h": float(i),
            "return_4h": 0.0,
            "features": {"feat_a": float(i)},
        })
    best, curve = grid_search_feature(signals, "feat_a", "long")
    if not best or not curve:
        _fail("grid search empty")
    print(f"OK: best threshold={best['threshold']} f1={best['f1']}")


def test_full_run():
    with tempfile.TemporaryDirectory() as tmp:
        # copy signals from data if available
        src = PKG / "data" / "zero_base" / "direction_champion_signals.csv"
        if src.exists():
            zb = Path(tmp) / "zero_base"
            zb.mkdir(parents=True)
            (zb / "direction_champion_signals.csv").write_text(
                src.read_text(encoding="utf-8"), encoding="utf-8",
            )
        runner = EntryFilterThresholdRunner(
            Path(tmp),
            pkg_root=PKG,
            candidates_path=PKG / "research_bundle" / "seed" / "candidates.jsonl",
            forward_path=PKG / "research_bundle" / "forward" / "forward_klines_15m.jsonl",
        )
        result = runner.run(max_rule_features=3)
        out = Path(tmp) / "zero_base"
        for fname in (
            "feature_best_threshold.csv",
            "entry_filter_rule_v1.md",
            "feature_threshold_curve.csv",
        ):
            if not (out / fname).exists():
                _fail(f"missing {fname}")
        if not result["long_rules"]:
            _fail("no long rules")
        print(f"OK: long_rules={len(result['long_rules'])}")


def main():
    test_metrics()
    test_grid_search()
    test_full_run()
    print("\nALL THRESHOLD OPTIMIZER V1 TESTS PASSED")


if __name__ == "__main__":
    main()
