"""Adaptive Feature Weight V1 tests."""

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


def test_conditions():
    from scout_auto_os.engine.research.adaptive_feature_weight.scan_conditions import classify_scan

    rows = [{"x": {"dna_1h_current_range_pct": 25.0, "dna_1h_current_return_pct": 5.0}}] * 10
    tags = classify_scan(rows, {"high_volatility": 15.0, "strong_trend": 2.0})
    if not tags:
        _fail("no tags")
    print(f"OK: tags={tags}")


def test_full_run():
    try:
        import sklearn  # noqa: F401
    except ImportError:
        print("SKIP: no sklearn")
        return

    from scout_auto_os.engine.research.adaptive_feature_weight.runner import AdaptiveFeatureWeightRunner

    runner = AdaptiveFeatureWeightRunner(
        PKG / "data",
        PKG,
        PKG / "research_bundle" / "seed" / "candidates.jsonl",
        PKG / "research_bundle" / "forward" / "forward_klines_15m.jsonl",
    )
    result = runner.run()
    if not result:
        _fail("run failed")
    out = PKG / "data" / "adaptive_feature_weight"
    for name in (
        "adaptive_feature_map.csv",
        "feature_importance_heatmap.csv",
        "adaptive_feature_report.md",
        "weight_comparison.csv",
    ):
        if not (out / name).exists():
            _fail(f"missing {name}")
    print("OK: adaptive feature weight complete")


def main():
    test_conditions()
    test_full_run()
    print("\nALL ADAPTIVE FEATURE WEIGHT V1 TESTS PASSED")


if __name__ == "__main__":
    main()
