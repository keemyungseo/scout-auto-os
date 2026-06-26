"""Signal Lifecycle Engine V1 tests."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PKG / "research_bundle"))

from scout_auto_os.engine.research.signal_lifecycle.shape_classifier import classify_lifecycle_shape
from scout_auto_os.engine.research.signal_lifecycle.timeline import build_signal_timeline


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def _sample_klines(n: int = 30, drift: float = 0.002) -> list:
    """Synthetic 15m bars with upward drift."""
    bars = []
    px = 100.0
    t0 = 1780239600000
    for i in range(n):
        o = px
        c = px * (1 + drift)
        h = max(o, c) * 1.003
        l = min(o, c) * 0.997
        bars.append([t0 + i * 900_000, o, h, l, c, 1_000_000.0 * (1 + i * 0.01)])
        px = c
    return bars


def test_timeline_build():
    klines = _sample_klines(49)
    timeline, summary = build_signal_timeline(
        klines, "long", "2026-06-01 00:00:00", "TESTUSDT", "long_test",
    )
    if len(timeline) < 24:
        _fail(f"expected >=24 bars, got {len(timeline)}")
    if summary["return_2h"] <= 0:
        _fail("uptrend should show positive 2h return")
    if not all("velocity_pct_per_hour" in r for r in timeline):
        _fail("missing velocity")
    print(f"OK: timeline bars={len(timeline)} return_6h={summary['return_6h']}")


def test_classifier_dead():
    label = classify_lifecycle_shape(
        {
            "peak_time_min": 30,
            "peak_return_pct": 0.3,
            "mfe_full": 0.4,
            "return_2h": 0.1,
            "return_6h": 0.0,
            "return_at_end": 0.0,
            "return_90m": 0.0,
            "early_mae_1h": -0.2,
            "peak_fraction": 0.2,
            "end_time_min": 720,
        },
    )
    if label != "Dead Signal":
        _fail(f"expected Dead Signal, got {label}")
    print("OK: Dead Signal")


def test_classifier_explosion():
    label = classify_lifecycle_shape(
        {
            "peak_time_min": 30,
            "peak_return_pct": 5.0,
            "mfe_full": 5.5,
            "return_2h": 4.0,
            "return_6h": 3.5,
            "return_at_end": 3.0,
            "return_90m": 4.0,
            "early_mae_1h": -0.5,
            "peak_fraction": 0.1,
            "end_time_min": 720,
        },
    )
    if label != "Immediate Explosion":
        _fail(f"expected Immediate Explosion, got {label}")
    print("OK: Immediate Explosion")


def test_full_run():
    with tempfile.TemporaryDirectory() as tmp:
        from scout_auto_os.engine.research.signal_lifecycle.runner import SignalLifecycleRunner

        runner = SignalLifecycleRunner(
            Path(tmp),
            PKG,
            PKG / "research_bundle" / "seed" / "candidates.jsonl",
            PKG / "research_bundle" / "forward" / "forward_klines_15m.jsonl",
        )
        result = runner.run()
        out = Path(tmp) / "signal_lifecycle"
        for name in (
            "signal_lifecycle.csv",
            "signal_timeline.csv",
            "signal_shape.csv",
            "lifecycle_cluster.csv",
            "lifecycle_report.md",
        ):
            if not (out / name).exists():
                _fail(f"missing {name}")
        print(
            f"OK: signals={result['meta']['long_signal_count']}+"
            f"{result['meta']['short_signal_count']} clusters={result['cluster_count']}",
        )


def main():
    test_timeline_build()
    test_classifier_dead()
    test_classifier_explosion()
    test_full_run()
    print("\nALL SIGNAL LIFECYCLE V1 TESTS PASSED")


if __name__ == "__main__":
    main()
