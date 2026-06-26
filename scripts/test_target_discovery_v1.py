"""Target Discovery V1 tests."""

from __future__ import annotations

import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PKG / "research_bundle"))


def test_label_metrics():
    from scout_auto_os.engine.research.target_discovery.label_builder import compute_label_metrics

    klines = [
        [0, 100.0, 101.0, 99.0, 100.5, 1000],
        [1, 100.5, 103.0, 100.0, 102.5, 1100],
        [2, 102.5, 104.0, 101.0, 103.0, 1200],
    ]
    m = compute_label_metrics(klines, 100.0)
    assert "max_up_30m" in m
    assert m["return_30m"] > 0
    print("OK: label metrics")


def test_candidates():
    from scout_auto_os.engine.research.target_discovery.candidate_generator import generate_label_candidates

    specs = generate_label_candidates()
    assert len(specs) >= 20
    assert specs[0].label_id == "baseline_max_up_4h"
    print(f"OK: {len(specs)} label candidates")


def test_full_run():
    try:
        import sklearn  # noqa: F401
    except ImportError:
        print("SKIP: no sklearn")
        return

    from scout_auto_os.engine.research.target_discovery.runner import TargetDiscoveryRunner

    runner = TargetDiscoveryRunner(
        PKG / "data",
        PKG,
        PKG / "research_bundle" / "seed" / "candidates.jsonl",
        PKG / "research_bundle" / "forward" / "forward_klines_15m.jsonl",
    )
    result = runner.run()
    if not result:
        raise SystemExit("run failed")
    out = PKG / "data" / "target_discovery"
    for name in ("target_discovery_report.md", "label_ranking.csv", "blind_comparison.csv"):
        if not (out / name).exists():
            raise SystemExit(f"missing {name}")
    print(f"OK: improved={result['decision'].get('improved')}")


def main():
    test_label_metrics()
    test_candidates()
    test_full_run()
    print("\nALL TARGET DISCOVERY V1 TESTS PASSED")


if __name__ == "__main__":
    main()
