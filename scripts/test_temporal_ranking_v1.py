"""Temporal Ranking V1 tests."""

from __future__ import annotations

import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PKG / "research_bundle"))


def test_temporal_features():
    from scout_auto_os.engine.research.temporal_ranking.features import build_temporal_features

    seq = [
        {"entry_score": 80.0, "direction_confidence": 0.8},
        {"entry_score": 70.0, "direction_confidence": 0.6},
        {"entry_score": 65.0, "direction_confidence": 0.5},
    ]
    out = build_temporal_features(seq, ("entry_score", "direction_confidence"))
    assert "ts_entry_score_delta" in out
    assert out["ts_entry_score_delta"] == 10.0
    print("OK: temporal features")


def test_full_run():
    try:
        import sklearn  # noqa: F401
    except ImportError:
        print("SKIP: no sklearn")
        return

    from scout_auto_os.engine.research.temporal_ranking.runner import TemporalRankingRunner

    runner = TemporalRankingRunner(
        PKG / "data",
        PKG,
        PKG / "research_bundle" / "seed" / "candidates.jsonl",
        PKG / "research_bundle" / "forward" / "forward_klines_15m.jsonl",
    )
    result = runner.run()
    if not result:
        raise SystemExit("run failed")
    out = PKG / "data" / "temporal_ranking"
    for name in ("temporal_ranking_report.md", "sequence_length_comparison.csv", "leak_check.csv"):
        if not (out / name).exists():
            raise SystemExit(f"missing {name}")
    print(f"OK: improved={result['decision'].get('improved')}")


def main():
    test_temporal_features()
    test_full_run()
    print("\nALL TEMPORAL RANKING V1 TESTS PASSED")


if __name__ == "__main__":
    main()
