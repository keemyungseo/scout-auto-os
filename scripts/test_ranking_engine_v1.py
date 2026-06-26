"""Ranking Engine V1 tests."""

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


def test_imports():
    try:
        import sklearn  # noqa: F401
        import lightgbm  # noqa: F401
    except ImportError:
        print("SKIP: ML deps not installed")
        return
    print("OK: ML imports")


def test_full_run():
    try:
        import sklearn  # noqa: F401
    except ImportError:
        print("SKIP: no sklearn")
        return

    from scout_auto_os.engine.research.ranking_engine.runner import RankingEngineRunner

    runner = RankingEngineRunner(
        PKG / "data",
        PKG,
        PKG / "research_bundle" / "seed" / "candidates.jsonl",
        PKG / "research_bundle" / "forward" / "forward_klines_15m.jsonl",
    )
    result = runner.run()
    if not result:
        _fail("run failed")
    out = PKG / "data" / "ranking_engine"
    for name in ("ranking_report.md", "model_comparison.csv", "feature_importance.csv"):
        if not (out / name).exists():
            _fail(f"missing {name}")
    print(f"OK: best={result['meta']['best_model']}")


def main():
    test_imports()
    test_full_run()
    print("\nALL RANKING ENGINE V1 TESTS PASSED")


if __name__ == "__main__":
    main()
