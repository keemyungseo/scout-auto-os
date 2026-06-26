"""Research Infrastructure V1 tests."""

from __future__ import annotations

import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PKG / "research_bundle"))


def test_forward_labels():
    from scout_auto_os.engine.research.infrastructure.forward_labeler import build_forward_labels

    klines = [
        [0, 100.0, 101.0, 99.0, 100.5, 1000],
        [1, 100.5, 103.0, 100.0, 102.5, 1100],
        [2, 102.5, 104.0, 101.0, 103.0, 1200],
    ]
    m = build_forward_labels(klines, 100.0)
    assert m and "return_minus_dd" in m
    print("OK: forward labels")


def test_database():
    from scout_auto_os.engine.research.infrastructure.dataset_manager import HistoryDatabase

    db_path = PKG / "data" / "research_infrastructure" / "_test_history.db"
    if db_path.exists():
        db_path.unlink()
    db = HistoryDatabase(db_path)
    db.upsert_scan("2026-06-01 00:00:00", "2026-06-01", 10, {"market_simple": "bull"})
    assert db.scan_count() == 1
    db_path.unlink()
    print("OK: sqlite database")


def test_full_run():
    try:
        import sklearn  # noqa: F401
    except ImportError:
        print("SKIP: no sklearn")
        return

    from scout_auto_os.engine.research.infrastructure.runner import ResearchInfrastructureRunner

    runner = ResearchInfrastructureRunner(
        PKG / "data",
        PKG,
        PKG / "research_bundle" / "seed" / "candidates.jsonl",
        PKG / "research_bundle" / "forward" / "forward_klines_15m.jsonl",
    )
    result = runner.run()
    if not result:
        raise SystemExit("run failed")
    out = PKG / "data" / "research_infrastructure"
    for name in ("research_dashboard.md", "dataset_status.csv", "history.db"):
        if not (out / name).exists():
            raise SystemExit(f"missing {name}")
    print(f"OK: days={result['meta']['status'].get('calendar_days')}")


def main():
    test_forward_labels()
    test_database()
    test_full_run()
    print("\nALL RESEARCH INFRASTRUCTURE V1 TESTS PASSED")


if __name__ == "__main__":
    main()
