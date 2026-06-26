"""Execution Engine V1 tests."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PKG / "research_bundle"))

from scout_auto_os.engine.research.execution_research.observation import compute_observation_features, execution_score


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def test_observation_features():
    klines = [[0, 100.0, 103.0, 99.0, 102.0, 2_000_000.0]]
    feats = compute_observation_features(klines, "long", {"15m_current_volume_ratio": 1.5, "1h_current_range_pct": 5.0})
    if not feats or feats["obs_return_pct"] <= 0:
        _fail("obs return")
    sc = execution_score(feats)
    print(f"OK: execution_score={sc}")


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
        from scout_auto_os.engine.research.execution_research.runner import ExecutionResearchRunner

        runner = ExecutionResearchRunner(
            data, PKG,
            PKG / "research_bundle" / "seed" / "candidates.jsonl",
            PKG / "research_bundle" / "forward" / "forward_klines_15m.jsonl",
        )
        result = runner.run()
        if not result:
            _fail("run failed")
        out = data / "execution_engine"
        if not (out / "execution_report.md").exists():
            _fail("report missing")
        print(f"OK: lift={result['combined'].get('lift_vs_entry_top2_pct')}%")


def main():
    test_observation_features()
    test_full_run()
    print("\nALL EXECUTION ENGINE V1 TESTS PASSED")


if __name__ == "__main__":
    main()
