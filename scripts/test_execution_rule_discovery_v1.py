"""Execution Rule Discovery V1 tests."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PKG / "research_bundle"))


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


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
        from scout_auto_os.engine.research.execution_rule_discovery.runner import ExecutionRuleDiscoveryRunner

        runner = ExecutionRuleDiscoveryRunner(
            data, PKG,
            PKG / "research_bundle" / "seed" / "candidates.jsonl",
            PKG / "research_bundle" / "forward" / "forward_klines_15m.jsonl",
        )
        result = runner.run()
        if not result:
            _fail("run failed")
        out = data / "execution_rule_discovery"
        if not (out / "execution_replacement_report.md").exists():
            _fail("report missing")
        if not (out / "recommended_execution_rule.json").exists():
            _fail("json missing")
        print(f"OK: decision={result['recommendation']['decision']}")


def main():
    test_full_run()
    print("\nALL EXECUTION RULE DISCOVERY V1 TESTS PASSED")


if __name__ == "__main__":
    main()
