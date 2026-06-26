"""Execution Generalization V1 tests."""

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


def test_rule_loader():
    from scout_auto_os.engine.research.execution_generalization.rule_loader import load_frozen_execution_rule

    p = PKG / "data" / "execution_rule_discovery" / "recommended_execution_rule.json"
    if not p.exists():
        p = PKG / "research_bundle" / "reports" / "recommended_execution_rule_v1.json"
    if not p.exists():
        print("SKIP: no frozen rule file")
        return
    rule = load_frozen_execution_rule(p)
    if rule.direction != "long":
        _fail("direction")
    print(f"OK: loaded {rule.rule_expr[:50]}")


def test_full_run():
    with tempfile.TemporaryDirectory() as tmp:
        data = Path(tmp)
        (data / "zero_base").mkdir(parents=True)
        (data / "execution_rule_discovery").mkdir(parents=True)
        for src in (
            PKG / "data" / "zero_base" / "entry_filter_rules_v2.json",
            PKG / "research_bundle" / "reports" / "entry_filter_rules_v2.json",
        ):
            if src.exists():
                (data / "zero_base" / "entry_filter_rules_v2.json").write_text(
                    src.read_text(encoding="utf-8"), encoding="utf-8",
                )
                break
        for src in (
            PKG / "data" / "execution_rule_discovery" / "recommended_execution_rule.json",
            PKG / "research_bundle" / "reports" / "recommended_execution_rule_v1.json",
        ):
            if src.exists():
                (data / "execution_rule_discovery" / "recommended_execution_rule.json").write_text(
                    src.read_text(encoding="utf-8"), encoding="utf-8",
                )
                break
        else:
            print("SKIP: no rule json")
            return

        from scout_auto_os.engine.research.execution_generalization.runner import ExecutionGeneralizationRunner

        runner = ExecutionGeneralizationRunner(
            data, PKG,
            PKG / "research_bundle" / "seed" / "candidates.jsonl",
            PKG / "research_bundle" / "forward" / "forward_klines_15m.jsonl",
        )
        result = runner.run()
        if not result:
            _fail("run failed")
        out = data / "execution_generalization"
        for name in ("generalization_report.md", "regime_report.md", "generalization_decision.json"):
            if not (out / name).exists():
                _fail(f"missing {name}")
        print(f"OK: {result['decision']['decision']}")


def main():
    test_rule_loader()
    test_full_run()
    print("\nALL EXECUTION GENERALIZATION V1 TESTS PASSED")


if __name__ == "__main__":
    main()
