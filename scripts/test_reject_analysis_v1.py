"""Reject Analysis V1 tests."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PKG / "research_bundle"))

from scout_auto_os.engine.portfolio.reject_analysis.rule_audit import audit_rule, classify_reject_tier
from scout_auto_os.engine.research.directional.entry_filter.rule_tree import Condition, and_node


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def test_near_pass_tier():
    failed = [{"gap_pct": 5.0}]
    if classify_reject_tier(failed) != "near_pass":
        _fail("near pass tier")
    print("OK: near pass tier")


def test_rule_audit_fail():
    tree = and_node([
        Condition("B", "1h_current_body_pct", "gte", 13.83),
        Condition("D", "2h_current_body_pct", "gte", 22.12),
    ])
    features = {"1h_current_body_pct": 13.72, "2h_current_body_pct": 25.0}
    audit = audit_rule(features, tree, "long")
    if audit["rule_pass"]:
        _fail("should fail rule")
    if not audit["failed_conditions"]:
        _fail("need failed cond")
    print(f"OK: gap={audit['failed_conditions'][0]['gap']}")


def test_full_run():
    with tempfile.TemporaryDirectory() as tmp:
        data = Path(tmp)
        (data / "zero_base").mkdir(parents=True)
        src = PKG / "research_bundle" / "reports" / "entry_filter_rules_v2.json"
        if not src.exists():
            src = PKG / "data" / "zero_base" / "entry_filter_rules_v2.json"
        (data / "zero_base" / "entry_filter_rules_v2.json").write_text(
            src.read_text(encoding="utf-8"), encoding="utf-8",
        )
        from scout_auto_os.engine.portfolio.reject_analysis.runner import RejectAnalysisRunner
        runner = RejectAnalysisRunner(
            data, PKG,
            PKG / "research_bundle" / "seed" / "candidates.jsonl",
        )
        result = runner.run()
        out = data / "portfolio"
        if not (out / "coverage_report.md").exists():
            _fail("report missing")
        print(f"OK: rule pass rate={result['meta']['funnel']['rule_pass_rate_pct']}%")


def main():
    test_near_pass_tier()
    test_rule_audit_fail()
    test_full_run()
    print("\nALL REJECT ANALYSIS V1 TESTS PASSED")


if __name__ == "__main__":
    main()
