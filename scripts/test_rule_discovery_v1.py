"""Rule Discovery Engine V1 tests."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PKG / "research_bundle"))

from scout_auto_os.engine.research.rule_discovery.discovered_rule import Predicate, pred_node
from scout_auto_os.engine.research.rule_discovery.discovered_rule import DiscoveredRule


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def test_predicate_eval():
    p = Predicate(kind="threshold", feature_a="1h_current_body_pct", operator="gte", threshold=10.0)
    rule = DiscoveredRule("T1", p.describe(), pred_node(p), "long")
    if not rule.evaluate({"1h_current_body_pct": 15.0}):
        _fail("should pass")
    if rule.evaluate({"1h_current_body_pct": 5.0}):
        _fail("should fail")
    print("OK: predicate")


def test_ratio():
    p = Predicate(kind="ratio", feature_a="1h_current_body_pct", feature_b="1h_current_range_pct", operator="gte", threshold=0.5)
    feats = {"1h_current_body_pct": 10, "1h_current_range_pct": 15}
    if not p.evaluate(feats):
        _fail("ratio fail")
    print("OK: ratio")


def test_full_run():
    with tempfile.TemporaryDirectory() as tmp:
        data = Path(tmp)
        (data / "zero_base").mkdir(parents=True)
        for name in ("entry_filter_rules_v2.json", "directional_dna_formulas.json"):
            pass
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
            PKG / "data" / "zero_base" / "entry_dna_feature_importance_long.csv",
            PKG / "research_bundle" / "reports" / "entry_dna_feature_importance_long_v1.csv",
        ):
            if src.exists():
                (data / "zero_base" / "entry_dna_feature_importance_long.csv").write_text(
                    src.read_text(encoding="utf-8"), encoding="utf-8",
                )
                break
        for src in (
            PKG / "data" / "zero_base" / "entry_dna_feature_importance_short.csv",
            PKG / "research_bundle" / "reports" / "entry_dna_feature_importance_short_v1.csv",
        ):
            if src.exists():
                (data / "zero_base" / "entry_dna_feature_importance_short.csv").write_text(
                    src.read_text(encoding="utf-8"), encoding="utf-8",
                )
                break

        from scout_auto_os.engine.research.rule_discovery.runner import RuleDiscoveryRunner

        runner = RuleDiscoveryRunner(
            data,
            PKG,
            PKG / "research_bundle" / "seed" / "candidates.jsonl",
            PKG / "research_bundle" / "forward" / "forward_klines_15m.jsonl",
        )
        result = runner.run()
        if not result:
            _fail("run returned None")
        out = data / "rule_discovery"
        for name in (
            "rule_discovery_report.md",
            "top20_candidate_rules.csv",
            "hybrid_rule_comparison.csv",
            "recommended_rule.json",
        ):
            if not (out / name).exists():
                _fail(f"missing {name}")
        print(f"OK: decision={result['recommendation'].get('decision')}")


def main():
    test_predicate_eval()
    test_ratio()
    test_full_run()
    print("\nALL RULE DISCOVERY V1 TESTS PASSED")


if __name__ == "__main__":
    main()
