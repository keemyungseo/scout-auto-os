"""Rule Portfolio Engine V1 tests."""

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


def test_collectors():
    from scout_auto_os.engine.research.rule_portfolio.collectors import collect_all_rules

    with tempfile.TemporaryDirectory() as tmp:
        data = Path(tmp)
        (data / "execution_rule_discovery").mkdir(parents=True)
        for src in (
            PKG / "data" / "execution_rule_discovery" / "top20_execution_rules.csv",
            PKG / "research_bundle" / "reports" / "top20_execution_rules_v1.csv",
        ):
            if src.exists():
                (data / "execution_rule_discovery" / "top20_execution_rules.csv").write_text(
                    src.read_text(encoding="utf-8"), encoding="utf-8",
                )
                break
        rules = collect_all_rules(data, PKG, [], [])
        if len(rules) < 2:
            _fail("expected baseline rules")
        print(f"OK: collected {len(rules)} rules")


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
        for name in ("top20_execution_rules.csv", "execution_rules_train_rank.csv", "recommended_execution_rule.json"):
            for base in (PKG / "data" / "execution_rule_discovery", PKG / "research_bundle" / "reports"):
                src = base / (name if name.endswith(".json") else name.replace(".csv", "_v1.csv") if "top20" in name else name)
                if "top20" in name and "reports" in str(base):
                    src = base / "top20_execution_rules_v1.csv"
                elif name.endswith(".json") and "reports" in str(base):
                    src = base / "recommended_execution_rule_v1.json"
                if src.exists():
                    dst = data / "execution_rule_discovery" / name
                    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
                    break

        from scout_auto_os.engine.research.rule_portfolio.runner import RulePortfolioRunner

        runner = RulePortfolioRunner(
            data, PKG,
            PKG / "research_bundle" / "seed" / "candidates.jsonl",
            PKG / "research_bundle" / "forward" / "forward_klines_15m.jsonl",
        )
        result = runner.run()
        if not result:
            _fail("run failed")
        out = data / "rule_portfolio"
        for name in (
            "rule_library.csv",
            "rule_metadata.csv",
            "rule_cluster.csv",
            "rule_activation_matrix.csv",
            "rule_portfolio.md",
        ):
            if not (out / name).exists():
                _fail(f"missing {name}")
        print(f"OK: {result['meta']['rule_count']} rules")


def main():
    test_collectors()
    test_full_run()
    print("\nALL RULE PORTFOLIO V1 TESTS PASSED")


if __name__ == "__main__":
    main()
