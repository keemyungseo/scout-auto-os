"""Entry Rule Optimizer V2 tests."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PKG / "research_bundle"))

from scout_auto_os.engine.research.directional.entry_filter.rule_combinator import generate_rule_trees
from scout_auto_os.engine.research.directional.entry_filter.rule_tree import Condition, and_node
from scout_auto_os.engine.research.directional.entry_filter.rule_evaluator import evaluate_rule_tree
from scout_auto_os.engine.research.directional.entry_filter.rule_optimizer_v2 import EntryRuleOptimizerV2


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def test_combinator_count():
    conds = [
        Condition("A", "f1", "gte", 1.0),
        Condition("B", "f2", "gte", 2.0),
        Condition("C", "f3", "gte", 3.0),
        Condition("D", "f4", "gte", 4.0),
    ]
    trees = generate_rule_trees(conds)
    # 15 AND subsets + multi-block partitions
    if len(trees) < 20:
        _fail(f"expected >=20 trees got {len(trees)}")
    ids = {t[0] for t in trees}
    if "AND_ABCD" not in ids:
        _fail(f"missing AND_ABCD in {ids}")
    print(f"OK: {len(trees)} rule trees")


def test_evaluate():
    tree = and_node([Condition("A", "x", "gte", 5.0)])
    signals = [
        {"cohort": "winner", "return_2h": 10, "return_4h": 5, "features": {"x": 6}, "scan_time_kst": "t1"},
        {"cohort": "loser", "return_2h": 1, "return_4h": 0, "features": {"x": 1}, "scan_time_kst": "t1"},
    ]
    row = evaluate_rule_tree(signals, tree, "AND_A", "test")
    if row["pass_count"] != 1:
        _fail("pass count")
    print(f"OK: precision={row['precision']}")


def test_full_run():
    with tempfile.TemporaryDirectory() as tmp:
        zb = Path(tmp) / "zero_base"
        zb.mkdir(parents=True)
        for fname in ("direction_champion_signals.csv", "entry_filter_rules_v1.json"):
            src = PKG / "data" / "zero_base" / fname
            if src.exists():
                (zb / fname).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        runner = EntryRuleOptimizerV2(
            Path(tmp),
            pkg_root=PKG,
            candidates_path=PKG / "research_bundle" / "seed" / "candidates.jsonl",
            forward_path=PKG / "research_bundle" / "forward" / "forward_klines_15m.jsonl",
        )
        result = runner.run()
        out = Path(tmp) / "zero_base"
        for fname in ("rule_combination_report.csv", "best_live_rule.csv", "live_entry_rule_v2.md"):
            if not (out / fname).exists():
                _fail(f"missing {fname}")
        if not result.get("best_long"):
            _fail("no best long")
        print(f"OK: long recall={result['best_long'].get('recall')}")


def main():
    test_combinator_count()
    test_evaluate()
    test_full_run()
    print("\nALL RULE OPTIMIZER V2 TESTS PASSED")


if __name__ == "__main__":
    main()
