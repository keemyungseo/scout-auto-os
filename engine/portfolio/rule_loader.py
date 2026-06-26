"""Load Entry Rule V2 trees for portfolio filtering."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from scout_auto_os.engine.research.directional.entry_filter.rule_combinator import generate_rule_trees
from scout_auto_os.engine.research.directional.entry_filter.rule_tree import (
    Condition,
    RuleNode,
    conditions_from_v1,
)


@dataclass
class PortfolioRules:
    long_tree: RuleNode
    short_tree: RuleNode
    long_meta: dict
    short_meta: dict
    pattern_trees: dict[str, RuleNode] = field(default_factory=dict)
    pattern_meta: dict[str, dict] = field(default_factory=dict)
    condition_map: dict[str, dict[str, Condition]] = field(default_factory=dict)


def _tree_for_rule_id(conditions: list[Condition], rule_id: str) -> RuleNode:
    for rid, tree in generate_rule_trees(conditions):
        if rid == rule_id:
            return tree
    if rule_id.startswith("AND_") and len(rule_id) > 4:
        letters = rule_id[4:]
        subset = [c for c in conditions if c.letter in letters]
        if subset:
            from scout_auto_os.engine.research.directional.entry_filter.rule_tree import and_node
            return and_node(subset)
    raise ValueError(f"rule tree not found: {rule_id}")


def load_portfolio_rules(data_dir: Path, pkg_root: Path) -> PortfolioRules:
    for path in (
        data_dir / "zero_base" / "entry_filter_rules_v2.json",
        pkg_root / "research_bundle" / "reports" / "entry_filter_rules_v2.json",
    ):
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            break
    else:
        raise FileNotFoundError("entry_filter_rules_v2.json not found")

    long_conds = conditions_from_v1(raw["long"]["conditions"])
    short_conds = conditions_from_v1(raw["short"]["conditions"])
    long_tree = _tree_for_rule_id(long_conds, raw["long"]["rule_id"])
    short_tree = _tree_for_rule_id(short_conds, raw["short"]["rule_id"])

    pattern_trees: dict[str, RuleNode] = {}
    pattern_meta: dict[str, dict] = {}
    for pname, pdata in raw.get("patterns", {}).items():
        direction = pdata.get("direction", "long")
        base = long_conds if direction == "long" else short_conds
        try:
            pattern_trees[pname] = _tree_for_rule_id(base, pdata["rule_id"])
            pattern_meta[pname] = pdata
        except ValueError:
            continue

    return PortfolioRules(
        long_tree=long_tree,
        short_tree=short_tree,
        long_meta=raw["long"],
        short_meta=raw["short"],
        pattern_trees=pattern_trees,
        pattern_meta=pattern_meta,
        condition_map={
            "long": {c.letter: c for c in long_conds},
            "short": {c.letter: c for c in short_conds},
        },
    )
