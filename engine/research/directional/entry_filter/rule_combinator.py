"""Generate all rule combinations and OR-tree partitions from V1 conditions."""

from __future__ import annotations

import itertools

from scout_auto_os.engine.research.directional.entry_filter.rule_tree import (
    Condition,
    RuleNode,
    and_node,
    or_node,
)


def _set_partitions(items: list[int]) -> list[list[list[int]]]:
    if not items:
        return []
    if len(items) == 1:
        return [[[items[0]]]]
    partitions: list[list[list[int]]] = []
    for partition in _set_partitions(items[:-1]):
        for i in range(len(partition)):
            new_part = [list(block) for block in partition]
            new_part[i] = new_part[i] + [items[-1]]
            partitions.append(new_part)
        partitions.append([list(block) for block in partition] + [[items[-1]]])
    return partitions


def generate_rule_trees(conditions: list[Condition]) -> list[tuple[str, RuleNode]]:
    """All AND subsets + OR-of-AND partition trees. Returns (rule_id, tree)."""
    n = len(conditions)
    indices = list(range(n))
    trees: list[tuple[str, RuleNode]] = []
    seen_desc: set[str] = set()

    def _add(rule_id: str, tree: RuleNode) -> None:
        desc = tree.describe()
        if desc in seen_desc:
            return
        seen_desc.add(desc)
        trees.append((rule_id, tree))

    # AND subsets (non-empty)
    for r in range(1, n + 1):
        for combo in itertools.combinations(indices, r):
            conds = [conditions[i] for i in combo]
            letters = "".join(conditions[i].letter for i in combo)
            _add(f"AND_{letters}", and_node(conds))

    # OR partitions (2+ groups)
    if n >= 2:
        for part in _set_partitions(indices):
            if len(part) < 2:
                continue
            groups = []
            part_letters: list[str] = []
            for block in part:
                block_conds = [conditions[i] for i in block]
                groups.append(and_node(block_conds))
                part_letters.append("".join(conditions[i].letter for i in sorted(block)))
            rule_id = "OR_" + "_".join(part_letters)
            _add(rule_id, or_node(groups))

    return trees
