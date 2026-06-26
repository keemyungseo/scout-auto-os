"""Rule tree AST — AND/OR over fixed V1 threshold conditions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

NodeType = Literal["cond", "and", "or"]


@dataclass
class Condition:
    letter: str
    feature: str
    operator: str
    threshold: float

    def evaluate(self, features: dict) -> bool:
        v = float(features.get(self.feature, 0))
        if self.operator == "gte":
            return v >= self.threshold
        return v <= self.threshold

    def describe(self) -> str:
        sym = ">=" if self.operator == "gte" else "<="
        return f"{self.letter}({self.feature} {sym} {self.threshold})"


@dataclass
class RuleNode:
    node_type: NodeType
    condition: Condition | None = None
    children: list[RuleNode] = field(default_factory=list)

    def evaluate(self, features: dict) -> bool:
        if self.node_type == "cond":
            assert self.condition is not None
            return self.condition.evaluate(features)
        if self.node_type == "and":
            return all(c.evaluate(features) for c in self.children)
        return any(c.evaluate(features) for c in self.children)

    def describe(self) -> str:
        if self.node_type == "cond":
            return self.condition.letter if self.condition else "?"
        if self.node_type == "and":
            parts = [c.describe() for c in self.children]
            if len(parts) == 1:
                return parts[0]
            return "(" + " AND ".join(parts) + ")"
        parts = [c.describe() for c in self.children]
        return "(" + " OR ".join(parts) + ")"


def and_node(conditions: list[Condition]) -> RuleNode:
    return RuleNode(
        node_type="and",
        children=[RuleNode(node_type="cond", condition=c) for c in conditions],
    )


def or_node(groups: list[RuleNode]) -> RuleNode:
    return RuleNode(node_type="or", children=groups)


def conditions_from_v1(v1_conditions: list[dict]) -> list[Condition]:
    letters = "ABCDEFGH"
    out: list[Condition] = []
    for i, row in enumerate(v1_conditions):
        out.append(Condition(
            letter=letters[i],
            feature=row["feature"],
            operator=row["operator"],
            threshold=float(row["threshold"]),
        ))
    return out
