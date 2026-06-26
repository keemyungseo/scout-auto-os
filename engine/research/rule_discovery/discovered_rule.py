"""Extended rule AST — threshold, ratio, diff, window, rank, AND/OR/NOT."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

NodeType = Literal["pred", "and", "or", "not"]
PredKind = Literal["threshold", "ratio", "diff", "window_increase", "rank_gte"]


@dataclass
class Predicate:
    kind: PredKind
    feature_a: str
    feature_b: str | None = None
    operator: str = "gte"
    threshold: float = 0.0
    predicate_id: str = ""

    def value(self, features: dict, ctx: dict | None = None) -> float:
        ctx = ctx or {}
        if self.kind == "threshold":
            return float(features.get(self.feature_a, 0))
        if self.kind == "ratio":
            denom = float(features.get(self.feature_b or "", 0)) or 1e-9
            return float(features.get(self.feature_a, 0)) / denom
        if self.kind == "diff":
            return float(features.get(self.feature_a, 0)) - float(features.get(self.feature_b or "", 0))
        if self.kind == "window_increase":
            return float(features.get(self.feature_a, 0)) - float(features.get(self.feature_b or "", 0))
        if self.kind == "rank_gte":
            ranks = ctx.get("scan_ranks", {})
            return float(ranks.get(self.feature_a, {}).get(ctx.get("_symbol", ""), 0))
        return 0.0

    def evaluate(self, features: dict, ctx: dict | None = None) -> bool:
        ctx = dict(ctx or {})
        ctx["_symbol"] = ctx.get("_symbol", "")
        v = self.value(features, ctx)
        if self.kind == "window_increase":
            return v > 0
        if self.kind == "rank_gte":
            return v >= self.threshold
        if self.operator == "gte":
            return v >= self.threshold
        return v <= self.threshold

    def describe(self) -> str:
        if self.kind == "threshold":
            sym = ">=" if self.operator == "gte" else "<="
            return f"{self.feature_a} {sym} {self.threshold:.4g}"
        if self.kind == "ratio":
            sym = ">=" if self.operator == "gte" else "<="
            return f"{self.feature_a}/{self.feature_b} {sym} {self.threshold:.4g}"
        if self.kind == "diff":
            sym = ">=" if self.operator == "gte" else "<="
            return f"({self.feature_a} - {self.feature_b}) {sym} {self.threshold:.4g}"
        if self.kind == "window_increase":
            return f"{self.feature_a} > {self.feature_b}"
        if self.kind == "rank_gte":
            return f"rank({self.feature_a}) >= {self.threshold:.4g}"
        return self.predicate_id


@dataclass
class DiscoveredRule:
    rule_id: str
    rule_expr: str
    root: "RuleExpr"
    direction: str = "long"

    def evaluate(self, features: dict, ctx: dict | None = None) -> bool:
        return self.root.evaluate(features, ctx)

    def to_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "rule_expr": self.rule_expr,
            "direction": self.direction,
            "ast": self.root.to_dict(),
        }


@dataclass
class RuleExpr:
    node_type: NodeType
    predicate: Predicate | None = None
    children: list["RuleExpr"] = field(default_factory=list)

    def evaluate(self, features: dict, ctx: dict | None = None) -> bool:
        if self.node_type == "pred":
            assert self.predicate is not None
            return self.predicate.evaluate(features, ctx)
        if self.node_type == "and":
            return all(c.evaluate(features, ctx) for c in self.children)
        if self.node_type == "or":
            return any(c.evaluate(features, ctx) for c in self.children)
        assert self.node_type == "not"
        return not self.children[0].evaluate(features, ctx)

    def describe(self) -> str:
        if self.node_type == "pred":
            return self.predicate.describe() if self.predicate else "?"
        if self.node_type == "and":
            return "(" + " AND ".join(c.describe() for c in self.children) + ")"
        if self.node_type == "or":
            return "(" + " OR ".join(c.describe() for c in self.children) + ")"
        return f"NOT ({self.children[0].describe()})"

    def to_dict(self) -> dict:
        if self.node_type == "pred" and self.predicate:
            return {
                "type": "pred",
                "kind": self.predicate.kind,
                "feature_a": self.predicate.feature_a,
                "feature_b": self.predicate.feature_b,
                "operator": self.predicate.operator,
                "threshold": self.predicate.threshold,
            }
        return {
            "type": self.node_type,
            "children": [c.to_dict() for c in self.children],
        }


def pred_node(p: Predicate) -> RuleExpr:
    return RuleExpr(node_type="pred", predicate=p)


def and_expr(items: list[RuleExpr]) -> RuleExpr:
    return RuleExpr(node_type="and", children=items)


def or_expr(items: list[RuleExpr]) -> RuleExpr:
    return RuleExpr(node_type="or", children=items)


def not_expr(child: RuleExpr) -> RuleExpr:
    return RuleExpr(node_type="not", children=[child])
