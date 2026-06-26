"""SearchFormula — AST predicate or linear weighted score."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from scout_auto_os.engine.research.rule_discovery.discovered_rule import RuleExpr

FormulaKind = Literal["ast", "linear", "baseline"]


@dataclass
class SearchFormula:
    formula_id: str
    formula_expr: str
    kind: FormulaKind = "ast"
    root: RuleExpr | None = None
    linear_terms: list[tuple[str, float]] = field(default_factory=list)
    pass_bonus: float = 50.0
    source: str = "generated"

    def to_dict(self) -> dict:
        return {
            "formula_id": self.formula_id,
            "formula_expr": self.formula_expr,
            "kind": self.kind,
            "pass_bonus": self.pass_bonus,
            "source": self.source,
            "linear_terms": self.linear_terms,
        }
