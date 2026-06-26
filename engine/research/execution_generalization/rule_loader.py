"""Load frozen execution rule from JSON — no modification."""

from __future__ import annotations

import json
from pathlib import Path

from scout_auto_os.engine.research.rule_discovery.discovered_rule import (
    DiscoveredRule,
    Predicate,
    RuleExpr,
    and_expr,
    not_expr,
    or_expr,
    pred_node,
)


def _build_expr(node: dict) -> RuleExpr:
    t = node["type"]
    if t == "pred":
        p = Predicate(
            kind=node["kind"],
            feature_a=node["feature_a"],
            feature_b=node.get("feature_b"),
            operator=node.get("operator", "gte"),
            threshold=float(node.get("threshold", 0)),
            predicate_id=f"FROZEN_{node['feature_a']}",
        )
        return pred_node(p)
    if t == "and":
        return and_expr([_build_expr(c) for c in node["children"]])
    if t == "or":
        return or_expr([_build_expr(c) for c in node["children"]])
    return not_expr(_build_expr(node["children"][0]))


def load_frozen_execution_rule(path: Path) -> DiscoveredRule:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rec = raw.get("recommended_rule") or raw
    ast = rec["ast"]
    return DiscoveredRule(
        rule_id=rec["rule_id"],
        rule_expr=rec["rule_expr"],
        root=_build_expr(ast),
        direction=rec.get("direction", "long"),
    )


def resolve_rule_path(data_dir: Path, pkg_root: Path) -> Path:
    for p in (
        data_dir / "execution_rule_discovery" / "recommended_execution_rule.json",
        pkg_root / "research_bundle" / "reports" / "recommended_execution_rule_v1.json",
    ):
        if p.exists():
            return p
    raise FileNotFoundError("recommended_execution_rule.json not found")
