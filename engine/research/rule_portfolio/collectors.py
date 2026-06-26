"""Collect all discovered execution rules and assign discovery status."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

from scout_auto_os.engine.research.execution_rule_discovery.generator import generate_execution_rules
from scout_auto_os.engine.research.rule_discovery.discovered_rule import DiscoveredRule
from scout_auto_os.engine.research.rule_portfolio.constants import BASELINE_RULE_EXPR, BASELINE_RULE_ID


@dataclass
class PortfolioRule:
    rule: DiscoveredRule | None
    rule_id: str
    rule_expr: str
    direction: str
    source: str
    status_tags: list[str] = field(default_factory=list)
    discovery_decision: str = ""
    blind_avg: float | None = None
    train_avg: float | None = None


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_paths(data_dir: Path, pkg_root: Path) -> dict[str, Path]:
    reports = pkg_root / "research_bundle" / "reports"
    disc = data_dir / "execution_rule_discovery"
    gen = data_dir / "execution_generalization"
    return {
        "top20": disc / "top20_execution_rules.csv",
        "top20_mirror": reports / "top20_execution_rules_v1.csv",
        "train_rank": disc / "execution_rules_train_rank.csv",
        "recommended": disc / "recommended_execution_rule.json",
        "recommended_mirror": reports / "recommended_execution_rule_v1.json",
        "generalization": gen / "generalization_decision.json",
        "generalization_mirror": reports / "execution_generalization_decision_v1.json",
    }


def _baseline_rule(direction: str = "long") -> PortfolioRule:
    return PortfolioRule(
        rule=None,
        rule_id=BASELINE_RULE_ID,
        rule_expr=BASELINE_RULE_EXPR,
        direction=direction,
        source="baseline",
        status_tags=["universal", "baseline"],
        discovery_decision="Keep current Execution Engine",
    )


def collect_all_rules(
    data_dir: Path,
    pkg_root: Path,
    long_train_groups: list[list[dict]],
    short_train_groups: list[list[dict]],
) -> list[PortfolioRule]:
    paths = _resolve_paths(data_dir, pkg_root)
    top20_rows = _read_csv(paths["top20"]) or _read_csv(paths["top20_mirror"])
    train_rows = _read_csv(paths["train_rank"])
    rec = _load_json(paths["recommended"]) or _load_json(paths["recommended_mirror"])
    gen = _load_json(paths["generalization"]) or _load_json(paths["generalization_mirror"])

    recommended_id = rec.get("best_rule_id") or (rec.get("recommended_rule") or {}).get("rule_id")
    gen_decision = gen.get("decision", "")
    top20_ids = {r["rule_id"] for r in top20_rows}
    train_by_id = {r["rule_id"]: r for r in train_rows}
    blind_by_expr = {r["rule_expr"]: r for r in top20_rows}

    mined: dict[str, DiscoveredRule] = {}
    for direction, tg in (("long", long_train_groups), ("short", short_train_groups)):
        for rule in generate_execution_rules(tg, direction):
            mined[rule.rule_expr] = rule

    ast_by_id: dict[str, dict] = {}
    rec_rule = rec.get("recommended_rule") or {}
    if rec_rule.get("rule_id"):
        ast_by_id[rec_rule["rule_id"]] = rec_rule

    portfolio: dict[str, PortfolioRule] = {}

    def _ensure(rule_id: str, rule_expr: str, direction: str, source: str) -> PortfolioRule:
        key = f"{direction}|{rule_expr}"
        if key in portfolio:
            return portfolio[key]
        dr = mined.get(rule_expr)
        if dr is None and rule_id in ast_by_id:
            from scout_auto_os.engine.research.execution_generalization.rule_loader import _build_expr

            ast = ast_by_id[rule_id]["ast"]
            dr = DiscoveredRule(
                rule_id=rule_id,
                rule_expr=rule_expr,
                root=_build_expr(ast),
                direction=direction,
            )
        pr = PortfolioRule(
            rule=dr,
            rule_id=rule_id,
            rule_expr=rule_expr,
            direction=direction,
            source=source,
        )
        portfolio[key] = pr
        return pr

    for row in train_rows:
        pr = _ensure(row["rule_id"], row["rule_expr"], row.get("direction", "long"), "train_rank")
        pr.train_avg = float(row.get("avg_return_2h", 0))

    for row in top20_rows:
        pr = _ensure(row["rule_id"], row["rule_expr"], row.get("direction", "long"), "blind_top20")
        pr.blind_avg = float(row.get("avg_return_2h", 0))

    for expr, rule in mined.items():
        if expr not in {p.rule_expr for p in portfolio.values() if p.direction == rule.direction}:
            _ensure(rule.rule_id, expr, rule.direction, "mined")

    for pr in list(portfolio.values()):
        tags: list[str] = []
        if pr.rule_id == recommended_id:
            tags.append("recommended")
            if gen_decision == "KEEP":
                tags.append("accepted")
            elif gen_decision == "REJECT":
                tags.append("rejected")
        if pr.rule_id in top20_ids:
            tags.append("candidate")
        if pr.source == "mined" and pr.rule_id not in top20_ids:
            tags.append("mined_pool")
        if pr.train_avg is not None and pr.train_avg < 0:
            tags.append("weak")
        if pr.blind_avg is not None and pr.blind_avg < 0:
            tags.append("weak")
        if "rank_obs_return_top5" in pr.rule_expr and "direction_confidence" in pr.rule_expr:
            tags.append("momentum_hybrid")
        if pr.rule_expr.count("rank_obs_return_top5") >= 2:
            tags.append("rank_only")
        pr.status_tags = sorted(set(tags) or ["discovered"])
        if pr.rule_id == recommended_id:
            pr.discovery_decision = rec.get("decision", "")

    rules = list(portfolio.values())
    rules.append(_baseline_rule("long"))
    rules.append(_baseline_rule("short"))
    return rules
