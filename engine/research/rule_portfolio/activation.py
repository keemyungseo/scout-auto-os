"""Rule activation matrix — regime x volatility activation rates."""

from __future__ import annotations

from collections import defaultdict

from scout_auto_os.engine.research.execution_rule_discovery.constants import TOP2_SIZE
from scout_auto_os.engine.research.execution_rule_discovery.generator import pick_top2_by_rule
from scout_auto_os.engine.research.rule_portfolio.collectors import PortfolioRule


def build_activation_matrix(
    portfolio_rules: list[PortfolioRule],
    groups: list[list[dict]],
) -> list[dict]:
    regimes = sorted({g[0].get("regime", "unknown") for g in groups})
    vols = sorted({g[0].get("volatility_band", "unknown") for g in groups})
    columns = regimes + [f"vol_{v}" for v in vols]

    rows: list[dict] = []
    for pr in portfolio_rules:
        if pr.rule is None:
            continue
        dir_groups = [g for g in groups if g[0].get("direction") == pr.direction and len(g) >= TOP2_SIZE]
        cell_pass: dict[str, list[bool]] = defaultdict(list)

        for g in dir_groups:
            regime = g[0].get("regime", "unknown")
            vol = g[0].get("volatility_band", "unknown")
            picks = pick_top2_by_rule(g, pr.rule)
            for p in picks:
                passed = pr.rule.evaluate(p["features"], p.get("ctx"))
                cell_pass[regime].append(passed)
                cell_pass[f"vol_{vol}"].append(passed)

        row: dict = {
            "rule_id": pr.rule_id,
            "rule_expr": pr.rule_expr[:120],
            "direction": pr.direction,
            "cluster_hint": pr.status_tags[0] if pr.status_tags else "",
        }
        for col in columns:
            vals = cell_pass.get(col, [])
            row[f"act_{col}_pct"] = round(sum(vals) / len(vals) * 100, 2) if vals else 0.0
            row[f"n_{col}"] = len(vals)
        rows.append(row)
    return rows
