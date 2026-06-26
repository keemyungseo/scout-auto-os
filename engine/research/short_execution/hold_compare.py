"""Research 4 — Early vs late vs dynamic exit comparison."""

from __future__ import annotations

from scout_auto_os.engine.research.short_execution.exit_simulator import (
    blind_exit_comparison,
    exit_rule_catalog,
    simulate_picks_with_rule,
)


def hold_strategy_compare(picks: list[dict], best_dynamic_rule: dict) -> list[dict]:
    rules = {r["rule_id"]: r for r in exit_rule_catalog()}
    candidates = [
        ("hold_2h_constitution", rules["hold_2h"]),
        ("hold_4h", rules["hold_4h"]),
        ("dynamic_best", best_dynamic_rule),
    ]
    rows: list[dict] = []
    all_blind = blind_exit_comparison(picks, [r for _, r in candidates])
    by_id = {r["rule_id"]: r for r in all_blind}
    for label, rule in candidates:
        m = by_id.get(rule["rule_id"], {})
        rows.append({
            "strategy": label,
            "rule_id": rule["rule_id"],
            **{k: v for k, v in m.items() if k not in ("rule_id", "category", "exit_rank")},
        })
    return rows


def pick_best_dynamic_rule(exit_ranking: list[dict], exclude: set[str] | None = None) -> dict:
    exclude = exclude or {"hold_2h", "hold_4h", "hold_1h"}
    rules = {r["rule_id"]: r for r in exit_rule_catalog()}
    for row in exit_ranking:
        rid = row["rule_id"]
        if rid in exclude:
            continue
        if rid in rules:
            return rules[rid]
    return rules["roi_trail5"]
