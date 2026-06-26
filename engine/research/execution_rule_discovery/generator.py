"""Generate and evaluate execution rule candidates."""

from __future__ import annotations

import itertools
import statistics

from scout_auto_os.engine.portfolio.diversification import diversify_select
from scout_auto_os.engine.research.execution_rule_discovery.constants import (
    EXEC_DIFF_PAIRS,
    EXEC_NUMERIC_FEATURES,
    EXEC_RATIO_PAIRS,
    MAX_RULES,
    MIN_TRAIN_GROUPS,
    MIN_TRAIN_PASS,
    TOP2_SIZE,
    TOP_ATOMS,
)
from scout_auto_os.engine.research.rule_discovery.discovered_rule import (
    DiscoveredRule,
    Predicate,
    RuleExpr,
    and_expr,
    not_expr,
    or_expr,
    pred_node,
)


def _grid_thresholds(values: list[float], steps: int = 10) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if abs(hi - lo) < 1e-9:
        return [round(lo, 6)]
    return [round(lo + (hi - lo) * i / steps, 6) for i in range(steps + 1)]


def pick_top2_by_rule(group: list[dict], rule: DiscoveredRule) -> list[dict]:
    ranked = sorted(
        group,
        key=lambda r: (
            rule.evaluate(r["features"], r.get("ctx")),
            float(r["features"].get("entry_score", 0)),
        ),
        reverse=True,
    )
    pool = [{**r, "entry_score": r["features"]["entry_score"]} for r in ranked]
    return diversify_select(pool, TOP2_SIZE)


def avg_top2_return(groups: list[list[dict]], rule: DiscoveredRule | None, baseline_fn=None) -> dict:
    rets: list[float] = []
    pass_count = 0
    n_groups = 0
    for g in groups:
        if len(g) < TOP2_SIZE:
            continue
        n_groups += 1
        if rule is not None:
            top2 = pick_top2_by_rule(g, rule)
        else:
            top2 = baseline_fn(g) if baseline_fn else g[:TOP2_SIZE]
        for r in top2:
            rets.append(float(r["return_2h"]))
            if rule and rule.evaluate(r["features"], r.get("ctx")):
                pass_count += 1
    n = len(rets)
    wins = sum(1 for r in rets if r >= 3.0)
    days = len({g[0]["scan_time_kst"][:10] for g in groups if g}) or 1
    return {
        "avg_return_2h": round(statistics.mean(rets), 4) if rets else 0.0,
        "trade_count": n,
        "win_rate_pct": round(wins / n * 100, 2) if n else 0.0,
        "rule_pass_count": pass_count,
        "scan_groups": n_groups,
        "pass_per_day": round(pass_count / days, 4),
        "precision_proxy_pct": round(pass_count / n * 100, 2) if n else 0.0,
    }


def _mine_atoms(train_flat: list[dict], features: list[str]) -> list[tuple[Predicate, float]]:
    atoms: list[tuple[Predicate, float]] = []
    for feat in features:
        vals = [float(s["features"].get(feat, 0)) for s in train_flat]
        for op in ("gte", "lte"):
            for thr in _grid_thresholds(vals, 8):
                pred = Predicate(
                    kind="threshold",
                    feature_a=feat,
                    operator=op,
                    threshold=thr,
                    predicate_id=f"E_{feat}_{op}_{thr}",
                )

                def _passes(s: dict, p: Predicate = pred) -> bool:
                    return p.evaluate(s["features"], s.get("ctx"))

                passed = [s for s in train_flat if _passes(s)]
                if len(passed) < MIN_TRAIN_PASS:
                    continue
                avg_r = statistics.mean(float(s["return_2h"]) for s in passed)
                atoms.append((pred, avg_r))
    for fa, fb in EXEC_RATIO_PAIRS:
        for thr in _grid_thresholds([
            float(s["features"].get(fa, 0)) / max(float(s["features"].get(fb, 0)), 1e-9) for s in train_flat
        ], 6):
            pred = Predicate(kind="ratio", feature_a=fa, feature_b=fb, operator="gte", threshold=thr, predicate_id=f"ER_{fa}_{fb}_{thr}")
            passed = [s for s in train_flat if pred.evaluate(s["features"], s.get("ctx"))]
            if len(passed) >= MIN_TRAIN_PASS:
                atoms.append((pred, statistics.mean(float(s["return_2h"]) for s in passed)))
    for fa, fb in EXEC_DIFF_PAIRS:
        for op in ("gte", "lte"):
            for thr in _grid_thresholds([
                float(s["features"].get(fa, 0)) - float(s["features"].get(fb, 0)) for s in train_flat
            ], 6):
                pred = Predicate(kind="diff", feature_a=fa, feature_b=fb, operator=op, threshold=thr, predicate_id=f"ED_{fa}_{fb}_{op}_{thr}")
                passed = [s for s in train_flat if pred.evaluate(s["features"], s.get("ctx"))]
                if len(passed) >= MIN_TRAIN_PASS:
                    atoms.append((pred, statistics.mean(float(s["return_2h"]) for s in passed)))
    for feat in ("obs_return_pct", "entry_score", "execution_score"):
        for thr in (0.4, 0.6, 0.8):
            pred = Predicate(kind="rank_gte", feature_a=feat, threshold=thr, predicate_id=f"RNK_{feat}_{thr}")
            passed = [s for s in train_flat if pred.evaluate(s["features"], s.get("ctx"))]
            if len(passed) >= MIN_TRAIN_PASS:
                atoms.append((pred, statistics.mean(float(s["return_2h"]) for s in passed)))
    atoms.sort(key=lambda x: -x[1])
    return atoms[:TOP_ATOMS * 2]


def _groups_to_flat(groups: list[list[dict]]) -> list[dict]:
    return [r for g in groups for r in g]


def generate_execution_rules(train_groups: list[list[dict]], direction: str) -> list[DiscoveredRule]:
    if len(train_groups) < MIN_TRAIN_GROUPS:
        return []
    train_flat = _groups_to_flat(train_groups)
    features = list(EXEC_NUMERIC_FEATURES)
    atoms = _mine_atoms(train_flat, features)
    top = atoms[:TOP_ATOMS]
    rules: list[DiscoveredRule] = []
    seen: set[str] = set()

    def _add(rid: str, root: RuleExpr) -> None:
        expr = root.describe()
        if expr in seen:
            return
        seen.add(expr)
        rules.append(DiscoveredRule(rule_id=rid, rule_expr=expr, root=root, direction=direction))

    for i, (pred, _) in enumerate(top):
        _add(f"EX_{i}", pred_node(pred))
    for (p1, _), (p2, _) in itertools.combinations(top[:18], 2):
        _add(f"EX_AND_{p1.predicate_id}_{p2.predicate_id}", and_expr([pred_node(p1), pred_node(p2)]))
    for (p1, _), (p2, _) in itertools.combinations(top[:14], 2):
        _add(f"EX_OR_{p1.predicate_id}_{p2.predicate_id}", or_expr([pred_node(p1), pred_node(p2)]))
    for p1, _ in top[:10]:
        _add(f"EX_NOT_{p1.predicate_id}", not_expr(pred_node(p1)))
    return rules[:MAX_RULES]


def rank_rules_on_train(
    train_groups: list[list[dict]],
    rules: list[DiscoveredRule],
) -> list[dict]:
    rows: list[dict] = []
    for rule in rules:
        m = avg_top2_return(train_groups, rule)
        rows.append({
            "rule_id": rule.rule_id,
            "rule_expr": rule.rule_expr,
            "direction": rule.direction,
            **m,
        })
    rows.sort(key=lambda x: (-float(x["avg_return_2h"]), -int(x["trade_count"])))
    return rows
