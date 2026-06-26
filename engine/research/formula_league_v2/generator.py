"""Auto-generate thousands of SearchFormula candidates."""

from __future__ import annotations

import itertools
import statistics

from scout_auto_os.engine.research.formula_league_v2.constants import (
    DIFF_PAIRS,
    LINEAR_WEIGHT_GRID,
    MAX_FORMULAS,
    MIN_TRAIN_PASS,
    RANK_FEATURES,
    RATIO_PAIRS,
    TOP_ATOMS,
    WINDOW_PAIRS,
)
from scout_auto_os.engine.research.formula_league_v2.features import build_feature_pool
from scout_auto_os.engine.research.formula_league_v2.formula import SearchFormula
from scout_auto_os.engine.research.rule_discovery.discovered_rule import (
    Predicate,
    RuleExpr,
    and_expr,
    not_expr,
    or_expr,
    pred_node,
)


def _grid_thresholds(values: list[float], steps: int = 8) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if abs(hi - lo) < 1e-9:
        return [round(lo, 6)]
    return [round(lo + (hi - lo) * i / steps, 6) for i in range(steps + 1)]


def _score_atom(train: list[dict], fn) -> dict | None:
    winners = sum(1 for s in train if s.get("cohort") == "winner")
    if not winners:
        return None
    baseline = winners / len(train)
    passed = [s for s in train if fn(s)]
    if len(passed) < MIN_TRAIN_PASS:
        return None
    tp = sum(1 for s in passed if s.get("cohort") == "winner")
    precision = tp / len(passed)
    recall = tp / winners
    lift = precision / baseline if baseline else 0
    if lift < 1.02:
        return None
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return {
        "precision": precision,
        "lift": lift,
        "train_score": f1 * 0.4 + lift * 0.4 + recall * 0.2,
        "pass_count": len(passed),
    }


def _mine_atoms(train: list[dict], feature_pool: list[str]) -> list[tuple[Predicate, dict]]:
    atoms: list[tuple[Predicate, dict]] = []
    priority = feature_pool[:45]

    for feat in priority:
        vals = [float(s["features"].get(feat, 0)) for s in train]
        for op in ("gte", "lte"):
            for thr in _grid_thresholds(vals, 7):
                pred = Predicate(
                    kind="threshold",
                    feature_a=feat,
                    operator=op,
                    threshold=thr,
                    predicate_id=f"SF_{feat}_{op}_{thr}",
                )

                def _fn(s, p=pred):
                    return p.evaluate(s["features"], s.get("ctx"))

                sc = _score_atom(train, _fn)
                if sc:
                    atoms.append((pred, sc))

    for fa, fb in RATIO_PAIRS:
        vals = [
            float(s["features"].get(fa, 0)) / max(float(s["features"].get(fb, 0)), 1e-9)
            for s in train
        ]
        for thr in _grid_thresholds(vals, 5):
            pred = Predicate(
                kind="ratio", feature_a=fa, feature_b=fb, operator="gte",
                threshold=thr, predicate_id=f"SR_{fa}_{fb}_{thr}",
            )
            sc = _score_atom(train, lambda s, p=pred: p.evaluate(s["features"], s.get("ctx")))
            if sc:
                atoms.append((pred, sc))

    for fa, fb in DIFF_PAIRS:
        for op in ("gte", "lte"):
            vals = [float(s["features"].get(fa, 0)) - float(s["features"].get(fb, 0)) for s in train]
            for thr in _grid_thresholds(vals, 5):
                pred = Predicate(
                    kind="diff", feature_a=fa, feature_b=fb, operator=op,
                    threshold=thr, predicate_id=f"SD_{fa}_{fb}_{op}_{thr}",
                )
                sc = _score_atom(train, lambda s, p=pred: p.evaluate(s["features"], s.get("ctx")))
                if sc:
                    atoms.append((pred, sc))

    for fa, fb in WINDOW_PAIRS:
        pred = Predicate(
            kind="window_increase", feature_a=fa, feature_b=fb,
            predicate_id=f"SW_{fa}_{fb}",
        )
        sc = _score_atom(train, lambda s, p=pred: p.evaluate(s["features"], s.get("ctx")))
        if sc:
            atoms.append((pred, sc))

    for feat in RANK_FEATURES:
        for thr in (0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
            pred = Predicate(
                kind="rank_gte", feature_a=feat, threshold=thr,
                predicate_id=f"RNK_{feat}_{thr}",
            )
            sc = _score_atom(train, lambda s, p=pred: p.evaluate(s["features"], s.get("ctx")))
            if sc:
                atoms.append((pred, sc))

    atoms.sort(key=lambda x: -x[1]["train_score"])
    return atoms[:TOP_ATOMS]


def _add_ast(
    formulas: list[SearchFormula],
    seen: set[str],
    rule_id: str,
    root: RuleExpr,
    source: str,
) -> None:
    expr = root.describe()
    if expr in seen:
        return
    seen.add(expr)
    formulas.append(SearchFormula(
        formula_id=rule_id,
        formula_expr=expr,
        kind="ast",
        root=root,
        source=source,
    ))


def _combine_atoms(atoms: list[tuple[Predicate, dict]]) -> list[SearchFormula]:
    formulas: list[SearchFormula] = []
    seen: set[str] = set()
    top = atoms[:TOP_ATOMS]

    for i, (pred, _) in enumerate(top):
        _add_ast(formulas, seen, f"SF_ATOM_{i}", pred_node(pred), "atom")

    for (p1, _), (p2, _) in itertools.combinations(top[:35], 2):
        _add_ast(formulas, seen, f"SF_AND_{p1.predicate_id}_{p2.predicate_id}",
                 and_expr([pred_node(p1), pred_node(p2)]), "and_pair")

    for (p1, _), (p2, _) in itertools.combinations(top[:30], 2):
        _add_ast(formulas, seen, f"SF_OR_{p1.predicate_id}_{p2.predicate_id}",
                 or_expr([pred_node(p1), pred_node(p2)]), "or_pair")

    for p1, _ in top[:25]:
        _add_ast(formulas, seen, f"SF_NOT_{p1.predicate_id}", not_expr(pred_node(p1)), "not")

    for (p1, _), (p2, _), (p3, _) in itertools.combinations(top[:18], 3):
        _add_ast(
            formulas, seen, f"SF_AND3_{p1.predicate_id}",
            and_expr([pred_node(p1), pred_node(p2), pred_node(p3)]),
            "and_triple",
        )

    for (p1, _), (p2, _) in itertools.combinations(top[:12], 2):
        _add_ast(
            formulas, seen, f"SF_ORAND_{p1.predicate_id}",
            or_expr([pred_node(p1), and_expr([pred_node(p2), pred_node(p1)])]),
            "or_and",
        )

    return formulas


def _linear_formulas(train: list[dict], feature_pool: list[str]) -> list[SearchFormula]:
    formulas: list[SearchFormula] = []
    seen: set[str] = set()
    top_feats = feature_pool[:25]

    for feat in top_feats:
        vals = [float(s["features"].get(feat, 0)) for s in train]
        pos = [v for s, v in zip(train, vals) if s.get("cohort") == "winner"]
        neg = [v for s, v in zip(train, vals) if s.get("cohort") != "winner"]
        if not pos or not neg:
            continue
        direction = 1.0 if statistics.mean(pos) >= statistics.mean(neg) else -1.0
        for w in LINEAR_WEIGHT_GRID:
            weight = w * direction
            expr = f"linear({feat} * {weight:.2f})"
            if expr in seen:
                continue
            seen.add(expr)
            formulas.append(SearchFormula(
                formula_id=f"SF_LIN_{feat}_{weight:.2f}".replace(".", "_").replace("-", "m"),
                formula_expr=expr,
                kind="linear",
                linear_terms=[(feat, weight)],
                source="linear_single",
            ))

    for (f1, f2) in itertools.combinations(top_feats[:15], 2):
        for w1 in (1.0, 2.0):
            for w2 in (0.5, 1.0, 1.5):
                expr = f"linear({f1}*{w1}+{f2}*{w2})"
                if expr in seen:
                    continue
                seen.add(expr)
                formulas.append(SearchFormula(
                    formula_id=f"SF_LIN2_{f1}_{f2}_{w1}_{w2}",
                    formula_expr=expr,
                    kind="linear",
                    linear_terms=[(f1, w1), (f2, w2)],
                    source="linear_pair",
                ))

    return formulas


def baseline_formula() -> SearchFormula:
    return SearchFormula(
        formula_id="A6_frozen",
        formula_expr="A6 = base + ig_a2*range_pct_rank + ig_a5*expansion_bonus",
        kind="baseline",
        source="frozen_baseline",
    )


def generate_search_formulas(train: list[dict]) -> list[SearchFormula]:
    if not train:
        return [baseline_formula()]

    sample = train[0]["features"]
    pool = build_feature_pool(sample)
    rank_feats = list(RANK_FEATURES) + [f for f in pool if f.startswith("derived_")][:10]
    for s in train:
        if "ctx" not in s:
            s["ctx"] = {"scan_ranks": {}}

    atoms = _mine_atoms(train, pool)
    formulas = _combine_atoms(atoms)
    formulas.extend(_linear_formulas(train, pool))
    formulas.append(baseline_formula())

    seen_ids: set[str] = set()
    unique: list[SearchFormula] = []
    for f in formulas:
        if f.formula_id in seen_ids:
            continue
        seen_ids.add(f.formula_id)
        unique.append(f)

    unique.sort(key=lambda x: (x.kind != "baseline", x.formula_id))
    return unique[:MAX_FORMULAS]
