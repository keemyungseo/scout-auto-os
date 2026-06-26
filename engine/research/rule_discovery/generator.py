"""Generate candidate rule predicates and combinations."""

from __future__ import annotations

import itertools
import statistics

from scout_auto_os.engine.research.directional.dna.collector import numeric_feature_keys
from scout_auto_os.engine.research.rule_discovery.constants import (
    DIFF_PAIRS,
    MAX_CANDIDATES,
    MIN_TRAIN_PASS,
    RATIO_PAIRS,
    TOP_ATOMS,
    WINDOW_PAIRS,
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


def _grid_thresholds(values: list[float], steps: int = 12) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if abs(hi - lo) < 1e-9:
        return [round(lo, 6)]
    return [round(lo + (hi - lo) * i / steps, 6) for i in range(steps + 1)]


def _score_predicate(train: list[dict], fn) -> dict | None:
    winners = sum(1 for s in train if s.get("cohort") == "winner")
    if not winners:
        return None
    baseline = winners / len(train)
    pass_s = [s for s in train if fn(s)]
    if len(pass_s) < MIN_TRAIN_PASS:
        return None
    tp = sum(1 for s in pass_s if s.get("cohort") == "winner")
    precision = tp / len(pass_s)
    recall = tp / winners
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    lift = precision / baseline if baseline else 0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "lift": lift,
        "pass_count": len(pass_s),
        "train_score": f1 * 0.5 + lift * 0.3 + recall * 0.2,
    }


def _mine_threshold_atoms(
    train: list[dict],
    features: list[str],
    direction: str,
) -> list[tuple[Predicate, dict]]:
    atoms: list[tuple[Predicate, dict]] = []
    for feat in features:
        vals = [float(s["features"].get(feat, 0)) for s in train]
        for op in ("gte", "lte"):
            for thr in _grid_thresholds(vals, 10):
                pred = Predicate(
                    kind="threshold",
                    feature_a=feat,
                    operator=op,
                    threshold=thr,
                    predicate_id=f"TH_{feat}_{op}",
                )

                def _mk(p: Predicate):
                    return lambda s, p=p: p.evaluate(s["features"], s.get("ctx"))

                sc = _score_predicate(train, _mk(pred))
                if sc and sc["lift"] >= 1.05:
                    atoms.append((pred, sc))
    atoms.sort(key=lambda x: -x[1]["train_score"])
    return atoms[:TOP_ATOMS * 2]


def _mine_derived_atoms(train: list[dict], direction: str) -> list[tuple[Predicate, dict]]:
    atoms: list[tuple[Predicate, dict]] = []

    for fa, fb in WINDOW_PAIRS:
        pred = Predicate(kind="window_increase", feature_a=fa, feature_b=fb, predicate_id=f"WIN_{fa}")
        sc = _score_predicate(train, lambda s, p=pred: p.evaluate(s["features"], s.get("ctx")))
        if sc:
            atoms.append((pred, sc))

    for fa, fb in RATIO_PAIRS:
        vals = []
        for s in train:
            denom = float(s["features"].get(fb, 0)) or 1e-9
            vals.append(float(s["features"].get(fa, 0)) / denom)
        for thr in _grid_thresholds(vals, 8):
            pred = Predicate(
                kind="ratio", feature_a=fa, feature_b=fb, operator="gte", threshold=thr,
                predicate_id=f"RAT_{fa}_{fb}",
            )
            sc = _score_predicate(train, lambda s, p=pred: p.evaluate(s["features"], s.get("ctx")))
            if sc and sc["lift"] >= 1.05:
                atoms.append((pred, sc))

    for fa, fb in DIFF_PAIRS:
        vals = [float(s["features"].get(fa, 0)) - float(s["features"].get(fb, 0)) for s in train]
        for op in ("gte", "lte"):
            for thr in _grid_thresholds(vals, 8):
                pred = Predicate(
                    kind="diff", feature_a=fa, feature_b=fb, operator=op, threshold=thr,
                    predicate_id=f"DIF_{fa}_{fb}",
                )
                sc = _score_predicate(train, lambda s, p=pred: p.evaluate(s["features"], s.get("ctx")))
                if sc and sc["lift"] >= 1.05:
                    atoms.append((pred, sc))

    for feat in ("1h_current_body_pct", "1h_current_range_pct", "1h_current_return_pct"):
        for thr in (0.5, 0.6, 0.7, 0.8):
            pred = Predicate(kind="rank_gte", feature_a=feat, threshold=thr, predicate_id=f"RNK_{feat}")
            sc = _score_predicate(train, lambda s, p=pred: p.evaluate(s["features"], s.get("ctx")))
            if sc:
                atoms.append((pred, sc))

    atoms.sort(key=lambda x: -x[1]["train_score"])
    return atoms[:TOP_ATOMS]


def _combine_atoms(
    atoms: list[tuple[Predicate, dict]],
    direction: str,
) -> list[DiscoveredRule]:
    rules: list[DiscoveredRule] = []
    seen: set[str] = set()
    top = atoms[:TOP_ATOMS]

    def _add(rule_id: str, root: RuleExpr) -> None:
        expr = root.describe()
        if expr in seen:
            return
        seen.add(expr)
        rules.append(DiscoveredRule(rule_id=rule_id, rule_expr=expr, root=root, direction=direction))

    for i, (pred, _) in enumerate(top):
        _add(f"ATOM_{i}", pred_node(pred))

    for (p1, _), (p2, _) in itertools.combinations(top[:20], 2):
        _add(f"AND_{p1.predicate_id}_{p2.predicate_id}", and_expr([pred_node(p1), pred_node(p2)]))

    for (p1, _), (p2, _) in itertools.combinations(top[:15], 2):
        _add(f"OR_{p1.predicate_id}_{p2.predicate_id}", or_expr([pred_node(p1), pred_node(p2)]))

    for p1, _ in top[:12]:
        _add(f"NOT_{p1.predicate_id}", not_expr(pred_node(p1)))

    for (p1, _), (p2, _), (p3, _) in itertools.combinations(top[:10], 3):
        _add(
            f"AND3_{p1.predicate_id}",
            and_expr([pred_node(p1), pred_node(p2), pred_node(p3)]),
        )

    return rules[:MAX_CANDIDATES]


def build_scan_rank_context(signals: list[dict], features: list[str]) -> None:
    """Attach ctx with per-scan feature rank percentiles (scan-time only)."""
    by_scan: dict[str, list[dict]] = {}
    for s in signals:
        by_scan.setdefault(s["scan_time_kst"], []).append(s)

    for scan_rows in by_scan.values():
        n = len(scan_rows)
        for feat in features:
            ranked = sorted(scan_rows, key=lambda r: float(r["features"].get(feat, 0)))
            for i, row in enumerate(ranked):
                pct = i / max(n - 1, 1)
                ctx = row.setdefault("ctx", {"scan_ranks": {}})
                ctx["_symbol"] = row["symbol"]
                ctx["scan_ranks"].setdefault(feat, {})[row["symbol"]] = pct


def generate_candidate_rules(
    train: list[dict],
    direction: str,
    dna_features: list[str] | None = None,
) -> list[DiscoveredRule]:
    if not train:
        return []
    all_feats = numeric_feature_keys(train[0]["features"])
    dna_set = set(dna_features or [])
    feat_pool = [f for f in all_feats if f in dna_set] or list(all_feats)
    priority = [f for f in feat_pool if any(k in f for k in ("body", "range", "return", "volume", "momentum"))]
    if len(priority) < 15:
        priority = feat_pool[:30]
    else:
        priority = priority[:30]

    rank_feats = ["1h_current_body_pct", "1h_current_range_pct", "1h_current_return_pct"]
    build_scan_rank_context(train, rank_feats)

    threshold_atoms = _mine_threshold_atoms(train, priority, direction)
    derived_atoms = _mine_derived_atoms(train, direction)
    merged: dict[str, tuple[Predicate, dict]] = {}
    for pred, sc in threshold_atoms + derived_atoms:
        key = pred.describe()
        if key not in merged or sc["train_score"] > merged[key][1]["train_score"]:
            merged[key] = (pred, sc)
    atoms = sorted(merged.values(), key=lambda x: -x[1]["train_score"])
    return _combine_atoms(atoms, direction)
