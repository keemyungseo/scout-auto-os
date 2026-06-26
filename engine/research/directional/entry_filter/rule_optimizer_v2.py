"""Entry Rule Optimizer V2 — combination + OR-tree search for LIVE rules."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scout_auto_os.engine.research.directional.entry_filter.pattern_labels import (
    LONG_LIVE_PATTERNS,
    SHORT_LIVE_PATTERNS,
    attach_live_pattern,
)
from scout_auto_os.engine.research.directional.entry_filter.rule_combinator import generate_rule_trees
from scout_auto_os.engine.research.directional.entry_filter.rule_evaluator import (
    evaluate_rule_tree,
    live_selection_score,
    select_best_live_rule,
)
from scout_auto_os.engine.research.directional.entry_filter.rule_tree import (
    and_node,
    conditions_from_v1,
)
from scout_auto_os.engine.research.directional.entry_filter.rule_v2_report import build_live_entry_rule_v2_report
from scout_auto_os.engine.research.directional.entry_filter.signals_loader import label_signals, resolve_signals
from scout_auto_os.engine.research.safe import research_safe

KST = timezone(timedelta(hours=9))
MIN_PATTERN_SIGNALS = 25


def _load_v1_rules(data_dir: Path, pkg_root: Path) -> dict:
    for path in (
        data_dir / "zero_base" / "entry_filter_rules_v1.json",
        pkg_root / "research_bundle" / "reports" / "entry_filter_rules_v1.json",
    ):
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    raise FileNotFoundError("entry_filter_rules_v1.json not found")


def _run_search(
    signals: list[dict],
    conditions: list,
    scope: str,
) -> list[dict]:
    trees = generate_rule_trees(conditions)
    rows: list[dict] = []
    for rule_id, tree in trees:
        row = evaluate_rule_tree(signals, tree, rule_id, scope=scope)
        row["direction"] = signals[0]["direction"] if signals else scope
        rows.append(row)
    return rows


class EntryRuleOptimizerV2:
    def __init__(
        self,
        data_dir: Path,
        pkg_root: Path,
        candidates_path: Path,
        forward_path: Path,
    ) -> None:
        self.data_dir = data_dir
        self.pkg_root = pkg_root
        self.out_dir = data_dir / "zero_base"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.candidates_path = candidates_path
        self.forward_path = forward_path

    @research_safe("entry_rule_optimizer_v2")
    def run(self, target_pass_per_day: float = 3.0) -> dict:
        print("[RULE OPTIMIZER V2] started")
        v1 = _load_v1_rules(self.data_dir, self.pkg_root)
        long_raw, short_raw, _ = resolve_signals(
            self.data_dir, self.candidates_path, self.forward_path,
        )
        long_signals = attach_live_pattern(label_signals(long_raw)[0])
        short_signals = attach_live_pattern(label_signals(short_raw)[0])

        long_conds = conditions_from_v1(v1["long"]["conditions"])
        short_conds = conditions_from_v1(v1["short"]["conditions"])

        combo_long = _run_search(long_signals, long_conds, "long_all")
        combo_short = _run_search(short_signals, short_conds, "short_all")

        v1_long_row = next((r for r in combo_long if r["rule_id"] == "AND_ABCD"), combo_long[-1])
        v1_short_row = next((r for r in combo_short if r["rule_id"] == "AND_ABCD"), combo_short[-1])

        for r in combo_long + combo_short:
            base = v1_long_row if r["direction"] == "long" else v1_short_row
            r["live_score"] = round(
                live_selection_score(r, float(base["precision"]), target_pass_per_day), 4,
            )
            r["v1_precision_baseline"] = base["precision"]

        best_long = select_best_live_rule(combo_long, float(v1_long_row["precision"]))
        best_short = select_best_live_rule(combo_short, float(v1_short_row["precision"]))

        pattern_rows: list[dict] = []
        for pattern in LONG_LIVE_PATTERNS:
            subset = [s for s in long_signals if s.get("live_pattern") == pattern]
            if len(subset) < MIN_PATTERN_SIGNALS:
                continue
            for row in _run_search(subset, long_conds, f"long_{pattern}"):
                row["direction"] = "long"
                row["live_pattern"] = pattern
                row["live_score"] = round(
                    live_selection_score(row, float(v1_long_row["precision"]), target_pass_per_day), 4,
                )
                pattern_rows.append(row)

        for pattern in SHORT_LIVE_PATTERNS:
            subset = [s for s in short_signals if s.get("live_pattern") == pattern]
            if len(subset) < MIN_PATTERN_SIGNALS:
                continue
            for row in _run_search(subset, short_conds, f"short_{pattern}"):
                row["direction"] = "short"
                row["live_pattern"] = pattern
                row["live_score"] = round(
                    live_selection_score(row, float(v1_short_row["precision"]), target_pass_per_day), 4,
                )
                pattern_rows.append(row)

        pattern_best: list[dict] = []
        for pattern in LONG_LIVE_PATTERNS + SHORT_LIVE_PATTERNS:
            cand = [r for r in pattern_rows if r.get("live_pattern") == pattern]
            if not cand:
                continue
            direction = "long" if pattern in LONG_LIVE_PATTERNS else "short"
            base = v1_long_row if direction == "long" else v1_short_row
            best = select_best_live_rule(cand, float(base["precision"]), min_pass_count=3)
            if best:
                best["live_pattern"] = pattern
                pattern_best.append(best)

        live_score_rows = sorted(
            combo_long + combo_short + pattern_rows,
            key=lambda x: x.get("live_score", 0),
            reverse=True,
        )

        best_live_rows = []
        if best_long:
            best_live_rows.append({**best_long, "tier": "direction_champion"})
        if best_short:
            best_live_rows.append({**best_short, "tier": "direction_champion"})
        for pb in pattern_best:
            best_live_rows.append({**pb, "tier": "pattern"})

        trees_count = len(generate_rule_trees(long_conds))
        meta = {
            "long_signals": len(long_signals),
            "short_signals": len(short_signals),
            "trees_per_direction": trees_count,
            "target_pass_per_day": target_pass_per_day,
            "v1_long_precision": v1_long_row["precision"],
            "v1_short_precision": v1_short_row["precision"],
            "generated_at": datetime.now(KST).isoformat(),
        }

        from season2_p37_scout_decision_hierarchy import write_csv

        write_csv(self.out_dir / "rule_combination_report.csv", combo_long + combo_short)
        write_csv(self.out_dir / "pattern_rule_report.csv", pattern_rows)
        write_csv(self.out_dir / "best_live_rule.csv", best_live_rows)
        write_csv(self.out_dir / "live_rule_score.csv", live_score_rows)

        rules_v2 = {
            "version": "v2",
            "long": _serialize_best(best_long, long_conds),
            "short": _serialize_best(best_short, short_conds),
            "patterns": {pb["live_pattern"]: pb for pb in pattern_best},
        }
        (self.out_dir / "entry_filter_rules_v2.json").write_text(
            json.dumps(rules_v2, indent=2, ensure_ascii=False), encoding="utf-8",
        )

        report = build_live_entry_rule_v2_report(
            best_long or v1_long_row,
            best_short or v1_short_row,
            v1_long_row,
            v1_short_row,
            meta,
        )
        (self.out_dir / "live_entry_rule_v2.md").write_text(report, encoding="utf-8")
        (self.out_dir / "entry_rule_v2_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        reports_dir = self.pkg_root / "research_bundle" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        for src, dst in {
            "rule_combination_report.csv": "rule_combination_report_v2.csv",
            "pattern_rule_report.csv": "pattern_rule_report_v2.csv",
            "best_live_rule.csv": "best_live_rule_v2.csv",
            "live_rule_score.csv": "live_rule_score_v2.csv",
            "live_entry_rule_v2.md": "live_entry_rule_v2.md",
            "entry_filter_rules_v2.json": "entry_filter_rules_v2.json",
        }.items():
            s = self.out_dir / src
            if s.exists():
                (reports_dir / dst).write_text(s.read_text(encoding="utf-8"), encoding="utf-8")

        print("[RULE OPTIMIZER V2] complete")
        return {
            "meta": meta,
            "best_long": best_long,
            "best_short": best_short,
            "pattern_best": pattern_best,
            "report_path": str(self.out_dir / "live_entry_rule_v2.md"),
        }


def _serialize_best(best: dict | None, conditions: list) -> dict:
    if not best:
        return {}
    return {
        "rule_id": best.get("rule_id"),
        "rule_expr": best.get("rule_expr"),
        "precision": best.get("precision"),
        "recall": best.get("recall"),
        "pass_count": best.get("pass_count"),
        "pass_per_day": best.get("pass_per_day"),
        "avg_return_2h": best.get("avg_return_2h"),
        "avg_return_4h": best.get("avg_return_4h"),
        "live_score": best.get("live_score"),
        "conditions": [
            {
                "letter": c.letter,
                "feature": c.feature,
                "operator": c.operator,
                "threshold": c.threshold,
            }
            for c in conditions
        ],
    }
