"""Entry Filter Threshold Optimizer V1 orchestrator."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scout_auto_os.engine.research.directional.entry_filter.constants import WINNER_QUANTILE
from scout_auto_os.engine.research.directional.entry_filter.rule_builder import (
    build_entry_filter_rule_markdown,
    build_rules_json,
    merge_both_directions,
    select_rule_features,
)
from scout_auto_os.engine.research.directional.entry_filter.signals_loader import (
    label_signals,
    load_dna_feature_sets,
    resolve_signals,
)
from scout_auto_os.engine.research.directional.entry_filter.threshold_optimizer import (
    evaluate_combined_rule,
    grid_search_all,
)
from scout_auto_os.engine.research.safe import research_safe

KST = timezone(timedelta(hours=9))


class EntryFilterThresholdRunner:
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

    @research_safe("entry_filter_threshold")
    def run(self, max_rule_features: int = 4) -> dict:
        print("[THRESHOLD OPTIMIZER] started")
        long_raw, short_raw, feature_keys = resolve_signals(
            self.data_dir, self.candidates_path, self.forward_path,
        )
        long_signals, long_split = label_signals(long_raw)
        short_signals, short_split = label_signals(short_raw)
        dna_sets = load_dna_feature_sets(self.data_dir, self.pkg_root)

        long_best, long_thr_curve, long_prec_curve, long_ret_curve = grid_search_all(
            long_signals, feature_keys, "long",
        )
        short_best, short_thr_curve, short_prec_curve, short_ret_curve = grid_search_all(
            short_signals, feature_keys, "short",
        )

        long_rules, short_rules = select_rule_features(
            long_best, short_best, dna_sets, max_rules=max_rule_features,
        )
        long_stats = evaluate_combined_rule(long_signals, long_rules)
        short_stats = evaluate_combined_rule(short_signals, short_rules)
        scope_rows = merge_both_directions(long_best, short_best)

        # Full threshold report = all best rows + scope
        threshold_report = []
        for row in long_best:
            threshold_report.append({**row, "dna_group": _dna_group(row["feature"], dna_sets, "long")})
        for row in short_best:
            threshold_report.append({**row, "dna_group": _dna_group(row["feature"], dna_sets, "short")})

        meta = {
            "long_signals": len(long_signals),
            "short_signals": len(short_signals),
            "feature_count": len(feature_keys),
            "winner_quantile": int(WINNER_QUANTILE * 100),
            "long_split": long_split,
            "short_split": short_split,
            "long_rule_count": len(long_rules),
            "short_rule_count": len(short_rules),
            "generated_at": datetime.now(KST).isoformat(),
        }

        from season2_p37_scout_decision_hierarchy import write_csv

        write_csv(self.out_dir / "feature_threshold_report.csv", threshold_report)
        write_csv(self.out_dir / "feature_best_threshold.csv", long_best + short_best)
        write_csv(self.out_dir / "feature_threshold_curve.csv", long_thr_curve + short_thr_curve)
        write_csv(self.out_dir / "feature_precision_curve.csv", long_prec_curve + short_prec_curve)
        write_csv(self.out_dir / "feature_return_curve.csv", long_ret_curve + short_ret_curve)
        write_csv(self.out_dir / "feature_threshold_scope.csv", scope_rows)

        rule_md = build_entry_filter_rule_markdown(long_rules, short_rules, long_stats, short_stats, meta)
        (self.out_dir / "entry_filter_rule_v1.md").write_text(rule_md, encoding="utf-8")
        rules_json = build_rules_json(long_rules, short_rules)
        (self.out_dir / "entry_filter_rules_v1.json").write_text(
            json.dumps(rules_json, indent=2), encoding="utf-8",
        )
        (self.out_dir / "entry_filter_threshold_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        reports_dir = self.pkg_root / "research_bundle" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        bundle_map = {
            "feature_threshold_report.csv": "feature_threshold_report_v1.csv",
            "feature_best_threshold.csv": "feature_best_threshold_v1.csv",
            "feature_threshold_curve.csv": "feature_threshold_curve_v1.csv",
            "feature_precision_curve.csv": "feature_precision_curve_v1.csv",
            "feature_return_curve.csv": "feature_return_curve_v1.csv",
            "entry_filter_rule_v1.md": "entry_filter_rule_v1.md",
            "entry_filter_rules_v1.json": "entry_filter_rules_v1.json",
        }
        for src_name, dst_name in bundle_map.items():
            src = self.out_dir / src_name
            if src.exists():
                (reports_dir / dst_name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

        print("[THRESHOLD OPTIMIZER] complete")
        return {
            "meta": meta,
            "long_rules": long_rules,
            "short_rules": short_rules,
            "long_stats": long_stats,
            "short_stats": short_stats,
            "report_path": str(self.out_dir / "entry_filter_rule_v1.md"),
        }


def _dna_group(feature: str, dna_sets: dict, direction: str) -> str:
    common = dna_sets.get("common", set())
    if feature in common:
        return "common"
    ranked: list[str] = list(dna_sets.get(direction, []))
    if feature in ranked[:15]:
        return f"{direction}_winner_dna_top15"
    dna_set = dna_sets.get(f"{direction}_set", set())
    if feature in dna_set:
        return f"{direction}_winner_dna"
    return "all_features"
