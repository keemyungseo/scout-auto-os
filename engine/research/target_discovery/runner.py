"""Target Discovery Engine V1 orchestrator."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from scout_auto_os.engine.portfolio.rule_loader import load_portfolio_rules
from scout_auto_os.engine.research.directional.prediction.loader import load_formulas, resolve_formulas_path
from scout_auto_os.engine.research.ranking_engine.dataset import prepare_annotated
from scout_auto_os.engine.research.ranking_engine.features import feature_matrix
from scout_auto_os.engine.research.safe import research_safe
from scout_auto_os.engine.research.target_discovery.analysis import (
    feature_importance_for_label,
    importance_shift,
    learnability_report,
    rank_labels,
    statistical_vs_baseline,
)
from scout_auto_os.engine.research.target_discovery.candidate_generator import generate_label_candidates
from scout_auto_os.engine.research.target_discovery.constants import BASELINE_LABEL_ID, BASELINE_MODEL, RANDOM_SEED, TRAIN_RATIO
from scout_auto_os.engine.research.target_discovery.dataset import collect_target_discovery_dataset, split_by_scans
from scout_auto_os.engine.research.target_discovery.evaluation import evaluate_label_candidate, regime_breakdown
from scout_auto_os.engine.research.target_discovery.report import build_decision, build_report
from scout_auto_os.engine.research.zero_base.runner import load_candidates_jsonl, load_forward_klines

KST = timezone(timedelta(hours=9))


class TargetDiscoveryRunner:
    def __init__(
        self,
        data_dir: Path,
        pkg_root: Path,
        candidates_path: Path,
        forward_path: Path,
    ) -> None:
        self.data_dir = data_dir
        self.pkg_root = pkg_root
        self.out_dir = data_dir / "target_discovery"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.candidates_path = candidates_path
        self.forward_path = forward_path

    @research_safe("target_discovery")
    def run(self) -> dict:
        print("[TARGET DISCOVERY] started")
        np.random.seed(RANDOM_SEED)

        by_scan = load_candidates_jsonl(self.candidates_path)
        fwd = load_forward_klines(self.forward_path)
        rules = load_portfolio_rules(self.data_dir, self.pkg_root)
        formulas = load_formulas(resolve_formulas_path(self.data_dir, self.pkg_root))

        annotated, th, stats = prepare_annotated(by_scan)
        dataset = collect_target_discovery_dataset(annotated, fwd, rules, formulas, th, stats)
        train_rows, blind_rows = split_by_scans(dataset, TRAIN_RATIO)
        feat_names, _ = feature_matrix(train_rows)
        specs = generate_label_candidates()
        print(f"[TARGET DISCOVERY] samples={len(dataset)} features={len(feat_names)} labels={len(specs)}")

        blind_results: list[dict] = []
        generalization_rows: list[dict] = []
        regime_rows: list[dict] = []
        bundles: dict = {}

        for i, spec in enumerate(specs, 1):
            print(f"[TARGET DISCOVERY] [{i}/{len(specs)}] {spec.label_id}")
            row, gen, bundle = evaluate_label_candidate(
                train_rows, blind_rows, feat_names, spec, BASELINE_MODEL,
            )
            blind_results.append(row)
            generalization_rows.extend(gen)
            if bundle and not row.get("error"):
                bundles[spec.label_id] = bundle
                regime_rows.extend(regime_breakdown(blind_rows, bundle, spec))

        label_ranking = rank_labels(blind_results, BASELINE_LABEL_ID)
        learnability = learnability_report(train_rows, specs)

        baseline_row = next(
            (r for r in blind_results if r.get("label_id") == BASELINE_LABEL_ID),
            blind_results[0] if blind_results else {},
        )
        best_row = label_ranking[0] if label_ranking else baseline_row
        if best_row.get("label_id") != baseline_row.get("label_id"):
            best_blind = next(
                (r for r in blind_results if r.get("label_id") == best_row.get("label_id")),
                best_row,
            )
        else:
            best_blind = baseline_row

        sig = statistical_vs_baseline(
            float(baseline_row.get("avg_return_2h", 0)),
            float(best_blind.get("avg_return_2h", 0)),
            int(baseline_row.get("trade_count", 0)),
        )
        decision = build_decision(baseline_row, best_blind, label_ranking, sig, learnability)

        baseline_bundle = bundles.get(BASELINE_LABEL_ID)
        baseline_spec = next(s for s in specs if s.label_id == BASELINE_LABEL_ID)
        best_bundle = bundles.get(best_blind.get("label_id", ""))
        imp_rows: list[dict] = []
        shift_rows: list[dict] = []
        if baseline_bundle:
            base_imp = feature_importance_for_label(baseline_bundle, train_rows, baseline_spec)
            imp_rows.extend(base_imp)
            if best_bundle and best_blind.get("label_id") != BASELINE_LABEL_ID:
                best_spec = next(s for s in specs if s.label_id == best_blind["label_id"])
                cand_imp = feature_importance_for_label(best_bundle, train_rows, best_spec)
                imp_rows.extend(cand_imp)
                shift_rows = importance_shift(base_imp, cand_imp, best_blind["label_id"])

        meta = {
            "candidate_count": len(specs),
            "baseline": baseline_row,
            "best": best_blind,
            "significance": sig,
            "decision": decision,
            "generated_at": datetime.now(KST).isoformat(),
        }

        from season2_p37_scout_decision_hierarchy import write_csv

        write_csv(self.out_dir / "label_ranking.csv", label_ranking)
        write_csv(self.out_dir / "blind_comparison.csv", blind_results)
        write_csv(self.out_dir / "baseline_vs_best_label.csv", [
            {"strategy": "baseline_label", **baseline_row},
            {"strategy": f"best_{best_blind.get('label_id')}", **best_blind},
        ])
        write_csv(self.out_dir / "generalization_report.csv", generalization_rows)
        write_csv(self.out_dir / "regime_comparison.csv", regime_rows)
        write_csv(self.out_dir / "learnability_report.csv", learnability)
        write_csv(self.out_dir / "feature_importance.csv", imp_rows)
        write_csv(self.out_dir / "feature_importance_shift.csv", shift_rows)

        report_md = build_report(meta, label_ranking, baseline_row, best_blind, decision)
        (self.out_dir / "target_discovery_report.md").write_text(report_md, encoding="utf-8")
        (self.out_dir / "target_discovery_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        reports_dir = self.pkg_root / "research_bundle" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        for src, dst in {
            "target_discovery_report.md": "target_discovery_v1_report.md",
            "label_ranking.csv": "target_discovery_label_ranking_v1.csv",
            "blind_comparison.csv": "target_discovery_blind_comparison_v1.csv",
            "baseline_vs_best_label.csv": "target_discovery_baseline_comparison_v1.csv",
            "generalization_report.csv": "target_discovery_generalization_v1.csv",
            "learnability_report.csv": "target_discovery_learnability_v1.csv",
            "feature_importance.csv": "target_discovery_feature_importance_v1.csv",
            "feature_importance_shift.csv": "target_discovery_importance_shift_v1.csv",
            "target_discovery_meta.json": "target_discovery_v1_meta.json",
        }.items():
            p = self.out_dir / src
            if p.exists():
                (reports_dir / dst).write_text(p.read_text(encoding="utf-8"), encoding="utf-8")

        print(
            f"[TARGET DISCOVERY] baseline={baseline_row.get('avg_return_2h')} "
            f"best={best_blind.get('avg_return_2h')} label={best_blind.get('label_id')} "
            f"improved={decision.get('improved')}",
        )
        return {"meta": meta, "decision": decision, "label_ranking": label_ranking}
