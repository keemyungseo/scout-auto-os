"""Temporal Ranking Engine V1 orchestrator."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from scout_auto_os.engine.portfolio.rule_loader import load_portfolio_rules
from scout_auto_os.engine.research.directional.prediction.loader import load_formulas, resolve_formulas_path
from scout_auto_os.engine.research.ranking_engine.dataset import (
    collect_ranking_dataset,
    prepare_annotated,
    split_by_scans,
)
from scout_auto_os.engine.research.ranking_engine.features import feature_matrix
from scout_auto_os.engine.research.ranking_engine.metrics import evaluate_strategy_on_blind
from scout_auto_os.engine.research.ranking_engine.models import predict_scores, train_model
from scout_auto_os.engine.research.ranking_engine.validation import split_validation_rows
from scout_auto_os.engine.research.safe import research_safe
from scout_auto_os.engine.research.temporal_ranking.analysis import importance_analysis, statistical_significance
from scout_auto_os.engine.research.temporal_ranking.constants import (
    BASELINE_MODEL,
    MODEL_NAMES,
    RANDOM_SEED,
    SEQUENCE_LENGTHS,
    TRAIN_RATIO,
)
from scout_auto_os.engine.research.temporal_ranking.dataset import build_temporal_dataset
from scout_auto_os.engine.research.temporal_ranking.report import build_decision, build_temporal_report
from scout_auto_os.engine.research.zero_base.runner import load_candidates_jsonl, load_forward_klines
from scout_auto_os.engine.research.zero_base.validation import classify_regime

KST = timezone(timedelta(hours=9))


class TemporalRankingRunner:
    def __init__(
        self,
        data_dir: Path,
        pkg_root: Path,
        candidates_path: Path,
        forward_path: Path,
    ) -> None:
        self.data_dir = data_dir
        self.pkg_root = pkg_root
        self.out_dir = data_dir / "temporal_ranking"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.candidates_path = candidates_path
        self.forward_path = forward_path

    @research_safe("temporal_ranking")
    def run(self) -> dict:
        print("[TEMPORAL RANKING] started")
        np.random.seed(RANDOM_SEED)

        by_scan = load_candidates_jsonl(self.candidates_path)
        fwd = load_forward_klines(self.forward_path)
        rules = load_portfolio_rules(self.data_dir, self.pkg_root)
        formulas = load_formulas(resolve_formulas_path(self.data_dir, self.pkg_root))

        annotated, th, stats = prepare_annotated(by_scan)
        snapshot_rows = collect_ranking_dataset(annotated, fwd, rules, formulas, th, stats)
        train_snap, blind_snap = split_by_scans(snapshot_rows, TRAIN_RATIO)

        snap_names, _ = feature_matrix(train_snap)
        print(f"[TEMPORAL RANKING] snapshot baseline training n={len(train_snap)}")
        baseline_bundle = train_model(BASELINE_MODEL, train_snap, snap_names)

        def snap_score(row, peers, b=baseline_bundle):
            return float(predict_scores(b, [row])[0])

        _, baseline_metrics, _ = evaluate_strategy_on_blind(blind_snap, snap_score)
        baseline_metrics["model"] = "ranking_v1_snapshot"
        baseline_metrics["seq_len"] = 1

        seq_comparison: list[dict] = []
        generalization_rows: list[dict] = []
        leak_report: dict = {"passed": True}
        best_temporal: dict | None = None
        best_temporal_bundle = None
        best_temporal_train: list[dict] = []

        for seq_len in SEQUENCE_LENGTHS:
            train_t, leak = build_temporal_dataset(train_snap, seq_len)
            blind_t, leak_b = build_temporal_dataset(blind_snap, seq_len)
            leak_report = leak
            if not leak.get("passed", True):
                print(f"[TEMPORAL RANKING] LEAK WARNING seq={seq_len}")

            feat_names, _ = feature_matrix(train_t)
            print(f"[TEMPORAL RANKING] seq={seq_len} features={len(feat_names)}")

            for model_name in MODEL_NAMES:
                try:
                    bundle = train_model(model_name, train_t, feat_names)

                    def score_fn(row, peers, b=bundle):
                        return float(predict_scores(b, [row])[0])

                    _, metrics, _ = evaluate_strategy_on_blind(blind_t, score_fn)
                    row = {
                        "seq_len": seq_len,
                        "model": model_name,
                        "feature_count": len(feat_names),
                        **metrics,
                    }
                    seq_comparison.append(row)
                    generalization_rows.append({**row, "split": "blind_temporal"})

                    for val_row in split_validation_rows(blind_t, feat_names, model_name, bundle):
                        generalization_rows.append({**val_row, "seq_len": seq_len})

                    if best_temporal is None or float(metrics.get("avg_return_2h", 0)) > float(
                        best_temporal.get("avg_return_2h", 0),
                    ):
                        best_temporal = row
                        best_temporal_bundle = bundle
                        best_temporal_train = train_t
                except Exception as exc:
                    print(f"[TEMPORAL RANKING] skip seq={seq_len} {model_name}: {exc}")

        best_row = best_temporal or {"model": "none", "seq_len": 0, "avg_return_2h": 0.0}
        imp_bundle = best_temporal_bundle or baseline_bundle
        imp_train = best_temporal_train or train_snap
        imp_rows, imp_summary = importance_analysis(imp_bundle, imp_train[:1500])
        from scout_auto_os.engine.research.ranking_engine.importance import shap_rows
        shap_out = shap_rows(imp_bundle, imp_train[:400])
        sig = statistical_significance(
            float(baseline_metrics.get("avg_return_2h", 0)),
            float(best_row.get("avg_return_2h", 0)),
            int(baseline_metrics.get("trade_count", 0)),
        )

        regime_rows: list[dict] = []
        eval_seq = int(best_row.get("seq_len") or SEQUENCE_LENGTHS[0])
        blind_t_best, _ = build_temporal_dataset(blind_snap, eval_seq)
        eval_bundle = best_temporal_bundle or baseline_bundle

        def best_score(row, peers, b=eval_bundle):
            return float(predict_scores(b, [row])[0])

        by_scan_blind: dict[str, list] = {}
        for r in blind_t_best:
            by_scan_blind.setdefault(r["scan_kst"], []).append(r)
        for scan, rows in by_scan_blind.items():
            regime = classify_regime([
                {"features": {k.replace("dna_", ""): v for k, v in (rows[0].get("x") or {}).items() if k.startswith("dna_")}}
            ])
            _, m, _ = evaluate_strategy_on_blind(rows, best_score)
            regime_rows.append({"regime": regime, "scan": scan, **m})

        decision = build_decision(baseline_metrics, best_row, seq_comparison, imp_summary, sig, leak_report)
        meta = {
            "baseline": baseline_metrics,
            "best": best_row,
            "leak_check": leak_report,
            "significance": sig,
            "importance_level_pct": imp_summary.get("level_pct"),
            "decision": decision,
            "generated_at": datetime.now(KST).isoformat(),
        }

        from season2_p37_scout_decision_hierarchy import write_csv

        write_csv(self.out_dir / "sequence_length_comparison.csv", seq_comparison)
        write_csv(self.out_dir / "baseline_vs_temporal.csv", [
            {"strategy": "ranking_v1_snapshot", **baseline_metrics},
            {"strategy": f"temporal_{best_row.get('model', 'none')}", **best_row},
        ])
        write_csv(self.out_dir / "feature_importance.csv", imp_rows)
        write_csv(self.out_dir / "feature_shap.csv", shap_out)
        write_csv(self.out_dir / "delta_vs_absolute.csv", imp_summary.get("delta_vs_absolute", []))
        write_csv(self.out_dir / "generalization_report.csv", generalization_rows)
        write_csv(self.out_dir / "regime_comparison.csv", regime_rows)
        write_csv(self.out_dir / "leak_check.csv", [leak_report])

        report_md = build_temporal_report(
            meta, seq_comparison, baseline_metrics, best_row, imp_summary, decision, leak_report,
        )
        (self.out_dir / "temporal_ranking_report.md").write_text(report_md, encoding="utf-8")
        (self.out_dir / "temporal_ranking_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        reports_dir = self.pkg_root / "research_bundle" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        for src, dst in {
            "temporal_ranking_report.md": "temporal_ranking_v1_report.md",
            "sequence_length_comparison.csv": "temporal_sequence_comparison_v1.csv",
            "baseline_vs_temporal.csv": "temporal_baseline_comparison_v1.csv",
            "feature_importance.csv": "temporal_feature_importance_v1.csv",
            "feature_shap.csv": "temporal_feature_shap_v1.csv",
            "delta_vs_absolute.csv": "temporal_delta_vs_absolute_v1.csv",
            "generalization_report.csv": "temporal_generalization_v1.csv",
            "leak_check.csv": "temporal_leak_check_v1.csv",
            "temporal_ranking_meta.json": "temporal_ranking_v1_meta.json",
        }.items():
            p = self.out_dir / src
            if p.exists():
                (reports_dir / dst).write_text(p.read_text(encoding="utf-8"), encoding="utf-8")

        print(
            f"[TEMPORAL RANKING] baseline={baseline_metrics.get('avg_return_2h')} "
            f"best={best_row.get('avg_return_2h')} seq={best_row.get('seq_len')} improved={decision.get('improved')}",
        )
        return {"meta": meta, "decision": decision}
