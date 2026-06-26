"""Adaptive Feature Weight Engine V1 orchestrator."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from scout_auto_os.engine.portfolio.rule_loader import load_portfolio_rules
from scout_auto_os.engine.research.adaptive_feature_weight.adaptive_scorer import (
    predict_adaptive,
    predict_uniform,
)
from scout_auto_os.engine.research.adaptive_feature_weight.analysis import (
    feature_drift,
    feature_stability,
    interaction_matrix,
)
from scout_auto_os.engine.research.adaptive_feature_weight.conditional_importance import (
    build_all_conditional_importance,
    compute_conditional_importance,
    importance_to_weight_map,
)
from scout_auto_os.engine.research.adaptive_feature_weight.constants import (
    CONDITION_DEFS,
    FROZEN_RANKING_MODEL,
    RANDOM_SEED,
    TOP_FEATURES,
    TRAIN_RATIO,
)
from scout_auto_os.engine.research.adaptive_feature_weight.report import (
    build_adaptive_report,
    build_decision,
    infer_patterns,
)
from scout_auto_os.engine.research.adaptive_feature_weight.scan_conditions import (
    build_scan_condition_map,
    primary_condition,
    _train_thresholds,
)
from scout_auto_os.engine.research.directional.prediction.loader import load_formulas, resolve_formulas_path
from scout_auto_os.engine.research.ranking_engine.dataset import (
    collect_ranking_dataset,
    prepare_annotated,
    split_by_scans,
)
from scout_auto_os.engine.research.ranking_engine.features import feature_matrix
from scout_auto_os.engine.research.ranking_engine.importance import merge_importance, gain_importance, permutation_importance_rows, shap_rows
from scout_auto_os.engine.research.ranking_engine.metrics import evaluate_strategy_on_blind
from scout_auto_os.engine.research.ranking_engine.models import train_model
from scout_auto_os.engine.research.safe import research_safe
from scout_auto_os.engine.research.zero_base.runner import load_candidates_jsonl, load_forward_klines

KST = timezone(timedelta(hours=9))


class AdaptiveFeatureWeightRunner:
    def __init__(
        self,
        data_dir: Path,
        pkg_root: Path,
        candidates_path: Path,
        forward_path: Path,
    ) -> None:
        self.data_dir = data_dir
        self.pkg_root = pkg_root
        self.out_dir = data_dir / "adaptive_feature_weight"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.candidates_path = candidates_path
        self.forward_path = forward_path

    @research_safe("adaptive_feature_weight")
    def run(self) -> dict:
        print("[ADAPTIVE FEATURE WEIGHT] started")
        np.random.seed(RANDOM_SEED)

        by_scan = load_candidates_jsonl(self.candidates_path)
        fwd = load_forward_klines(self.forward_path)
        rules = load_portfolio_rules(self.data_dir, self.pkg_root)
        formulas = load_formulas(resolve_formulas_path(self.data_dir, self.pkg_root))

        annotated, th, stats = prepare_annotated(by_scan)
        dataset = collect_ranking_dataset(annotated, fwd, rules, formulas, th, stats)
        train_rows, blind_rows = split_by_scans(dataset, TRAIN_RATIO)
        feat_names, _ = feature_matrix(train_rows)

        print(f"[ADAPTIVE FEATURE WEIGHT] train={len(train_rows)} blind={len(blind_rows)} feats={len(feat_names)}")

        bundle = train_model(FROZEN_RANKING_MODEL, train_rows, feat_names)
        print(f"[ADAPTIVE FEATURE WEIGHT] frozen model={FROZEN_RANKING_MODEL}")

        thresholds = _train_thresholds(train_rows)
        train_scan_cond = build_scan_condition_map(train_rows, thresholds)
        blind_scan_cond = build_scan_condition_map(blind_rows, thresholds)

        conditions = sorted(set(CONDITION_DEFS.keys()) | {"unclassified"})
        cond_imp_train, w_cond_map = build_all_conditional_importance(
            bundle, train_rows, train_scan_cond, conditions,
        )

        global_imp = merge_importance(
            gain_importance(bundle),
            permutation_importance_rows(bundle, train_rows),
            shap_rows(bundle, train_rows),
        )
        w_global = importance_to_weight_map(global_imp)

        cond_imp_blind: list[dict] = []
        for cid in conditions:
            cond_imp_blind.extend(
                compute_conditional_importance(bundle, blind_rows, blind_scan_cond, cid),
            )

        heatmap: list[dict] = []
        for r in cond_imp_train:
            heatmap.append({
                "condition_id": r["condition_id"],
                "feature": r["feature"],
                "combined_score": r.get("combined_score"),
                "shap_mean_abs": r.get("shap_mean_abs"),
                "gain_importance": r.get("gain_importance"),
                "permutation_importance": r.get("permutation_importance"),
            })

        interaction_rows: list[dict] = []
        by_scan_train: dict[str, list[dict]] = defaultdict(list)
        for r in train_rows:
            by_scan_train[r["scan_kst"]].append(r)
        for cid in conditions:
            scans = {s for s, t in train_scan_cond.items() if cid in t}
            subset = [r for r in train_rows if r["scan_kst"] in scans]
            interaction_rows.extend(interaction_matrix(subset, feat_names, cid))

        stability = feature_stability(cond_imp_train, feat_names)
        drift = feature_drift(cond_imp_train, cond_imp_blind)

        feature_map: list[dict] = []
        for cid, wmap in w_cond_map.items():
            for feat, wt in sorted(wmap.items(), key=lambda x: -x[1])[:TOP_FEATURES]:
                feature_map.append({
                    "condition_id": cid,
                    "feature": feat,
                    "conditional_weight": round(wt, 6),
                    "global_weight": round(w_global.get(feat, 0), 6),
                    "weight_ratio": round(wt / max(w_global.get(feat, 1e-9), 1e-9), 4),
                })

        top_conditional = sorted(cond_imp_train, key=lambda x: -float(x.get("combined_score", 0)))[:TOP_FEATURES]

        def uniform_fn(row, peers):
            return float(predict_uniform(bundle, [row])[0])

        def adaptive_fn(row, peers):
            tags = blind_scan_cond.get(row["scan_kst"], ["unclassified"])
            return float(predict_adaptive(bundle, [row], tags, w_global, w_cond_map)[0])

        _, uniform_metrics, _ = evaluate_strategy_on_blind(blind_rows, uniform_fn)
        _, adaptive_metrics, _ = evaluate_strategy_on_blind(blind_rows, adaptive_fn)

        comparison = {"uniform": uniform_metrics, "adaptive": adaptive_metrics}
        improved = float(adaptive_metrics.get("avg_return_2h", 0)) > float(uniform_metrics.get("avg_return_2h", 0))
        patterns = infer_patterns(cond_imp_train)
        decision = build_decision(comparison, improved, patterns)

        meta = {
            "feature_count": len(feat_names),
            "condition_count": len(w_cond_map),
            "train_scans": len(train_scan_cond),
            "blind_scans": len(blind_scan_cond),
            "frozen_model": FROZEN_RANKING_MODEL,
            "random_seed": RANDOM_SEED,
            "comparison": comparison,
            "decision": decision,
            "generated_at": datetime.now(KST).isoformat(),
        }

        from season2_p37_scout_decision_hierarchy import write_csv

        write_csv(self.out_dir / "adaptive_feature_map.csv", feature_map)
        write_csv(self.out_dir / "feature_importance_heatmap.csv", heatmap)
        write_csv(self.out_dir / "feature_interaction_matrix.csv", interaction_rows)
        write_csv(self.out_dir / "conditional_shap.csv", [
            {k: v for k, v in r.items() if k in (
                "condition_id", "feature", "shap_mean_abs", "combined_score",
                "gain_importance", "permutation_importance", "sample_count",
            )} for r in cond_imp_train
        ])
        write_csv(self.out_dir / "feature_stability.csv", stability)
        write_csv(self.out_dir / "feature_drift.csv", drift)
        write_csv(self.out_dir / "top20_conditional_features.csv", top_conditional)
        write_csv(self.out_dir / "weight_comparison.csv", [
            {"strategy": "uniform_frozen_ranker", **uniform_metrics},
            {"strategy": "adaptive_conditional_weight", **adaptive_metrics},
        ])

        report_md = build_adaptive_report(meta, comparison, top_conditional, patterns, decision)
        (self.out_dir / "adaptive_feature_report.md").write_text(report_md, encoding="utf-8")
        (self.out_dir / "adaptive_feature_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        reports_dir = self.pkg_root / "research_bundle" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        for src, dst in {
            "adaptive_feature_report.md": "adaptive_feature_weight_v1_report.md",
            "adaptive_feature_map.csv": "adaptive_feature_map_v1.csv",
            "feature_importance_heatmap.csv": "feature_importance_heatmap_v1.csv",
            "weight_comparison.csv": "adaptive_weight_comparison_v1.csv",
            "adaptive_feature_meta.json": "adaptive_feature_weight_v1_meta.json",
        }.items():
            p = self.out_dir / src
            if p.exists():
                (reports_dir / dst).write_text(p.read_text(encoding="utf-8"), encoding="utf-8")

        print(f"[ADAPTIVE FEATURE WEIGHT] improved={improved} uniform={uniform_metrics.get('avg_return_2h')} adaptive={adaptive_metrics.get('avg_return_2h')}")
        return {"meta": meta, "decision": decision}
