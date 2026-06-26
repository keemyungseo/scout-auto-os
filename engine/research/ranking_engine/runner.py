"""Ranking Engine V1 orchestrator."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from scout_auto_os.engine.portfolio.rule_loader import load_portfolio_rules
from scout_auto_os.engine.research.directional.prediction.loader import load_formulas, resolve_formulas_path
from scout_auto_os.engine.research.ranking_engine.baselines import BASELINE_SCORERS
from scout_auto_os.engine.research.ranking_engine.calibration import calibration_bins
from scout_auto_os.engine.research.ranking_engine.constants import MODEL_NAMES, RANDOM_SEED, TRAIN_RATIO
from scout_auto_os.engine.research.ranking_engine.dataset import (
    collect_ranking_dataset,
    prepare_annotated,
    split_by_scans,
)
from scout_auto_os.engine.research.ranking_engine.features import feature_matrix
from scout_auto_os.engine.research.ranking_engine.importance import (
    gain_importance,
    merge_importance,
    permutation_importance_rows,
    shap_rows,
)
from scout_auto_os.engine.research.ranking_engine.metrics import evaluate_strategy_on_blind
from scout_auto_os.engine.research.ranking_engine.models import predict_scores, train_model
from scout_auto_os.engine.research.ranking_engine.report import build_decision, build_ranking_report
from scout_auto_os.engine.research.ranking_engine.validation import split_validation_rows, walk_forward_validation
from scout_auto_os.engine.research.safe import research_safe
from scout_auto_os.engine.research.zero_base.runner import load_candidates_jsonl, load_forward_klines

KST = timezone(timedelta(hours=9))


class RankingEngineRunner:
    def __init__(
        self,
        data_dir: Path,
        pkg_root: Path,
        candidates_path: Path,
        forward_path: Path,
    ) -> None:
        self.data_dir = data_dir
        self.pkg_root = pkg_root
        self.out_dir = data_dir / "ranking_engine"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.candidates_path = candidates_path
        self.forward_path = forward_path

    @research_safe("ranking_engine")
    def run(self) -> dict:
        print("[RANKING ENGINE] started")
        np.random.seed(RANDOM_SEED)

        by_scan = load_candidates_jsonl(self.candidates_path)
        fwd = load_forward_klines(self.forward_path)
        rules = load_portfolio_rules(self.data_dir, self.pkg_root)
        formulas = load_formulas(resolve_formulas_path(self.data_dir, self.pkg_root))

        annotated, th, stats = prepare_annotated(by_scan)
        dataset = collect_ranking_dataset(annotated, fwd, rules, formulas, th, stats, direction="long")
        train_rows, blind_rows = split_by_scans(dataset, TRAIN_RATIO)
        feat_names, _ = feature_matrix(train_rows)
        print(f"[RANKING ENGINE] samples={len(dataset)} features={len(feat_names)} blind={len(blind_rows)}")

        model_comparison: list[dict] = []
        bundles: dict = {}
        trained = 0

        for model_name in MODEL_NAMES:
            try:
                print(f"[RANKING ENGINE] training {model_name}")
                bundle = train_model(model_name, train_rows, feat_names)
                bundles[model_name] = bundle
                trained += 1

                def score_fn(row, peers, b=bundle):
                    return float(predict_scores(b, [row])[0])

                picks, metrics, _ = evaluate_strategy_on_blind(blind_rows, score_fn)
                model_comparison.append({"split": "blind", "model": model_name, "strategy": model_name, **metrics})
                model_comparison.extend(split_validation_rows(blind_rows, feat_names, model_name, bundle))
                model_comparison.extend(walk_forward_validation(dataset, feat_names, model_name, TRAIN_RATIO))
            except Exception as exc:
                print(f"[RANKING ENGINE] skip {model_name}: {exc}")

        baseline_rows: list[dict] = []
        all_picks: list[dict] = []
        for strategy, scorer in BASELINE_SCORERS.items():
            picks, metrics, _ = evaluate_strategy_on_blind(
                blind_rows,
                lambda row, peers, fn=scorer: fn(row, peers),
            )
            baseline_rows.append({"split": "blind", "strategy": strategy, **metrics})
            if strategy == "current_search_a6":
                all_picks = picks

        blind_models = sorted(
            [m for m in model_comparison if m.get("split") == "blind" and m.get("model")],
            key=lambda x: (-float(x.get("rank_ndcg5", 0)), -float(x.get("avg_return_2h", 0))),
        )
        best_name = blind_models[0]["model"] if blind_models else "lightgbm_ranker"
        best_bundle = bundles.get(best_name) or next(iter(bundles.values()), None)

        ranking_picks: list[dict] = []
        ranking_top2: list[dict] = []
        ranking_top5: list[dict] = []
        pred_rows: list[dict] = []

        if best_bundle:
            def best_score(row, peers, b=best_bundle):
                return float(predict_scores(b, [row])[0])

            ranking_picks, _, _ = evaluate_strategy_on_blind(blind_rows, best_score)
            ranking_top5 = [p for p in ranking_picks if int(p.get("rank", 99)) <= 5]
            ranking_top2 = [p for p in ranking_picks if int(p.get("rank", 99)) <= 2]

            if best_bundle.kind == "classifier":
                for r in blind_rows:
                    prob = float(predict_scores(best_bundle, [r])[0])
                    pred_rows.append({**r, "pred_prob": prob})

        gain = gain_importance(best_bundle) if best_bundle else []
        perm = permutation_importance_rows(best_bundle, blind_rows) if best_bundle else []
        shap = shap_rows(best_bundle, blind_rows) if best_bundle else []
        importance = merge_importance(gain, perm, shap)
        calibration = calibration_bins(pred_rows) if pred_rows else calibration_bins([
            {**r, "pred_prob": float(predict_scores(best_bundle, [r])[0]), "label_top3": r["label_top3"]}
            for r in blind_rows[:500]
        ]) if best_bundle else []

        decision = build_decision(model_comparison, baseline_rows, importance)
        meta = {
            "sample_count": len(dataset),
            "feature_count": len(feat_names),
            "blind_scans": len({r["scan_kst"] for r in blind_rows}),
            "models_trained": trained,
            "random_seed": RANDOM_SEED,
            "best_model": best_name,
            "decision": decision,
            "generated_at": datetime.now(KST).isoformat(),
        }

        from season2_p37_scout_decision_hierarchy import write_csv

        write_csv(self.out_dir / "ranking_predictions.csv", [
            {
                "scan_kst": p["scan_kst"],
                "symbol": p["symbol"],
                "rank": p.get("rank"),
                "score": p.get("score"),
                "return_2h": p.get("return_2h"),
                "outcome_rank": p.get("outcome_rank"),
                "model": best_name,
            }
            for p in ranking_picks
        ])
        write_csv(self.out_dir / "ranking_top5.csv", ranking_top5)
        write_csv(self.out_dir / "ranking_top2.csv", ranking_top2)
        write_csv(self.out_dir / "feature_importance.csv", importance)
        write_csv(self.out_dir / "feature_shap.csv", shap)
        write_csv(self.out_dir / "model_comparison.csv", model_comparison + baseline_rows)
        write_csv(self.out_dir / "ranking_metrics.csv", [
            {**m, "kind": "model"} for m in model_comparison if m.get("split") == "blind"
        ] + [{**b, "kind": "baseline"} for b in baseline_rows])
        write_csv(self.out_dir / "probability_calibration.csv", calibration)

        report_md = build_ranking_report(meta, model_comparison, baseline_rows, importance, decision)
        (self.out_dir / "ranking_report.md").write_text(report_md, encoding="utf-8")
        (self.out_dir / "ranking_engine_meta.json").write_text(
            json.dumps({**meta, "gain_top5": gain[:5]}, indent=2), encoding="utf-8",
        )

        reports_dir = self.pkg_root / "research_bundle" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        mirror = {
            "ranking_report.md": "ranking_engine_v1_report.md",
            "model_comparison.csv": "ranking_model_comparison_v1.csv",
            "feature_importance.csv": "ranking_feature_importance_v1.csv",
            "ranking_engine_meta.json": "ranking_engine_meta_v1.json",
        }
        for src, dst in mirror.items():
            p = self.out_dir / src
            if p.exists():
                (reports_dir / dst).write_text(p.read_text(encoding="utf-8"), encoding="utf-8")

        print(f"[RANKING ENGINE] best={best_name} trained={trained}")
        return {"meta": meta, "decision": decision, "model_comparison": model_comparison}
