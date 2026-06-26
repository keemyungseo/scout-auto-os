"""Trade DNA Predictor V1 runner."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from scout_auto_os.engine.research.safe import research_safe
from scout_auto_os.engine.research.trade_dna.predictor.dataset import build_entry_dataset
from scout_auto_os.engine.research.trade_dna.predictor.importance import compute_importance, top_features_for_class
from scout_auto_os.engine.research.trade_dna.predictor.models import compare_classifiers, cross_cv_proba
from scout_auto_os.engine.research.trade_dna.predictor.report import PredictorReport, false_case_rows, search_score_rank


class TradeDNAPredictorRunner:
    def __init__(self, data_dir: Path, pkg_root: Path, candidates_path: Path) -> None:
        self.data_dir = data_dir
        self.out_dir = data_dir / "trade_dna"
        self.pkg_root = pkg_root
        self.candidates_path = candidates_path
        self.cluster_path = data_dir / "trade_dna" / "trade_cluster.csv"

    @research_safe("trade_dna_predictor_v1")
    def run(self) -> dict:
        print("[DNA PREDICTOR] V1 started")
        if not self.cluster_path.exists():
            raise FileNotFoundError(f"Missing {self.cluster_path} — run trade_dna_v1 first")

        rows = build_entry_dataset(
            self.cluster_path, self.data_dir, self.pkg_root, self.candidates_path,
        )
        print(f"[DNA PREDICTOR] samples={len(rows)}")

        y = np.array([r["cluster_id"] for r in rows])
        importance = compute_importance(rows, y)
        runner_top = top_features_for_class(importance, rows, class_label=0, top_n=20)
        failed_top = top_features_for_class(importance, rows, class_label=1, top_n=20)

        classifier_rows, meta, y_pred, feat_names, best_pipe = compare_classifiers(rows)
        best = next(c for c in classifier_rows if c["model"] == meta["best_model"])

        X = np.array([[float(r["x"].get(n, 0)) for n in feat_names] for r in rows])
        n_splits = min(5, min(np.bincount(y)) * 2)
        cv = StratifiedKFold(n_splits=max(2, n_splits), shuffle=True, random_state=42)
        proba_failed = cross_cv_proba(best_pipe, X, y, cv)

        prediction_rows = []
        for r, pred, p_fail in zip(rows, y_pred, proba_failed):
            p_runner = 1.0 - p_fail
            prediction_rows.append({
                "trade_key": r["trade_key"],
                "symbol": r["symbol"],
                "direction": r["direction"],
                "actual_type": r["trade_type_id"],
                "predicted_type": f"TYPE_{pred}",
                "runner_probability": round(p_runner, 4),
                "failed_probability": round(float(p_fail), 4),
                "confidence": round(max(p_runner, p_fail), 4),
                "correct": int(pred == r["cluster_id"]),
                "entry_score": r["x"].get("entry_score"),
                "search_h4_score": r["x"].get("search_h4_score"),
            })

        false_runner = false_case_rows(rows, y_pred, "false_runner")
        false_failed = false_case_rows(rows, y_pred, "false_failed")
        sr = search_score_rank(importance)

        reporter = PredictorReport(self.out_dir)
        report_path = reporter.write_all(
            importance, runner_top, failed_top, classifier_rows,
            prediction_rows, false_runner, false_failed,
            meta, best, sr,
        )
        print(f"[DNA PREDICTOR] best={meta['best_model']} acc={best['accuracy']}")
        print(f"[DNA PREDICTOR] search rank=#{sr['search_rank']} ({sr['best_search_feature']})")
        print(f"[DNA PREDICTOR] report: {report_path}")
        return {
            "n_samples": len(rows),
            "best_model": meta["best_model"],
            "accuracy": best["accuracy"],
            "search_rank": sr["search_rank"],
            "report_path": str(report_path),
        }
