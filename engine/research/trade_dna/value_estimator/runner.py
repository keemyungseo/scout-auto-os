"""Trade Value Estimator V1 runner."""

from __future__ import annotations

from pathlib import Path

from scout_auto_os.engine.research.safe import research_safe
from scout_auto_os.engine.research.trade_dna.value_estimator.dataset import build_value_dataset
from scout_auto_os.engine.research.trade_dna.value_estimator.models import TARGETS, evaluate_regressors
from scout_auto_os.engine.research.trade_dna.value_estimator.report import ValueEstimatorReport
from scout_auto_os.engine.research.trade_dna.value_estimator.simulation import simulate_sizing
from scout_auto_os.engine.research.trade_dna.value_estimator.value_score import fit_norm_params


class TradeValueEstimatorRunner:
    def __init__(self, data_dir: Path, pkg_root: Path, candidates_path: Path) -> None:
        self.data_dir = data_dir
        self.out_dir = data_dir / "trade_dna"
        self.pkg_root = pkg_root
        self.candidates_path = candidates_path
        self.cluster_path = data_dir / "trade_dna" / "trade_cluster.csv"

    @research_safe("trade_value_estimator_v1")
    def run(self) -> dict:
        print("[VALUE ESTIMATOR] V1 started")
        if not self.cluster_path.exists():
            raise FileNotFoundError(f"Missing {self.cluster_path} — run trade_dna_v1 first")

        rows = build_value_dataset(
            self.cluster_path, self.data_dir, self.pkg_root, self.candidates_path,
        )
        print(f"[VALUE ESTIMATOR] samples={len(rows)}")

        model_results: list[dict] = []
        oof_preds: dict = {}
        for target in TARGETS:
            evals = evaluate_regressors(rows, target)
            for e in evals:
                model_results.append({k: v for k, v in e.items() if k != "oof_predictions"})
            best = max(evals, key=lambda x: x["r2"])
            oof_preds[target] = best["oof_predictions"]
            print(f"[VALUE ESTIMATOR] {target}: best={best['model']} R2={best['r2']} MAE={best['mae']}")

        norms = {
            "roi_p95": fit_norm_params(rows, "expected_roi")["p95"],
            "dd_p95": fit_norm_params(rows, "expected_drawdown")["p95"],
            "sharpe_p95": fit_norm_params(rows, "expected_sharpe_contrib")["p95"],
        }
        sim = simulate_sizing(rows, oof_preds)

        reporter = ValueEstimatorReport(self.out_dir)
        report_path = reporter.write_all(rows, model_results, oof_preds, sim, norms)

        best_roi = max((r for r in model_results if r["target"] == "expected_roi"), key=lambda x: x["r2"])
        print(f"[VALUE ESTIMATOR] report: {report_path}")
        return {
            "n_samples": len(rows),
            "roi_r2": best_roi["r2"],
            "roi_mae": best_roi["mae"],
            "sharpe_delta": sim["improvement"]["sharpe_delta"],
            "report_path": str(report_path),
        }
