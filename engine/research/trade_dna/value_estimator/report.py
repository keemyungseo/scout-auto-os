"""Trade Value Estimator reports and CSV exports."""

from __future__ import annotations

import csv
import statistics
from pathlib import Path

import numpy as np

from scout_auto_os.engine.research.trade_dna.value_estimator.models import TARGETS
from scout_auto_os.engine.research.trade_dna.value_estimator.value_score import compute_value_score, size_multiplier


class ValueEstimatorReport:
    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def write_all(
        self,
        rows: list[dict],
        model_results: list[dict],
        oof_preds: dict,
        sim: dict,
        norms: dict,
    ) -> Path:
        self._write_predictions(rows, oof_preds, norms)
        self._write_model_comparison(model_results)
        self._write_value_score_table(rows, oof_preds, norms)
        self._write_position_simulation(sim)
        self._write_calibration(rows, oof_preds)
        self._write_error_distribution(rows, oof_preds)
        arch_path = self._write_architecture(rows, model_results, sim)
        report_path = self._write_report(rows, model_results, sim, norms)
        return report_path

    def _write_predictions(self, rows: list[dict], oof_preds: dict, norms: dict) -> None:
        path = self.out_dir / "value_prediction.csv"
        fields = [
            "trade_key", "symbol", "direction", "trade_type_id",
            "value_score", "size_multiplier",
        ] + [f"pred_{t}" for t in TARGETS] + [f"actual_{t}" for t in TARGETS]
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for i, r in enumerate(rows):
                pred = {t: round(float(oof_preds[t][i]), 4) for t in TARGETS}
                score = compute_value_score(pred, norms)
                row = {
                    "trade_key": r["trade_key"],
                    "symbol": r["symbol"],
                    "direction": r["direction"],
                    "trade_type_id": r["trade_type_id"],
                    "value_score": score,
                    "size_multiplier": size_multiplier(score),
                }
                for t in TARGETS:
                    row[f"pred_{t}"] = pred[t]
                    row[f"actual_{t}"] = r["y"][t]
                w.writerow(row)

    def _write_model_comparison(self, model_results: list[dict]) -> None:
        path = self.out_dir / "value_model_comparison.csv"
        fields = ["model", "target", "mae", "rmse", "r2", "mape_pct"]
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in model_results:
                w.writerow({k: r[k] for k in fields})

    def _write_value_score_table(self, rows: list[dict], oof_preds: dict, norms: dict) -> None:
        path = self.out_dir / "value_score_table.csv"
        buckets = {"90+": [], "80-89": [], "70-79": [], "60-69": [], "<50": []}
        for i, r in enumerate(rows):
            pred = {t: float(oof_preds[t][i]) for t in TARGETS}
            score = compute_value_score(pred, norms)
            roi = float(r["y"]["expected_roi"])
            if score >= 90:
                buckets["90+"].append(roi)
            elif score >= 80:
                buckets["80-89"].append(roi)
            elif score >= 70:
                buckets["70-79"].append(roi)
            elif score >= 60:
                buckets["60-69"].append(roi)
            else:
                buckets["<50"].append(roi)

        fields = ["score_band", "size_mult", "trade_count", "avg_actual_roi", "win_rate_pct"]
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            mults = {"90+": 1.0, "80-89": 0.8, "70-79": 0.6, "60-69": 0.3, "<50": 0.0}
            for band, rois in buckets.items():
                if not rois:
                    w.writerow({"score_band": band, "size_mult": mults[band], "trade_count": 0,
                                "avg_actual_roi": 0, "win_rate_pct": 0})
                    continue
                wins = sum(1 for x in rois if x >= 3.0)
                w.writerow({
                    "score_band": band,
                    "size_mult": mults[band],
                    "trade_count": len(rois),
                    "avg_actual_roi": round(statistics.mean(rois), 4),
                    "win_rate_pct": round(wins / len(rois) * 100, 2),
                })

    def _write_position_simulation(self, sim: dict) -> None:
        path = self.out_dir / "position_size_simulation.csv"
        fields = [
            "strategy", "total_roi", "avg_roi", "sharpe", "mdd", "win_rate",
            "trade_count", "skipped_count", "total_roi_delta", "sharpe_delta", "mdd_improvement",
        ]
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerow(sim["full"])
            row = dict(sim["dynamic"])
            row.update(sim["improvement"])
            w.writerow(row)

    def _write_calibration(self, rows: list[dict], oof_preds: dict) -> None:
        path = self.out_dir / "win_prob_calibration.csv"
        bins = np.linspace(0, 1, 11)
        actual = np.array([float(r["y"]["expected_win_prob"]) for r in rows])
        pred = np.array([float(oof_preds["expected_win_prob"][i]) for i in range(len(rows))])
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["bin_lo", "bin_hi", "mean_pred", "mean_actual", "count"])
            for i in range(len(bins) - 1):
                mask = (pred >= bins[i]) & (pred < bins[i + 1])
                if not mask.any():
                    continue
                w.writerow([
                    round(bins[i], 2), round(bins[i + 1], 2),
                    round(float(pred[mask].mean()), 4),
                    round(float(actual[mask].mean()), 4),
                    int(mask.sum()),
                ])

    def _write_error_distribution(self, rows: list[dict], oof_preds: dict) -> None:
        path = self.out_dir / "prediction_error_distribution.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["target", "error_p25", "error_p50", "error_p75", "error_mean", "error_std"])
            for t in TARGETS:
                actual = np.array([float(r["y"][t]) for r in rows])
                pred = np.array([float(oof_preds[t][i]) for i in range(len(rows))])
                err = pred - actual
                w.writerow([
                    t,
                    round(float(np.percentile(err, 25)), 4),
                    round(float(np.percentile(err, 50)), 4),
                    round(float(np.percentile(err, 75)), 4),
                    round(float(err.mean()), 4),
                    round(float(err.std()), 4),
                ])

    def _write_architecture(self, rows: list[dict], model_results: list[dict], sim: dict) -> Path:
        path = self.out_dir / "season3_value_architecture.md"
        best_by_target = {}
        for t in TARGETS:
            subset = [r for r in model_results if r["target"] == t]
            best_by_target[t] = max(subset, key=lambda x: x["r2"])

        lines = [
            "# Season3 Value Architecture",
            "",
            "## Pipeline",
            "",
            "```",
            "Search (A6) -> DNA Predictor (TYPE0/1) -> Value Estimator -> Value Score -> Dynamic Size -> Exit",
            "```",
            "",
            "## Regression Targets (entry-only features, replay labels)",
            "",
        ]
        for t in TARGETS:
            b = best_by_target[t]
            lines.append(f"- **{t}**: best={b['model']} MAE={b['mae']} R2={b['r2']}")

        lines.extend([
            "",
            "## Value Score (0-100)",
            "",
            "Weights: ROI 30%, WinProb 25%, Drawdown 20%, HoldEfficiency 15%, Sharpe 10%",
            "",
            "| Score | Size |",
            "|-------|------|",
            "| 90+ | 1.0x |",
            "| 80-89 | 0.8x |",
            "| 70-79 | 0.6x |",
            "| 60-69 | 0.3x |",
            "| <50 | Skip |",
            "",
            "## Portfolio Simulation",
            "",
            f"- Full size: ROI={sim['full']['total_roi']}% Sharpe={sim['full']['sharpe']} MDD={sim['full']['mdd']}%",
            f"- Dynamic: ROI={sim['dynamic']['total_roi']}% Sharpe={sim['dynamic']['sharpe']} MDD={sim['dynamic']['mdd']}%",
            f"- Delta: ROI {sim['improvement']['total_roi_delta']:+.2f} Sharpe {sim['improvement']['sharpe_delta']:+.2f}",
            "",
            f"Samples: {len(rows)} replay trades (15-day seed=42)",
        ])
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def _write_report(
        self,
        rows: list[dict],
        model_results: list[dict],
        sim: dict,
        norms: dict,
    ) -> Path:
        path = self.out_dir / "trade_value_estimator.md"
        best_roi = max((r for r in model_results if r["target"] == "expected_roi"), key=lambda x: x["r2"])
        best_hold = max((r for r in model_results if r["target"] == "expected_hold_time"), key=lambda x: x["r2"])
        best_win = max((r for r in model_results if r["target"] == "expected_win_prob"), key=lambda x: x["r2"])

        imp = sim["improvement"]
        dyn_better = imp["sharpe_delta"] > 0 or imp["mdd_improvement"] > 0

        lines = [
            "# Trade Value Estimator V1",
            "",
            "## Mission",
            "",
            "Entry-time features only. Labels from replay forward curves (no leakage).",
            "",
            f"**Samples:** {len(rows)}",
            "",
            "## Model Comparison (5-fold CV, best R2 per target)",
            "",
            "| Target | Best Model | MAE | RMSE | R2 | MAPE% |",
            "|--------|------------|-----|------|-----|-------|",
        ]
        for t in TARGETS:
            b = max((r for r in model_results if r["target"] == t), key=lambda x: x["r2"])
            lines.append(f"| {t} | {b['model']} | {b['mae']} | {b['rmse']} | {b['r2']} | {b['mape_pct']} |")

        lines.extend([
            "",
            "## Final Questions",
            "",
            f"1. **Trade Value prediction possible?** {'Partially' if best_roi['r2'] > 0.1 else 'Weak signal'} (ROI R2={best_roi['r2']})",
            f"2. **ROI prediction error?** MAE={best_roi['mae']}% RMSE={best_roi['rmse']}%",
            f"3. **Holding time predictable?** R2={best_hold['r2']} MAE={best_hold['mae']} min",
            f"4. **Value Score improves portfolio?** {'Yes (risk-adjusted)' if dyn_better else 'Mixed / needs tuning'} "
            f"(Sharpe delta {imp['sharpe_delta']:+.2f}, MDD improve {imp['mdd_improvement']:+.2f}%)",
            f"5. **Season3 structure complete with sizing?** Architecture defined; exit tuning + Short validation still required before LIVE",
            "",
            "## Position Size Simulation",
            "",
            "| Strategy | Total ROI | Avg ROI | Sharpe | MDD | WinRate | Trades |",
            "|----------|-----------|---------|--------|-----|---------|--------|",
            f"| Full | {sim['full']['total_roi']} | {sim['full']['avg_roi']} | {sim['full']['sharpe']} | {sim['full']['mdd']} | {sim['full']['win_rate']} | {sim['full']['trade_count']} |",
            f"| Dynamic | {sim['dynamic']['total_roi']} | {sim['dynamic']['avg_roi']} | {sim['dynamic']['sharpe']} | {sim['dynamic']['mdd']} | {sim['dynamic']['win_rate']} | {sim['dynamic']['trade_count']} |",
            "",
            f"Skipped (score<60): {sim['dynamic'].get('skipped_count', 0)}",
            "",
            "## Outputs",
            "",
            "- value_prediction.csv",
            "- value_model_comparison.csv",
            "- value_score_table.csv",
            "- position_size_simulation.csv",
            "- win_prob_calibration.csv",
            "- prediction_error_distribution.csv",
            "- season3_value_architecture.md",
        ])
        path.write_text("\n".join(lines), encoding="utf-8")
        return path
