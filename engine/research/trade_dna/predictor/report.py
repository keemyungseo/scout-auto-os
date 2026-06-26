"""Trade DNA predictor reports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone, timedelta
from pathlib import Path

KST = timezone(timedelta(hours=9))


def _now_kst() -> str:
    return datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def search_score_rank(importance_rows: list[dict]) -> dict:
    search_keys = (
        "entry_score", "search_h4_score", "search_a6_proxy",
        "constitution_entry_score", "direction_confidence",
    )
    ranks = []
    for key in search_keys:
        for row in importance_rows:
            if row["feature"] == key:
                ranks.append({"feature": key, "overall_rank": row["overall_rank"]})
    best = min(ranks, key=lambda r: r["overall_rank"]) if ranks else {"feature": "n/a", "overall_rank": 999}
    return {"best_search_feature": best["feature"], "search_rank": best["overall_rank"], "all_search_ranks": ranks}


def false_case_rows(
    rows: list[dict],
    y_pred,
    case: str,
) -> list[dict]:
    out: list[dict] = []
    for r, pred in zip(rows, y_pred):
        actual = int(r["cluster_id"])
        if case == "false_runner" and pred == 0 and actual == 1:
            out.append({
                "trade_key": r["trade_key"],
                "symbol": r["symbol"],
                "direction": r["direction"],
                "scan_kst": r["scan_kst"],
                "predicted": "TYPE_0",
                "actual": r["trade_type_id"],
                "entry_score": r["x"].get("entry_score"),
                "search_h4_score": r["x"].get("search_h4_score"),
                "live_pattern": r.get("live_pattern"),
            })
        if case == "false_failed" and pred == 1 and actual == 0:
            out.append({
                "trade_key": r["trade_key"],
                "symbol": r["symbol"],
                "direction": r["direction"],
                "scan_kst": r["scan_kst"],
                "predicted": "TYPE_1",
                "actual": r["trade_type_id"],
                "entry_score": r["x"].get("entry_score"),
                "search_h4_score": r["x"].get("search_h4_score"),
                "live_pattern": r.get("live_pattern"),
            })
    return out


class PredictorReport:
    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def write_all(
        self,
        importance_rows: list[dict],
        runner_top: list[dict],
        failed_top: list[dict],
        classifier_rows: list[dict],
        prediction_rows: list[dict],
        false_runner: list[dict],
        false_failed: list[dict],
        meta: dict,
        best_metrics: dict,
        search_rank: dict,
    ) -> Path:
        _write_csv(self.out_dir / "feature_importance.csv", importance_rows)
        _write_csv(self.out_dir / "classifier_comparison.csv", classifier_rows)
        _write_csv(self.out_dir / "dna_prediction_model.csv", prediction_rows)
        _write_csv(self.out_dir / "false_runner_cases.csv", false_runner)
        _write_csv(self.out_dir / "false_failed_cases.csv", false_failed)

        top20 = importance_rows[:20]
        lines = [
            "# Trade DNA Predictor V1",
            "",
            f"_Generated: {_now_kst()}_",
            "",
            "## Principle",
            "",
            "Entry-time features ONLY. No forward ROI/volume/drawdown curves used as inputs.",
            "",
            "## Eight Questions",
            "",
            "### 1. Entry 시점에서 Trade Type 예측이 가능한가?",
            f"**{'예' if best_metrics.get('accuracy', 0) >= 0.65 else '부분적으로'}** — "
            f"best model {meta.get('best_model')} CV accuracy {best_metrics.get('accuracy')}",
            "",
            "### 2. Prediction Accuracy?",
            f"**{best_metrics.get('accuracy')}** (F1={best_metrics.get('f1')}, ROC-AUC={best_metrics.get('roc_auc')})",
            "",
            "### 3. 가장 중요한 Feature Top20",
            "",
        ]
        for r in top20:
            lines.append(
                f"- `{r['feature']}` — composite rank {r['overall_rank']} "
                f"(IG={r['information_gain']}, SHAP={r['shap_mean_abs']})"
            )

        sr = search_rank
        lines.extend([
            "",
            "### 4. Search Score 실제 영향력 순위?",
            f"**{sr.get('best_search_feature')}** overall rank **#{sr.get('search_rank')}** / {len(importance_rows)} features",
            "",
            "Search-related ranks:",
        ])
        for r in sr.get("all_search_ranks", []):
            lines.append(f"- `{r['feature']}`: rank #{r['overall_rank']}")

        lines.extend([
            "",
            "### 5. Runner(TYPE_0)를 가장 잘 설명하는 Feature Top20",
            "",
        ])
        for r in runner_top:
            lines.append(f"- `{r['feature']}` diff={r['mean_diff']} (class_mean={r['class_mean']})")

        lines.extend([
            "",
            "### 6. Failed(TYPE_1)를 가장 잘 설명하는 Feature Top20",
            "",
        ])
        for r in failed_top:
            lines.append(f"- `{r['feature']}` diff={r['mean_diff']} (class_mean={r['class_mean']})")

        lines.extend([
            "",
            "### 7. Prediction Confidence → Position Size?",
            (
                f"**가능 (조건부)** — TYPE_0 precision {best_metrics.get('type0_precision')} "
                f"but false_runner_rate {best_metrics.get('false_runner_rate_pct')}% "
                f"→ size scaling for high-confidence TYPE_0 only"
            ),
            "",
            "### 8. Season3 Search → DNA Predictor → Entry 구조?",
            "**가능** — entry features available at scan time; predictor runs before slot fill; "
            "TYPE_0 → full size + hold exit; TYPE_1 → skip or reduced size.",
            "",
            "## Critical: Does Search Score select Runners?",
            "",
        ])
        entry_rank = next((r for r in importance_rows if r["feature"] == "entry_score"), None)
        h4_rank = next((r for r in importance_rows if r["feature"] == "search_h4_score"), None)
        if entry_rank and h4_rank:
            if entry_rank["overall_rank"] <= 10:
                lines.append(
                    f"Search `entry_score` rank **#{entry_rank['overall_rank']}** — "
                    "Search contributes but is NOT the sole driver."
                )
            else:
                lines.append(
                    f"Search `entry_score` rank **#{entry_rank['overall_rank']}** — "
                    "**other features dominate Runner selection.**"
                )
        lines.extend([
            "",
            "## Classifier Comparison",
            "",
            "| Model | Acc | F1 | ROC-AUC | TYPE0 Prec | TYPE1 Prec | False Runner% | False Failed% |",
            "|-------|-----|-----|---------|------------|------------|---------------|---------------|",
        ])
        for c in classifier_rows:
            lines.append(
                f"| {c['model']} | {c['accuracy']} | {c['f1']} | {c['roc_auc']} | "
                f"{c['type0_precision']} | {c['type1_precision']} | "
                f"{c['false_runner_rate_pct']} | {c['false_failed_rate_pct']} |"
            )
        path = self.out_dir / "trade_dna_predictor.md"
        path.write_text("\n".join(lines), encoding="utf-8")
        return path
