"""Lifecycle Classifier V1 orchestrator."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from scout_auto_os.engine.portfolio.rule_loader import load_portfolio_rules
from scout_auto_os.engine.research.directional.entry_filter.collector import filter_scans_last_months
from scout_auto_os.engine.research.directional.entry_filter.constants import CHAMPION_TOP_K, LOOKBACK_MONTHS
from scout_auto_os.engine.research.directional.prediction.loader import load_formulas, resolve_formulas_path
from scout_auto_os.engine.research.lifecycle_classifier.constants import LABEL_SHORT, PROB_DISPLAY_ORDER
from scout_auto_os.engine.research.lifecycle_classifier.dataset import (
    active_class_names,
    collect_lifecycle_dataset,
    split_dataset,
)
from scout_auto_os.engine.research.lifecycle_classifier.features import feature_matrix
from scout_auto_os.engine.research.lifecycle_classifier.metrics import (
    aggregate_metrics,
    binary_discrimination,
    confusion_matrix,
    per_class_metrics,
    top_k_accuracy,
)
from scout_auto_os.engine.research.lifecycle_classifier.model import MultinomialLifecycleClassifier
from scout_auto_os.engine.research.lifecycle_classifier.report import build_classifier_report
from scout_auto_os.engine.research.safe import research_safe
from scout_auto_os.engine.research.zero_base.runner import load_candidates_jsonl, load_forward_klines

KST = timezone(timedelta(hours=9))


def _label_index(name: str, class_names: list[str]) -> int:
    return class_names.index(name)


def _train_direction_model(
    train: list[dict],
    val: list[dict],
    direction: str,
) -> tuple[MultinomialLifecycleClassifier, dict]:
    train_d = [r for r in train if r["direction"] == direction]
    val_d = [r for r in val if r["direction"] == direction]
    class_names = active_class_names(train_d)
    if len(class_names) < 2:
        raise ValueError(f"{direction}: need >=2 lifecycle classes with enough train samples")

    feat_names, _ = feature_matrix(train_d)
    X_train = np.array([[float(r["x"].get(n, 0.0)) for n in feat_names] for r in train_d], dtype=float)
    y_train = np.array([_label_index(r["lifecycle_label"], class_names) for r in train_d], dtype=int)

    model = MultinomialLifecycleClassifier(class_names)
    fit_info = model.fit(X_train, y_train)

    X_val = np.array([[float(r["x"].get(n, 0.0)) for n in feat_names] for r in val_d], dtype=float)
    y_true = [r["lifecycle_label"] for r in val_d]
    y_pred_idx = model.predict(X_val) if len(val_d) else np.array([], dtype=int)
    y_pred = [class_names[i] for i in y_pred_idx]
    proba_rows = model.proba_dicts(X_val) if len(val_d) else []

    per_class = per_class_metrics(y_true, y_pred, class_names)
    agg = aggregate_metrics(per_class)
    acc = round(float((np.array(y_pred) == np.array(y_true)).mean()), 4) if y_true else 0.0
    top2 = top_k_accuracy(proba_rows, y_true, k=2) if y_true else 0.0

    fake_vs_cont = binary_discrimination(
        y_true, y_pred, positive="Fake Breakout", negative="Continuous Trend",
    )

    metrics = {
        "direction": direction,
        "class_names": class_names,
        "feature_names": feat_names,
        "fit_info": fit_info,
        "accuracy": acc,
        "top2_accuracy": top2,
        "per_class": per_class,
        "aggregate": agg,
        "confusion": confusion_matrix(y_true, y_pred, class_names),
        "fake_vs_continuous": fake_vs_cont,
        "val_count": len(val_d),
    }
    return model, metrics


def _prediction_rows(
    records: list[dict],
    model: MultinomialLifecycleClassifier,
    feat_names: list[str],
    split: str,
) -> list[dict]:
    if not records:
        return []
    X = np.array([[float(r["x"].get(n, 0.0)) for n in feat_names] for r in records], dtype=float)
    pred_idx = model.predict(X)
    probas = model.proba_dicts(X)
    rows: list[dict] = []
    for r, pi, probs in zip(records, pred_idx, probas):
        pred_label = model.class_names[int(pi)]
        row = {
            "signal_id": r["signal_id"],
            "direction": r["direction"],
            "scan_time_kst": r["scan_time_kst"],
            "symbol": r["symbol"],
            "split": split,
            "true_lifecycle_label": r["lifecycle_label"],
            "pred_lifecycle_label": pred_label,
            "pred_confidence_pct": round(max(probs.values()), 2),
        }
        for lab in PROB_DISPLAY_ORDER:
            short = LABEL_SHORT.get(lab, lab.lower().replace(" ", "_"))
            row[f"prob_{short}"] = probs.get(lab, 0.0)
        for lab, pct in probs.items():
            if lab not in PROB_DISPLAY_ORDER:
                short = LABEL_SHORT.get(lab, lab.lower().replace(" ", "_"))
                row[f"prob_{short}"] = pct
        rows.append(row)
    return rows


class LifecycleClassifierRunner:
    def __init__(
        self,
        data_dir: Path,
        pkg_root: Path,
        candidates_path: Path,
        forward_path: Path,
        lookback_months: int = LOOKBACK_MONTHS,
    ) -> None:
        self.data_dir = data_dir
        self.pkg_root = pkg_root
        self.out_dir = data_dir / "lifecycle_classifier"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.candidates_path = candidates_path
        self.forward_path = forward_path
        self.lookback_months = lookback_months

    @research_safe("lifecycle_classifier")
    def run(self, top_k: int = CHAMPION_TOP_K) -> dict:
        print("[LIFECYCLE CLASSIFIER] started")
        by_scan = load_candidates_jsonl(self.candidates_path)
        fwd = load_forward_klines(self.forward_path)
        all_scans = sorted(by_scan.keys())
        scans = filter_scans_last_months(all_scans, self.lookback_months)

        rules = load_portfolio_rules(self.data_dir, self.pkg_root)
        formulas_path = resolve_formulas_path(self.data_dir, self.pkg_root)
        formulas = load_formulas(formulas_path)

        records = collect_lifecycle_dataset(by_scan, fwd, scans, rules, formulas, top_k=top_k)
        train, val = split_dataset(records)
        train_scans = sorted({r["scan_time_kst"] for r in train})
        val_scans = sorted({r["scan_time_kst"] for r in val})

        long_model, long_metrics = _train_direction_model(train, val, "long")
        short_model, short_metrics = _train_direction_model(train, val, "short")

        pred_rows = (
            _prediction_rows(train, long_model, long_metrics["feature_names"], "train")
            + _prediction_rows(val, long_model, long_metrics["feature_names"], "val")
            + _prediction_rows(train, short_model, short_metrics["feature_names"], "train")
            + _prediction_rows(val, short_model, short_metrics["feature_names"], "val")
        )

        dates = [s[:10] for s in scans]
        meta = {
            "lookback_months": self.lookback_months,
            "date_min": min(dates) if dates else None,
            "date_max": max(dates) if dates else None,
            "total_signals": len(records),
            "train_scan_count": len(train_scans),
            "val_scan_count": len(val_scans),
            "feature_dim": len(long_metrics["feature_names"]),
            "long_class_count": len(long_metrics["class_names"]),
            "short_class_count": len(short_metrics["class_names"]),
            "generated_at": datetime.now(KST).isoformat(),
        }

        from season2_p37_scout_decision_hierarchy import write_csv

        write_csv(self.out_dir / "lifecycle_classifier_predictions.csv", pred_rows)
        write_csv(
            self.out_dir / "lifecycle_classifier_metrics.csv",
            [{**r, "direction": "long"} for r in long_metrics["per_class"]]
            + [{**r, "direction": "short"} for r in short_metrics["per_class"]],
        )
        write_csv(self.out_dir / "lifecycle_confusion_matrix_long.csv", long_metrics["confusion"])
        write_csv(self.out_dir / "lifecycle_confusion_matrix_short.csv", short_metrics["confusion"])

        summary_rows = [
            {"direction": "long", "metric": "accuracy", "value": long_metrics["accuracy"]},
            {"direction": "long", "metric": "top2_accuracy", "value": long_metrics["top2_accuracy"]},
            {"direction": "long", "metric": "macro_f1", "value": long_metrics["aggregate"]["macro_f1"]},
            {"direction": "long", "metric": "weighted_f1", "value": long_metrics["aggregate"]["weighted_f1"]},
            {"direction": "short", "metric": "accuracy", "value": short_metrics["accuracy"]},
            {"direction": "short", "metric": "top2_accuracy", "value": short_metrics["top2_accuracy"]},
            {"direction": "short", "metric": "macro_f1", "value": short_metrics["aggregate"]["macro_f1"]},
            {"direction": "short", "metric": "weighted_f1", "value": short_metrics["aggregate"]["weighted_f1"]},
        ]
        write_csv(self.out_dir / "lifecycle_classifier_summary.csv", summary_rows)

        report = build_classifier_report(
            meta,
            long_metrics,
            short_metrics,
            long_metrics["fake_vs_continuous"],
            short_metrics["fake_vs_continuous"],
        )
        report_path = self.out_dir / "lifecycle_classifier_report.md"
        report_path.write_text(report, encoding="utf-8")
        (self.out_dir / "lifecycle_classifier_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        reports_dir = self.pkg_root / "research_bundle" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        bundle_map = {
            "lifecycle_classifier_report.md": "lifecycle_classifier_v1_report.md",
            "lifecycle_classifier_predictions.csv": "lifecycle_classifier_v1_predictions.csv",
            "lifecycle_confusion_matrix_long.csv": "lifecycle_confusion_matrix_long_v1.csv",
            "lifecycle_confusion_matrix_short.csv": "lifecycle_confusion_matrix_short_v1.csv",
            "lifecycle_classifier_summary.csv": "lifecycle_classifier_v1_summary.csv",
            "lifecycle_classifier_meta.json": "lifecycle_classifier_v1_meta.json",
        }
        for src_name, dst_name in bundle_map.items():
            src = self.out_dir / src_name
            if src.exists():
                (reports_dir / dst_name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

        print("[LIFECYCLE CLASSIFIER] report generated")
        return {
            "meta": meta,
            "long_metrics": long_metrics,
            "short_metrics": short_metrics,
            "report_path": str(report_path),
        }
