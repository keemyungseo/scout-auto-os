"""Constitution Validation V1 orchestrator."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from scout_auto_os.engine.research.constitution_validation.constants import (
    CONSTITUTION,
    RANDOM_SEED,
    TRAIN_RATIO,
)
from scout_auto_os.engine.research.constitution_validation.long_calendar import (
    load_constitution_dataset,
    split_chronological,
)
from scout_auto_os.engine.research.constitution_validation.regime_validator import (
    aggregate_by_regime_axis,
    build_regime_index,
)
from scout_auto_os.engine.research.constitution_validation.report import build_decision, build_report
from scout_auto_os.engine.research.constitution_validation.rolling_validation import all_rolling_validations
from scout_auto_os.engine.research.constitution_validation.stability import (
    drift_summary,
    importance_stability,
    performance_drift,
)
from scout_auto_os.engine.research.constitution_validation.validator import (
    evaluate_constitution,
    metrics_from_picks,
    train_frozen_constitution,
)
from scout_auto_os.engine.research.ranking_engine.dataset import prepare_annotated
from scout_auto_os.engine.research.ranking_engine.importance import gain_importance, merge_importance, shap_rows
from scout_auto_os.engine.research.safe import research_safe
from scout_auto_os.engine.research.target_discovery.candidate_generator import generate_label_candidates
from scout_auto_os.engine.research.target_discovery.label_builder import apply_label
from scout_auto_os.engine.research.zero_base.runner import load_candidates_jsonl

KST = timezone(timedelta(hours=9))

REGIME_AXES = (
    "market_simple",
    "market_mapped",
    "market_ecology",
    "volatility",
    "structure",
    "dynamics",
)


class ConstitutionValidationRunner:
    def __init__(
        self,
        data_dir: Path,
        pkg_root: Path,
        candidates_path: Path,
        forward_path: Path,
    ) -> None:
        self.data_dir = data_dir
        self.pkg_root = pkg_root
        self.out_dir = data_dir / "constitution_validation"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.candidates_path = candidates_path
        self.forward_path = forward_path

    @research_safe("constitution_validation")
    def run(self) -> dict:
        print("[CONSTITUTION VALIDATION] started - frozen stack, no tuning")
        np.random.seed(RANDOM_SEED)

        dataset, calendar, feat_names = load_constitution_dataset(
            self.candidates_path, self.forward_path, self.data_dir, self.pkg_root,
        )
        train_rows, blind_rows = split_chronological(dataset, TRAIN_RATIO)
        print(
            f"[CONSTITUTION VALIDATION] calendar={calendar.get('calendar_days')}d "
            f"scans={calendar.get('scan_count')} blind_scans={len({r['scan_kst'] for r in blind_rows})}",
        )

        bundle = train_frozen_constitution(train_rows, feat_names)
        blind_picks, blind_metrics, calibration = evaluate_constitution(
            blind_rows, bundle, split_name="blind_holdout",
        )

        by_scan = load_candidates_jsonl(self.candidates_path)
        annotated, _, _ = prepare_annotated(by_scan)
        regime_index = build_regime_index(annotated)

        regime_rows: list[dict] = []
        for axis in REGIME_AXES:
            buckets = aggregate_by_regime_axis(blind_picks, regime_index, axis)
            for regime, picks in sorted(buckets.items()):
                m = metrics_from_picks(picks, split_name=f"regime_{axis}_{regime}")
                regime_rows.append({
                    "regime_axis": axis,
                    "regime": regime,
                    **m,
                })

        rolling_rows = all_rolling_validations(
            dataset, blind_rows, feat_names, TRAIN_RATIO, bundle,
        )
        drift_rows = performance_drift(blind_picks)
        drift_meta = drift_summary(drift_rows)
        imp_shift, imp_stability = importance_stability(train_rows, feat_names)

        spec = next(s for s in generate_label_candidates() if s.label_id == "return_minus_dd")
        labeled_train = apply_label(train_rows[:800], spec)
        gain = gain_importance(bundle)
        shap = shap_rows(bundle, labeled_train[:300])
        importance = merge_importance(gain, [], shap)
        for r in importance:
            r["constitution"] = "frozen_v1"

        decision = build_decision(
            calendar, blind_metrics, drift_meta, imp_stability, regime_rows, rolling_rows,
        )
        meta = {
            "constitution": CONSTITUTION,
            "calendar": calendar,
            "blind": blind_metrics,
            "drift": drift_meta,
            "importance_stability": imp_stability,
            "decision": decision,
            "generated_at": datetime.now(KST).isoformat(),
        }

        from season2_p37_scout_decision_hierarchy import write_csv

        write_csv(self.out_dir / "calendar_meta.csv", [calendar])
        write_csv(self.out_dir / "blind_report.csv", [blind_metrics])
        write_csv(self.out_dir / "rolling_validation.csv", rolling_rows)
        write_csv(self.out_dir / "regime_report.csv", regime_rows)
        write_csv(self.out_dir / "stability_report.csv", drift_rows)
        write_csv(self.out_dir / "importance_stability.csv", imp_shift)
        write_csv(self.out_dir / "calibration_report.csv", calibration)
        write_csv(self.out_dir / "feature_importance.csv", importance)

        report_md = build_report(meta, calendar, blind_metrics, decision)
        (self.out_dir / "constitution_validation_report.md").write_text(report_md, encoding="utf-8")
        (self.out_dir / "final_constitution_report.md").write_text(report_md, encoding="utf-8")
        (self.out_dir / "constitution_validation_meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8",
        )

        reports_dir = self.pkg_root / "research_bundle" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        for src, dst in {
            "final_constitution_report.md": "constitution_validation_v1_report.md",
            "blind_report.csv": "constitution_blind_report_v1.csv",
            "rolling_validation.csv": "constitution_rolling_validation_v1.csv",
            "regime_report.csv": "constitution_regime_report_v1.csv",
            "stability_report.csv": "constitution_stability_report_v1.csv",
            "importance_stability.csv": "constitution_importance_stability_v1.csv",
            "calibration_report.csv": "constitution_calibration_v1.csv",
            "feature_importance.csv": "constitution_feature_importance_v1.csv",
            "constitution_validation_meta.json": "constitution_validation_v1_meta.json",
        }.items():
            p = self.out_dir / src
            if p.exists():
                (reports_dir / dst).write_text(p.read_text(encoding="utf-8"), encoding="utf-8")

        print(
            f"[CONSTITUTION VALIDATION] blind_avg={blind_metrics.get('avg_return_2h')} "
            f"confidence={decision.get('confidence_tier')} core_ready={decision.get('core_ready')}",
        )
        return {"meta": meta, "decision": decision}
