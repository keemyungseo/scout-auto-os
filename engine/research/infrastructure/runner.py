"""Research Infrastructure V1 orchestrator."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scout_auto_os.engine.research.infrastructure.calendar_monitor import (
    calendar_status,
    coverage_report,
    dataset_status,
    regime_gaps,
    validation_readiness,
)
from scout_auto_os.engine.research.infrastructure.constants import CONSTITUTION, DATASET_VERSION
from scout_auto_os.engine.research.infrastructure.dataset_manager import HistoryDatabase
from scout_auto_os.engine.research.infrastructure.history_builder import import_seed_bundle
from scout_auto_os.engine.research.infrastructure.quality_checker import run_quality_checks
from scout_auto_os.engine.research.infrastructure.report import (
    build_decision,
    build_quality_report,
    build_research_dashboard,
)
from scout_auto_os.engine.research.safe import research_safe

KST = timezone(timedelta(hours=9))


class ResearchInfrastructureRunner:
    def __init__(
        self,
        data_dir: Path,
        pkg_root: Path,
        candidates_path: Path,
        forward_path: Path,
    ) -> None:
        self.data_dir = data_dir
        self.pkg_root = pkg_root
        self.out_dir = data_dir / "research_infrastructure"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.out_dir / "history.db"
        self.parquet_dir = self.out_dir / "parquet"
        self.candidates_path = candidates_path
        self.forward_path = forward_path

    @research_safe("research_infrastructure")
    def run(self) -> dict:
        print("[RESEARCH INFRA] started - blind dataset builder")
        db = HistoryDatabase(self.db_path)

        import_result = import_seed_bundle(
            db, self.candidates_path, self.forward_path, self.data_dir, self.pkg_root,
        )
        db.set_meta("dataset_version", DATASET_VERSION)
        db.set_meta("constitution", json.dumps(CONSTITUTION))

        parquet_paths = db.export_parquet(self.parquet_dir)
        status = dataset_status(db, DATASET_VERSION)
        cal_rows = calendar_status(db)
        cov_rows = coverage_report(db)
        gaps = regime_gaps(db)
        readiness = validation_readiness(db)
        quality = run_quality_checks(db)
        decision = build_decision(status, readiness, quality, gaps)

        meta = {
            "constitution": CONSTITUTION,
            "import": import_result,
            "status": status,
            "readiness": readiness,
            "quality": {k: quality[k] for k in quality if k != "issues"},
            "decision": decision,
            "parquet_files": [str(p) for p in parquet_paths],
            "generated_at": datetime.now(KST).isoformat(),
        }

        from season2_p37_scout_decision_hierarchy import write_csv

        write_csv(self.out_dir / "dataset_status.csv", [status])
        write_csv(self.out_dir / "calendar_status.csv", cal_rows)
        write_csv(self.out_dir / "coverage_report.csv", cov_rows)
        if quality.get("issues"):
            write_csv(self.out_dir / "quality_issues.csv", quality["issues"])

        dashboard = build_research_dashboard(status, cal_rows, cov_rows, quality, decision)
        quality_md = build_quality_report(quality)
        (self.out_dir / "research_dashboard.md").write_text(dashboard, encoding="utf-8")
        (self.out_dir / "quality_report.md").write_text(quality_md, encoding="utf-8")
        (self.out_dir / "infrastructure_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        reports_dir = self.pkg_root / "research_bundle" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        for src, dst in {
            "research_dashboard.md": "research_dashboard_v1.md",
            "quality_report.md": "research_quality_report_v1.md",
            "dataset_status.csv": "research_dataset_status_v1.csv",
            "calendar_status.csv": "research_calendar_status_v1.csv",
            "coverage_report.csv": "research_coverage_report_v1.csv",
            "infrastructure_meta.json": "research_infrastructure_v1_meta.json",
        }.items():
            p = self.out_dir / src
            if p.exists():
                (reports_dir / dst).write_text(p.read_text(encoding="utf-8"), encoding="utf-8")

        print(
            f"[RESEARCH INFRA] days={status.get('calendar_days')} scans={status.get('scan_count')} "
            f"labeled={status.get('labeled_count')} quality={quality.get('passed')}",
        )
        return {"meta": meta, "decision": decision}
