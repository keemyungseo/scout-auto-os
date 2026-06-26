"""Append-only runtime shadow CSV + summary JSON."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from scout_auto_os.storage.db import now_kst

SHADOW_FIELDS = (
    "timestamp", "scan_id", "trade_key", "prediction_key", "symbol", "side",
    "baseline_decision", "baseline_size",
    "policy_b_decision", "policy_b_size",
    "value_score", "runner_prob", "predicted_dna_type",
    "predicted_roi", "predicted_peak_roi", "predicted_drawdown", "predicted_win_prob",
    "reason", "manual_lock", "source", "auto_manage",
    "actual_roi_2h", "actual_roi_4h", "actual_peak_roi", "actual_drawdown",
    "actual_dna_type", "false_skip", "false_accept",
)

WATCH_FIELDS = (
    "timestamp", "scan_id", "symbol", "side", "value_score", "runner_prob",
    "predicted_dna_type", "predicted_drawdown", "predicted_win_prob", "reason",
    "actual_after_label_if_available",
)


class ValueGateShadowLogger:
    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.shadow_path = self.out_dir / "value_gate_runtime_shadow.csv"
        self.summary_path = self.out_dir / "value_gate_shadow_summary.json"
        self._ensure_header()

    def _ensure_header(self) -> None:
        if not self.shadow_path.exists():
            with self.shadow_path.open("w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=SHADOW_FIELDS).writeheader()

    def append(self, row: dict) -> None:
        with self.shadow_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=SHADOW_FIELDS, extrasaction="ignore")
            w.writerow({k: row.get(k, "") for k in SHADOW_FIELDS})

    def read_all(self) -> list[dict]:
        if not self.shadow_path.exists():
            return []
        with self.shadow_path.open(encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def reset(self) -> None:
        """Truncate shadow CSV (replay rebuild)."""
        with self.shadow_path.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=SHADOW_FIELDS).writeheader()

    def backup_to(self, filename: str) -> Path | None:
        if not self.shadow_path.exists():
            return None
        dst = self.out_dir / filename
        import shutil
        shutil.copy2(self.shadow_path, dst)
        return dst

    def rows_today(self) -> list[dict]:
        if not self.shadow_path.exists():
            return []
        today = now_kst()[:10]
        with self.shadow_path.open(encoding="utf-8") as f:
            return [r for r in csv.DictReader(f) if (r.get("timestamp") or "").startswith(today)]

    def write_summary(self, summary: dict) -> None:
        self.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def read_summary(self) -> dict:
        if not self.summary_path.exists():
            return {}
        return json.loads(self.summary_path.read_text(encoding="utf-8"))
