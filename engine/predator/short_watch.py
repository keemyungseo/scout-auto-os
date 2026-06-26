"""Short false-accept watch list for Policy B shadow."""

from __future__ import annotations

import csv
from pathlib import Path

from scout_auto_os.engine.predator.value_gate import GateAction
from scout_auto_os.engine.predator.value_gate_shadow_logger import WATCH_FIELDS


def short_watch_match(
    side: str,
    policy_decision: str,
    value_score: float,
    predicted_dna_type: str,
    runner_prob: float,
) -> bool:
    if side.upper() != "SHORT":
        return False
    if policy_decision != GateAction.ENTER.value:
        return False
    if not (60 <= value_score < 70):
        return False
    if predicted_dna_type != "TYPE_0":
        return False
    if not (0.5 <= runner_prob < 0.7):
        return False
    return True


class ShortFalseAcceptWatch:
    def __init__(self, out_dir: Path) -> None:
        self.path = out_dir / "short_false_accept_watch.csv"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=WATCH_FIELDS).writeheader()

    def maybe_record(self, row: dict) -> bool:
        if not short_watch_match(
            row.get("side", ""),
            row.get("policy_b_decision", ""),
            float(row.get("value_score", 0)),
            row.get("predicted_dna_type", ""),
            float(row.get("runner_prob", 0)),
        ):
            return False
        watch = {
            "timestamp": row.get("timestamp", ""),
            "scan_id": row.get("scan_id", ""),
            "symbol": row.get("symbol", ""),
            "side": row.get("side", ""),
            "value_score": row.get("value_score", ""),
            "runner_prob": row.get("runner_prob", ""),
            "predicted_dna_type": row.get("predicted_dna_type", ""),
            "predicted_drawdown": row.get("predicted_drawdown", ""),
            "predicted_win_prob": row.get("predicted_win_prob", ""),
            "reason": "short_false_accept_risk_band",
            "actual_after_label_if_available": row.get("actual_after_label_if_available", ""),
        }
        with self.path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=WATCH_FIELDS, extrasaction="ignore")
            w.writerow(watch)
        return True

    def count_today(self, today_prefix: str) -> int:
        if not self.path.exists():
            return 0
        with self.path.open(encoding="utf-8") as f:
            return sum(1 for r in csv.DictReader(f) if (r.get("timestamp") or "").startswith(today_prefix))
