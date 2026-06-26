"""Labeled Policy B re-evaluation tests."""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(ROOT))

from scout_auto_os.engine.predator.labeled_reevaluation import (
    analyze_false_skip_gap,
    band_calibration,
    compute_labeled_metrics,
    decide_verdict,
    false_skip_detail,
    labeled_score_band,
    run_labeled_reevaluation,
    score_monotonicity,
)


def _sample_row(**kw) -> dict:
    base = {
        "scan_id": "2026-06-01 00:00:00|BTCUSDT|long",
        "symbol": "BTCUSDT", "side": "LONG",
        "policy_b_decision": "SKIP", "policy_b_size": "0",
        "value_score": "45", "runner_prob": "0.9", "predicted_dna_type": "TYPE_0",
        "actual_roi_2h": "12", "actual_roi_4h": "11", "actual_peak_roi": "16",
        "actual_drawdown": "-2", "actual_dna_type": "TYPE_0",
        "false_skip": "1", "false_accept": "0", "reason": "test",
        "predicted_drawdown": "5",
    }
    base.update(kw)
    return base


class LabeledReevaluationTests(unittest.TestCase):
    def test_score_bands(self) -> None:
        self.assertEqual(labeled_score_band(35), "<40")
        self.assertEqual(labeled_score_band(55), "50-59")
        self.assertEqual(labeled_score_band(85), "80+")

    def test_metrics(self) -> None:
        rows = [
            _sample_row(policy_b_decision="ENTER", policy_b_size="0.3", actual_roi_2h="20", false_skip="0"),
            _sample_row(actual_roi_2h="-5", false_skip="1"),
        ]
        m = compute_labeled_metrics(rows)
        self.assertEqual(m["enter_count"], 1)
        self.assertEqual(m["false_skip_count"], 1)
        self.assertLess(m["skipped_avg_roi"], 0)

    def test_false_skip_detail(self) -> None:
        d = false_skip_detail([_sample_row()])
        self.assertEqual(len(d), 1)
        self.assertIn("false_skip_reason", d[0])
        self.assertIn("timestamp", d[0])

    def test_band_calibration(self) -> None:
        bands = band_calibration([_sample_row(), _sample_row(value_score="70", policy_b_decision="ENTER")])
        self.assertTrue(any(b["count"] > 0 for b in bands))

    def test_run_outputs(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        data = Path(tmp.name)
        out = data / "runtime_shadow"
        out.mkdir(parents=True)
        policy = data / "value_gate_policy"
        policy.mkdir()
        (policy / "policy_comparison.csv").write_text(
            "policy,policy_name,false_skip_count,false_accept_count,total_roi,avg_roi,win_rate,sharpe,mdd,trade_count,skipped_count,skipped_avg_roi,accepted_avg_roi\n"
            "B,Soft 50s,11,5,409,18,93,10.7,-6,74,83,-14,18\n",
            encoding="utf-8",
        )
        (policy / "policy_false_skip.csv").write_text(
            "policy,trade_key\nB,2026-06-01 00:00:00|BTCUSDT|long\n",
            encoding="utf-8",
        )
        fields = [
            "scan_id", "symbol", "side", "policy_b_decision", "policy_b_size",
            "value_score", "runner_prob", "predicted_dna_type", "predicted_drawdown",
            "actual_roi_2h", "actual_roi_4h", "actual_peak_roi", "actual_drawdown",
            "actual_dna_type", "false_skip", "false_accept", "reason",
        ]
        with (out / "value_gate_runtime_shadow_labeled.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerow({k: _sample_row().get(k, "") for k in fields})
        result = run_labeled_reevaluation(data)
        self.assertTrue(result["ok"])
        self.assertTrue((out / "value_gate_labeled_reevaluation_report.md").exists())
        tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
