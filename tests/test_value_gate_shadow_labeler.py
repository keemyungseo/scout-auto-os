"""Shadow Labeler V1 tests."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(ROOT))

from scout_auto_os.engine.predator.shadow_labeler import (  # noqa: E402
    LabelerConfig,
    build_summary,
    compute_false_flags,
    compute_side_aware_labels,
    label_shadow_row,
    run_shadow_labeler,
)
from scout_auto_os.engine.predator.value_gate_shadow_logger import SHADOW_FIELDS as LOGGER_FIELDS  # noqa: E402


def make_klines(entry: float, closes: list[float]) -> list:
    bars: list[list] = []
    base_ms = 1_700_000_000_000
    for i, c in enumerate(closes):
        o = entry if i == 0 else closes[i - 1]
        bars.append([
            base_ms + i * 900_000,
            str(o), str(max(o, c)), str(min(o, c)), str(c), "1000",
        ])
    return bars


def _flat_closes(entry: float, target: float, n: int) -> list[float]:
    return [entry] + [target] * (n - 1)


class RoiCalculationTests(unittest.TestCase):
    def test_long_roi(self) -> None:
        klines = make_klines(100.0, _flat_closes(100.0, 110.0, 16))
        labels = compute_side_aware_labels(klines, "LONG", hours=4)
        self.assertAlmostEqual(float(labels["actual_roi_2h"]), 10.0, places=2)
        self.assertAlmostEqual(float(labels["actual_roi_4h"]), 10.0, places=2)

    def test_short_roi(self) -> None:
        klines = make_klines(100.0, _flat_closes(100.0, 90.0, 16))
        labels = compute_side_aware_labels(klines, "SHORT", hours=4)
        self.assertAlmostEqual(float(labels["actual_roi_2h"]), 10.0, places=2)
        self.assertAlmostEqual(float(labels["actual_roi_4h"]), 10.0, places=2)


class LabelTimingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = LabelerConfig()
        self.klines = make_klines(100.0, _flat_closes(100.0, 105.0, 16))

    def _fetch(self, _sym: str, _s: int, _e: int) -> list:
        return self.klines

    def test_waiting_before_2h(self) -> None:
        row = {
            "timestamp": "2026-06-01 13:30:00",
            "symbol": "BTCUSDT",
            "side": "LONG",
            "policy_b_decision": "ENTER",
        }
        updated, status = label_shadow_row(
            row,
            now_kst_str="2026-06-01 14:00:00",
            cfg=self.cfg,
            klines_fetcher=self._fetch,
        )
        self.assertEqual(status, "waiting")
        self.assertEqual(updated.get("actual_roi_2h", ""), "")

    def test_2h_label_partial(self) -> None:
        row = {
            "timestamp": "2026-06-01 10:00:00",
            "symbol": "BTCUSDT",
            "side": "LONG",
            "policy_b_decision": "SKIP",
        }
        updated, status = label_shadow_row(
            row,
            now_kst_str="2026-06-01 12:30:00",
            cfg=self.cfg,
            klines_fetcher=self._fetch,
            hours=4,
        )
        self.assertEqual(status, "partial")
        self.assertTrue(updated["actual_roi_2h"])
        self.assertEqual(updated["actual_roi_4h"], "WAITING")

    def test_4h_full_label(self) -> None:
        row = {
            "timestamp": "2026-06-01 10:00:00",
            "symbol": "BTCUSDT",
            "side": "LONG",
            "policy_b_decision": "ENTER",
        }
        updated, status = label_shadow_row(
            row,
            now_kst_str="2026-06-01 15:00:00",
            cfg=self.cfg,
            klines_fetcher=self._fetch,
            hours=4,
        )
        self.assertEqual(status, "full")
        self.assertTrue(updated.get("actual_peak_roi"))
        self.assertTrue(updated.get("actual_dna_type"))


class FalseFlagTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = LabelerConfig()

    def test_false_skip(self) -> None:
        row = {"policy_b_decision": "SKIP"}
        labels = {"actual_roi_2h": "12.0", "actual_peak_roi": "8.0"}
        fs, fa = compute_false_flags(row, labels, self.cfg)
        self.assertEqual(fs, "1")
        self.assertEqual(fa, "0")

    def test_false_accept(self) -> None:
        row = {"policy_b_decision": "ENTER"}
        labels = {"actual_roi_2h": "-2.0", "actual_drawdown": "-12.0"}
        fs, fa = compute_false_flags(row, labels, self.cfg)
        self.assertEqual(fs, "0")
        self.assertEqual(fa, "1")


class IncrementalTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self._tmp.name)
        self.shadow_dir = self.data_dir / "runtime_shadow"
        self.shadow_dir.mkdir(parents=True)
        self.klines = make_klines(100.0, _flat_closes(100.0, 108.0, 16))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write_source(self, rows: list[dict]) -> None:
        path = self.shadow_dir / "value_gate_runtime_shadow.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=LOGGER_FIELDS, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in LOGGER_FIELDS})

    def _fetch(self, _sym: str, _s: int, _e: int) -> list:
        return self.klines

    def test_skip_already_labeled(self) -> None:
        row = {
            "timestamp": "2026-06-01 10:00:00",
            "scan_id": "2026-06-01 10:00:00|BTCUSDT|long",
            "symbol": "BTCUSDT",
            "side": "LONG",
            "policy_b_decision": "ENTER",
            "actual_roi_2h": "8.0",
            "actual_roi_4h": "8.0",
            "actual_peak_roi": "9.0",
            "actual_drawdown": "-1.0",
            "actual_dna_type": "TYPE_0",
            "false_skip": "0",
            "false_accept": "0",
        }
        self._write_source([row])
        labeled = self.shadow_dir / "value_gate_runtime_shadow_labeled.csv"
        with labeled.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=LOGGER_FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerow({k: row.get(k, "") for k in LOGGER_FIELDS})

        result = run_shadow_labeler(
            self.data_dir,
            now_kst_str="2026-06-01 16:00:00",
            klines_fetcher=self._fetch,
        )
        self.assertEqual(result["stats"]["skipped"], 1)
        self.assertEqual(result["stats"]["new_full"], 0)

    def test_force_relabel(self) -> None:
        row = {
            "timestamp": "2026-06-01 10:00:00",
            "scan_id": "2026-06-01 10:00:00|BTCUSDT|long",
            "symbol": "BTCUSDT",
            "side": "LONG",
            "policy_b_decision": "ENTER",
            "actual_roi_2h": "1.0",
            "actual_roi_4h": "1.0",
        }
        self._write_source([row])
        labeled = self.shadow_dir / "value_gate_runtime_shadow_labeled.csv"
        with labeled.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=LOGGER_FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerow({k: row.get(k, "") for k in LOGGER_FIELDS})

        result = run_shadow_labeler(
            self.data_dir,
            now_kst_str="2026-06-01 16:00:00",
            klines_fetcher=self._fetch,
            force=True,
        )
        out = list(csv.DictReader(labeled.open(encoding="utf-8")))
        self.assertNotEqual(out[0]["actual_roi_2h"], "1.0")
        self.assertAlmostEqual(float(out[0]["actual_roi_2h"]), 8.0, places=1)

    def test_summary_json(self) -> None:
        self._write_source([{
            "timestamp": "2026-06-01 10:00:00",
            "scan_id": "2026-06-01 10:00:00|BTCUSDT|long",
            "symbol": "BTCUSDT",
            "side": "LONG",
            "policy_b_decision": "SKIP",
        }])
        run_shadow_labeler(
            self.data_dir,
            now_kst_str="2026-06-01 16:00:00",
            klines_fetcher=self._fetch,
        )
        summary_path = self.shadow_dir / "value_gate_shadow_label_summary.json"
        self.assertTrue(summary_path.exists())
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertIn("false_skip_count", summary)
        self.assertIn("policy_enter_avg_roi_2h", summary)
        self.assertEqual(summary["labeled_4h_rows"], 1)


class SummaryBuilderTests(unittest.TestCase):
    def test_summary_fields(self) -> None:
        rows = [
            {"policy_b_decision": "ENTER", "side": "SHORT", "actual_roi_2h": "5", "false_accept": "1"},
            {"policy_b_decision": "SKIP", "side": "LONG", "actual_roi_2h": "12", "false_skip": "1"},
        ]
        s = build_summary(rows, LabelerConfig())
        self.assertEqual(s["false_skip_count"], 1)
        self.assertEqual(s["short_false_accept_count"], 1)


if __name__ == "__main__":
    unittest.main()
