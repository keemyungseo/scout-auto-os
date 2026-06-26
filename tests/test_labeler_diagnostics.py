"""Shadow labeler diagnostics and replay backfill tests."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(ROOT))

from scout_auto_os.engine.predator.labeler_diagnostics import (
    ReplaySources,
    classify_timestamp,
    diagnose_row,
    resolve_replay_labels,
    run_diagnostics,
    trade_key_from_row,
)
from scout_auto_os.engine.predator.shadow_labeler import (
    LabelerConfig,
    effective_scan_time,
    label_shadow_row,
    run_shadow_labeler,
)
from scout_auto_os.engine.predator.value_gate_shadow_logger import SHADOW_FIELDS


class TimestampDiagnosticsTests(unittest.TestCase):
    def test_parse_ok(self) -> None:
        row = {"timestamp": "2026-06-01 10:00:00", "scan_id": "2026-06-01 10:00:00|BTCUSDT|long"}
        diag, _, info = classify_timestamp(row, "2026-06-01 14:00:00")
        self.assertEqual(diag, "OK")
        self.assertTrue(info["can_label_2h"])

    def test_future_timestamp(self) -> None:
        row = {"timestamp": "2026-06-30 10:00:00", "scan_id": "2026-06-30 10:00:00|BTCUSDT|long"}
        diag, _, _ = classify_timestamp(row, "2026-06-01 10:00:00")
        self.assertEqual(diag, "FUTURE_TIMESTAMP")

    def test_too_recent(self) -> None:
        row = {"timestamp": "2026-06-01 13:00:00", "scan_id": "2026-06-01 13:00:00|BTCUSDT|long"}
        diag, _, _ = classify_timestamp(row, "2026-06-01 14:00:00")
        self.assertEqual(diag, "TOO_RECENT")

    def test_timestamp_mismatch_effective(self) -> None:
        row = {
            "timestamp": "2026-06-26 11:06:35",
            "scan_id": "2026-06-01 00:00:00|AIAUSDT|long",
        }
        self.assertEqual(effective_scan_time(row), "2026-06-01 00:00:00")


class ReplayJoinTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.pkg = Path(self._tmp.name)
        dna = self.pkg / "data" / "trade_dna"
        dna.mkdir(parents=True)
        cluster_path = dna / "trade_cluster.csv"
        with cluster_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=[
                "trade_key", "final_roi_2h", "final_roi_4h", "peak_roi", "max_drawdown",
                "trade_type_id", "roi_5m", "roi_120m",
            ])
            w.writeheader()
            w.writerow({
                "trade_key": "2026-06-01 00:00:00|AIAUSDT|long",
                "final_roi_2h": "8.5",
                "final_roi_4h": "10.0",
                "peak_roi": "12.0",
                "max_drawdown": "5.0",
                "trade_type_id": "TYPE_0",
                "roi_5m": "2.0",
                "roi_120m": "8.5",
            })
        value_path = dna / "value_prediction.csv"
        with value_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["trade_key", "symbol", "direction", "actual_expected_roi"])
            w.writeheader()
            w.writerow({
                "trade_key": "2026-06-01 00:00:00|AIAUSDT|long",
                "symbol": "AIAUSDT",
                "direction": "long",
                "actual_expected_roi": "8.5",
            })
        self.store = ReplaySources.discover(self.pkg)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_value_prediction_join(self) -> None:
        row = {"scan_id": "2026-06-01 00:00:00|AIAUSDT|long", "symbol": "AIAUSDT", "side": "LONG"}
        labels, source, key = resolve_replay_labels(row, self.store)
        self.assertIsNotNone(labels)
        self.assertIn(source, ("local_forward_klines", "trade_cluster", "value_prediction"))
        self.assertEqual(key, trade_key_from_row(row))

    def test_trade_cluster_join(self) -> None:
        labels, source, _ = resolve_replay_labels(
            {"scan_id": "2026-06-01 00:00:00|AIAUSDT|long", "symbol": "AIAUSDT", "side": "LONG"},
            self.store,
        )
        self.assertEqual(float(labels["actual_roi_2h"]), 8.5)
        self.assertEqual(labels["actual_dna_type"], "TYPE_0")


class ModeIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.pkg = Path(self._tmp.name)
        self.data = self.pkg / "data"
        shadow = self.data / "runtime_shadow"
        shadow.mkdir(parents=True)
        dna = self.data / "trade_dna"
        dna.mkdir(parents=True)
        row = {
            "timestamp": "2026-06-26 11:06:35",
            "scan_id": "2026-06-01 00:00:00|AIAUSDT|long",
            "symbol": "AIAUSDT",
            "side": "LONG",
            "policy_b_decision": "SKIP",
        }
        with (shadow / "value_gate_runtime_shadow.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=SHADOW_FIELDS, extrasaction="ignore")
            w.writeheader()
            full = {k: "" for k in SHADOW_FIELDS}
            full.update(row)
            w.writerow(full)
        with (dna / "trade_cluster.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=[
                "trade_key", "final_roi_2h", "final_roi_4h", "peak_roi", "max_drawdown",
                "trade_type_id", "roi_5m", "roi_240m",
            ])
            w.writeheader()
            w.writerow({
                "trade_key": "2026-06-01 00:00:00|AIAUSDT|long",
                "final_roi_2h": "12.0",
                "final_roi_4h": "11.0",
                "peak_roi": "16.0",
                "max_drawdown": "4.0",
                "trade_type_id": "TYPE_0",
                "roi_5m": "1.0",
                "roi_240m": "11.0",
            })

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_replay_mode_labels(self) -> None:
        result = run_shadow_labeler(self.data, mode="replay", pkg_root=self.pkg)
        labeled = list(csv.DictReader(
            (self.data / "runtime_shadow" / "value_gate_runtime_shadow_labeled.csv").open(encoding="utf-8")
        ))
        self.assertEqual(result["summary"]["labeled_4h_rows"], 1)
        self.assertEqual(float(labeled[0]["actual_roi_2h"]), 12.0)
        self.assertEqual(labeled[0]["false_skip"], "1")

    def test_live_mode_waiting(self) -> None:
        result = run_shadow_labeler(
            self.data,
            mode="live",
            now_kst_str="2026-06-01 01:00:00",
            pkg_root=self.pkg,
            klines_fetcher=lambda *_: [],
        )
        self.assertEqual(result["summary"]["waiting_rows"], 1)

    def test_diagnostics_csv(self) -> None:
        run_diagnostics(self.data, self.pkg, mode="replay", now_s="2026-06-01 14:00:00")
        path = self.data / "runtime_shadow" / "labeler_diagnostics.csv"
        self.assertTrue(path.exists())
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        self.assertEqual(rows[0]["diagnosis"], "REPLAY_JOIN_OK")

    def test_summary_json(self) -> None:
        run_diagnostics(self.data, self.pkg, mode="replay")
        summary = json.loads(
            (self.data / "runtime_shadow" / "labeler_diagnostics_summary.json").read_text(encoding="utf-8")
        )
        self.assertIn("diagnosis_counts", summary)
        self.assertEqual(summary["replay_join_ok"], 1)


class KlineEmptyTests(unittest.TestCase):
    def test_empty_kline_waiting(self) -> None:
        row = {
            "timestamp": "2026-06-01 10:00:00",
            "scan_id": "2026-06-01 10:00:00|BTCUSDT|long",
            "symbol": "BTCUSDT",
            "side": "LONG",
            "policy_b_decision": "ENTER",
        }
        updated, status = label_shadow_row(
            row,
            now_kst_str="2026-06-01 16:00:00",
            cfg=LabelerConfig(),
            klines_fetcher=lambda *_: [],
        )
        self.assertEqual(status, "waiting")


class MissingEntryPriceTests(unittest.TestCase):
    def test_missing_entry_in_diagnosis(self) -> None:
        store = ReplaySources.discover(Path(__file__).resolve().parents[1])
        d = diagnose_row(
            {
                "timestamp": "2026-06-01 10:00:00",
                "scan_id": "2026-06-01 10:00:00|NOPE|long",
                "symbol": "NOPE",
                "side": "LONG",
                "policy_b_decision": "ENTER",
            },
            now_s="2026-06-01 16:00:00",
            mode="live",
            store=store,
            cfg=LabelerConfig(),
            klines_fetcher=lambda *_: [],
        )
        self.assertEqual(d["diagnosis"], "SYMBOL_FORMAT_ERROR")


if __name__ == "__main__":
    unittest.main()
