"""Runtime shadow timestamp fix tests."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(ROOT))

from scout_auto_os.engine.predator.runtime_shadow import (
    ValueGateRuntimeShadow,
    make_scan_id,
)
from scout_auto_os.engine.predator.shadow_labeler import run_shadow_labeler
from scout_auto_os.engine.predator.timestamp_fix import (
    resolve_replay_timestamp,
    run_timestamp_validation,
    validate_shadow_timestamps,
)
from scout_auto_os.engine.predator.value_gate_shadow_logger import SHADOW_FIELDS


def _setup_data_dir(base: Path) -> Path:
    policy_dir = base / "value_gate_policy"
    policy_dir.mkdir(parents=True)
    (policy_dir / "recommended_policy.json").write_text(
        json.dumps({"policy": "B", "policy_name": "Soft 50s"}),
        encoding="utf-8",
    )
    dna = base / "trade_dna"
    dna.mkdir(parents=True)
    cluster_fields = [
        "trade_key", "final_roi_2h", "final_roi_4h", "peak_roi", "max_drawdown",
        "trade_type_id", "roi_5m", "roi_240m", "symbol", "direction",
    ]
    rows = [
        {
            "trade_key": "2026-06-01 00:00:00|AIAUSDT|long",
            "symbol": "AIAUSDT", "direction": "long",
            "final_roi_2h": "8.0", "final_roi_4h": "9.0", "peak_roi": "10.0",
            "max_drawdown": "3.0", "trade_type_id": "TYPE_0", "roi_5m": "1.0", "roi_240m": "9.0",
        },
        {
            "trade_key": "2026-06-01 02:00:00|STGUSDT|long",
            "symbol": "STGUSDT", "direction": "long",
            "final_roi_2h": "5.0", "final_roi_4h": "6.0", "peak_roi": "7.0",
            "max_drawdown": "2.0", "trade_type_id": "TYPE_0", "roi_5m": "0.5", "roi_240m": "6.0",
        },
    ]
    with (dna / "trade_cluster.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cluster_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with (dna / "value_prediction.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["trade_key", "symbol", "direction", "value_score"])
        w.writeheader()
        for r in rows:
            w.writerow({
                "trade_key": r["trade_key"],
                "symbol": r["symbol"],
                "direction": r["direction"],
                "value_score": "60",
            })
    return base


class ReplayTimestampTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.data_dir = _setup_data_dir(Path(self._tmp.name))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    @patch("scout_auto_os.engine.predator.runtime_shadow.now_kst")
    def test_replay_does_not_use_now(self, mock_now) -> None:
        mock_now.return_value = "2099-01-01 00:00:00"
        shadow = ValueGateRuntimeShadow(self.data_dir, mode="replay")
        replay_rows = [
            {"trade_key": "2026-06-01 00:00:00|AIAUSDT|long", "symbol": "AIAUSDT", "direction": "long"},
        ]
        logged, skipped = shadow.replay_backfill(replay_rows)
        self.assertEqual(len(logged), 1)
        self.assertEqual(logged[0]["timestamp"], "2026-06-01 00:00:00")
        self.assertNotEqual(logged[0]["timestamp"], "2099-01-01 00:00:00")

    def test_live_uses_scan_timestamp(self) -> None:
        shadow = ValueGateRuntimeShadow(self.data_dir, mode="live")
        row = shadow.record_candidate("2026-06-02 08:00:00", {"symbol": "BTCUSDT"}, side="long")
        self.assertIsNotNone(row)
        self.assertEqual(row["timestamp"], "2026-06-02 08:00:00")

    def test_original_timestamp_recorded(self) -> None:
        shadow = ValueGateRuntimeShadow(self.data_dir, mode="replay")
        row = shadow.record_candidate(
            "2026-06-01 02:00:00",
            {"symbol": "STGUSDT", "trade_key": "2026-06-01 02:00:00|STGUSDT|long"},
            side="long",
        )
        self.assertEqual(row["timestamp"], "2026-06-01 02:00:00")
        self.assertEqual(row["scan_id"], "2026-06-01 02:00:00|STGUSDT|long")

    def test_missing_timestamp_skipped(self) -> None:
        shadow = ValueGateRuntimeShadow(self.data_dir, mode="replay")
        _, skipped = shadow.replay_backfill([{"symbol": "NOPE", "direction": "long"}])
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0]["diagnosis"], "MISSING_ORIGINAL_TIMESTAMP")

    def test_scan_id_deterministic(self) -> None:
        a = make_scan_id("2026-06-01 10:00:00", "BTCUSDT", "LONG")
        b = make_scan_id("2026-06-01 10:00:00", "BTCUSDT", "LONG")
        self.assertEqual(a, b)
        self.assertEqual(a, "2026-06-01 10:00:00|BTCUSDT|long")

    def test_replay_twice_same_scan_id(self) -> None:
        shadow = ValueGateRuntimeShadow(self.data_dir, mode="replay")
        row1, _ = shadow.replay_backfill([
            {"trade_key": "2026-06-01 00:00:00|AIAUSDT|long", "symbol": "AIAUSDT", "direction": "long"},
        ])
        id1 = row1[0]["scan_id"]
        shadow2 = ValueGateRuntimeShadow(self.data_dir, mode="replay")
        row2, _ = shadow2.replay_backfill([
            {"trade_key": "2026-06-01 00:00:00|AIAUSDT|long", "symbol": "AIAUSDT", "direction": "long"},
        ])
        self.assertEqual(row2[0]["scan_id"], id1)


class ValidationTests(unittest.TestCase):
    def test_all_same_timestamps_fail(self) -> None:
        rows = [
            {"timestamp": "2026-06-26 11:06:35", "scan_id": "a"},
            {"timestamp": "2026-06-26 11:06:35", "scan_id": "b"},
        ]
        v = validate_shadow_timestamps(rows, now_s="2026-06-30 00:00:00")
        self.assertFalse(v["ok"])
        self.assertTrue(v["all_timestamps_identical"])

    def test_future_timestamp_fail(self) -> None:
        rows = [{"timestamp": "2026-12-01 00:00:00", "scan_id": "x"}]
        v = validate_shadow_timestamps(rows, now_s="2026-06-01 00:00:00")
        self.assertFalse(v["ok"])
        self.assertEqual(v["future_timestamp_count"], 1)

    def test_valid_replay_timestamps(self) -> None:
        rows = [
            {"timestamp": "2026-06-01 00:00:00", "scan_id": "a"},
            {"timestamp": "2026-06-01 02:00:00", "scan_id": "b"},
        ]
        v = validate_shadow_timestamps(rows, now_s="2026-06-30 00:00:00")
        self.assertTrue(v["ok"])
        self.assertEqual(v["unique_timestamp_count"], 2)

    def test_resolve_replay_timestamp_priority(self) -> None:
        self.assertEqual(
            resolve_replay_timestamp({"scan_kst": "2026-06-01 00:00:00", "trade_key": "x|Y|long"}),
            "2026-06-01 00:00:00",
        )
        self.assertIsNone(resolve_replay_timestamp({"symbol": "X"}))


class LabelerAfterFixTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.pkg = Path(self._tmp.name)
        self.data = self.pkg / "data"
        _setup_data_dir(self.data)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_labeler_labels_fixed_timestamps(self) -> None:
        shadow = ValueGateRuntimeShadow(self.data, mode="replay")
        shadow.replay_backfill([
            {"trade_key": "2026-06-01 00:00:00|AIAUSDT|long", "symbol": "AIAUSDT", "direction": "long"},
        ])
        result = run_shadow_labeler(self.data, mode="replay", force=True, pkg_root=self.pkg)
        self.assertEqual(result["summary"]["labeled_4h_rows"], 1)

    def test_diagnostics_output(self) -> None:
        shadow_dir = self.data / "runtime_shadow"
        shadow_dir.mkdir(parents=True, exist_ok=True)
        with (shadow_dir / "value_gate_runtime_shadow.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=SHADOW_FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerow({k: "" for k in SHADOW_FIELDS} | {
                "timestamp": "2026-06-01 00:00:00",
                "scan_id": "2026-06-01 00:00:00|AIAUSDT|long",
                "symbol": "AIAUSDT",
                "side": "LONG",
            })
        run_timestamp_validation(shadow_dir, now_s="2026-06-30 00:00:00")
        self.assertTrue((shadow_dir / "timestamp_fix_report.md").exists())
        self.assertTrue((shadow_dir / "timestamp_diagnostics.csv").exists())


if __name__ == "__main__":
    unittest.main()
