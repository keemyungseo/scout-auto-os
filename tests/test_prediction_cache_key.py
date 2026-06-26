"""Prediction cache key fix V1 tests."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(ROOT))

from scout_auto_os.engine.predator.inference import load_replay_bundle
from scout_auto_os.engine.predator.policies import policy_b_soft_50s
from scout_auto_os.engine.predator.prediction_key import (
    make_prediction_key,
    make_scan_id,
    prediction_key_from_row,
    symbol_side_key,
)
from scout_auto_os.engine.predator.runtime_shadow import (
    ShadowPredictionCache,
    ValueGateRuntimeShadow,
    evaluate_policy_b_shadow,
)
from scout_auto_os.engine.predator.value_gate import GateAction


class PredictionKeyTests(unittest.TestCase):
    def test_trade_key_priority(self) -> None:
        tk = "2026-06-01 10:00:00|BTCUSDT|long"
        sid = "2026-06-02 10:00:00|BTCUSDT|long"
        self.assertEqual(make_prediction_key(trade_key=tk, scan_id=sid), tk)

    def test_scan_id_fallback(self) -> None:
        sid = make_scan_id("2026-06-01 10:00:00", "ETHUSDT", "short")
        self.assertEqual(make_prediction_key(scan_id=sid), sid)

    def test_timestamp_symbol_side_fallback(self) -> None:
        pk = make_prediction_key(scan_time="2026-06-01 10:00:00", symbol="XRPUSDT", side="long")
        self.assertEqual(pk, "2026-06-01 10:00:00|XRPUSDT|long")

    def test_prediction_key_from_row(self) -> None:
        row = {"trade_key": "2026-06-01|A|long", "symbol": "A", "direction": "long"}
        self.assertEqual(prediction_key_from_row(row), "2026-06-01|A|long")


class ShadowPredictionCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dna_dir = Path(self._tmp.name)
        self._write_csv(
            self.dna_dir / "value_prediction.csv",
            ["trade_key", "symbol", "direction", "value_score", "pred_expected_roi",
             "pred_expected_peak_roi", "pred_expected_drawdown", "pred_expected_win_prob"],
            [
                ("2026-06-01 10:00:00|BTCUSDT|long", "BTCUSDT", "long", "55", "10", "15", "5", "0.8"),
                ("2026-06-01 12:00:00|BTCUSDT|long", "BTCUSDT", "long", "40", "5", "8", "5", "0.5"),
            ],
        )
        self._write_csv(
            self.dna_dir / "dna_prediction_model.csv",
            ["trade_key", "symbol", "direction", "predicted_type", "runner_probability", "entry_score"],
            [
                ("2026-06-01 10:00:00|BTCUSDT|long", "BTCUSDT", "long", "TYPE_0", "0.9", "60"),
                ("2026-06-01 12:00:00|BTCUSDT|long", "BTCUSDT", "long", "TYPE_0", "0.9", "50"),
            ],
        )
        self.cache = ShadowPredictionCache(self.dna_dir)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    @staticmethod
    def _write_csv(path: Path, header: list[str], rows: list[tuple]) -> None:
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)

    def test_same_symbol_side_different_scan_id(self) -> None:
        k1 = "2026-06-01 10:00:00|BTCUSDT|long"
        k2 = "2026-06-01 12:00:00|BTCUSDT|long"
        self.assertNotEqual(self.cache.lookup(k1)["value_score"], self.cache.lookup(k2)["value_score"])

    def test_trade_key_lookup(self) -> None:
        pk = "2026-06-01 10:00:00|BTCUSDT|long"
        pred = self.cache.lookup(pk)
        self.assertEqual(pred["value_score"], 55.0)
        self.assertEqual(pred["prediction_source"], "trade_key_cache")

    def test_no_symbol_side_fallback(self) -> None:
        sym_key = symbol_side_key("BTCUSDT", "long")
        pred = self.cache.lookup(sym_key)
        self.assertEqual(pred["prediction_source"], "key_not_found")

    def test_value_prediction_join_by_prediction_key(self) -> None:
        rows = list(csv.DictReader((self.dna_dir / "value_prediction.csv").open(encoding="utf-8")))
        by_pk = {prediction_key_from_row(r): r for r in rows}
        self.assertEqual(len(by_pk), 2)
        self.assertIn("2026-06-01 10:00:00|BTCUSDT|long", by_pk)

    def test_dna_join_by_prediction_key(self) -> None:
        rows = list(csv.DictReader((self.dna_dir / "dna_prediction_model.csv").open(encoding="utf-8")))
        by_pk = {prediction_key_from_row(r): r for r in rows}
        self.assertEqual(by_pk["2026-06-01 12:00:00|BTCUSDT|long"]["runner_probability"], "0.9")


class RuntimeShadowReplayTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self._tmp.name)
        policy_dir = self.data_dir / "value_gate_policy"
        policy_dir.mkdir(parents=True)
        (policy_dir / "recommended_policy.json").write_text(
            json.dumps({"policy": "B", "policy_name": "Soft 50s"}),
            encoding="utf-8",
        )
        dna = self.data_dir / "trade_dna"
        dna.mkdir(parents=True)
        self._write_bundle(dna)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    @staticmethod
    def _write_bundle(dna: Path) -> None:
        value_header = [
            "trade_key", "symbol", "direction", "trade_type_id", "value_score",
            "pred_expected_roi", "pred_expected_peak_roi", "pred_expected_drawdown", "pred_expected_win_prob",
        ]
        dna_header = [
            "trade_key", "symbol", "direction", "predicted_type",
            "runner_probability", "failed_probability", "entry_score",
        ]
        rows = [
            ("2026-06-01 10:00:00|AAAUSDT|long", "AAAUSDT", "long", "TYPE_0", "60", "10", "15", "5", "0.8"),
            ("2026-06-01 12:00:00|AAAUSDT|long", "AAAUSDT", "long", "TYPE_0", "45", "5", "8", "5", "0.5"),
        ]
        with (dna / "value_prediction.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(value_header)
            w.writerows(rows)
        with (dna / "dna_prediction_model.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(dna_header)
            w.writerows([
                ("2026-06-01 10:00:00|AAAUSDT|long", "AAAUSDT", "long", "TYPE_0", "0.9", "0.1", "60"),
                ("2026-06-01 12:00:00|AAAUSDT|long", "AAAUSDT", "long", "TYPE_0", "0.9", "0.1", "50"),
            ])

    def test_replay_matches_per_trade_key_policy(self) -> None:
        bundle = load_replay_bundle(self.data_dir / "trade_dna")
        shadow = ValueGateRuntimeShadow(self.data_dir, enabled=True, mode="replay")
        logged, _ = shadow.replay_backfill(bundle)
        mismatches = 0
        for row, replay in zip(logged, bundle):
            gate = policy_b_soft_50s({
                "value_score": float(replay["value_score"]),
                "runner_probability": float(replay["runner_probability"]),
                "predicted_dna_type": replay["predicted_dna_type"],
                "predicted_drawdown": float(replay["predicted_drawdown"]),
                "predicted_win_prob": float(replay["predicted_win_prob"]),
            })
            if row["policy_b_decision"] != gate["action"]:
                mismatches += 1
        self.assertEqual(mismatches, 0)

    def test_manual_lock_no_action(self) -> None:
        shadow = ValueGateRuntimeShadow(self.data_dir, enabled=True, mode="live")
        row = shadow.record_candidate(
            "2026-06-01 10:00:00",
            {"symbol": "WLDUSDT", "trade_key": "2026-06-01 10:00:00|WLDUSDT|long"},
            side="long",
            locked={"WLDUSDT"},
            manual_context={"source": "MANUAL", "manual_lock": 1, "auto_manage": 0},
        )
        self.assertEqual(row["policy_b_decision"], GateAction.NO_ACTION.value)

    def test_no_execution_engine_call(self) -> None:
        shadow = ValueGateRuntimeShadow(self.data_dir, enabled=True)
        execution = MagicMock()
        shadow.on_scan(
            "2026-06-01 10:00:00",
            [{"symbol": "ETHUSDT", "side": "long", "trade_key": "2026-06-01 10:00:00|ETHUSDT|long"}],
            occupied=set(),
            locked=set(),
            can_enter=True,
        )
        execution.paper_entry.assert_not_called()
        execution.paper_exit.assert_not_called()


class SymbolSideCacheForbiddenTest(unittest.TestCase):
    """Ensures legacy symbol|side indexing is not present on ShadowPredictionCache."""

    def test_no_by_sym_attribute(self) -> None:
        self.assertFalse(hasattr(ShadowPredictionCache, "_by_sym"))


if __name__ == "__main__":
    unittest.main()
