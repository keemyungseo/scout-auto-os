"""Value Gate runtime shadow V1 tests."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(ROOT))

from scout_auto_os.engine.predator.policies import policy_b_soft_50s
from scout_auto_os.engine.predator.runtime_shadow import (
    ValueGateRuntimeShadow,
    evaluate_policy_b_shadow,
    load_recommended_policy,
)
from scout_auto_os.engine.predator.short_watch import short_watch_match
from scout_auto_os.engine.predator.value_gate import GateAction


class PolicyBShadowTests(unittest.TestCase):
    def _row(self, score: float, runner: float = 0.9, dna: str = "TYPE_0") -> dict:
        return {
            "value_score": score,
            "runner_probability": runner,
            "predicted_dna_type": dna,
            "predicted_drawdown": 10,
            "predicted_win_prob": 0.8,
        }

    def test_policy_b_50s_enter(self) -> None:
        g = policy_b_soft_50s(self._row(55))
        self.assertEqual(g["action"], GateAction.ENTER.value)
        self.assertEqual(g["recommended_size"], 0.2)

    def test_policy_b_skip_low_score(self) -> None:
        g = policy_b_soft_50s(self._row(40))
        self.assertEqual(g["action"], GateAction.SKIP.value)

    def test_policy_b_skip_type1(self) -> None:
        g = policy_b_soft_50s(self._row(80, runner=0.2, dna="TYPE_1"))
        self.assertEqual(g["action"], GateAction.SKIP.value)

    def test_manual_no_action(self) -> None:
        g = evaluate_policy_b_shadow(
            self._row(90),
            manual_context={"source": "MANUAL", "manual_lock": 1, "auto_manage": 0},
        )
        self.assertEqual(g["action"], GateAction.NO_ACTION.value)


class ShortWatchTests(unittest.TestCase):
    def test_short_watch_match(self) -> None:
        self.assertTrue(short_watch_match("SHORT", "ENTER", 65, "TYPE_0", 0.55))
        self.assertFalse(short_watch_match("LONG", "ENTER", 65, "TYPE_0", 0.55))
        self.assertFalse(short_watch_match("SHORT", "ENTER", 65, "TYPE_0", 0.75))


class RuntimeShadowIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self._tmp.name)
        policy_dir = self.data_dir / "value_gate_policy"
        policy_dir.mkdir(parents=True)
        (policy_dir / "recommended_policy.json").write_text(
            json.dumps({"policy": "B", "policy_name": "Soft 50s"}),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_load_policy_b(self) -> None:
        rec = load_recommended_policy(self.data_dir)
        self.assertEqual(rec["policy"], "B")

    def test_shadow_log_and_summary(self) -> None:
        from scout_auto_os.storage.db import now_kst

        shadow = ValueGateRuntimeShadow(self.data_dir, enabled=True, mode="live")
        shadow.record_candidate(
            now_kst(),
            {"symbol": "BTCUSDT"},
            side="long",
            can_enter=True,
        )
        path = self.data_dir / "runtime_shadow" / "value_gate_runtime_shadow.csv"
        self.assertTrue(path.exists())
        summary = shadow.refresh_summary()
        self.assertEqual(summary["policy_name"], "Soft 50s")
        self.assertGreaterEqual(summary["total_candidates_today"], 1)
        sj = self.data_dir / "runtime_shadow" / "value_gate_shadow_summary.json"
        self.assertTrue(sj.exists())

    def test_short_watch_logged(self) -> None:
        shadow = ValueGateRuntimeShadow(self.data_dir, enabled=True)
        shadow.pred_cache.lookup = lambda prediction_key, **_kw: {
            "value_score": 65,
            "runner_probability": 0.55,
            "predicted_dna_type": "TYPE_0",
            "predicted_roi": 10,
            "predicted_peak_roi": 15,
            "predicted_drawdown": 8,
            "predicted_win_prob": 0.7,
            "entry_score": 60,
            "prediction_source": "test",
        }
        shadow.record_candidate(
            "2026-06-01 12:00:00",
            {"symbol": "CLOUSDT"},
            side="short",
            can_enter=True,
        )
        watch_path = self.data_dir / "runtime_shadow" / "short_false_accept_watch.csv"
        self.assertTrue(watch_path.exists())
        self.assertIn("CLOUSDT", watch_path.read_text(encoding="utf-8"))

    def test_no_execution_engine_call(self) -> None:
        shadow = ValueGateRuntimeShadow(self.data_dir, enabled=True)
        execution = MagicMock()
        shadow.on_scan(
            "2026-06-01 10:00:00",
            [{"symbol": "ETHUSDT", "side": "long"}],
            occupied=set(),
            locked=set(),
            can_enter=True,
        )
        execution.paper_entry.assert_not_called()
        execution.paper_exit.assert_not_called()

    def test_wld_manual_lock_no_action(self) -> None:
        shadow = ValueGateRuntimeShadow(self.data_dir, enabled=True)
        row = shadow.record_candidate(
            "2026-06-01 10:00:00",
            {"symbol": "WLDUSDT"},
            side="long",
            locked={"WLDUSDT"},
            manual_context={"source": "MANUAL", "manual_lock": 1, "auto_manage": 0},
        )
        self.assertEqual(row["policy_b_decision"], GateAction.NO_ACTION.value)
        self.assertEqual(row["baseline_decision"], GateAction.NO_ACTION.value)

    def test_join_keys_present(self) -> None:
        shadow = ValueGateRuntimeShadow(self.data_dir, enabled=True)
        row = shadow.record_candidate("2026-06-01 10:00:00", {"symbol": "XRPUSDT"}, side="long")
        self.assertIn("scan_id", row)
        self.assertIn("2026-06-01 10:00:00|XRPUSDT|long", row["scan_id"])


if __name__ == "__main__":
    unittest.main()
