"""Guardian Position Panel tests."""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(ROOT))

from scout_auto_os.engine.control.control_api import ControlService, create_control_app
from scout_auto_os.engine.control.dashboard import load_template
from scout_auto_os.engine.control.manual_close import DryRunCloseExecutor
from scout_auto_os.engine.control.position_status import build_guardian_positions
from scout_auto_os.storage.db import Database, now_kst

fastapi = unittest.skipUnless(
    __import__("importlib").util.find_spec("fastapi") is not None,
    "fastapi not installed",
)


def _seed_review_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "timestamp", "thesis_id", "position_id", "symbol", "side", "source", "auto_manage",
        "entry_time", "entry_price", "current_price", "roi", "elapsed_minutes",
        "expected_horizon", "expected_return", "success_probability", "mfe", "mae",
        "peak_roi", "drawdown_from_peak", "thesis_validity_score", "exit_pressure_score",
        "hold_confidence", "action", "action_reason", "thesis_update_reason", "should_exit",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def _seed_expectation_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "timestamp", "thesis_id", "position_id", "symbol", "side", "elapsed_minutes",
        "expected_roi", "current_roi", "expected_progress", "progress_ratio",
        "progress_delta", "expectation_score", "expectation_label", "thesis_state",
        "thesis_version", "curve_version", "thesis_transition_reason",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


class PositionStatusTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.control_dir = self.root / "control"
        self.data_dir = self.root / "data"
        self.control_dir.mkdir()
        self.data_dir.mkdir()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _svc(self) -> ControlService:
        return ControlService(
            self.control_dir,
            executor=DryRunCloseExecutor(self.control_dir),
            data_dir=self.data_dir,
        )

    def test_empty_positions_safe(self) -> None:
        payload = build_guardian_positions(self.data_dir, self.control_dir)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["count"], 0)
        self.assertEqual(payload["positions"], [])

    def test_wld_manual_lock(self) -> None:
        self._svc().manual_lock("WLDUSDT", "test")
        payload = build_guardian_positions(self.data_dir, self.control_dir)
        wld = next(p for p in payload["positions"] if p["symbol"] == "WLDUSDT")
        self.assertEqual(wld["source"], "MANUAL")
        self.assertFalse(wld["auto_manage"])
        self.assertTrue(wld["manual_lock"])
        self.assertEqual(wld["guardian_action"], "NO_ACTION_MANUAL_POSITION")

    def test_bot_position_from_db_and_review(self) -> None:
        db = Database(self.data_dir / "trades.db")
        db.execute(
            """INSERT INTO positions
            (position_id,symbol,side,source,engine,entry_time,entry_price,current_price,
             unrealized_pnl_pct,status,manual_lock,auto_manage,last_update_time,thesis_id)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            ("p1", "BTCUSDT", "LONG", "AUTO", "A6_LONG", "2026-06-01 10:00:00", 100, 103, 3.0,
             "OPEN", 0, 1, now_kst(), "th_btc"),
        )
        db.conn.commit()
        db.conn.close()
        _seed_review_csv(self.data_dir / "position_evaluation" / "position_review.csv", [{
            "timestamp": now_kst(), "thesis_id": "th_btc", "position_id": "p1", "symbol": "BTCUSDT",
            "side": "LONG", "source": "AUTO", "auto_manage": 1, "entry_time": "2026-06-01 10:00:00",
            "entry_price": 100, "current_price": 103, "roi": 3, "elapsed_minutes": 45,
            "expected_horizon": 120, "expected_return": 5, "mfe": 4, "mae": -1,
            "peak_roi": 4, "drawdown_from_peak": 1, "exit_pressure_score": 20,
            "hold_confidence": 70, "action": "HOLD", "action_reason": "ok",
        }])
        p = build_guardian_positions(self.data_dir, self.control_dir)["positions"][0]
        self.assertEqual(p["source"], "BOT")
        self.assertEqual(p["elapsed_minutes"], 45)
        self.assertEqual(p["expected_horizon"], 120)

    def test_exit_ready_and_expectation_fields(self) -> None:
        _seed_review_csv(self.data_dir / "position_evaluation" / "position_review.csv", [{
            "timestamp": now_kst(), "thesis_id": "th_met", "position_id": "p2", "symbol": "METUSDT",
            "side": "LONG", "source": "AUTO", "auto_manage": 1, "entry_time": "2026-06-01 08:00:00",
            "entry_price": 1, "current_price": 1.02, "roi": 2, "elapsed_minutes": 300,
            "expected_horizon": 120, "action": "EXIT", "action_reason": "horizon exceeded",
        }])
        _seed_expectation_csv(self.data_dir / "expectation" / "expectation_review.csv", [{
            "timestamp": now_kst(), "thesis_id": "th_met", "position_id": "p2", "symbol": "METUSDT",
            "side": "LONG", "elapsed_minutes": 300, "expected_roi": 3, "current_roi": 2,
            "expected_progress": 2.5, "progress_ratio": 85, "expectation_score": 42,
            "thesis_state": "EXIT_READY",
        }])
        p = build_guardian_positions(self.data_dir, self.control_dir)["positions"][0]
        self.assertEqual(p["thesis_state"], "EXIT_READY")
        self.assertEqual(p["expectation_score"], 42.0)
        self.assertEqual(p["progress_ratio"], 85.0)
        self.assertIn(p["guardian_action"], ("EXIT", "EXIT_READY"))


@fastapi
class GuardianPanelAPITests(unittest.TestCase):
    def setUp(self) -> None:
        from fastapi.testclient import TestClient

        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.control_dir = self.root / "control"
        self.data_dir = self.root / "data"
        self.control_dir.mkdir()
        self.data_dir.mkdir()
        self.svc = ControlService(
            self.control_dir,
            executor=DryRunCloseExecutor(self.control_dir),
            data_dir=self.data_dir,
        )
        self.client = TestClient(create_control_app(self.svc))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_positions_api(self) -> None:
        r = self.client.get("/control/positions")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(data["ok"])
        self.assertIn("positions", data)
        self.assertTrue(data["dry_run"])

    def test_html_has_guardian_panel(self) -> None:
        html = load_template()
        self.assertIn("Guardian Position Panel", html)
        self.assertIn("guardian-position-panel", html)
        self.assertIn("/control/positions", html)


if __name__ == "__main__":
    unittest.main()
