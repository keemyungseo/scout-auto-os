"""V1.2 review layer dry-run tests (paper-safe)."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from scout_auto_os.engine.alert_manager import AlertManager
from scout_auto_os.engine.daily_trade_report import DailyTradeReportService
from scout_auto_os.engine.position_report import PositionReportService
from scout_auto_os.engine.review_layer import ReviewLayer
from scout_auto_os.engine.telegram_commands import TelegramCommandBot
from scout_auto_os.engine.trade_record import TradeRecordService
from scout_auto_os.engine.binance_client import BinanceClient
from scout_auto_os.storage.db import Database


def main() -> None:
    td = Path(tempfile.mkdtemp(prefix="scout_v12_"))
    data_dir = td / "data"
    data_dir.mkdir()
    cfg = {
        "paper_mode": True,
        "execution": {"order_size_usdt": 7, "leverage": 3},
        "alerts": {"telegram": False},
        "risk": {"kill_switch": False},
        "position": {"max_long_slots": 2},
    }
    trade_rec = TradeRecordService(data_dir / "trades.db", 7, 3)
    pos_rep = PositionReportService(BinanceClient())
    review = ReviewLayer(data_dir, trade_rec, pos_rep, lambda s: 100.0, cfg)
    os_db = Database(td / "os.db")
    alerts = AlertManager(cfg, os_db, td / "logs")

    top5 = [
        {"rank": 1, "symbol": "AAAUSDT", "a6_score": 6.5, "expected_ev": 1.2,
         "reason": "1h=Acceleration 2h=TrendAlive | 1h_rng=8.0% 30m_ret=2.0%",
         "entry_price": 10.0, "volume_state": "ok", "trend_alive": "alive", "acceleration": "acc"},
        {"rank": 2, "symbol": "BBBUSDT", "a6_score": 5.1, "expected_ev": 0.8,
         "reason": "1h=ExpansionStart 2h=TrendAlive | 1h_rng=5.0% 30m_ret=1.0%",
         "entry_price": 20.0, "volume_state": "weak", "trend_alive": "alive", "acceleration": "exp"},
    ]
    review.prepare_scan(top5, set(), set(), 2, False)
    review.complete_scan("2026-06-24 10:00:00", top5, {"AAAUSDT"}, set(), set(), 2, False)

    trade_rec.set_entry_context("AAAUSDT", {"scan_rank": 1, "score": 6.5, "top5_snapshot_id": "x"})
    trade_rec.record_open("AAAUSDT", "LONG", 10.0, 0.7, "A6_test")
    trade_rec.record_close("AAAUSDT", 10.5, "efr_exit", 5.0)

    review.capture_positions([
        {"symbol": "AAAUSDT", "side": "LONG", "entry_price": 10, "current_price": 10.2, "unrealized_pnl_pct": 2.0},
    ])
    review.update_snapshot()

    dtr = DailyTradeReportService(cfg, trade_rec.db, pos_rep, alerts, os_db, td / "logs", data_dir, review.get_snapshot)
    report = dtr.build_report("2026-06-24")
    print(report[:400])

    bot = TelegramCommandBot(cfg, data_dir, trade_rec.db, pos_rep, review.get_snapshot, os_db, lambda: {"bot_state": "test"})
    print(bot._cmd_status())
    print(bot._cmd_report())

    snap = json.loads((data_dir / "review_snapshot.json").read_text(encoding="utf-8"))
    assert snap["total_scans"] >= 1
    assert (data_dir / "scans.csv").exists()
    assert (data_dir / "scans.jsonl").exists()
    assert (data_dir / "positions_snapshot.json").exists()
    print("V1.2 review tests PASSED")


if __name__ == "__main__":
    main()
