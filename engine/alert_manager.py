"""Console + CSV alerts (Telegram/Discord hooks)."""

from __future__ import annotations

import csv
import os
import urllib.parse
import urllib.request
from pathlib import Path

from scout_auto_os.storage.db import Database, now_kst


class AlertManager:
    def __init__(self, config: dict, db: Database, csv_dir: Path) -> None:
        self.config = config
        self.db = db
        self.csv_dir = csv_dir
        self.csv_dir.mkdir(parents=True, exist_ok=True)
        self.console = bool(config["alerts"].get("console", True))
        self.csv_enabled = bool(config["alerts"].get("csv", True))
        self.telegram = bool(config["alerts"].get("telegram", False))

    def send_telegram_report(self, message: str) -> bool:
        """Daily report only — does not duplicate console TOP5/ENTRY/EXIT alerts."""
        if not self.telegram:
            return False
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
        if not token or not chat_id:
            return False
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        data = urllib.parse.urlencode({
            "chat_id": chat_id,
            "text": message[:4000],
        }).encode()
        try:
            req = urllib.request.Request(url, data=data, method="POST")
            urllib.request.urlopen(req, timeout=10)
            return True
        except Exception:
            return False

    def _telegram_send(self, message: str) -> None:
        if not self.telegram:
            return
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
        if not token or not chat_id:
            return
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        data = urllib.parse.urlencode({
            "chat_id": chat_id,
            "text": message[:4000],
        }).encode()
        try:
            req = urllib.request.Request(url, data=data, method="POST")
            urllib.request.urlopen(req, timeout=10)
        except Exception:
            pass

    def _emit(self, alert_type: str, message: str) -> None:
        ts = now_kst()
        if self.console:
            print(message)
        self._telegram_send(message)
        self.db.execute(
            "INSERT INTO alerts (timestamp, alert_type, message) VALUES (?,?,?)",
            (ts, alert_type, message),
        )
        if self.csv_enabled:
            path = self.csv_dir / "alerts.csv"
            write_header = not path.exists()
            with path.open("a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["timestamp", "alert_type", "message"])
                if write_header:
                    w.writeheader()
                w.writerow({"timestamp": ts, "alert_type": alert_type, "message": message})

    def top5_alert(self, top5: list[dict], scan_time: str) -> None:
        lines = ["[SCAN TOP5]", f"time: {scan_time}"]
        for r in top5:
            lines.append(
                f"{r['rank']}. {r['symbol']} A6 {r['a6_score']:.2f} expected_ev {r['expected_ev']:.2f}%"
            )
        self._emit("SCAN_TOP5", "\n".join(lines))

    def entry_alert(self, symbol: str, side: str, entry_price: float, reason: str, expected_ev: float) -> None:
        msg = (
            f"[ENTRY]\nsymbol: {symbol}\nside: {side}\nentry_price: {entry_price}\n"
            f"reason: {reason}\nexpected_ev: {expected_ev:.2f}%"
        )
        self._emit("ENTRY", msg)

    def exit_alert(self, symbol: str, exit_price: float, pnl: float, hold_minutes: int, exit_reason: str) -> None:
        msg = (
            f"[EXIT]\nsymbol: {symbol}\nexit_price: {exit_price}\npnl: {pnl:.2f}%\n"
            f"hold_minutes: {hold_minutes}\nexit_reason: {exit_reason}"
        )
        self._emit("EXIT", msg)

    def manual_alert(self, symbol: str, action: str) -> None:
        self._emit("MANUAL", f"[MANUAL LOCK]\nsymbol: {symbol}\naction: {action}")

    def risk_guard_exit_alert(
        self, symbol: str, roi: float, pnl_usdt: float, reason: str,
    ) -> None:
        msg = (
            f"[RISK GUARD EXIT]\nsymbol: {symbol}\nroi: {roi:.2f}%\n"
            f"pnl_usdt: {pnl_usdt:.4f}\nreason: {reason}"
        )
        if self.console:
            print(msg)
        self._telegram_send(msg)
        self.db.execute(
            "INSERT INTO alerts (timestamp, alert_type, message) VALUES (?,?,?)",
            (now_kst(), "RISK_GUARD_EXIT", msg),
        )

    def entry_block_summary_alert(self, message: str) -> None:
        """30m entry block summary — Telegram only, no console spam."""
        self._telegram_send(message)

    def error_alert(self, module: str, message: str) -> None:
        self._emit("ERROR", f"[ERROR]\nmodule: {module}\nmessage: {message}")
