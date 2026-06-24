"""Telegram command handler — /status /report /export /positions /missed (V1.2)."""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

from scout_auto_os.engine.review.safe import review_safe
from scout_auto_os.engine.position_report import PositionReportService
from scout_auto_os.engine.review.missed_winners import MissedWinnersStore
from scout_auto_os.storage.db import Database, now_kst
from scout_auto_os.storage.trade_record_db import TradeRecordDB


class TelegramCommandBot:
    COMMANDS = ("/status", "/report", "/export", "/positions", "/missed")

    def __init__(
        self,
        config: dict,
        data_dir: Path,
        trade_db: TradeRecordDB,
        position_report: PositionReportService,
        review_snapshot_fn,
        os_db: Database,
        engine_state_fn,
    ) -> None:
        self.config = config
        self.data_dir = data_dir
        self.trade_db = trade_db
        self.position_report = position_report
        self.review_snapshot_fn = review_snapshot_fn
        self.os_db = os_db
        self.engine_state_fn = engine_state_fn
        self.missed_store = MissedWinnersStore(data_dir, lambda _s: 0.0)
        self.token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        self.chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
        self.enabled = bool(config.get("alerts", {}).get("telegram")) and bool(self.token and self.chat_id)
        self._offset = 0
        self._thread: threading.Thread | None = None
        self._running = False

    def start(self) -> None:
        if not self.enabled:
            return
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True, name="telegram-cmd")
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def _api(self, method: str, params: dict | None = None, multipart: bool = False) -> dict:
        url = f"https://api.telegram.org/bot{self.token}/{method}"
        if multipart:
            req = urllib.request.Request(url, data=params.get("_body"), method="POST")
            req.add_header("Content-Type", params.get("_content_type", ""))
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
        data = urllib.parse.urlencode(params or {}).encode()
        req = urllib.request.Request(url, data=data, method="POST")
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())

    def send_text(self, text: str) -> bool:
        if not self.enabled:
            return False
        try:
            self._api("sendMessage", {"chat_id": self.chat_id, "text": text[:4000]})
            return True
        except Exception:
            return False

    def send_document(self, path: Path, caption: str = "") -> bool:
        if not self.enabled or not path.exists() or path.stat().st_size > 4_000_000:
            return False
        try:
            boundary = "----ScoutBoundary"
            body = (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="chat_id"\r\n\r\n{self.chat_id}\r\n'
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="caption"\r\n\r\n{caption[:200]}\r\n'
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="document"; filename="{path.name}"\r\n'
                f"Content-Type: text/csv\r\n\r\n"
            ).encode() + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
            self._api("sendDocument", {
                "_body": body,
                "_content_type": f"multipart/form-data; boundary={boundary}",
            }, multipart=True)
            return True
        except Exception:
            return False

    def _poll_loop(self) -> None:
        while self._running:
            try:
                data = self._api("getUpdates", {"offset": self._offset, "timeout": 10})
                for upd in data.get("result", []):
                    self._offset = upd["update_id"] + 1
                    msg = upd.get("message") or {}
                    text = (msg.get("text") or "").strip().split()[0] if msg.get("text") else ""
                    chat = str(msg.get("chat", {}).get("id", ""))
                    if chat != self.chat_id:
                        continue
                    if text in self.COMMANDS:
                        self._handle_command(text)
            except Exception:
                pass
            time.sleep(2)

    @review_safe("telegram_cmd")
    def _handle_command(self, cmd: str) -> None:
        print(f"[TELEGRAM CMD] {cmd}")
        if cmd == "/status":
            self.send_text(self._cmd_status())
        elif cmd == "/report":
            self.send_text(self._cmd_report())
        elif cmd == "/export":
            self._cmd_export()
        elif cmd == "/positions":
            self.send_text(self._cmd_positions())
        elif cmd == "/missed":
            self.send_text(self._cmd_missed())

    def _cmd_status(self) -> str:
        snap = self.review_snapshot_fn() or {}
        state = self.engine_state_fn()
        today = now_kst()[:10]
        today_closed = self.trade_db.fetchall(
            """SELECT * FROM trades WHERE status IN ('CLOSED','CLOSED_BY_USER')
            AND exit_time LIKE ?""",
            (f"{today}%",),
        )
        today_pnl = sum(t.get("realized_pnl_usdt") or 0 for t in today_closed)
        lines = [
            "SCOUT /status",
            f"State: {state.get('bot_state', 'unknown')}",
            f"Mode: {'PAPER' if self.config.get('paper_mode') else 'LIVE'}",
            f"Open trades: {snap.get('open_trades', 0)}",
            f"Unrealized USDT: {snap.get('unrealized_pnl_usdt', 0):+.4f}",
            f"Today realized USDT: {today_pnl:+.4f}",
            "",
            "Last TOP5:",
        ]
        for r in (snap.get("last_top5") or [])[:5]:
            lines.append(f"  #{r.get('rank')} {r.get('symbol')} score={r.get('score')}")
        return "\n".join(lines)

    def _cmd_report(self) -> str:
        snap = self.review_snapshot_fn() or {}
        best = snap.get("best_trade") or {}
        worst = snap.get("worst_trade") or {}
        lines = [
            "SCOUT /report (since start)",
            f"Since: {snap.get('since_start_time', 'n/a')}",
            f"Scans: {snap.get('total_scans', 0)} | Entries: {snap.get('total_entries', 0)}",
            f"Closed: {snap.get('closed_trades', 0)} | Win rate: {snap.get('win_rate', 0)}%",
            f"Realized USDT: {snap.get('realized_pnl_usdt', 0):+.4f}",
            f"Unrealized USDT: {snap.get('unrealized_pnl_usdt', 0):+.4f}",
            f"Total USDT: {snap.get('total_pnl_usdt', 0):+.4f}",
        ]
        if best:
            lines.append(f"Best: {best.get('symbol')} {best.get('realized_pnl_pct', 0):+.2f}%")
        if worst:
            lines.append(f"Worst: {worst.get('symbol')} {worst.get('realized_pnl_pct', 0):+.2f}%")
        return "\n".join(lines)

    def _cmd_positions(self) -> str:
        positions = self.position_report.fetch_open_positions()
        if not positions:
            paper_snap = self.data_dir / "positions_snapshot.json"
            if paper_snap.exists():
                positions = json.loads(paper_snap.read_text(encoding="utf-8")).get("positions", [])
        lines = ["SCOUT /positions"]
        if not positions:
            lines.append("(none)")
            return "\n".join(lines)
        for p in positions:
            lines.append(
                f"{p['symbol']} {p['side']} entry={p.get('entry_price')} mark={p.get('mark_price')} "
                f"upnl={p.get('unrealized_pnl_usdt', 0):+.4f} ({p.get('unrealized_pnl_pct', 0):+.2f}%) "
                f"hold={p.get('hold_minutes', 0)}m"
            )
        return "\n".join(lines)

    def _cmd_missed(self) -> str:
        rows = self.missed_store.recent(10)
        lines = ["SCOUT /missed (recent 10)"]
        if not rows:
            lines.append("(none yet — checks run 30m/1h/2h after scan)")
            return "\n".join(lines)
        for r in rows:
            lines.append(
                f"{r.get('scan_time_kst')} {r.get('symbol')} "
                f"30m={r.get('forward_30m_return')}% 1h={r.get('forward_1h_return')}% "
                f"block={r.get('reason_not_entered')}"
            )
        return "\n".join(lines)

    def _cmd_export(self) -> None:
        lines = ["SCOUT /export — recent summary", ""]
        trades = self.trade_db.fetchall("SELECT * FROM trades ORDER BY id DESC LIMIT 20")
        lines.append("[Trades last 20]")
        for t in trades:
            lines.append(
                f"  {t.get('entry_time')} {t['symbol']} {t['status']} "
                f"pnl={t.get('realized_pnl_pct', 'n/a')}%"
            )
        scans_path = self.data_dir / "scans.jsonl"
        if scans_path.exists():
            scan_lines = scans_path.read_text(encoding="utf-8").strip().splitlines()[-20:]
            lines.append("\n[Scans last 20 rows]")
            for sl in scan_lines:
                try:
                    r = json.loads(sl)
                    lines.append(f"  {r.get('scan_time_kst')} #{r.get('rank')} {r.get('symbol')}")
                except json.JSONDecodeError:
                    pass
        missed = self.missed_store.recent(20)
        lines.append("\n[Missed winners last 20]")
        for m in missed:
            lines.append(f"  {m.get('symbol')} 2h={m.get('forward_2h_return')}%")
        text = "\n".join(lines)
        self.send_text(text[:4000])
        csv_path = self.data_dir / "scans.csv"
        if csv_path.exists():
            self.send_document(csv_path, "scans.csv export")
