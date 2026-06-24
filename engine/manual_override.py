"""Manual override via JSON/CSV files."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from scout_auto_os.storage.db import Database, now_kst


class ManualOverride:
    def __init__(self, config: dict, db: Database, csv_dir: Path, data_dir: Path) -> None:
        self.db = db
        self.csv_dir = csv_dir
        self.csv_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.json_path = self.data_dir / "manual_override.json"
        self.positions_csv = self.data_dir / "manual_positions.csv"
        self.log_path = self.csv_dir / "manual_override_log.csv"
        if not self.json_path.exists():
            self.json_path.write_text(json.dumps({"locks": [], "events": []}, indent=2), encoding="utf-8")

    def load(self) -> dict:
        return json.loads(self.json_path.read_text(encoding="utf-8"))

    def save(self, data: dict) -> None:
        self.json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def locked_symbols(self) -> set[str]:
        data = self.load()
        locks = {e["symbol"] for e in data.get("locks", []) if e.get("active", True)}
        if self.positions_csv.exists():
            with self.positions_csv.open(encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row.get("manual_lock", "").lower() in ("1", "true", "yes"):
                        locks.add(row["symbol"])
        return locks

    def queue_force_close(self, symbol: str, position_id: str = "") -> None:
        data = self.load()
        ev = {"symbol": symbol, "action": "force_close"}
        if position_id:
            ev["position_id"] = position_id
        data.setdefault("events", []).append(ev)
        self.save(data)

    def apply_events(self, position_mgr, execution, alert_mgr) -> list[dict]:
        data = self.load()
        applied: list[dict] = []
        remaining: list[dict] = []
        for ev in data.get("events", []):
            action = ev.get("action", "")
            sym = ev.get("symbol", "")
            if action == "manual_lock":
                self._lock(sym, ev.get("details", ""))
                alert_mgr.manual_alert(sym, "manual_lock")
                applied.append(ev)
            elif action == "unlock":
                self._unlock(sym)
                alert_mgr.manual_alert(sym, "unlock")
                applied.append(ev)
            elif action == "force_close":
                pos_id = ev.get("position_id", "")
                if pos_id:
                    position_mgr.close_by_user(pos_id, execution, alert_mgr)
                else:
                    for p in position_mgr.open_positions():
                        if p["symbol"] == sym and p["source"] == "AUTO":
                            position_mgr.close_by_user(p["position_id"], execution, alert_mgr)
                alert_mgr.manual_alert(sym or pos_id, "force_close")
                applied.append(ev)
            elif action == "convert_to_manual":
                self.db.execute(
                    "UPDATE positions SET source='MANUAL', manual_lock=1, auto_manage=0 WHERE symbol=? AND status='OPEN'",
                    (sym,),
                )
                alert_mgr.manual_alert(sym, "convert_to_manual")
                applied.append(ev)
            elif action == "enable_auto_manage":
                self.db.execute(
                    "UPDATE positions SET auto_manage=1, manual_lock=0 WHERE symbol=? AND status='OPEN'",
                    (sym,),
                )
                alert_mgr.manual_alert(sym, "enable_auto_manage")
                applied.append(ev)
            else:
                remaining.append(ev)
        if applied:
            data["events"] = remaining
            self.save(data)
            for ev in applied:
                self._log(ev)
        return applied

    def _lock(self, symbol: str, details: str) -> None:
        data = self.load()
        locks = [l for l in data.get("locks", []) if l["symbol"] != symbol]
        locks.append({"symbol": symbol, "active": True, "details": details})
        data["locks"] = locks
        self.save(data)
        self.db.execute(
            "INSERT INTO manual_overrides (timestamp, symbol, action, details) VALUES (?,?,?,?)",
            (now_kst(), symbol, "manual_lock", details),
        )

    def _unlock(self, symbol: str) -> None:
        data = self.load()
        data["locks"] = [l for l in data.get("locks", []) if l["symbol"] != symbol]
        self.save(data)
        self.db.execute(
            "INSERT INTO manual_overrides (timestamp, symbol, action, details) VALUES (?,?,?,?)",
            (now_kst(), symbol, "unlock", ""),
        )

    def _log(self, ev: dict) -> None:
        write_header = not self.log_path.exists()
        with self.log_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["timestamp", "symbol", "action", "details"])
            if write_header:
                w.writeheader()
            w.writerow({
                "timestamp": now_kst(),
                "symbol": ev.get("symbol", ""),
                "action": ev.get("action", ""),
                "details": json.dumps(ev.get("details", "")),
            })

    def sync_manual_positions(self, position_mgr) -> None:
        if not self.positions_csv.exists():
            return
        occupied = position_mgr.occupied_symbols()
        with self.positions_csv.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                sym = row["symbol"]
                if sym in occupied:
                    continue
                entry_px = float(row.get("entry_price", 0))
                side = row.get("side", "LONG").upper()
                manual_lock = row.get("manual_lock", "true").lower() in ("1", "true", "yes")
                position_mgr.create_position(
                    sym, side, entry_px, "MANUAL", "USER",
                    manual_lock=manual_lock, auto_manage=not manual_lock,
                )
