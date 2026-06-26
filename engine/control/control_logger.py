"""Append-only control action log."""

from __future__ import annotations

import csv
from pathlib import Path

from scout_auto_os.storage.db import now_kst

FIELDS = ("timestamp", "action", "symbol", "requested_by", "status", "reason", "confirm_id")


class ControlLogger:
    def __init__(self, control_dir: Path) -> None:
        self.control_dir = control_dir
        self.control_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.control_dir / "control_log.csv"

    def log(
        self,
        action: str,
        *,
        symbol: str = "",
        requested_by: str = "operator",
        status: str = "ok",
        reason: str = "",
        confirm_id: str = "",
    ) -> None:
        write_header = not self.path.exists()
        with self.path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if write_header:
                w.writeheader()
            w.writerow({
                "timestamp": now_kst(),
                "action": action,
                "symbol": symbol,
                "requested_by": requested_by,
                "status": status,
                "reason": reason,
                "confirm_id": confirm_id,
            })

    def rows(self) -> list[dict]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as f:
            return list(csv.DictReader(f))
