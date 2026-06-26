"""Security event log — security.log."""

from __future__ import annotations

from pathlib import Path

from scout_auto_os.storage.db import now_kst

LOG_FILENAME = "security.log"


class SecurityLogger:
    def __init__(self, control_dir: Path) -> None:
        self.path = control_dir / LOG_FILENAME
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, ip: str, detail: str = "") -> None:
        line = f"{now_kst()}\t{event}\t{ip}\t{detail}\n"
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line)

    def rows(self) -> list[str]:
        if not self.path.exists():
            return []
        return self.path.read_text(encoding="utf-8").splitlines()
