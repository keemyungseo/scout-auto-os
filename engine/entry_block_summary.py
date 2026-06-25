"""Entry block aggregation + 30m Telegram summary (V1.3)."""

from __future__ import annotations

import time
from collections import Counter

from scout_auto_os.storage.db import now_kst


class EntryBlockSummary:
    def __init__(self, interval_sec: int = 1800) -> None:
        self.interval_sec = interval_sec
        self._blocked_count = 0
        self._reasons: Counter = Counter()
        self._passed_count = 0
        self._last_send = time.time()

    def record_block(self, reason: str) -> None:
        self._blocked_count += 1
        self._reasons[reason or "unknown"] += 1

    def record_pass(self) -> None:
        self._passed_count += 1

    def maybe_telegram_summary(self, send_fn) -> bool:
        now = time.time()
        if now - self._last_send < self.interval_sec:
            return False
        if self._blocked_count == 0 and self._passed_count == 0:
            self._last_send = now
            return False
        top = self._reasons.most_common(5)
        lines = [
            "[ENTRY BLOCK SUMMARY]",
            f"time: {now_kst()}",
            f"blocked_count: {self._blocked_count}",
            f"passed_count: {self._passed_count}",
            "top_block_reasons:",
        ]
        for reason, cnt in top:
            lines.append(f"  {reason}: {cnt}")
        if not top:
            lines.append("  (none)")
        send_fn("\n".join(lines))
        self._blocked_count = 0
        self._reasons.clear()
        self._passed_count = 0
        self._last_send = now
        return True
