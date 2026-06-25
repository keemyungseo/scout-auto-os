"""Forward outcome tracker for research candidates."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

KST = timezone(timedelta(hours=9))

HORIZONS = (
    ("30m", 30),
    ("1h", 60),
    ("2h", 120),
    ("4h", 240),
    ("6h", 360),
    ("12h", 720),
)


def _parse_kst(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=KST)


def _ret(price_at: float, price_now: float) -> float:
    if price_at <= 0:
        return 0.0
    return round((price_now - price_at) / price_at * 100, 4)


class ForwardTracker:
    def __init__(self, pending_path: Path) -> None:
        self.pending_path = pending_path
        self.pending_path.parent.mkdir(parents=True, exist_ok=True)
        self._pending: list[dict] = self._load()

    def _load(self) -> list[dict]:
        if not self.pending_path.exists():
            return []
        out: list[dict] = []
        for line in self.pending_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
        return out

    def _save(self) -> None:
        self.pending_path.write_text(
            "\n".join(json.dumps(x, ensure_ascii=False) for x in self._pending) + ("\n" if self._pending else ""),
            encoding="utf-8",
        )

    def register(self, scan_time_kst: str, symbol: str, rank: int, price_at_scan: float) -> None:
        key = (scan_time_kst, symbol)
        if any(p["scan_time_kst"] == scan_time_kst and p["symbol"] == symbol for p in self._pending):
            return
        self._pending.append({
            "scan_time_kst": scan_time_kst,
            "symbol": symbol,
            "rank": rank,
            "price_at_scan": price_at_scan,
            "scan_ms": int(_parse_kst(scan_time_kst).timestamp() * 1000),
            "filled": {},
            "peak_2h": price_at_scan,
            "trough_2h": price_at_scan,
            "done": False,
        })
        self._save()

    def update(self, price_fn, now: datetime | None = None) -> list[dict]:
        now = now or datetime.now(KST)
        completed: list[dict] = []
        still: list[dict] = []

        for item in self._pending:
            if item.get("done"):
                continue
            scan_dt = _parse_kst(item["scan_time_kst"])
            elapsed_min = (now - scan_dt).total_seconds() / 60.0
            sym = item["symbol"]
            try:
                px = float(price_fn(sym))
            except Exception:
                still.append(item)
                continue
            if px <= 0:
                still.append(item)
                continue

            if elapsed_min <= 120:
                item["peak_2h"] = max(item.get("peak_2h", px), px)
                item["trough_2h"] = min(item.get("trough_2h", px), px)

            filled = item.setdefault("filled", {})
            for label, minutes in HORIZONS:
                if label in filled:
                    continue
                if elapsed_min >= minutes:
                    filled[label] = px

            if elapsed_min >= 720 and len(filled) >= len(HORIZONS):
                item["done"] = True
                completed.append(self._to_csv_row(item))
            else:
                still.append(item)

        self._pending = still
        self._save()
        return completed

    def placeholders(self, scan_time_kst: str, symbol: str, rank: int, price_at_scan: float) -> dict:
        """Immediate forward row shell for new candidates."""
        return {
            "scan_time_kst": scan_time_kst,
            "symbol": symbol,
            "rank": rank,
            "price_at_scan": price_at_scan,
            "price_30m": "",
            "return_30m": "",
            "price_1h": "",
            "return_1h": "",
            "price_2h": "",
            "return_2h": "",
            "price_4h": "",
            "return_4h": "",
            "price_6h": "",
            "return_6h": "",
            "price_12h": "",
            "return_12h": "",
            "max_return_2h": "",
            "min_return_2h": "",
            "max_drawdown_2h": "",
            "label_success_2h": "",
            "label_big_winner": "",
            "label_trap": "",
        }

    def _to_csv_row(self, item: dict) -> dict:
        p0 = float(item["price_at_scan"])
        filled = item.get("filled", {})
        row: dict = {
            "scan_time_kst": item["scan_time_kst"],
            "symbol": item["symbol"],
            "rank": item["rank"],
            "price_at_scan": p0,
        }
        for label, _ in HORIZONS:
            px = float(filled.get(label, p0))
            row[f"price_{label}"] = px
            row[f"return_{label}"] = _ret(p0, px)

        peak = float(item.get("peak_2h", p0))
        trough = float(item.get("trough_2h", p0))
        max_up = _ret(p0, peak)
        min_up = _ret(p0, trough)
        r2h = float(row.get("return_2h") or 0)
        row["max_return_2h"] = max_up
        row["min_return_2h"] = min_up
        row["max_drawdown_2h"] = round(min_up, 4)
        row["label_success_2h"] = r2h >= 3.0
        row["label_big_winner"] = max_up >= 4.0 or r2h >= 4.0
        row["label_trap"] = (max_up >= 2.0 and r2h < 0) or r2h <= -2.0
        return row

    @property
    def pending_count(self) -> int:
        return len(self._pending)
