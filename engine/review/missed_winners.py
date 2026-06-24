"""Missed winners — forward return checks after scan (V1.2)."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scout_auto_os.engine.review.parse_reason import parse_reason_fields
from scout_auto_os.storage.db import now_kst

KST = timezone(timedelta(hours=9))
THRESHOLDS = {"30m": 3.0, "1h": 5.0, "2h": 7.0}
HORIZON_MIN = {"30m": 30, "1h": 60, "2h": 120}

MISSED_FIELDS = [
    "scan_time_kst", "symbol", "forward_30m_return", "forward_1h_return", "forward_2h_return",
    "was_in_top5", "top5_rank", "was_entered", "reason_not_entered",
    "observed_pattern", "suggested_feature_hint", "recorded_at_kst",
]


class MissedWinnersStore:
    def __init__(self, data_dir: Path, price_fn) -> None:
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = data_dir / "missed_winners.csv"
        self.jsonl_path = data_dir / "missed_winners.jsonl"
        self.queue_path = data_dir / "missed_winners_queue.json"
        self.price_fn = price_fn
        self._queue: list[dict] = self._load_queue()

    def _load_queue(self) -> list[dict]:
        if self.queue_path.exists():
            try:
                return json.loads(self.queue_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return []
        return []

    def _save_queue(self) -> None:
        self.queue_path.write_text(json.dumps(self._queue, indent=2), encoding="utf-8")

    def schedule(self, scan_time_kst: str, top5: list[dict], entered: set[str], block_map: dict[str, str]) -> None:
        base_prices: dict[str, float] = {}
        for row in top5:
            sym = row["symbol"]
            base_prices[sym] = float(row.get("entry_price") or 0)
        self._queue.append({
            "scan_time_kst": scan_time_kst,
            "scheduled_at": now_kst(),
            "base_prices": base_prices,
            "top5": [
                {
                    "symbol": r["symbol"],
                    "rank": r.get("rank"),
                    "reason": r.get("reason", ""),
                    "entered": r["symbol"] in entered,
                    "block": block_map.get(r["symbol"], ""),
                }
                for r in top5
            ],
            "checked": {"30m": False, "1h": False, "2h": False},
            "returns": {},
        })
        self._save_queue()

    @staticmethod
    def _forward_return(base: float, current: float) -> float:
        if base <= 0 or current <= 0:
            return 0.0
        return round((current - base) / base * 100, 4)

    def _pattern_hint(self, reason: str) -> tuple[str, str]:
        p = parse_reason_fields(reason)
        pattern = f"1h={p['reason_1h']} 2h={p['reason_2h']} rng={p['range_pct']}%"
        hint = "high_range_acceleration" if p["range_pct"] >= 10 else "state_mismatch_review"
        return pattern, hint

    def _save_missed(self, rec: dict) -> None:
        write_header = not self.csv_path.exists()
        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=MISSED_FIELDS)
            if write_header:
                w.writeheader()
            w.writerow(rec)
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[REVIEW] missed winner saved symbol={rec['symbol']}")

    def process_due(self) -> int:
        if not self._queue:
            return 0
        now = datetime.now(KST)
        saved = 0
        remaining: list[dict] = []

        for job in self._queue:
            try:
                scan_dt = datetime.strptime(job["scan_time_kst"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=KST)
            except ValueError:
                continue
            elapsed_min = (now - scan_dt).total_seconds() / 60
            rets = job.setdefault("returns", {})

            for label, minutes in HORIZON_MIN.items():
                if job["checked"].get(label) or elapsed_min < minutes:
                    continue
                job["checked"][label] = True
                for item in job["top5"]:
                    sym = item["symbol"]
                    base = float(job["base_prices"].get(sym) or 0)
                    px = self.price_fn(sym)
                    rets.setdefault(sym, {})[label] = self._forward_return(base, px)

            if elapsed_min >= HORIZON_MIN["2h"]:
                for item in job["top5"]:
                    sym = item["symbol"]
                    sym_rets = rets.get(sym, {})
                    hit = any(
                        sym_rets.get(h, 0) >= THRESHOLDS[h]
                        for h in ("30m", "1h", "2h")
                        if h in sym_rets
                    )
                    if not hit:
                        continue
                    if item["entered"]:
                        continue
                    pattern, hint = self._pattern_hint(item.get("reason", ""))
                    rec = {
                        "scan_time_kst": job["scan_time_kst"],
                        "symbol": sym,
                        "forward_30m_return": sym_rets.get("30m", 0),
                        "forward_1h_return": sym_rets.get("1h", 0),
                        "forward_2h_return": sym_rets.get("2h", 0),
                        "was_in_top5": True,
                        "top5_rank": item.get("rank"),
                        "was_entered": False,
                        "reason_not_entered": item.get("block", ""),
                        "observed_pattern": pattern,
                        "suggested_feature_hint": hint,
                        "recorded_at_kst": now_kst(),
                    }
                    self._save_missed(rec)
                    saved += 1
            else:
                remaining.append(job)

        self._queue = remaining
        self._save_queue()
        return saved

    def recent(self, limit: int = 10) -> list[dict]:
        if not self.jsonl_path.exists():
            return []
        lines = self.jsonl_path.read_text(encoding="utf-8").strip().splitlines()
        out = []
        for line in lines[-limit:]:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        return out
