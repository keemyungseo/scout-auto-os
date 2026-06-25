"""Position Evolution timeline — LIVE reviews + research replay checkpoints."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from scout_research_r006_pilot_execution_engine import load_forward_bars
from scout_research_r008_exit_engine import state_snapshot

from scout_auto_os.engine.state_engine import LIVE_STATE_FORMULA, compute_alive_from_snap

EVOLUTION_FIELDS = [
    "record_time_kst", "source", "position_id", "symbol", "checkpoint_min",
    "entry_time_kst", "exit_time_kst", "realized_pnl_pct",
    "alive_score", "alive_delta", "trend_alive", "momentum_alive",
    "volume_alive", "expansion_alive", "acceleration", "exhaustion",
    "recommendation", "review_reason", "formula_name",
]

CHECKPOINTS_MIN = (0, 30, 60, 90, 120)


class PositionEvolutionStore:
    def __init__(self, research_dir: Path) -> None:
        self.path = research_dir / "position_evolution.jsonl"
        self.csv_path = research_dir / "position_evolution.csv"
        self.research_dir = research_dir
        self.research_dir.mkdir(parents=True, exist_ok=True)
        self._seen: set[str] = set()
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        r = json.loads(line)
                        self._seen.add(self._key(r))
                    except json.JSONDecodeError:
                        pass
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=EVOLUTION_FIELDS).writeheader()

    @staticmethod
    def _key(r: dict) -> str:
        return f"{r.get('source')}|{r.get('position_id')}|{r.get('symbol')}|{r.get('checkpoint_min')}|{r.get('record_time_kst')}"

    def append(self, row: dict) -> bool:
        k = self._key(row)
        if k in self._seen:
            return False
        self._seen.add(k)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=EVOLUTION_FIELDS, extrasaction="ignore").writerow(
                {fld: row.get(fld, "") for fld in EVOLUTION_FIELDS}
            )
        return True

    def ingest_live_reviews(self, data_dir: Path, record_time: str) -> int:
        """Copy LIVE position_review.csv into evolution store (read-only bridge)."""
        review_path = data_dir / "position_review.csv"
        if not review_path.exists():
            return 0
        n = 0
        with review_path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rec = {
                    "record_time_kst": row.get("review_time_kst") or record_time,
                    "source": "live_review",
                    "position_id": row.get("position_id", ""),
                    "symbol": row.get("symbol", ""),
                    "checkpoint_min": row.get("hold_minutes", ""),
                    "entry_time_kst": row.get("entry_time_kst", ""),
                    "exit_time_kst": "",
                    "realized_pnl_pct": row.get("unrealized_pnl_pct", ""),
                    "alive_score": row.get("current_alive_score", ""),
                    "alive_delta": row.get("alive_delta", ""),
                    "trend_alive": row.get("trend_alive_current", ""),
                    "momentum_alive": row.get("momentum_alive_current", ""),
                    "volume_alive": row.get("volume_alive_current", ""),
                    "expansion_alive": row.get("expansion_alive_current", ""),
                    "acceleration": "",
                    "exhaustion": row.get("exhaustion_current", ""),
                    "recommendation": row.get("hold_recommendation", ""),
                    "review_reason": row.get("review_reason", ""),
                    "formula_name": "LIVE_V14",
                }
                if self.append(rec):
                    n += 1
        return n

    def build_replay_checkpoints(self, forward_rows: list[dict], record_time: str) -> int:
        """Historical replay: entry → 30/60/90/120m checkpoints."""
        n = 0
        for row in forward_rows:
            if row.get("return_2h") in ("", None):
                continue
            sym = row["symbol"]
            scan = row["scan_time_kst"]
            try:
                bars = load_forward_bars(sym, scan)
            except Exception:
                continue
            if not bars or len(bars) < 12:
                continue
            entry_alive = None
            for cp_min in CHECKPOINTS_MIN:
                bar_i = min(len(bars) - 1, cp_min // 5)
                snap = state_snapshot(bars, bar_i, 0)
                alive = compute_alive_from_snap(snap, LIVE_STATE_FORMULA)
                if cp_min == 0:
                    entry_alive = alive.alive_score
                delta = alive.alive_score - entry_alive if entry_alive is not None else 0
                rec = {
                    "record_time_kst": record_time,
                    "source": "replay_forward",
                    "position_id": f"replay_{scan}_{sym}",
                    "symbol": sym,
                    "checkpoint_min": cp_min,
                    "entry_time_kst": scan,
                    "exit_time_kst": "",
                    "realized_pnl_pct": row.get(f"return_{cp_min}m", row.get("return_2h", "")) if cp_min else "0",
                    "alive_score": alive.alive_score,
                    "alive_delta": round(delta, 2),
                    "trend_alive": alive.trend_alive,
                    "momentum_alive": alive.momentum_alive,
                    "volume_alive": alive.volume_alive,
                    "expansion_alive": alive.expansion_alive,
                    "acceleration": alive.acceleration_bonus,
                    "exhaustion": alive.exhaustion,
                    "recommendation": alive.hold_recommendation,
                    "review_reason": "replay_checkpoint",
                    "formula_name": "LIVE_V14",
                }
                if self.append(rec):
                    n += 1
        return n

    def read_all(self, limit: int = 2000) -> list[dict]:
        if not self.path.exists():
            return []
        rows: list[dict] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows[-limit:]
