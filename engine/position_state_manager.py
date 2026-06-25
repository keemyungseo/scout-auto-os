"""Position State Manager — entry baseline, 30m review, live state cache (V1.4)."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path

from scout_research_r006_pilot_execution_engine import Bar

from scout_auto_os.engine.position_review_store import PositionReviewStore
from scout_auto_os.engine.state_engine import AliveScore, compute_alive_score
from scout_auto_os.engine.state_exit_engine import StateExitEngine, StateExitDecision
from scout_auto_os.storage.db import now_kst

KST_FMT = "%Y-%m-%d %H:%M:%S"


class PositionStateManager:
    """State Engine facade — wired to PositionManager, readable by Research later."""

    def __init__(self, config: dict, data_dir: Path, get_bars_fn) -> None:
        self.config = config
        self.get_bars_fn = get_bars_fn
        sc = config.get("state_engine", {})
        self.review_interval_sec = int(
            os.environ.get("STATE_REVIEW_INTERVAL_SEC", sc.get("review_interval_sec", 1800))
        )
        self.hold_alive = float(sc.get("hold_alive_score", 70))
        self.exit_alive = float(sc.get("exit_alive_score", 45))
        self.store = PositionReviewStore(data_dir)
        self.exit_engine = StateExitEngine(config)
        self.cache_path = data_dir / "position_state_cache.json"
        self._entry: dict[str, dict] = {}
        self._current: dict[str, dict] = {}
        self._last_review: dict[str, float] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        if not self.cache_path.exists():
            return
        try:
            data = json.loads(self.cache_path.read_text(encoding="utf-8"))
            self._entry = data.get("entry", {})
            self._current = data.get("current", {})
        except json.JSONDecodeError:
            pass

    def _save_cache(self) -> None:
        self.cache_path.write_text(
            json.dumps({"entry": self._entry, "current": self._current}, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def hold_minutes(entry_time: str, now: str | None = None) -> int:
        now = now or now_kst()
        try:
            t0 = datetime.strptime(entry_time, KST_FMT)
            t1 = datetime.strptime(now, KST_FMT)
            return int((t1 - t0).total_seconds() / 60)
        except ValueError:
            return 0

    def _score(self, bars: list[Bar]) -> AliveScore | None:
        return compute_alive_score(bars, 0, self.hold_alive, self.exit_alive)

    def register_entry(self, position_id: str, symbol: str, entry_time: str, bars: list[Bar]) -> AliveScore | None:
        score = self._score(bars)
        if not score:
            return None
        self._entry[position_id] = {
            "symbol": symbol,
            "entry_time": entry_time,
            "score": score.to_dict(),
        }
        self._current[position_id] = score.to_dict()
        self._last_review[position_id] = time.time()
        self._save_cache()
        print(
            f"[STATE ENGINE] entry registered {symbol} alive_score={score.alive_score} "
            f"rec={score.hold_recommendation}"
        )
        return score

    def bootstrap_missing(self, position_id: str, symbol: str, entry_time: str) -> None:
        """Rehydrate entry baseline for positions opened before restart."""
        if position_id in self._entry:
            return
        bars = self.get_bars_fn(symbol, entry_time)
        if bars:
            self.register_entry(position_id, symbol, entry_time, bars)

    def update_current(self, position_id: str, symbol: str, entry_time: str, bars: list[Bar]) -> AliveScore | None:
        self.bootstrap_missing(position_id, symbol, entry_time)
        score = self._score(bars)
        if score:
            self._current[position_id] = score.to_dict()
            self._save_cache()
        return score

    def maybe_review(
        self,
        position: dict,
        bars: list[Bar],
        pnl_pct: float,
    ) -> StateExitDecision | None:
        pid = position["position_id"]
        sym = position["symbol"]
        entry_time = position["entry_time"]
        self.bootstrap_missing(pid, sym, entry_time)

        entry_raw = self._entry.get(pid, {}).get("score", {}) if pid in self._entry else {}
        if not entry_raw:
            return None
        entry_score = AliveScore.from_dict(entry_raw)
        current = self.update_current(pid, sym, entry_time, bars)
        if not current or not entry_score:
            return None

        hold = self.hold_minutes(entry_time)
        now = time.time()
        due = now - self._last_review.get(pid, 0) >= self.review_interval_sec

        decision = self.exit_engine.evaluate(bars, position["entry_price"], entry_score, current, hold)
        review_reason = decision.review_reason or current.hold_recommendation

        if due or decision.should_exit:
            delta = round(current.alive_score - entry_score.alive_score, 2)
            row = {
                "review_time_kst": now_kst(),
                "position_id": pid,
                "symbol": sym,
                "entry_time_kst": entry_time,
                "hold_minutes": hold,
                "entry_alive_score": entry_score.alive_score,
                "current_alive_score": current.alive_score,
                "alive_delta": delta,
                "trend_alive_entry": entry_score.trend_alive,
                "trend_alive_current": current.trend_alive,
                "momentum_alive_entry": entry_score.momentum_alive,
                "momentum_alive_current": current.momentum_alive,
                "volume_alive_entry": entry_score.volume_alive,
                "volume_alive_current": current.volume_alive,
                "expansion_alive_current": current.expansion_alive,
                "exhaustion_current": current.exhaustion,
                "hold_recommendation": current.hold_recommendation,
                "review_reason": review_reason,
                "exit_reason": decision.reason if decision.should_exit else "",
                "unrealized_pnl_pct": round(pnl_pct, 4),
            }
            self.store.append(row)
            self._last_review[pid] = now
            print(
                f"[STATE REVIEW] {sym} hold={hold}m alive={current.alive_score} "
                f"delta={delta:+.1f} rec={current.hold_recommendation}"
            )

        return decision if decision.should_exit else None

    def on_close(self, position_id: str) -> None:
        self._entry.pop(position_id, None)
        self._current.pop(position_id, None)
        self._last_review.pop(position_id, None)
        self._save_cache()

    def live_summary(self, position_id: str, symbol: str, entry_time: str, hold_min: int) -> dict:
        cur = self._current.get(position_id, {})
        ent = self._entry.get(position_id, {}).get("score", {})
        if not cur:
            return {
                "symbol": symbol,
                "alive_score": "n/a",
                "hold_recommendation": "UNKNOWN",
                "alive_delta": "n/a",
                "hold_minutes": hold_min,
            }
        delta = round(float(cur.get("alive_score", 0)) - float(ent.get("alive_score", 0)), 1) if ent else "n/a"
        return {
            "symbol": symbol,
            "alive_score": cur.get("alive_score", "n/a"),
            "hold_recommendation": cur.get("hold_recommendation", "UNKNOWN"),
            "alive_delta": delta,
            "exhaustion": cur.get("exhaustion", 0),
            "hold_minutes": hold_min,
        }

    def summaries_for_open(self, open_positions: list[dict]) -> list[dict]:
        out: list[dict] = []
        for p in open_positions:
            hold = self.hold_minutes(p.get("entry_time", ""))
            out.append(self.live_summary(p["position_id"], p["symbol"], p.get("entry_time", ""), hold))
        return out
