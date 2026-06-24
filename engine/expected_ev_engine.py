"""Live expected EV / RemainingEV from R010 entry + R008/R009 state."""

from __future__ import annotations

import csv
from pathlib import Path

from scout_research_r006_pilot_execution_engine import Bar
from scout_research_r008_exit_engine import state_snapshot
from scout_research_r009_dynamic_exit_engine import efr_from_bar
from scout_research_r010_dynamic_entry_engine import ENTRY_A, simulate_entry
from scout_auto_os.storage.db import now_kst


def compute_live_ev(
    symbol: str,
    bars: list[Bar],
    a6_score: float = 0.0,
    scan_kst: str | None = None,
    entry_ms: int | None = None,
) -> dict:
    """
    Compute expected_ev from live forward bars.
    Interface stable: returns expected_ev, remaining_ev, trend_alive, acceleration, volume_state.
    """
    if entry_ms is not None:
        bars = [b for b in bars if b.t_ms >= entry_ms]
    if not bars:
        return {
            "expected_ev": 0.0,
            "remaining_ev": 0.0,
            "trend_alive": "unknown",
            "acceleration": "unknown",
            "volume_state": "unknown",
            "efr_60": 0.0,
            "current_ret_pct": 0.0,
        }

    meta = {
        "symbol": symbol,
        "search_time": scan_kst or now_kst(),
        "a6_score": a6_score,
    }
    sim = simulate_entry(bars, meta, "A", list(ENTRY_A))
    i = len(bars) - 1
    snap = state_snapshot(bars, i, 0)
    entry_px = bars[0].o if bars[0].o > 0 else bars[0].c
    current_ret = (bars[i].c - entry_px) / entry_px * 100 if entry_px > 0 else 0.0
    efr60 = efr_from_bar(bars, i, 12)

    if sim:
        expected_ev = sim.return_pct
        remaining_ev = sim.upside_remaining_pct
    else:
        expected_ev = current_ret
        remaining_ev = max(0.0, efr60 - current_ret) if efr60 > current_ret else 0.0

    return {
        "expected_ev": round(expected_ev, 4),
        "remaining_ev": round(remaining_ev, 4),
        "trend_alive": "alive" if snap.trend_alive else "exhausted",
        "acceleration": "on" if snap.acceleration else "off",
        "volume_state": "weak" if snap.volume_weak else "ok",
        "efr_60": round(efr60, 4),
        "current_ret_pct": round(current_ret, 4),
    }


class ExpectedEVLogger:
    def __init__(self, log_dir: Path) -> None:
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.log_dir / "expected_ev.log"

    def log(self, symbol: str, ev: dict) -> None:
        write_header = not self.path.exists()
        with self.path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=[
                "timestamp", "symbol", "expected_ev", "remaining_ev",
                "trend_alive", "efr_60", "current_ret_pct",
            ])
            if write_header:
                w.writeheader()
            w.writerow({
                "timestamp": now_kst(),
                "symbol": symbol,
                "expected_ev": ev.get("expected_ev", 0),
                "remaining_ev": ev.get("remaining_ev", 0),
                "trend_alive": ev.get("trend_alive", ""),
                "efr_60": ev.get("efr_60", 0),
                "current_ret_pct": ev.get("current_ret_pct", 0),
            })
