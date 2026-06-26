"""LIVE entry via Portfolio Engine Long3/Short3."""

from __future__ import annotations

from pathlib import Path

from scout_auto_os.engine.portfolio.live_scan import run_portfolio_scan
from scout_auto_os.engine.position_evaluation.runner import PositionEvaluationRunner


class PortfolioEntryBridge:
    """Wire Portfolio Engine into main tick_scan (when portfolio_engine.enabled)."""

    def __init__(self, config: dict, data_dir: Path, pkg_root: Path, cache_dir: Path, live_engine=None) -> None:
        self.config = config
        self.data_dir = data_dir
        self.pkg_root = pkg_root
        self.cache_dir = cache_dir
        self.live_engine = live_engine
        pe = config.get("portfolio_engine", {})
        self.max_long = int(pe.get("max_long_slots", 3))
        self.max_short = int(pe.get("max_short_slots", 3))
        self.max_symbols = int(pe.get("universe_max_symbols", 120))

    def run_selection(self, scan_kst: str) -> dict:
        return run_portfolio_scan(
            scan_kst,
            self.cache_dir,
            self.data_dir,
            self.pkg_root,
            live_engine=self.live_engine,
            max_symbols=self.max_symbols,
        )

    def try_fill(
        self,
        selection: dict,
        positions,
        execution,
        alerts,
        risk_mgr,
        occupied: set[str],
        locked: set[str],
    ) -> list[str]:
        """Enter up to max long/short slots from portfolio selection."""
        entered: list[str] = []
        long_used = positions.long_slots_used()
        short_used = getattr(positions, "short_slots_used", lambda: 0)()

        for c in selection.get("long", []):
            if long_used >= self.max_long:
                break
            sym = c["symbol"]
            if sym in occupied or sym in locked:
                continue
            if not PositionEvaluationRunner.entry_allowed(
                sym, occupied, locked, positions.open_positions(),
            ):
                continue
            ok, _ = risk_mgr.can_enter_long(long_used)
            if not ok:
                break
            px = float(c.get("features", {}).get("price", 0)) or 0.0
            if px <= 0 and self.live_engine:
                px = self.live_engine.get_price(sym)
            if px <= 0:
                continue
            positions.create_position(
                sym, "LONG", px, "AUTO", "PORTFOLIO_LONG",
                a6_score=c.get("entry_score", 0),
                expected_ev=0.0,
            )
            entered.append(sym)
            occupied.add(sym)
            long_used += 1

        for c in selection.get("short", []):
            if short_used >= self.max_short:
                break
            sym = c["symbol"]
            if sym in occupied or sym in locked:
                continue
            if not PositionEvaluationRunner.entry_allowed(
                sym, occupied, locked, positions.open_positions(),
            ):
                continue
            if not getattr(risk_mgr, "can_enter_short", lambda _n: (False, ""))(short_used)[0]:
                continue
            ok, _ = risk_mgr.can_enter_short(short_used)
            if not ok:
                continue
            px = float(c.get("features", {}).get("price", 0)) or 0.0
            if px <= 0 and self.live_engine:
                px = self.live_engine.get_price(sym)
            if px <= 0:
                continue
            positions.create_position(
                sym, "SHORT", px, "AUTO", "PORTFOLIO_SHORT",
                a6_score=c.get("entry_score", 0),
                expected_ev=0.0,
            )
            entered.append(sym)
            occupied.add(sym)
            short_used += 1

        return entered
