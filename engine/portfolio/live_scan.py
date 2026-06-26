"""LIVE portfolio scan helper."""

from __future__ import annotations

from pathlib import Path

from scout_auto_os.engine.portfolio.engine import PortfolioEngine
from scout_auto_os.engine.strategy_core import run_universe_features


def run_portfolio_scan(
    scan_kst: str,
    cache_dir: Path,
    data_dir: Path,
    pkg_root: Path,
    live_engine=None,
    max_symbols: int = 120,
) -> dict:
    rows = run_universe_features(scan_kst, cache_dir, max_symbols=max_symbols, live_engine=live_engine)
    engine = PortfolioEngine.from_paths(data_dir, pkg_root)
    return engine.select_for_entry(rows, scan_kst)
