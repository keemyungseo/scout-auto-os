"""Collect Direction Champion signals with multi-horizon forward outcomes."""

from __future__ import annotations

from datetime import datetime, timedelta

from scout_auto_os.engine.research.directional.engines import rank_long, rank_short
from scout_auto_os.engine.research.directional.entry_filter.constants import (
    CHAMPION_TOP_K,
    LONG_DIRECTION_CHAMPION,
    RETURN_HORIZONS,
    SHORT_DIRECTION_CHAMPION,
)
from scout_auto_os.engine.research.directional.evaluation import to_long_metrics, to_short_metrics
from scout_auto_os.engine.research.zero_base.forward_eval import compute_forward_metrics


def filter_scans_last_months(scans: list[str], months: int) -> list[str]:
    """Keep scans within `months` before the latest scan date in the set."""
    if not scans:
        return []
    dates = [datetime.strptime(s[:10], "%Y-%m-%d") for s in scans]
    max_dt = max(dates)
    min_dt = max_dt - timedelta(days=months * 30)
    return sorted(s for s in scans if datetime.strptime(s[:10], "%Y-%m-%d") >= min_dt)


def _outcome_row(
    direction: str,
    engine: str,
    scan_kst: str,
    symbol: str,
    features: dict,
    metrics: dict,
) -> dict:
    if direction == "short":
        perf = {
            "return_30m": float(metrics.get("return_30m", 0)),
            "return_1h": float(metrics.get("return_1h", 0)),
            "return_2h": float(metrics.get("short_return_2h", metrics.get("return_2h", 0))),
            "return_4h": float(metrics.get("return_4h", 0)),
        }
        rank_key = "return_2h"
    else:
        perf = {h: float(metrics.get(h, 0)) for h in RETURN_HORIZONS}
        rank_key = "return_2h"

    return {
        "direction": direction,
        "engine": engine,
        "scan_time_kst": scan_kst,
        "symbol": symbol,
        "rank_metric": rank_key,
        **perf,
        "label_trap": bool(metrics.get("label_trap")),
        "label_big_winner": bool(metrics.get("label_big_winner")),
        "features": dict(features),
    }


def collect_direction_champion_signals(
    by_scan: dict[str, list[dict]],
    fwd: dict,
    scans: list[str],
    top_k: int = CHAMPION_TOP_K,
) -> tuple[list[dict], list[dict]]:
    """All Direction Champion picks with scan-time features and forward returns."""
    long_signals: list[dict] = []
    short_signals: list[dict] = []

    for scan_kst in scans:
        rows = by_scan.get(scan_kst, [])
        if not rows:
            continue

        def metric_fn(sym: str) -> dict | None:
            klines = fwd.get((scan_kst, sym))
            if not klines:
                return None
            raw = compute_forward_metrics(klines)
            return raw if raw else None

        for sym in rank_long(rows, LONG_DIRECTION_CHAMPION, top_k):
            raw = metric_fn(sym)
            if not raw:
                continue
            feat = next(r["features"] for r in rows if r["symbol"] == sym)
            long_signals.append(
                _outcome_row("long", LONG_DIRECTION_CHAMPION, scan_kst, sym, feat, to_long_metrics(raw)),
            )

        for sym in rank_short(rows, SHORT_DIRECTION_CHAMPION, top_k):
            raw = metric_fn(sym)
            if not raw:
                continue
            feat = next(r["features"] for r in rows if r["symbol"] == sym)
            short_signals.append(
                _outcome_row("short", SHORT_DIRECTION_CHAMPION, scan_kst, sym, feat, to_short_metrics(raw)),
            )

    return long_signals, short_signals
