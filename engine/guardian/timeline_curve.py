"""Position ROI snapshots along hold — from forward klines or cluster checkpoints."""

from __future__ import annotations

from datetime import datetime, timedelta

from scout_auto_os.engine.research.short_execution.constants import BAR_MINUTES, CHECKPOINT_MINUTES
from scout_auto_os.engine.runtime_audit.ablation_runner import _peak_and_mdd, _roi_at

KST_FMT = "%Y-%m-%d %H:%M:%S"
DEFAULT_MAX_MINUTES = 240


def _add_minutes(scan_kst: str, minutes: int) -> str:
    dt = datetime.strptime(scan_kst, KST_FMT)
    return (dt + timedelta(minutes=minutes)).strftime(KST_FMT)


def snapshots_from_klines(
    klines: list,
    direction: str,
    scan_kst: str,
    *,
    max_minutes: int = DEFAULT_MAX_MINUTES,
) -> list[dict]:
    """One snapshot per 15m bar — native data interval."""
    if not klines:
        return []
    out: list[dict] = []
    max_bar = len(klines) - 1
    for i in range(max_bar + 1):
        elapsed = (i + 1) * BAR_MINUTES
        if elapsed > max_minutes:
            break
        current = _roi_at(klines, i, direction)
        peak, _mdd = _peak_and_mdd(klines, i, direction)
        dd_from_peak = round(max(0.0, peak - current), 4)
        out.append({
            "timestamp": _add_minutes(scan_kst, elapsed),
            "elapsed_minutes": elapsed,
            "current_roi": current,
            "peak_roi": peak,
            "drawdown_from_peak": dd_from_peak,
        })
    return out


def snapshots_from_cluster(
    cluster_row: dict,
    direction: str,
    scan_kst: str,
    *,
    final_roi: float = 0.0,
    final_peak: float = 0.0,
    max_minutes: int = DEFAULT_MAX_MINUTES,
) -> list[dict]:
    """Fallback: CHECKPOINT_MINUTES ROI columns from trade_cluster."""
    points: list[tuple[int, float]] = [(0, 0.0)]
    for mins in CHECKPOINT_MINUTES:
        if mins > max_minutes:
            break
        key = f"roi_{mins}m"
        if cluster_row.get(key) not in ("", None):
            points.append((mins, float(cluster_row[key])))
    if max_minutes >= 240 and final_roi:
        points.append((max_minutes, final_roi))

    points = sorted(set(points), key=lambda x: x[0])
    out: list[dict] = []
    peak = 0.0
    for elapsed, roi in points:
        peak = max(peak, roi, final_peak)
        out.append({
            "timestamp": _add_minutes(scan_kst, elapsed),
            "elapsed_minutes": elapsed,
            "current_roi": round(roi, 4),
            "peak_roi": round(peak, 4),
            "drawdown_from_peak": round(max(0.0, peak - roi), 4),
        })
    return out


def build_trade_snapshots(
    trade_row: dict,
    *,
    klines: list | None = None,
    cluster_row: dict | None = None,
    max_minutes: int = DEFAULT_MAX_MINUTES,
) -> list[dict]:
    scan_kst = trade_row.get("scan_kst") or trade_row.get("trade_key", "").split("|")[0]
    direction = trade_row.get("direction", trade_row.get("side", "long")).lower()
    if klines:
        snaps = snapshots_from_klines(klines, direction, scan_kst, max_minutes=max_minutes)
        if snaps:
            return snaps
    cluster = cluster_row or {}
    return snapshots_from_cluster(
        cluster,
        direction,
        scan_kst,
        final_roi=float(trade_row.get("actual_roi", 0)),
        final_peak=float(trade_row.get("actual_peak_roi", 0)),
        max_minutes=max_minutes,
    )
