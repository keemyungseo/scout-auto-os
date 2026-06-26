"""Trade DNA — curve extraction from forward klines."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from scout_auto_os.engine.research.short_execution.constants import BAR_MINUTES, CHECKPOINT_MINUTES
from scout_auto_os.engine.runtime_audit.ablation_runner import _peak_and_mdd, _roi_at, simulate_exit


def _bar_idx(minutes: int) -> int:
    return max(0, minutes // BAR_MINUTES - 1)


def _vol_ratio(klines: list, i: int, lookback: int = 4) -> float:
    if i < 1 or not klines:
        return 1.0
    start = max(0, i - lookback)
    cur = float(klines[i][5])
    hist = [float(klines[j][5]) for j in range(start, i)]
    avg = statistics.mean(hist) if hist else cur or 1.0
    return round(cur / avg if avg else 1.0, 4)


def _momentum(klines: list, i: int, direction: str, bars: int = 2) -> float:
    if i < bars:
        return 0.0
    s = 0.0
    for j in range(i - bars + 1, i + 1):
        o, c = float(klines[j][1]), float(klines[j][4])
        raw = (c - o) / o * 100 if o else 0
        s += raw if direction == "long" else -raw
    return round(s, 4)


@dataclass
class TradeDNARecord:
    trade_key: str
    scan_kst: str
    symbol: str
    direction: str
    entry_score: float
    live_pattern: str
    features: dict = field(default_factory=dict)
    klines: list = field(default_factory=list)
    roi_curve: dict = field(default_factory=dict)
    volume_curve: dict = field(default_factory=dict)
    drawdown_curve: dict = field(default_factory=dict)
    peak_timing_min: int = 0
    peak_roi: float = 0.0
    final_roi_2h: float = 0.0
    final_roi_4h: float = 0.0
    max_drawdown: float = 0.0
    alive_delta_proxy: float = 0.0
    exit_pressure_proxy: float = 0.0
    is_winner: bool = False
    cluster_features: list[float] = field(default_factory=list)


def build_trade_dna(
    scan_kst: str,
    symbol: str,
    direction: str,
    entry_score: float,
    live_pattern: str,
    features: dict,
    klines: list,
) -> TradeDNARecord | None:
    if not klines or len(klines) < 2:
        return None
    max_bar = len(klines) - 1
    rois: list[tuple[int, float]] = []
    roi_curve: dict[str, float] = {}
    vol_curve: dict[str, float] = {}
    dd_curve: dict[str, float] = {}

    for mins in CHECKPOINT_MINUTES:
        bi = min(_bar_idx(mins), max_bar)
        roi = _roi_at(klines, bi, direction)
        key = f"roi_{mins}m"
        roi_curve[key] = roi
        vol_curve[f"vol_{mins}m"] = _vol_ratio(klines, bi)
        peak_to_here, _ = _peak_and_mdd(klines, bi, direction)
        cur = roi
        dd_curve[f"dd_{mins}m"] = round(peak_to_here - cur, 4)
        rois.append((mins, roi))

    peak_timing_min = 0
    peak_roi = -999.0
    for mins, roi in rois:
        if roi > peak_roi:
            peak_roi = roi
            peak_timing_min = mins

    end_2h = min(_bar_idx(120), max_bar)
    end_4h = min(_bar_idx(240), max_bar)
    final_2h = _roi_at(klines, end_2h, direction)
    final_4h = _roi_at(klines, end_4h, direction)
    _, max_dd = _peak_and_mdd(klines, end_2h, direction)

    mom_early = _momentum(klines, min(_bar_idx(30), max_bar), direction)
    mom_late = _momentum(klines, end_2h, direction)
    alive_delta = round(mom_late - mom_early, 4)
    exit_pressure = round(max(dd_curve.values()) if dd_curve else 0.0, 4)

    cluster_keys = []
    for mins in CHECKPOINT_MINUTES:
        cluster_keys.extend([
            roi_curve.get(f"roi_{mins}m", 0),
            vol_curve.get(f"vol_{mins}m", 1),
            dd_curve.get(f"dd_{mins}m", 0),
        ])
    cluster_keys.extend([
        peak_timing_min / 240.0,
        peak_roi,
        final_2h,
        max_dd,
        alive_delta,
        exit_pressure,
    ])

    return TradeDNARecord(
        trade_key=f"{scan_kst}|{symbol}|{direction}",
        scan_kst=scan_kst,
        symbol=symbol,
        direction=direction,
        entry_score=entry_score,
        live_pattern=live_pattern,
        features=features,
        klines=klines,
        roi_curve=roi_curve,
        volume_curve=vol_curve,
        drawdown_curve=dd_curve,
        peak_timing_min=peak_timing_min,
        peak_roi=round(peak_roi, 4),
        final_roi_2h=round(final_2h, 4),
        final_roi_4h=round(final_4h, 4),
        max_drawdown=round(max_dd, 4),
        alive_delta_proxy=alive_delta,
        exit_pressure_proxy=exit_pressure,
        is_winner=final_2h >= 3.0,
        cluster_features=cluster_keys,
    )


EXIT_MODES = (
    ("hold_60m", "hold_2h"),  # mapped via minutes below
    ("hold_90m", "expectation_proxy"),
    ("hold_120m", "hold_2h"),
    ("hold_180m", "hold_2h"),
    ("hold_240m", "hold_2h"),
    ("state_exit", "state_exit"),
    ("roi_trail", "pe_proxy"),
    ("expectation", "expectation_proxy"),
    ("full_dynamic", "full_exit"),
)


def _hold_return(klines: list, direction: str, minutes: int) -> tuple[float, int]:
    bi = min(_bar_idx(minutes), len(klines) - 1)
    return _roi_at(klines, bi, direction), bi * BAR_MINUTES


def simulate_exit_mode(klines: list, direction: str, mode_id: str) -> tuple[float, int, str]:
    if mode_id == "hold_60m":
        r, h = _hold_return(klines, direction, 60)
        return r, h, "hold_60m"
    if mode_id == "hold_180m":
        r, h = _hold_return(klines, direction, 180)
        return r, h, "hold_180m"
    if mode_id == "hold_240m":
        r, h = _hold_return(klines, direction, 240)
        return r, h, "hold_240m"
    mode_map = {
        "hold_90m": "expectation_proxy",
        "hold_120m": "hold_2h",
        "state_exit": "state_exit",
        "roi_trail": "pe_proxy",
        "expectation": "expectation_proxy",
        "full_dynamic": "full_exit",
    }
    internal = mode_map.get(mode_id, "hold_2h")
    ret, hold, reason, _, _ = simulate_exit(klines, direction, internal)
    return ret, hold, reason
