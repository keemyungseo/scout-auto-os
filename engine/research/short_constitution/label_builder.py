"""Short forward label metrics — independent from long, not sign-flip of long labels."""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from scout_auto_os.engine.research.directional.evaluation import to_short_metrics
from scout_auto_os.engine.research.short_constitution.constants import RELEVANCE_GRADES
from scout_auto_os.engine.research.zero_base.forward_eval import BAR_MINUTES, compute_forward_metrics


def _bar_idx(minutes: int) -> int:
    return max(0, minutes // BAR_MINUTES - 1)


def _min_return_up_to(klines: list, entry: float, minutes: int) -> float:
    if not klines or entry <= 0:
        return 0.0
    idx = min(_bar_idx(minutes), len(klines) - 1)
    lows = [float(k[3]) for k in klines[: idx + 1]]
    if not lows:
        return 0.0
    return round((min(lows) - entry) / entry * 100, 4)


def _max_return_up_to(klines: list, entry: float, minutes: int) -> float:
    if not klines or entry <= 0:
        return 0.0
    idx = min(_bar_idx(minutes), len(klines) - 1)
    highs = [float(k[2]) for k in klines[: idx + 1]]
    if not highs:
        return 0.0
    return round((max(highs) - entry) / entry * 100, 4)


def compute_short_label_metrics(klines: list, entry: float | None = None) -> dict:
    if not klines:
        return {}
    entry_px = entry or float(klines[0][1])
    if entry_px <= 0:
        return {}

    long_m = compute_forward_metrics(klines, entry_px)
    m = to_short_metrics(long_m)

    min2 = float(long_m.get("min_return_2h", 0))
    max2 = float(long_m.get("max_return_2h", 0))
    sr2 = float(m.get("short_return_2h", -float(long_m.get("return_2h", 0))))

    m["max_down_2h"] = round(-min2, 4) if min2 < 0 else 0.0
    m["max_down_4h"] = round(-_min_return_up_to(klines, entry_px, 240), 4)
    m["max_down_6h"] = round(-_min_return_up_to(klines, entry_px, 360), 4)
    m["max_up_adverse_2h"] = max2
    m["mae_short_2h"] = max2
    m["mfe_short_2h"] = m["max_down_2h"]

    m["return_plus_dd"] = round(sr2 - max2, 4)
    m["drawup_resilience"] = round(sr2 - abs(max2), 4)
    m["return_per_risk_short"] = round(sr2 / max(max2, 0.01), 4)
    m["risk_adjusted_short"] = round(sr2 - max2 * 0.5, 4)

    idx_2h = min(_bar_idx(120), len(klines) - 1)
    bar_rets: list[float] = []
    for i in range(1, idx_2h + 1):
        prev = float(klines[i - 1][4])
        cur = float(klines[i][4])
        if prev > 0:
            bar_rets.append(-(cur - prev) / prev * 100)

    if len(bar_rets) >= 2:
        mu = statistics.mean(bar_rets)
        sd = statistics.pstdev(bar_rets)
        m["intrabar_sharpe_short"] = round(mu / sd, 4) if sd > 1e-9 else 0.0
    else:
        m["intrabar_sharpe_short"] = 0.0

    m["hit_short_3pct"] = 1.0 if sr2 >= 3.0 else 0.0
    m["distribution_success"] = 1.0 if min2 <= -3.0 and sr2 >= 2.0 else 0.0
    m["capitulation_fade"] = 1.0 if min2 <= -4.0 and max2 < 2.0 else 0.0
    return m


@dataclass(frozen=True)
class ShortLabelSpec:
    label_id: str
    name: str
    rank_key: str
    category: str
    invert: bool = False


def _rank_val(row: dict, spec: ShortLabelSpec) -> float:
    lm = row.get("short_label_metrics") or {}
    if spec.rank_key == "max_down_4h_seed":
        val = float(row.get("max_up_4h") or 0)
        return val if spec.invert else -val
    val = float(lm.get(spec.rank_key, 0))
    return -val if spec.invert else val


def apply_short_label(rows: list[dict], spec: ShortLabelSpec) -> list[dict]:
    from collections import defaultdict

    by_scan: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_scan[r["scan_kst"]].append(dict(r))

    out: list[dict] = []
    for scan in sorted(by_scan):
        chunk = by_scan[scan]
        ranked = sorted(chunk, key=lambda r: (-_rank_val(r, spec), r["symbol"]))
        for i, row in enumerate(ranked, 1):
            row["outcome_rank"] = i
            row["relevance"] = RELEVANCE_GRADES.get(i, 0)
            row["label_top3"] = 1 if i <= 3 else 0
            row["label_id"] = spec.label_id
            out.append(row)
    return out
