"""Forward-derived label metrics and per-scan ranking targets."""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from scout_auto_os.engine.research.formula_league_v2.metrics import return_12h_from_klines
from scout_auto_os.engine.research.target_discovery.constants import RELEVANCE_GRADES
from scout_auto_os.engine.research.zero_base.forward_eval import BAR_MINUTES, compute_forward_metrics


def _bar_idx(minutes: int) -> int:
    return max(0, minutes // BAR_MINUTES - 1)


def _max_return_up_to(klines: list, entry: float, minutes: int) -> float:
    if not klines or entry <= 0:
        return 0.0
    idx = min(_bar_idx(minutes), len(klines) - 1)
    highs = [float(k[2]) for k in klines[: idx + 1]]
    if not highs:
        return 0.0
    return round((max(highs) - entry) / entry * 100, 4)


def _uptrend_duration(klines: list, entry: float, minutes: int) -> float:
    if not klines or entry <= 0:
        return 0.0
    idx = min(_bar_idx(minutes), len(klines) - 1)
    count = 0
    prev = entry
    for i in range(idx + 1):
        close = float(klines[i][4])
        if close >= prev:
            count += 1
            prev = close
        else:
            break
    return float(count)


def compute_label_metrics(klines: list, entry: float | None = None) -> dict:
    """Extended forward metrics used as candidate label scores (label-only, not features)."""
    if not klines:
        return {}
    entry_px = entry or float(klines[0][1])
    if entry_px <= 0:
        return {}

    base = compute_forward_metrics(klines, entry_px)
    m = dict(base)
    m["return_12h"] = return_12h_from_klines(klines, entry_px)

    m["max_up_30m"] = _max_return_up_to(klines, entry_px, 30)
    m["max_up_1h"] = _max_return_up_to(klines, entry_px, 60)
    m["max_up_2h"] = float(m.get("max_return_2h", 0))
    m["max_up_6h"] = _max_return_up_to(klines, entry_px, 360)
    m["max_up_12h"] = _max_return_up_to(klines, entry_px, 720)

    m["mfe_2h"] = m["max_up_2h"]
    m["mae_2h"] = abs(float(m.get("min_return_2h", 0)))
    m["return_minus_dd"] = round(float(m["return_2h"]) + float(m["max_drawdown_2h"]), 4)
    m["return_per_risk"] = round(
        float(m["return_2h"]) / max(abs(float(m["max_drawdown_2h"])), 0.01), 4,
    )
    m["peak_efficiency"] = round(
        float(m["return_2h"]) / max(float(m["max_up_2h"]), 0.01), 4,
    )
    m["recovery_speed"] = round(float(m["return_2h"]) - float(m.get("min_return_2h", 0)), 4)

    idx_2h = min(_bar_idx(120), len(klines) - 1)
    bar_rets: list[float] = []
    for i in range(1, idx_2h + 1):
        prev = float(klines[i - 1][4])
        cur = float(klines[i][4])
        if prev > 0:
            bar_rets.append((cur - prev) / prev * 100)

    if len(bar_rets) >= 2:
        mu = statistics.mean(bar_rets)
        sd = statistics.pstdev(bar_rets)
        downs = [min(r, 0.0) for r in bar_rets]
        dd = statistics.pstdev(downs) if any(d < 0 for d in downs) else 1e-9
        m["intrabar_sharpe"] = round(mu / sd, 4) if sd > 1e-9 else 0.0
        m["intrabar_sortino"] = round(mu / dd, 4) if dd > 1e-9 else 0.0
        m["intrabar_volatility"] = round(sd, 4)
    else:
        m["intrabar_sharpe"] = 0.0
        m["intrabar_sortino"] = 0.0
        m["intrabar_volatility"] = 0.0

    m["hit_3pct"] = 1.0 if float(m["return_2h"]) >= 3.0 else 0.0
    m["hit_max_3pct"] = 1.0 if float(m["max_up_2h"]) >= 3.0 else 0.0
    m["breakout_success"] = (
        1.0 if float(m["max_up_2h"]) >= 4.0 and float(m["return_2h"]) >= 2.0 else 0.0
    )
    m["momentum_persist"] = round(
        min(float(m.get("return_1h", 0)), float(m["max_up_1h"])) if float(m.get("return_1h", 0)) > 0 else 0.0,
        4,
    )
    m["uptrend_duration"] = _uptrend_duration(klines, entry_px, 120)
    m["time_to_peak_score"] = round(
        1.0 - min(float(m.get("time_to_peak", 120)), 120.0) / 120.0, 4,
    )
    tt3 = m.get("time_to_3pct")
    m["time_to_3pct_score"] = round(
        1.0 - min(float(tt3 if tt3 is not None else 120), 120.0) / 120.0, 4,
    )

    horizons = [
        float(m.get("return_30m", 0)),
        float(m.get("return_1h", 0)),
        float(m.get("return_2h", 0)),
        float(m.get("return_4h", 0)),
        float(m.get("return_6h", 0)),
        float(m.get("return_12h", 0)),
    ]
    m["avg_return_multi"] = round(statistics.mean(horizons), 4)
    m["max_return_multi"] = round(max(horizons), 4)
    m["drawdown_resilience"] = round(float(m["return_2h"]) - abs(float(m["max_drawdown_2h"])), 4)
    return m


@dataclass(frozen=True)
class LabelSpec:
    label_id: str
    name: str
    rank_key: str
    category: str
    invert: bool = False
    description: str = ""


def _rank_value(row: dict, spec: LabelSpec) -> float:
    if spec.rank_key == "max_up_4h":
        val = float(row.get("max_up_4h") or 0)
    else:
        lm = row.get("label_metrics") or {}
        val = float(lm.get(spec.rank_key, 0))
    return -val if spec.invert else val


def apply_label_to_scan(rows: list[dict], spec: LabelSpec) -> None:
    """Assign training relevance/outcome_rank from label spec (mutates rows)."""
    ranked = sorted(rows, key=lambda r: (-_rank_value(r, spec), r["symbol"]))
    for i, row in enumerate(ranked, 1):
        row["outcome_rank"] = i
        row["relevance"] = RELEVANCE_GRADES.get(i, 0)
        row["label_top1"] = 1 if i == 1 else 0
        row["label_top2"] = 1 if i <= 2 else 0
        row["label_top3"] = 1 if i <= 3 else 0
        row["label_top5"] = 1 if i <= 5 else 0
        row["label_id"] = spec.label_id


def apply_label(rows: list[dict], spec: LabelSpec) -> list[dict]:
    """Return shallow copies with training labels applied per scan."""
    from collections import defaultdict

    by_scan: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_scan[r["scan_kst"]].append(dict(r))

    out: list[dict] = []
    for scan in sorted(by_scan):
        chunk = by_scan[scan]
        apply_label_to_scan(chunk, spec)
        out.extend(chunk)
    return out


def label_learnability(rows: list[dict], spec: LabelSpec) -> dict:
    """How separable is this label vs baseline within scans."""
    from collections import defaultdict

    by_scan: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_scan[r["scan_kst"]].append(r)

    top3_agree = 0
    rank_corr_sum = 0.0
    n_scans = 0
    for chunk in by_scan.values():
        if len(chunk) < 3:
            continue
        n_scans += 1
        labeled = apply_label(list(chunk), spec)
        base_top3 = {r["symbol"] for r in chunk if int(r.get("baseline_outcome_rank", 99)) <= 3}
        lab_top3 = {r["symbol"] for r in labeled if int(r.get("outcome_rank", 99)) <= 3}
        top3_agree += len(base_top3 & lab_top3) / max(len(base_top3 | lab_top3), 1)

        base_ranks = {r["symbol"]: int(r["baseline_outcome_rank"]) for r in chunk}
        lab_ranks = {r["symbol"]: int(r["outcome_rank"]) for r in labeled}
        syms = list(base_ranks.keys())
        if len(syms) >= 2:
            bm = statistics.mean(base_ranks[s] for s in syms)
            lm = statistics.mean(lab_ranks[s] for s in syms)
            num = sum((base_ranks[s] - bm) * (lab_ranks[s] - lm) for s in syms)
            den_b = sum((base_ranks[s] - bm) ** 2 for s in syms) ** 0.5
            den_l = sum((lab_ranks[s] - lm) ** 2 for s in syms) ** 0.5
            if den_b > 1e-9 and den_l > 1e-9:
                rank_corr_sum += num / (den_b * den_l)

    return {
        "label_id": spec.label_id,
        "top3_overlap_pct": round(top3_agree / max(n_scans, 1) * 100, 2),
        "rank_correlation": round(rank_corr_sum / max(n_scans, 1), 4),
        "scan_count": n_scans,
    }
