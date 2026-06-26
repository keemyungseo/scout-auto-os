"""3+3 slot simulation using prediction engine scores."""

from __future__ import annotations

import statistics

from scout_auto_os.engine.research.directional.dna.formulas import ClusterFormula
from scout_auto_os.engine.research.directional.evaluation import MAX_LONG_SLOTS, MAX_SHORT_SLOTS
from scout_auto_os.engine.research.directional.prediction.engine import predict_symbol
from scout_auto_os.engine.research.directional.evaluation import to_long_metrics, to_short_metrics
from scout_auto_os.engine.research.zero_base.forward_eval import compute_forward_metrics


def simulate_prediction_slots(
    by_scan: dict,
    fwd: dict,
    scans: list[str],
    long_formulas: list[ClusterFormula],
    short_formulas: list[ClusterFormula],
    expected_returns: dict[str, dict],
    long_slots: int = MAX_LONG_SLOTS,
    short_slots: int = MAX_SHORT_SLOTS,
) -> tuple[dict, list[dict]]:
    long_rets: list[float] = []
    short_rets: list[float] = []
    detail_rows: list[dict] = []

    for scan_kst in scans:
        rows = by_scan[scan_kst]
        long_ranked: list[tuple[str, float, str]] = []
        short_ranked: list[tuple[str, float, str]] = []

        for row in rows:
            pred = predict_symbol(row["features"], long_formulas, short_formulas, expected_returns)
            long_ranked.append((row["symbol"], pred["long_score"], pred["top_long_cluster"]))
            short_ranked.append((row["symbol"], pred["short_score"], pred["top_short_cluster"]))

        long_ranked.sort(key=lambda x: x[1], reverse=True)
        short_ranked.sort(key=lambda x: x[1], reverse=True)

        for sym, score, cluster in long_ranked[:long_slots]:
            klines = fwd.get((scan_kst, sym))
            if not klines:
                continue
            raw = compute_forward_metrics(klines)
            if not raw:
                continue
            m = to_long_metrics(raw)
            long_rets.append(float(m["return_2h"]))
            detail_rows.append({
                "scan_time_kst": scan_kst,
                "direction": "long",
                "symbol": sym,
                "score": score,
                "cluster": cluster,
                "return_2h": m["return_2h"],
            })

        for sym, score, cluster in short_ranked[:short_slots]:
            klines = fwd.get((scan_kst, sym))
            if not klines:
                continue
            raw = compute_forward_metrics(klines)
            if not raw:
                continue
            m = to_short_metrics(raw)
            short_rets.append(float(m.get("short_return_2h", -float(m.get("return_2h", 0)))))
            detail_rows.append({
                "scan_time_kst": scan_kst,
                "direction": "short",
                "symbol": sym,
                "score": score,
                "cluster": cluster,
                "return_2h": m.get("short_return_2h", -float(m.get("return_2h", 0))),
            })

    combined = long_rets + short_rets
    summary = {
        "long_slots": long_slots,
        "short_slots": short_slots,
        "validation_scans": len(scans),
        "long_picks": len(long_rets),
        "short_picks": len(short_rets),
        "long_avg_return_2h": round(statistics.mean(long_rets), 4) if long_rets else 0,
        "short_avg_return_2h": round(statistics.mean(short_rets), 4) if short_rets else 0,
        "combined_avg_return_2h": round(statistics.mean(combined), 4) if combined else 0,
        "combined_sample_count": len(combined),
    }
    return summary, detail_rows
