"""Per Trade Type optimal exit selection."""

from __future__ import annotations

import statistics

from scout_auto_os.engine.research.trade_dna.curve_builder import TradeDNARecord, simulate_exit_mode

EXIT_CANDIDATES = (
    "hold_60m",
    "hold_90m",
    "hold_120m",
    "hold_180m",
    "hold_240m",
    "state_exit",
    "roi_trail",
    "expectation",
    "full_dynamic",
)


def best_exit_per_cluster(
    records: list[TradeDNARecord],
    cluster_labels: list[int],
) -> list[dict]:
    by_cluster: dict[int, list[TradeDNARecord]] = {}
    for rec, label in zip(records, cluster_labels):
        by_cluster.setdefault(label, []).append(rec)

    rows: list[dict] = []
    for cid, cluster_recs in sorted(by_cluster.items()):
        mode_stats: dict[str, list[float]] = {m: [] for m in EXIT_CANDIDATES}
        for rec in cluster_recs:
            for mode in EXIT_CANDIDATES:
                ret, hold, _ = simulate_exit_mode(rec.klines, rec.direction, mode)
                mode_stats[mode].append(ret)

        best_mode = "hold_120m"
        best_avg = -999.0
        for mode, rets in mode_stats.items():
            if not rets:
                continue
            avg = statistics.mean(rets)
            if avg > best_avg:
                best_avg = avg
                best_mode = mode

        baseline = mode_stats.get("hold_120m", [0])
        baseline_avg = statistics.mean(baseline) if baseline else 0.0
        lift = round(best_avg - baseline_avg, 4)

        rows.append({
            "trade_type_id": f"TYPE_{cid}",
            "cluster_id": cid,
            "trade_count": len(cluster_recs),
            "best_exit_mode": best_mode,
            "best_exit_avg_roi": round(best_avg, 4),
            "hold_2h_avg_roi": round(baseline_avg, 4),
            "exit_lift_vs_hold2h": lift,
            **{f"avg_{m}": round(statistics.mean(v), 4) if v else 0 for m, v in mode_stats.items()},
        })
    return rows


def estimate_portfolio_lift(
    records: list[TradeDNARecord],
    cluster_labels: list[int],
    exit_table: list[dict],
) -> dict:
    exit_by_cluster = {r["cluster_id"]: r["best_exit_mode"] for r in exit_table}
    baseline_total = 0.0
    optimized_total = 0.0
    for rec, label in zip(records, cluster_labels):
        baseline_total += rec.final_roi_2h
        mode = exit_by_cluster.get(label, "hold_120m")
        ret, _, _ = simulate_exit_mode(rec.klines, rec.direction, mode)
        optimized_total += ret
    n = len(records) or 1
    return {
        "baseline_avg_roi": round(baseline_total / n, 4),
        "type_exit_avg_roi": round(optimized_total / n, 4),
        "expected_lift_pct": round((optimized_total - baseline_total) / abs(baseline_total) * 100, 2)
        if baseline_total else 0.0,
        "expected_lift_pp": round((optimized_total - baseline_total) / n, 4),
    }
