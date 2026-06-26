"""Unsupervised Trade Type discovery via curve clustering."""

from __future__ import annotations

import statistics

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from scout_auto_os.engine.research.trade_dna.curve_builder import TradeDNARecord


def choose_cluster_count(X: np.ndarray, min_k: int = 2, max_k: int = 10) -> tuple[int, list[dict]]:
    n = len(X)
    if n < 4:
        return 1, [{"k": 1, "silhouette": 0.0, "inertia": 0.0}]
    max_k = min(max_k, n // 3, 10)
    max_k = max(max_k, min_k)
    scores: list[dict] = []
    best_k = min_k
    best_score = -1.0
    for k in range(min_k, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        if len(set(labels)) < 2:
            continue
        sil = float(silhouette_score(X, labels))
        scores.append({"k": k, "silhouette": round(sil, 4), "inertia": round(float(km.inertia_), 2)})
        if sil > best_score:
            best_score = sil
            best_k = k
    if not scores:
        return 1, [{"k": 1, "silhouette": 0.0, "inertia": 0.0}]
    return best_k, scores


def cluster_trades(records: list[TradeDNARecord]) -> tuple[list[dict], dict]:
    if not records:
        return [], {"n_clusters": 0, "silhouette_scores": []}

    X = np.array([r.cluster_features for r in records], dtype=float)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    n_clusters, sil_scores = choose_cluster_count(Xs)

    if n_clusters <= 1:
        labels = [0] * len(records)
        centroids = [np.mean(Xs, axis=0).tolist()]
    else:
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(Xs).tolist()
        centroids = km.cluster_centers_.tolist()

    rows: list[dict] = []
    for rec, label in zip(records, labels):
        rows.append({
            "trade_key": rec.trade_key,
            "scan_kst": rec.scan_kst,
            "symbol": rec.symbol,
            "direction": rec.direction,
            "trade_type_id": f"TYPE_{label}",
            "cluster_id": int(label),
            "entry_score": rec.entry_score,
            "live_pattern": rec.live_pattern,
            "peak_timing_min": rec.peak_timing_min,
            "peak_roi": rec.peak_roi,
            "final_roi_2h": rec.final_roi_2h,
            "final_roi_4h": rec.final_roi_4h,
            "max_drawdown": rec.max_drawdown,
            "alive_delta_proxy": rec.alive_delta_proxy,
            "exit_pressure_proxy": rec.exit_pressure_proxy,
            "is_winner": int(rec.is_winner),
            **{k: v for k, v in rec.roi_curve.items()},
            **{k: v for k, v in rec.volume_curve.items()},
            **{k: v for k, v in rec.drawdown_curve.items()},
        })

    meta = {
        "n_clusters": n_clusters,
        "silhouette_scores": sil_scores,
        "n_trades": len(records),
        "scaler_mean": scaler.mean_.tolist() if len(records) > 1 else [],
        "centroids": centroids,
    }
    return rows, meta


def infer_archetype_label(cluster_rows: list[dict]) -> str:
    """Data-derived descriptor (not threshold-tuned naming)."""
    if not cluster_rows:
        return "unknown"
    avg_peak_time = statistics.mean(int(r["peak_timing_min"]) for r in cluster_rows)
    avg_peak = statistics.mean(float(r["peak_roi"]) for r in cluster_rows)
    avg_2h = statistics.mean(float(r["final_roi_2h"]) for r in cluster_rows)
    avg_dd = statistics.mean(float(r["max_drawdown"]) for r in cluster_rows)
    roi_30 = statistics.mean(float(r.get("roi_30m", 0)) for r in cluster_rows)

    if avg_2h < 0 and avg_peak < 2:
        return "failed_momentum"
    if avg_peak_time >= 120 and avg_peak >= 8 and avg_2h >= 3:
        return "late_peak_runner"
    if roi_30 >= 5 and avg_peak_time <= 60:
        return "early_burst"
    if avg_dd >= 8 and avg_2h < avg_peak * 0.5:
        return "peak_fade"
    if abs(avg_2h) < 2 and avg_dd < 3:
        return "slow_grinder"
    if avg_2h >= 5 and avg_peak_time <= 90:
        return "trend_continuation"
    if avg_peak >= 6 and avg_2h >= 4:
        return "breakout_hold"
    return "mixed_profile"
