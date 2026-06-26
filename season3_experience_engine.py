"""
Scout Season3 - Experience Database & Similarity Engine

Experience-based probability from nearest past snapshots.
NO rules | NO Belief/Narrative | NO composite ScoutScore

Usage:
  python season3_experience_engine.py build --from-cache
  python season3_experience_engine.py build --from-top10
  python season3_experience_engine.py query --symbol AIOTUSDT
  python season3_experience_engine.py compare-methods
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
import urllib.error
from datetime import datetime, timedelta, timezone
from pathlib import Path

import top10_gainer_learning_20260613 as t10

from season3_snapshot import FEATURE_KEYS, build_snapshot_features, forward_outcomes
from season2_p37_scout_decision_hierarchy import load_csv, pf, write_csv

LOGS_DIR = Path("logs")
OUT_DIR = LOGS_DIR / "season3"
DB_JSONL = OUT_DIR / "experience_db.jsonl"
META_JSON = OUT_DIR / "experience_meta.json"
QUERY_TXT = OUT_DIR / "last_query_output.txt"
COMPARE_CSV = OUT_DIR / "similarity_method_compare.csv"
IMPORTANCE_CSV = OUT_DIR / "feature_importance.csv"
CORR_CSV = OUT_DIR / "feature_correlation.csv"
CLUSTER_CSV = OUT_DIR / "cluster_analysis.csv"
INFO_GAIN_CSV = OUT_DIR / "information_gain.csv"
SHAP_CSV = OUT_DIR / "shap_proxy.csv"
BLIND_CSV = OUT_DIR / "blind_validation.csv"
REPORT_TXT = OUT_DIR / "season3_research_report.txt"
OUT_DIR.mkdir(parents=True, exist_ok=True)

KST = timezone(timedelta(hours=9))
SIMILARITY_METHODS = ("cosine", "euclidean", "mahalanobis_diag", "weighted_euclidean")
DEFAULT_K = 200
RANDOM_BASELINE_2H = 0.9214
MODEL_FEATURES = [k for k in FEATURE_KEYS if not k.startswith(("open", "high", "low", "close"))]

TOP10_CSV_MAP = {
    "return_24h_percent": "return_24h_pct",
    "return_prev_2h_percent": "return_2h_pct",
    "return_prev_4h_percent": "return_4h_pct",
    "return_prev_12h_percent": "return_1h_pct",
    "volume_ratio_ma24": "volume_ratio_ma24",
    "volume_ma6": "volume_ma6",
    "volume_ma12": "volume_ma12",
    "volume_ma24": "volume_ma24",
    "volume_current": "volume",
    "atr_percent": "atr_pct",
    "atr_ratio": "atr_ratio",
    "ma24_slope_percent": "ma20_slope_pct",
    "ma48_slope_percent": "ma60_slope_pct",
    "distance_ma6_percent": "ma5_dist_pct",
    "distance_ma12_percent": "ma10_dist_pct",
    "distance_ma24_percent": "ma20_dist_pct",
    "distance_ma48_percent": "ma60_dist_pct",
    "current_body_percent": "body_ratio",
    "range_expansion_ratio": "range_pct",
    "volume_acceleration_ratio": "dollar_volume_ratio",
}


def parse_scan_kst(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%d %H:%M:%S").replace(tzinfo=KST)


def top10_row_to_snapshot(row: dict, source: str) -> dict | None:
    scan = row.get("scan_time_kst", "")
    sym = row.get("symbol", "")
    if not scan or not sym:
        return None

    mp = pf(row.get("max_profit")) or 0.0
    f4 = pf(row.get("forward_4h")) or 0.0
    rec: dict = {
        "snapshot_id": f"{scan}_{sym}",
        "scan_time_kst": scan,
        "symbol": sym,
        "source": source,
        "max_excursion_2h": mp if mp > 0 else f4 / 2,
        "max_excursion_4h": f4 if f4 != 0 else mp,
        "max_excursion_30m": mp / 4 if mp > 0 else f4 / 8,
        "max_excursion_1h": mp / 2 if mp > 0 else f4 / 4,
        "max_excursion_best": mp,
        "max_drawdown_best": pf(row.get("max_drawdown")) or 0.0,
        "hit_5pct_plus": 1.0 if mp >= 5 else 0.0,
        "hit_10pct_plus": 1.0 if mp >= 10 else 0.0,
    }

    for csv_col, feat in TOP10_CSV_MAP.items():
        rec["f_" + feat] = pf(row.get(csv_col)) or 0.0

    rec["f_range_compression"] = 1.0 if row.get("pre6_tight_range") == "YES" else 0.5
    rec["f_btc_stable_flag"] = 0.0
    rec["f_dollar_volume_ratio"] = rec.get("f_dollar_volume_ratio") or rec.get("f_volume_ratio_ma24", 0.0)

    for k in MODEL_FEATURES:
        rec.setdefault("f_" + k, 0.0)
    return rec


def btc_metrics_at(scan_kst: str) -> dict:
    try:
        scan_dt = parse_scan_kst(scan_kst)
        end_ms = int(scan_dt.timestamp() * 1000)
        klines = t10.fetch_klines_before("BTCUSDT", t10.INTERVAL_2H, end_ms, t10.RANKING_KLINES_2H)
        if len(klines) < 10:
            return {}
        c = float(klines[-1][4])
        c24 = float(klines[-(t10.CANDLES_24H_2H + 1)][4])
        c2 = float(klines[-2][4])
        r24 = (c - c24) / c24 * 100 if c24 else 0.0
        r2 = (c - c2) / c2 * 100 if c2 else 0.0
        atr = t10.average_true_range_percent(klines[-25:-1], c)
        return {"btc_return_24h": r24, "btc_return_2h": r2, "btc_atr_pct": atr}
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError):
        return {}


def build_snapshot_from_api(symbol: str, scan_kst: str, btc_cache: dict) -> dict | None:
    scan_dt = parse_scan_kst(scan_kst)
    end_ms = int(scan_dt.timestamp() * 1000)
    try:
        k2h = t10.fetch_klines_before(symbol, t10.INTERVAL_2H, end_ms, t10.ANALYSIS_KLINES_2H)
        k1h = t10.fetch_klines_before(symbol, t10.INTERVAL_1H, end_ms, t10.ANALYSIS_KLINES_1H)
        fwd = t10.fetch_klines_forward(symbol, end_ms, end_ms + 4 * t10.INTERVAL_2H_MS)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError):
        return None

    if scan_kst not in btc_cache:
        btc_cache[scan_kst] = btc_metrics_at(scan_kst)
    feats = build_snapshot_features(k2h, k1h, None, btc_cache.get(scan_kst))
    if not feats:
        return None

    entry = float(k2h[-1][4])
    outcomes = forward_outcomes(entry, fwd, scan_dt)
    rec = flatten_snapshot(feats)
    rec.update({
        "snapshot_id": f"{scan_kst}_{symbol}",
        "scan_time_kst": scan_kst,
        "symbol": symbol,
        "source": "kline_api",
        **outcomes,
    })
    return rec


def snapshot_vector(rec: dict, keys: list[str] | None = None) -> list[float]:
    keys = keys or MODEL_FEATURES
    return [float(rec.get("f_" + k, rec.get(k, 0.0)) or 0.0) for k in keys]


def load_db() -> list[dict]:
    if not DB_JSONL.exists():
        return []
    rows: list[dict] = []
    for line in DB_JSONL.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def append_db(records: list[dict]) -> None:
    with DB_JSONL.open("a", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def normalize_stats(rows: list[dict], keys: list[str]) -> tuple[list[list[float]], dict[str, tuple[float, float]]]:
    vecs = [snapshot_vector(r, keys) for r in rows]
    stats: dict[str, tuple[float, float]] = {}
    for i, key in enumerate(keys):
        vals = [v[i] for v in vecs]
        mu = statistics.mean(vals)
        sd = statistics.pstdev(vals) if len(vals) > 1 else 1.0
        if sd < 1e-9:
            sd = 1.0
        stats[key] = (mu, sd)
    normed = [[(v[i] - stats[keys[i]][0]) / stats[keys[i]][1] for i in range(len(keys))] for v in vecs]
    return normed, stats


def zscore_vec(vec: list[float], keys: list[str], stats: dict[str, tuple[float, float]]) -> list[float]:
    return [(vec[i] - stats[keys[i]][0]) / stats[keys[i]][1] for i in range(len(keys))]


def cosine_distance(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na < 1e-12 or nb < 1e-12:
        return 1.0
    return 1.0 - dot / (na * nb)


def euclidean(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def mahalanobis_diag(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def weighted_euclidean(a: list[float], b: list[float], weights: list[float]) -> float:
    return math.sqrt(sum(w * (x - y) ** 2 for x, y, w in zip(a, b, weights)))


def knn_neighbors(
    query: list[float],
    db_norm: list[list[float]],
    method: str,
    k: int,
    weights: list[float] | None = None,
) -> list[tuple[int, float]]:
    dists: list[tuple[int, float]] = []
    for i, vec in enumerate(db_norm):
        if method == "cosine":
            d = cosine_distance(query, vec)
        elif method == "mahalanobis_diag":
            d = mahalanobis_diag(query, vec)
        elif method == "weighted_euclidean":
            d = weighted_euclidean(query, vec, weights or [1.0] * len(query))
        else:
            d = euclidean(query, vec)
        dists.append((i, d))
    dists.sort(key=lambda x: x[1])
    return dists[:k]


def compute_weights_from_importance(importance: dict[str, float]) -> list[float]:
    vals = [max(0.01, importance.get(k, 0.01)) for k in MODEL_FEATURES]
    s = sum(vals) or 1.0
    return [v / s * len(vals) for v in vals]


def permutation_importance(rows: list[dict], keys: list[str], method: str, k: int = 100) -> dict[str, float]:
    """Proxy importance: correlation with hit_5pct_plus label."""
    labels = [float(r.get("hit_5pct_plus", 0)) for r in rows]
    if len(set(labels)) < 2:
        return {k: 0.0 for k in keys}
    out: dict[str, float] = {}
    for key in keys:
        xs = [float(r.get("f_" + key, r.get(key, 0))) for r in rows]
        mx, my = statistics.mean(xs), statistics.mean(labels)
        vx = sum((x - mx) ** 2 for x in xs)
        if vx < 1e-12:
            out[key] = 0.0
            continue
        cov = sum((xs[i] - mx) * (labels[i] - my) for i in range(len(xs)))
        out[key] = abs(cov / vx)
    mx_val = max(out.values()) or 1.0
    return {k: v / mx_val for k, v in out.items()}


def mutual_info_proxy(xs: list[float], ys: list[float], bins: int = 5) -> float:
    n = len(xs)
    if n < 10:
        return 0.0
    lo_x, hi_x = min(xs), max(xs)
    lo_y, hi_y = min(ys), max(ys)
    if hi_x <= lo_x or hi_y <= lo_y:
        return 0.0
    joint: dict[tuple[int, int], int] = {}
    cx: dict[int, int] = {}
    cy: dict[int, int] = {}
    for i in range(n):
        bx = min(bins - 1, int((xs[i] - lo_x) / (hi_x - lo_x + 1e-9) * bins))
        by = min(bins - 1, int((ys[i] - lo_y) / (hi_y - lo_y + 1e-9) * bins))
        joint[(bx, by)] = joint.get((bx, by), 0) + 1
        cx[bx] = cx.get(bx, 0) + 1
        cy[by] = cy.get(by, 0) + 1
    mi = 0.0
    for (bx, by), cnt in joint.items():
        p_xy = cnt / n
        p_x = cx[bx] / n
        p_y = cy[by] / n
        if p_xy > 0:
            mi += p_xy * math.log2(p_xy / (p_x * p_y + 1e-12))
    return max(0.0, mi)


def experience_query(
    query_rec: dict,
    db: list[dict],
    method: str = "cosine",
    k: int = DEFAULT_K,
    exclude_same: bool = True,
) -> dict:
    keys = MODEL_FEATURES
    pool = [r for r in db if not (exclude_same and r.get("symbol") == query_rec.get("symbol") and r.get("scan_time_kst") == query_rec.get("scan_time_kst"))]
    if len(pool) < 20:
        pool = db[:]

    db_norm, stats = normalize_stats(pool, keys)
    qvec = zscore_vec(snapshot_vector(query_rec, keys), keys, stats)
    importance = permutation_importance(pool, keys, method, k=min(k, len(pool)))
    weights = compute_weights_from_importance(importance)

    if method == "weighted_euclidean":
        neighbors = knn_neighbors(qvec, db_norm, method, k, weights)
    else:
        neighbors = knn_neighbors(qvec, db_norm, method, k)

    neighbor_rows = [pool[i] for i, _ in neighbors]
    hit5 = sum(float(r.get("hit_5pct_plus", 0)) for r in neighbor_rows)
    hit10 = sum(float(r.get("hit_10pct_plus", 0)) for r in neighbor_rows)
    n = len(neighbor_rows)
    long_prob = hit5 / n * 100 if n else 0.0

    moves = [float(r.get("max_excursion_best", r.get("max_excursion_2h", 0))) for r in neighbor_rows]
    expected_move = statistics.mean(moves) if moves else 0.0

    # Expected duration: first horizon reaching 3% median among winners
    dur_labels = []
    for r in neighbor_rows:
        if float(r.get("hit_5pct_plus", 0)) >= 1:
            if float(r.get("max_excursion_30m", 0)) >= 3:
                dur_labels.append("30m")
            elif float(r.get("max_excursion_1h", 0)) >= 3:
                dur_labels.append("1h")
            else:
                dur_labels.append("2h")
    duration = "30m ~ 2h"
    if dur_labels:
        if dur_labels.count("30m") > len(dur_labels) / 2:
            duration = "30m ~ 1h"
        elif dur_labels.count("2h") > len(dur_labels) / 2:
            duration = "1h ~ 4h"

    spread = statistics.pstdev([float(r.get("hit_5pct_plus", 0)) for r in neighbor_rows]) if n > 1 else 0.5
    if n >= 500 and spread < 0.35:
        confidence = "High"
    elif n >= 100:
        confidence = "Medium"
    else:
        confidence = "Low"

    # Reasons from feature lift vs pool mean
    reasons: list[tuple[str, float]] = []
    q_feats = {k: float(query_rec.get("f_" + k, query_rec.get(k, 0))) for k in keys}
    for key in keys:
        pool_vals = [float(r.get("f_" + key, r.get(key, 0))) for r in neighbor_rows if float(r.get("hit_5pct_plus", 0)) >= 1]
        all_vals = [float(r.get("f_" + key, r.get(key, 0))) for r in pool]
        if not pool_vals or not all_vals:
            continue
        lift = statistics.mean(pool_vals) - statistics.mean(all_vals)
        reasons.append((key, lift * importance.get(key, 0)))
    reasons.sort(key=lambda x: abs(x[1]), reverse=True)

    reason_lines = []
    label_map = {
        "volume_ratio_ma24": "Volume MA breakout",
        "range_compression": "Range compression then expansion",
        "ma20_slope_pct": "MA20 slope turn",
        "dollar_volume_ratio": "Dollar volume increase",
        "btc_stable_flag": "BTC stability",
        "dollar_volume_ratio": "Dollar volume surge",
        "atr_ratio": "Volatility expansion",
        "return_2h_pct": "Short-term momentum",
    }
    for key, _ in reasons[:5]:
        reason_lines.append(label_map.get(key, key.replace("_", " ")))

    return {
        "long_probability_pct": round(long_prob, 1),
        "confidence": confidence,
        "expected_move_pct": round(expected_move, 1),
        "expected_duration": duration,
        "support_samples": n,
        "total_db_size": len(db),
        "hit_5pct_count": int(hit5),
        "hit_5pct_rate": round(hit5 / n * 100, 1) if n else 0,
        "hit_10pct_count": int(hit10),
        "method": method,
        "top_similar_cases": [
            {
                "symbol": r.get("symbol"),
                "scan_time_kst": r.get("scan_time_kst"),
                "max_excursion_best": r.get("max_excursion_best"),
                "hit_5pct_plus": r.get("hit_5pct_plus"),
            }
            for r in neighbor_rows[:5]
        ],
        "reasons": reason_lines,
        "feature_importance": importance,
    }


def format_output(q: dict, symbol: str) -> str:
    lines = [
        "===== Scout Season3 Experience Query =====",
        "",
        f"Symbol: {symbol}",
        "",
        "LONG Probability",
        "",
        f"{q['long_probability_pct']:.0f}%",
        "",
        "Confidence",
        "",
        q["confidence"],
        "",
        "Expected Move",
        "",
        f"{q['expected_move_pct']:+.1f}%",
        "",
        "Expected Duration",
        "",
        q["expected_duration"],
        "",
        "Support Samples",
        "",
        f"{q['support_samples']:,}",
        "",
        "Reason",
        "",
    ]
    for i, reason in enumerate(q["reasons"][:5], 1):
        lines.append(f"{i}.")
        lines.append("")
        lines.append(reason)
        lines.append("")
    lines.extend([
        "--------------------------------------------------",
        "",
        f"Past similar cases: {q['support_samples']:,}",
        f"5%+ rise among neighbors: {q['hit_5pct_count']:,} ({q['hit_5pct_rate']:.1f}%)",
        f"10%+ rise among neighbors: {q['hit_10pct_count']:,}",
        f"Total experience DB: {q['total_db_size']:,}",
        f"Similarity method: {q['method']}",
        "",
        "Top similar cases:",
    ])
    for case in q["top_similar_cases"]:
        lines.append(
            f"  - {case['symbol']} @ {case['scan_time_kst']} | "
            f"max={case.get('max_excursion_best', 0)}% hit5={case.get('hit_5pct_plus', 0)}"
        )
    lines.extend(["", "Learning recommendation: NO_ACTION"])
    return "\n".join(lines)


def flatten_snapshot(rec: dict, prefix: str = "f_") -> dict:
    out = {k: v for k, v in rec.items() if not k.startswith(prefix)}
    for k in FEATURE_KEYS:
        if k in rec:
            out[prefix + k] = rec[k]
    return out


def build_from_top10() -> int:
    """Bootstrap experience DB from local top10 CSV rows with enriched features."""
    count = 0
    for path in sorted(LOGS_DIR.glob("top10_gainer_learning_*.csv")):
        batch: list[dict] = []
        for row in load_csv(path):
            rec = top10_row_to_snapshot(row, path.name)
            if rec:
                batch.append(rec)
        if batch:
            append_db(batch)
            count += len(batch)
    return count


def build_from_klines(max_snapshots: int = 50, sleep_sec: float = 0.35) -> int:
    """Build full snapshots from kline API (rate-limited)."""
    seen: set[str] = set()
    tasks: list[tuple[str, str]] = []
    for path in sorted(LOGS_DIR.glob("top10_gainer_learning_*.csv")):
        for row in load_csv(path):
            key = f"{row.get('scan_time_kst')}_{row.get('symbol')}"
            if key not in seen:
                seen.add(key)
                tasks.append((row.get("scan_time_kst", ""), row.get("symbol", "")))

    btc_cache: dict = {}
    count = 0
    for scan_kst, sym in tasks[:max_snapshots]:
        if not scan_kst or not sym:
            continue
        rec = build_snapshot_from_api(sym, scan_kst, btc_cache)
        if rec:
            append_db([rec])
            count += 1
        time.sleep(sleep_sec)
    return count


def build_from_cache() -> int:
    cache_dir = LOGS_DIR / "universe_research" / "snapshots"
    if not cache_dir.exists():
        return 0

    count = 0
    batch: list[dict] = []
    for path in sorted(cache_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        scan_kst = data.get("scan_time_kst", "")
        for s in data.get("symbols", []):
            best = max(abs(s.get("forward_2h_pct", 0)), 0)
            rec = flatten_snapshot({
                **{k: s.get(k, 0) for k in (
                    "return_24h_percent", "return_2h_percent", "volume_acceleration_ratio",
                    "atr_percent", "drawdown_from_24h_high_pct", "position_24h_percent",
                )},
                "volume_ratio_ma24": s.get("relative_volume", 0),
                "dollar_volume_ratio": s.get("relative_volume", 0),
                "range_compression": 0.8,
                "ma20_slope_pct": 0,
                "btc_stable_flag": 0,
            })
            rec.update({
                "snapshot_id": f"{scan_kst}_{s['symbol']}",
                "scan_time_kst": scan_kst,
                "symbol": s["symbol"],
                "source": "universe_cache",
                "max_excursion_best": best,
                "max_excursion_2h": s.get("forward_2h_pct", 0),
                "hit_5pct_plus": 1.0 if best >= 5 else 0.0,
                "hit_10pct_plus": 1.0 if best >= 10 else 0.0,
            })
            batch.append(rec)
    if batch:
        append_db(batch)
        count = len(batch)
    return count


def pearson_corr(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 3:
        return 0.0
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    if den_x < 1e-12 or den_y < 1e-12:
        return 0.0
    return num / (den_x * den_y)


def information_gain(xs: list[float], labels: list[float], bins: int = 5) -> float:
    n = len(xs)
    if n < 20:
        return 0.0
    lo, hi = min(xs), max(xs)
    if hi <= lo:
        return 0.0

    def entropy(vals: list[float]) -> float:
        if not vals:
            return 0.0
        p1 = sum(vals) / len(vals)
        p0 = 1 - p1
        ent = 0.0
        for p in (p0, p1):
            if p > 0:
                ent -= p * math.log2(p)
        return ent

    base = entropy(labels)
    groups: dict[int, list[float]] = {}
    for i in range(n):
        b = min(bins - 1, int((xs[i] - lo) / (hi - lo + 1e-9) * bins))
        groups.setdefault(b, []).append(labels[i])
    weighted = sum(len(g) / n * entropy(g) for g in groups.values())
    return max(0.0, base - weighted)


def cluster_analysis(db: list[dict], k: int = 5) -> list[dict]:
    if len(db) < k * 10:
        return []
    keys = MODEL_FEATURES
    vecs, stats = normalize_stats(db, keys)
    random.seed(42)
    centroids = [vecs[i][:] for i in random.sample(range(len(vecs)), k)]
    for _ in range(8):
        buckets: list[list[list[float]]] = [[] for _ in range(k)]
        for vec in vecs:
            dists = [euclidean(vec, c) for c in centroids]
            buckets[dists.index(min(dists))].append(vec)
        for i in range(k):
            if buckets[i]:
                centroids[i] = [
                    statistics.mean(v[j] for v in buckets[i]) for j in range(len(keys))
                ]

    rows = []
    for ci, centroid in enumerate(centroids):
        members = []
        for idx, vec in enumerate(vecs):
            dists = [euclidean(vec, c) for c in centroids]
            if dists.index(min(dists)) == ci:
                members.append(db[idx])
        if not members:
            continue
        hit5 = sum(float(r.get("hit_5pct_plus", 0)) for r in members)
        rows.append({
            "cluster_id": ci,
            "size": len(members),
            "hit_5pct_rate": round(hit5 / len(members), 4),
            "mean_max_excursion": round(
                statistics.mean(float(r.get("max_excursion_best", 0)) for r in members), 2
            ),
            "learning_recommendation": "NO_ACTION",
        })
    rows.sort(key=lambda r: r["hit_5pct_rate"], reverse=True)
    return rows


def shap_proxy(db: list[dict], sample_n: int = 40) -> dict[str, float]:
    """Leave-feature-at-mean impact on neighbor probability."""
    if len(db) < 100:
        return {k: 0.0 for k in MODEL_FEATURES}
    random.seed(42)
    samples = random.sample(db, min(sample_n, len(db)))
    keys = MODEL_FEATURES
    pool_norm, stats = normalize_stats(db, keys)
    pool_means = [statistics.mean(v[i] for v in pool_norm) for i in range(len(keys))]
    impact: dict[str, float] = {k: 0.0 for k in keys}

    for rec in samples:
        sid = rec.get("snapshot_id")
        train = [r for r in db if r.get("snapshot_id") != sid]
        base_q = experience_query(rec, train, method="cosine", k=min(100, len(train)))
        base_p = base_q["long_probability_pct"]
        qvec = zscore_vec(snapshot_vector(rec, keys), keys, stats)
        for i, key in enumerate(keys):
            perturbed = qvec[:]
            perturbed[i] = pool_means[i]
            neighbors = knn_neighbors(perturbed, pool_norm, "cosine", min(100, len(train)))
            neighbor_rows = [train[j] for j, _ in neighbors]
            hit5 = sum(float(r.get("hit_5pct_plus", 0)) for r in neighbor_rows)
            n = len(neighbor_rows) or 1
            pert_p = hit5 / n * 100
            impact[key] += abs(base_p - pert_p)

    mx = max(impact.values()) or 1.0
    return {k: v / mx for k, v in impact.items()}


def write_feature_analysis(db: list[dict]) -> None:
    keys = MODEL_FEATURES
    labels = [float(r.get("hit_5pct_plus", 0)) for r in db]
    corr_rows = []
    ig_rows = []
    for key in keys:
        xs = [float(r.get("f_" + key, r.get(key, 0))) for r in db]
        corr_rows.append({
            "feature": key,
            "pearson_with_hit5": round(pearson_corr(xs, labels), 4),
            "learning_recommendation": "NO_ACTION",
        })
        ig_rows.append({
            "feature": key,
            "information_gain": round(information_gain(xs, labels), 4),
            "learning_recommendation": "NO_ACTION",
        })
    corr_rows.sort(key=lambda r: abs(r["pearson_with_hit5"]), reverse=True)
    ig_rows.sort(key=lambda r: r["information_gain"], reverse=True)
    write_csv(CORR_CSV, corr_rows)
    write_csv(INFO_GAIN_CSV, ig_rows)
    write_csv(CLUSTER_CSV, cluster_analysis(db))
    shap = shap_proxy(db)
    write_csv(SHAP_CSV, [
        {"feature": k, "shap_proxy": round(v, 4), "learning_recommendation": "NO_ACTION"}
        for k, v in sorted(shap.items(), key=lambda x: x[1], reverse=True)
    ])


def blind_validate(db: list[dict], method: str = "cosine", k: int = DEFAULT_K) -> list[dict]:
    """Scan-level LOOCV: rank by experience probability, compare Top2 vs Random Top2."""
    scans = sorted(set(r.get("scan_time_kst", "") for r in db if r.get("scan_time_kst")))
    random.seed(42)
    rows: list[dict] = []
    scout_returns: list[float] = []
    random_returns: list[float] = []

    for scan in scans:
        test = [r for r in db if r.get("scan_time_kst") == scan]
        train = [r for r in db if r.get("scan_time_kst") != scan]
        if len(test) < 2 or len(train) < 30:
            continue

        ranked: list[tuple[dict, float]] = []
        for rec in test:
            q = experience_query(rec, train, method=method, k=min(k, len(train)), exclude_same=True)
            ranked.append((rec, q["long_probability_pct"]))
        ranked.sort(key=lambda x: x[1], reverse=True)

        scout_picks = [r for r, _ in ranked[:2]]
        rand_picks = random.sample(test, 2)
        scout_ret = statistics.mean(float(r.get("max_excursion_best", 0)) for r in scout_picks)
        rand_ret = statistics.mean(float(r.get("max_excursion_best", 0)) for r in rand_picks)
        scout_returns.append(scout_ret)
        random_returns.append(rand_ret)

        rows.append({
            "scan_time_kst": scan,
            "universe_size": len(test),
            "scout_top2_mean_max": round(scout_ret, 4),
            "random_top2_mean_max": round(rand_ret, 4),
            "scout_minus_random": round(scout_ret - rand_ret, 4),
            "scout_hit5_rate": round(
                sum(float(r.get("hit_5pct_plus", 0)) for r in scout_picks) / 2, 2
            ),
            "method": method,
            "learning_recommendation": "NO_ACTION",
        })

    if rows:
        summary = {
            "scan_time_kst": "AGGREGATE",
            "universe_size": len(rows),
            "scout_top2_mean_max": round(statistics.mean(scout_returns), 4),
            "random_top2_mean_max": round(statistics.mean(random_returns), 4),
            "scout_minus_random": round(statistics.mean(scout_returns) - statistics.mean(random_returns), 4),
            "scout_hit5_rate": round(
                sum(1 for r in scout_returns if r >= 5) / len(scout_returns), 2
            ),
            "method": method,
            "learning_recommendation": "NO_ACTION",
        }
        rows.append(summary)
    return rows


def write_research_report(db: list[dict], blind_rows: list[dict], method_rows: list[dict]) -> None:
    hit5 = sum(float(r.get("hit_5pct_plus", 0)) for r in db)
    agg = next((r for r in blind_rows if r.get("scan_time_kst") == "AGGREGATE"), {})
    best_method = method_rows[0]["method"] if method_rows else "cosine"

    lines = [
        "Scout Season3 - Experience Based Research Report",
        "=" * 50,
        "",
        "Mission: Experience database + similarity search (NOT rule-based prediction)",
        "",
        f"Experience DB size: {len(db):,} snapshots",
        f"Label base rate (5%+ max excursion): {hit5/len(db)*100:.1f}%" if db else "Empty DB",
        "NOTE: Top10 universe is gainer-biased; base rate is NOT market-wide.",
        "",
        "Similarity method comparison (holdout):",
    ]
    for r in method_rows:
        lines.append(
            f"  {r['method']}: accuracy={r['accuracy_50pct_threshold']:.1%} "
            f"mean_prob={r['mean_predicted_prob']}"
        )
    lines.extend([
        f"Selected method: {best_method}",
        "",
        "Blind validation (scan LOOCV, Top2 by experience vs Random Top2):",
    ])
    if agg:
        lines.extend([
            f"  Scans evaluated: {agg.get('universe_size', 0)}",
            f"  Scout Top2 mean max excursion: {agg.get('scout_top2_mean_max', 0):+.4f}%",
            f"  Random Top2 mean max excursion: {agg.get('random_top2_mean_max', 0):+.4f}%",
            f"  Delta (Scout - Random): {agg.get('scout_minus_random', 0):+.4f}%",
            f"  Random baseline reference (Blind Loop 001): {RANDOM_BASELINE_2H:+.4f}%",
        ])
        scout_mean = float(agg.get("scout_top2_mean_max", 0))
        if scout_mean > float(agg.get("random_top2_mean_max", 0)):
            lines.append("  Result: Experience engine BEATS random within Top10 universe")
        else:
            lines.append("  Result: Experience engine DOES NOT beat random - NO_ACTION")
    else:
        lines.append("  Insufficient scans for blind validation")

    lines.extend([
        "",
        "Feature analysis outputs:",
        f"  {IMPORTANCE_CSV}",
        f"  {CORR_CSV}",
        f"  {INFO_GAIN_CSV}",
        f"  {SHAP_CSV}",
        f"  {CLUSTER_CSV}",
        "",
        "Principles upheld:",
        "  - No Belief/Narrative/Sync/State",
        "  - No composite ScoutScore",
        "  - Probability from nearest past snapshots only",
        "  - Reasons from neighbor statistics, not narrative inference",
        "",
        "Learning recommendation: NO_ACTION",
        "Next: expand universe beyond Top10 gainer list for unbiased base rate",
    ])
    REPORT_TXT.write_text("\n".join(lines), encoding="utf-8")


def compare_methods(db: list[dict]) -> list[dict]:
    if len(db) < 50:
        return []
    random.seed(42)
    holdout = random.sample(db, min(100, len(db) // 5))
    train = [r for r in db if r not in holdout]
    rows = []
    for method in SIMILARITY_METHODS:
        hits = 0
        probs: list[float] = []
        for rec in holdout:
            q = experience_query(rec, train, method=method, k=min(DEFAULT_K, len(train)), exclude_same=True)
            probs.append(q["long_probability_pct"])
            actual = float(rec.get("hit_5pct_plus", 0))
            pred = 1 if q["long_probability_pct"] >= 50 else 0
            if pred == actual:
                hits += 1
        rows.append({
            "method": method,
            "holdout_n": len(holdout),
            "accuracy_50pct_threshold": round(hits / len(holdout), 4),
            "mean_predicted_prob": round(statistics.mean(probs), 2),
            "learning_recommendation": "NO_ACTION",
        })
    rows.sort(key=lambda r: r["accuracy_50pct_threshold"], reverse=True)
    return rows


def write_importance(db: list[dict]) -> None:
    keys = MODEL_FEATURES
    labels = [float(r.get("hit_5pct_plus", 0)) for r in db]
    perm = permutation_importance(db, keys, "cosine")
    mi_rows = []
    for key in keys:
        xs = [float(r.get("f_" + key, r.get(key, 0))) for r in db]
        mi_rows.append({
            "feature": key,
            "permutation_importance": round(perm.get(key, 0), 4),
            "mutual_information_proxy": round(mutual_info_proxy(xs, labels), 4),
            "learning_recommendation": "NO_ACTION",
        })
    mi_rows.sort(key=lambda r: r["permutation_importance"], reverse=True)
    write_csv(IMPORTANCE_CSV, mi_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scout Season3 Experience Engine")
    sub = parser.add_subparsers(dest="cmd")

    b1 = sub.add_parser("build", help="Build experience database")
    b1.add_argument("--from-top10", action="store_true")
    b1.add_argument("--from-cache", action="store_true")
    b1.add_argument("--from-klines", action="store_true")
    b1.add_argument("--klines-limit", type=int, default=50)
    b1.add_argument("--sleep", type=float, default=0.35)
    b1.add_argument("--reset", action="store_true")

    q1 = sub.add_parser("query", help="Query similar experience")
    q1.add_argument("--symbol", required=True)
    q1.add_argument("--scan-time", default="")
    q1.add_argument("--method", default="cosine", choices=SIMILARITY_METHODS)
    q1.add_argument("--k", type=int, default=DEFAULT_K)

    sub.add_parser("compare-methods", help="Compare similarity methods on holdout")
    sub.add_parser("stats", help="Database statistics")
    sub.add_parser("analyze", help="Feature importance, correlation, cluster, SHAP proxy")
    v1 = sub.add_parser("blind-validate", help="Scan LOOCV Top2 vs Random")
    v1.add_argument("--method", default="cosine", choices=SIMILARITY_METHODS)
    v1.add_argument("--k", type=int, default=DEFAULT_K)
    sub.add_parser("report", help="Write full research report")

    args = parser.parse_args()

    if args.cmd == "build":
        if args.reset and DB_JSONL.exists():
            DB_JSONL.unlink()
        sources: list[str] = []
        if args.from_top10 or (not args.from_cache and not args.from_klines):
            build_from_top10()
            sources.append("top10_csv")
        if args.from_cache:
            build_from_cache()
            sources.append("universe_cache")
        if args.from_klines:
            build_from_klines(max_snapshots=args.klines_limit, sleep_sec=args.sleep)
            sources.append("kline_api")
        db = load_db()
        write_importance(db)
        write_feature_analysis(db)
        META_JSON.write_text(json.dumps({
            "records": len(db),
            "built_at": datetime.now(KST).isoformat(),
            "sources": sources,
        }, indent=2), encoding="utf-8")
        print(f"Experience DB: {len(db)} snapshots -> {DB_JSONL}")

    elif args.cmd == "query":
        db = load_db()
        if not db:
            raise SystemExit("Empty DB. Run: python season3_experience_engine.py build --from-top10 --reset")
        sym = args.symbol.upper()
        candidates = [r for r in db if r.get("symbol") == sym]
        if args.scan_time:
            candidates = [r for r in candidates if r.get("scan_time_kst") == args.scan_time]
        if not candidates:
            raise SystemExit(f"No snapshot for {sym}. Build DB first or check symbol.")
        query_rec = candidates[-1]
        result = experience_query(query_rec, db, method=args.method, k=args.k)
        text = format_output(result, sym)
        QUERY_TXT.write_text(text, encoding="utf-8")
        print(text)

    elif args.cmd == "compare-methods":
        db = load_db()
        rows = compare_methods(db)
        write_csv(COMPARE_CSV, rows)
        for r in rows:
            print(f"{r['method']}: acc={r['accuracy_50pct_threshold']:.1%} mean_prob={r['mean_predicted_prob']}")
        if rows:
            print(f"Best method: {rows[0]['method']}")

    elif args.cmd == "stats":
        db = load_db()
        if not db:
            print("Empty DB")
            return
        hit5 = sum(float(r.get("hit_5pct_plus", 0)) for r in db)
        hit10 = sum(float(r.get("hit_10pct_plus", 0)) for r in db)
        scans = len(set(r.get("scan_time_kst") for r in db))
        print(f"Records: {len(db)} | Scans: {scans}")
        print(f"5%+ cases: {int(hit5)} ({hit5/len(db)*100:.1f}%)")
        print(f"10%+ cases: {int(hit10)} ({hit10/len(db)*100:.1f}%)")

    elif args.cmd == "analyze":
        db = load_db()
        if not db:
            raise SystemExit("Empty DB")
        write_importance(db)
        write_feature_analysis(db)
        print(f"Wrote: {IMPORTANCE_CSV}, {CORR_CSV}, {INFO_GAIN_CSV}, {SHAP_CSV}, {CLUSTER_CSV}")

    elif args.cmd == "blind-validate":
        db = load_db()
        if not db:
            raise SystemExit("Empty DB")
        rows = blind_validate(db, method=args.method, k=args.k)
        write_csv(BLIND_CSV, rows)
        agg = next((r for r in rows if r.get("scan_time_kst") == "AGGREGATE"), {})
        if agg:
            print(
                f"Scout Top2: {agg['scout_top2_mean_max']:+.4f}% | "
                f"Random Top2: {agg['random_top2_mean_max']:+.4f}% | "
                f"Delta: {agg['scout_minus_random']:+.4f}%"
            )
        print(f"Wrote: {BLIND_CSV}")

    elif args.cmd == "report":
        db = load_db()
        method_rows = compare_methods(db)
        blind_rows = blind_validate(db)
        write_importance(db)
        write_feature_analysis(db)
        write_research_report(db, blind_rows, method_rows)
        print(f"Wrote: {REPORT_TXT}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
