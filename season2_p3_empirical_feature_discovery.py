"""
Scout Learning Season2 - P3 Empirical Feature Discovery

No hypothesis defense. Measure what predicts, discard what does not.
Compare simple candle/structure vs complex indicators by LOO prediction gain.
Research only - no condition scanner output.
"""

import csv
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from season2_p1_supply_probability_panel import (
    DATASETS,
    LOGS_DIR,
    build_records,
    load_rows,
    pb,
    pf,
    state_at_forward,
)
from season2_scout_probability_engine import attach_drawdown

REPORT_TXT = LOGS_DIR / "season2_p3_feature_discovery_report.txt"
RANKINGS_CSV = LOGS_DIR / "season2_p3_feature_rankings.csv"
ABLATION_CSV = LOGS_DIR / "season2_p3_ablation_results.csv"
PATTERNS_CSV = LOGS_DIR / "season2_p3_robust_patterns.csv"
SCORES_V3_CSV = LOGS_DIR / "season2_scout_probability_scores_v3.csv"
MIN_BUCKET = 3


def full_enrich(records: list[dict]) -> None:
    fields = [
        "forward_24h", "max_drawdown", "max_profit_24h",
        "position_24h_percent", "position_48h_percent",
        "body_expansion_ratio", "current_body_percent", "avg_body_24_percent",
        "range_expansion_ratio", "atr_ratio", "atr_percent",
        "ma24_slope_percent", "ma48_slope_percent", "ma84_slope_percent",
        "volume_ratio_ma24", "volume_acceleration_ratio",
        "distance_ma6_percent", "distance_ma24_percent", "distance_ma48_percent",
        "return_prev_2h_percent", "return_prev_4h_percent", "return_prev_6h_percent",
        "return_prev_12h_percent", "return_prev_48h_percent",
        "break_24h_highest_close", "break_48h_highest_close",
        "pre6_tight_range", "pre6_body_compression", "pre6_volatility_compression",
        "pre6_volume_contraction", "pre12_tight_range",
        "group",
    ]
    extra: dict[tuple[str, str], dict] = {}
    for path in DATASETS:
        for row in load_rows(path):
            scan = row.get("scan_time_kst", "")
            symbol = row.get("symbol", "")
            if not scan or not symbol:
                continue
            payload = {}
            for field in fields:
                raw = row.get(field)
                if field.startswith("break_") or field.startswith("pre"):
                    payload[field] = pb(raw) if isinstance(raw, str) else bool(raw)
                elif field == "group":
                    payload[field] = raw or ""
                else:
                    payload[field] = pf(raw)
            extra[(scan, symbol)] = payload

    for record in records:
        data = extra.get((record["scan_time"], record["symbol"]), {})
        record.update({k: v for k, v in data.items() if v is not None and v != ""})
        record["state_4h"] = state_at_forward(record.get("forward_4h"), record.get("forward_2h"))
        record["collapse"] = 1 if record.get("collapse_label") == "YES" else 0
        record["persist_4h"] = (
            1 if record.get("forward_4h") is not None and record["forward_4h"] >= 0 else 0
        )
        record["persist_6h"] = (
            1 if record.get("forward_6h") is not None and record["forward_6h"] >= 0 else 0
        )


# --- Derived simple structures (candidate discoveries) ---

def simple_rank_bucket(record: dict) -> str:
    rank = record.get("rank") or 99
    if rank <= 2:
        return "rank_top2"
    if rank <= 5:
        return "rank_3_5"
    return "rank_6_10"


def simple_return_24h_bucket(record: dict) -> str:
    value = record.get("return_24h_at_scan")
    if value is None:
        return "ret_unknown"
    if value >= 40:
        return "ret_extreme_40+"
    if value >= 25:
        return "ret_high_25_40"
    if value >= 15:
        return "ret_mid_15_25"
    return "ret_low_<15"


def simple_momentum_2h(record: dict) -> str:
    value = record.get("return_prev_2h_percent")
    if value is None:
        return "mom2_unknown"
    if value >= 3:
        return "mom2_strong_up"
    if value >= 0:
        return "mom2_flat_up"
    if value >= -3:
        return "mom2_flat_down"
    return "mom2_strong_down"


def simple_body_surge(record: dict) -> str:
    body = record.get("body_expansion_ratio")
    if body is None:
        return "body_unknown"
    if body >= 2.0:
        return "body_surge_2+"
    if body >= 1.2:
        return "body_normal"
    return "body_compressed"


def simple_compression(record: dict) -> str:
    if record.get("pre6_tight_range") or record.get("pre6_volatility_compression"):
        return "compress_yes"
    return "compress_no"


def simple_break_24h(record: dict) -> str:
    return "break24_yes" if record.get("break_24h_highest_close") else "break24_no"


def simple_pos7d(record: dict) -> str:
    value = record.get("position_7d_percent")
    if value is None:
        return "pos7_unknown"
    if value >= 95:
        return "pos7_extreme_95+"
    if value >= 80:
        return "pos7_high_80_95"
    return "pos7_low_<80"


def simple_late_leader(record: dict) -> str:
    rank = record.get("rank") or 99
    ret = record.get("return_24h_at_scan") or 0
    if rank <= 3 and ret >= 30:
        return "late_leader_yes"
    return "late_leader_no"


def simple_candle_chain(record: dict) -> str:
    """Minimal candle structure: compression + momentum + body."""
    return f"{simple_compression(record)}|{simple_momentum_2h(record)}|{simple_body_surge(record)}"


def complex_ma24_slope(record: dict) -> str:
    value = record.get("ma24_slope_percent")
    if value is None:
        return "slope_unknown"
    if value >= 8:
        return "slope_high_8+"
    if value >= 4:
        return "slope_mid_4_8"
    return "slope_low_<4"


def complex_volume(record: dict) -> str:
    value = record.get("volume_ratio_ma24")
    if value is None:
        return "vol_unknown"
    if value >= 3:
        return "vol_extreme_3+"
    if value >= 1.2:
        return "vol_mid_1.2_3"
    return "vol_low_<1.2"


def complex_state_scan(record: dict) -> str:
    return record.get("state_scan") or "unknown"


def complex_trigger_bundle(record: dict) -> str:
    return record.get("trigger_bundle") or "unknown"


def oracle_state_4h(record: dict) -> str:
    return record.get("state_4h") or "unknown"


def context_regime(record: dict) -> str:
    return record.get("market_regime") or "unknown"


def context_scan_hour(record: dict) -> str:
    hour = record.get("scan_time", "")[11:13]
    return f"hour_{hour}"


FEATURE_CATALOG: list[tuple[str, str, callable]] = [
    ("simple", "rank_bucket", simple_rank_bucket),
    ("simple", "return_24h_bucket", simple_return_24h_bucket),
    ("simple", "momentum_2h", simple_momentum_2h),
    ("simple", "momentum_4h", lambda r: _bucket_numeric(r.get("return_prev_4h_percent"), [-3, 0, 3], "mom4")),
    ("simple", "body_surge", simple_body_surge),
    ("simple", "compression", simple_compression),
    ("simple", "break_24h", simple_break_24h),
    ("simple", "pos7d", simple_pos7d),
    ("simple", "late_leader", simple_late_leader),
    ("simple", "candle_chain", simple_candle_chain),
    ("simple", "range_expansion", lambda r: _bucket_numeric(r.get("range_expansion_ratio"), [0.8, 1.2, 1.8], "range")),
    ("simple", "distance_ma24", lambda r: _bucket_numeric(r.get("distance_ma24_percent"), [10, 25, 45], "dma24")),
    ("complex", "ma24_slope", complex_ma24_slope),
    ("complex", "ma48_slope", lambda r: _bucket_numeric(r.get("ma48_slope_percent"), [2, 5, 10], "ma48")),
    ("complex", "volume_ratio", complex_volume),
    ("complex", "state_scan", complex_state_scan),
    ("complex", "trigger_bundle", complex_trigger_bundle),
    ("complex", "dominant_trigger", lambda r: r.get("dominant_trigger") or "unknown"),
    ("context", "market_regime", context_regime),
    ("context", "scan_hour", context_scan_hour),
    ("oracle", "state_4h", oracle_state_4h),
]


def _bucket_numeric(value: float | None, cuts: list[float], prefix: str) -> str:
    if value is None:
        return f"{prefix}_unknown"
    tags = [f"<{cuts[0]}"] + [f"{cuts[i]}_{cuts[i+1]}" for i in range(len(cuts) - 1)] + [f">{cuts[-1]}"]
    bounds = [-math.inf] + cuts + [math.inf]
    for index in range(len(bounds) - 1):
        if bounds[index] <= value < bounds[index + 1]:
            return f"{prefix}_{tags[index]}"
    return f"{prefix}_unknown"


@dataclass
class FeatureScore:
    tier: str
    name: str
    n_buckets: int
    spread_f6: float
    spread_persist4: float
    spread_collapse: float
    loo_mae_f6: float
    loo_top1: str
    verdict: str


def bucket_stats(train: list[dict], key_fn: callable) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in train:
        groups[key_fn(record)].append(record)
    return groups


def cohort_median_f6(group: list[dict]) -> float | None:
    values = [record["forward_6h"] for record in group if record.get("forward_6h") is not None]
    return statistics.median(values) if values else None


def loo_predict(records: list[dict], key_fn: callable, min_n: int = MIN_BUCKET) -> tuple[float, int, int]:
    dates = sorted({record["date"] for record in records})
    errors: list[float] = []
    top1_hits = 0
    scan_total = 0

    for holdout in dates:
        train = [record for record in records if record["date"] != holdout]
        test = [record for record in records if record["date"] == holdout]
        groups = bucket_stats(train, key_fn)
        bucket_medians = {
            key: cohort_median_f6(group)
            for key, group in groups.items()
            if len(group) >= min_n and cohort_median_f6(group) is not None
        }
        global_med = cohort_median_f6(train) or 0.0

        by_scan: dict[str, list[dict]] = defaultdict(list)
        for record in test:
            by_scan[record["scan_time"]].append(record)

        for scan_time, scan_rows in by_scan.items():
            scan_total += 1
            preds: list[tuple[str, float]] = []
            for record in scan_rows:
                key = key_fn(record)
                pred = bucket_medians.get(key, global_med)
                actual = record.get("forward_6h")
                if actual is not None:
                    errors.append(abs(pred - actual))
                preds.append((record["symbol"], pred))
            if preds:
                best_pred = max(preds, key=lambda item: item[1])
                best_actual = max(
                    scan_rows,
                    key=lambda item: item.get("forward_6h") or -9999,
                )
                if best_pred[0] == best_actual["symbol"]:
                    top1_hits += 1

    mae = statistics.mean(errors) if errors else 999.0
    return mae, top1_hits, scan_total


def signal_spread(records: list[dict], key_fn: callable) -> tuple[float, float, float, int]:
    groups = bucket_stats(records, key_fn)
    f6_medians: list[float] = []
    persist_rates: list[float] = []
    collapse_rates: list[float] = []

    for group in groups.values():
        if len(group) < MIN_BUCKET:
            continue
        f6_vals = [record["forward_6h"] for record in group if record.get("forward_6h") is not None]
        if f6_vals:
            f6_medians.append(statistics.median(f6_vals))
        persist_rates.append(statistics.mean([record["persist_4h"] for record in group]) * 100)
        collapse_rates.append(statistics.mean([record["collapse"] for record in group]) * 100)

    if len(f6_medians) < 2:
        return 0.0, 0.0, 0.0, len(groups)

    return (
        max(f6_medians) - min(f6_medians),
        max(persist_rates) - min(persist_rates) if persist_rates else 0.0,
        max(collapse_rates) - min(collapse_rates) if collapse_rates else 0.0,
        len([g for g in groups.values() if len(g) >= MIN_BUCKET]),
    )


def score_feature(records: list[dict], tier: str, name: str, key_fn: callable) -> FeatureScore:
    spread_f6, spread_p4, spread_col, n_buckets = signal_spread(records, key_fn)
    loo_mae, top1, scan_total = loo_predict(records, key_fn)
    top1_str = f"{top1}/{scan_total}"

    baseline_mae = _global_mae(records)

    if tier == "oracle":
        verdict = "oracle_ceiling"
    elif loo_mae < baseline_mae * 0.95 and spread_f6 >= 5:
        verdict = "KEEP_strong"
    elif loo_mae < baseline_mae * 0.98 and spread_f6 >= 3:
        verdict = "KEEP_moderate"
    elif loo_mae > baseline_mae * 1.02:
        verdict = "DISCARD_noise"
    elif spread_f6 < 2:
        verdict = "DISCARD_weak_signal"
    else:
        verdict = "HYPOTHESIS"

    return FeatureScore(tier, name, n_buckets, spread_f6, spread_p4, spread_col, loo_mae, top1_str, verdict)


def _global_mae(records: list[dict]) -> float:
    dates = sorted({record["date"] for record in records})
    errors: list[float] = []
    for holdout in dates:
        train = [record for record in records if record["date"] != holdout]
        test = [record for record in records if record["date"] == holdout]
        global_med = cohort_median_f6(train) or 0.0
        for record in test:
            actual = record.get("forward_6h")
            if actual is not None:
                errors.append(abs(global_med - actual))
    return statistics.mean(errors) if errors else 999.0


def greedy_ablation(records: list[dict], ranked: list[FeatureScore], catalog_map: dict) -> list[dict]:
    baseline = _global_mae(records)
    selected: list[str] = []
    current_mae = baseline
    rows: list[dict] = [{"step": 0, "feature": "baseline_global_median", "mae": round(baseline, 2), "gain": 0.0}]

    candidates = [
        f for f in ranked
        if f.tier != "oracle" and f.loo_mae_f6 <= baseline * 1.001
    ]
    candidates.sort(key=lambda item: item.loo_mae_f6)

    for step in range(1, 8):
        best_name = None
        best_mae = current_mae
        for candidate in candidates:
            if candidate.name in selected:
                continue
            trial = selected + [candidate.name]
            mae = loo_combo_mae(records, trial, catalog_map)
            if mae < best_mae - 0.02:
                best_mae = mae
                best_name = candidate.name
        if best_name is None:
            break
        selected.append(best_name)
        gain = current_mae - best_mae
        current_mae = best_mae
        rows.append(
            {
                "step": step,
                "feature": best_name,
                "mae": round(current_mae, 2),
                "gain": round(gain, 2),
                "selected_combo": "+".join(selected),
            }
        )

    oracle_mae, _, _ = loo_predict(records, catalog_map["state_4h"])
    rows.append({"step": "oracle", "feature": "state_4h", "mae": round(oracle_mae, 2), "gain": round(baseline - oracle_mae, 2)})
    return rows


def loo_combo_mae(records: list[dict], feature_names: list[str], catalog_map: dict[str, callable]) -> float:
    key_fns = [catalog_map[name] for name in feature_names]

    def combo_key(record: dict) -> str:
        return "|".join(fn(record) for fn in key_fns)

    mae, _, _ = loo_predict(records, combo_key, min_n=2)
    return mae


def discover_robust_patterns(records: list[dict]) -> list[dict]:
    patterns: list[dict] = []
    combos = [
        ("late_leader_no + mom2_strong_up", lambda r: simple_late_leader(r) == "late_leader_no" and simple_momentum_2h(r) == "mom2_strong_up"),
        ("compress_yes + mom2_strong_up", lambda r: simple_compression(r) == "compress_yes" and simple_momentum_2h(r) == "mom2_strong_up"),
        ("rank_6_10 + ret_mid", lambda r: simple_rank_bucket(r) == "rank_6_10" and simple_return_24h_bucket(r) == "ret_mid_15_25"),
        ("pos7_low + body_surge", lambda r: simple_pos7d(r) == "pos7_low_<80" and simple_body_surge(r) == "body_surge_2+"),
        ("break24_no + late_leader", lambda r: simple_break_24h(r) == "break24_no" and simple_late_leader(r) == "late_leader_yes"),
        ("state_4h Acceleration", lambda r: r.get("state_4h") == "Acceleration"),
        ("state_4h Collapse", lambda r: r.get("state_4h") == "Collapse"),
        ("state_4h Continuation", lambda r: r.get("state_4h") == "Continuation"),
        ("exhaustion scan + NOT late_leader", lambda r: r.get("state_scan") == "Exhaustion" and simple_late_leader(r) == "late_leader_no"),
    ]

    for label, predicate in combos:
        group = [record for record in records if predicate(record)]
        if len(group) < MIN_BUCKET:
            continue
        f6 = [record["forward_6h"] for record in group if record.get("forward_6h") is not None]
        patterns.append(
            {
                "pattern": label,
                "n": len(group),
                "median_f6": round(statistics.median(f6), 2) if f6 else "",
                "persist_4h_pct": round(statistics.mean([record["persist_4h"] for record in group]) * 100, 1),
                "collapse_pct": round(statistics.mean([record["collapse"] for record in group]) * 100, 1),
                "median_drawdown": round(
                    statistics.median(
                        [record["max_drawdown"] for record in group if record.get("max_drawdown") is not None]
                    ),
                    2,
                ) if any(record.get("max_drawdown") for record in group) else "",
                "confidence": "confirmed" if len(group) >= 10 else "hypothesis_only",
            }
        )

    return sorted(patterns, key=lambda item: -(item.get("median_f6") or -999 if item.get("median_f6") != "" else -999))


def build_v3_scores(records: list[dict], selected_features: list[str], catalog_map: dict[str, callable]) -> list[dict]:
    dates = sorted({record["date"] for record in records})
    rows: list[dict] = []

    for holdout in dates:
        train = [record for record in records if record["date"] != holdout]
        test = [record for record in records if record["date"] == holdout]
        key_fns = [catalog_map[name] for name in selected_features]

        def combo_key(record: dict) -> str:
            return "|".join(fn(record) for fn in key_fns)

        groups = bucket_stats(train, combo_key)
        global_med = cohort_median_f6(train) or 0.0
        bucket_stats_map: dict[str, dict] = {}
        for key, group in groups.items():
            if len(group) < 2:
                continue
            f6_vals = [record["forward_6h"] for record in group if record.get("forward_6h") is not None]
            dds = [record["max_drawdown"] for record in group if record.get("max_drawdown") is not None]
            bucket_stats_map[key] = {
                "median_f6": statistics.median(f6_vals) if f6_vals else global_med,
                "persist_4h": statistics.mean([record["persist_4h"] for record in group]) * 100,
                "persist_6h": statistics.mean([record["persist_6h"] for record in group]) * 100,
                "collapse": statistics.mean([record["collapse"] for record in group]) * 100,
                "drawdown": statistics.median(dds) if dds else None,
                "n": len(group),
            }

        by_scan: dict[str, list[dict]] = defaultdict(list)
        for record in test:
            by_scan[record["scan_time"]].append(record)

        for scan_time, scan_rows in by_scan.items():
            scored: list[dict] = []
            for record in scan_rows:
                key = combo_key(record)
                stats = bucket_stats_map.get(key, {})
                n = stats.get("n", 0)
                conf = "production" if n >= 30 else "high" if n >= 15 else "medium" if n >= 8 else "low" if n >= 3 else "hypothesis"
                exp_ret = stats.get("median_f6", global_med)
                exp_dd = stats.get("drawdown")
                collapse = stats.get("collapse", statistics.mean([record["collapse"] for record in train]) * 100)
                scored.append(
                    {
                        "date": record["date"],
                        "symbol": record["symbol"],
                        "scan_time": record["scan_time"],
                        "rank": record["rank"],
                        "feature_combo": key,
                        "features_used": "+".join(selected_features),
                        "expected_return_pct": round(exp_ret, 2),
                        "expected_drawdown_pct": round(exp_dd, 2) if exp_dd is not None else "",
                        "persist_4h_pct": round(stats.get("persist_4h", 50), 1),
                        "persist_6h_pct": round(stats.get("persist_6h", 50), 1),
                        "collapse_probability_pct": round(collapse, 1),
                        "confidence": conf,
                        "cohort_n": n,
                        "actual_forward_6h": record.get("forward_6h"),
                        "actual_supply_label": record.get("supply_label"),
                        "_sort": exp_ret - collapse * 0.15,
                    }
                )

            scored.sort(key=lambda item: item["_sort"], reverse=True)
            total = len(scored)
            for index, row in enumerate(scored):
                rank_pct = (total - index) / total * 100
                row["market_relative_rank_pct"] = round(rank_pct, 1)
                row["recommendation_score"] = round(min(max(row["_sort"] / 2 + rank_pct / 20, 0), 10), 1)
                row["action"] = (
                    "Prioritize" if row["recommendation_score"] >= 7 and row["collapse_probability_pct"] < 20
                    else "Consider" if row["recommendation_score"] >= 5.5
                    else "Watch" if row["recommendation_score"] >= 4
                    else "Avoid"
                )
                del row["_sort"]
            rows.extend(scored)

    return rows


def hypothesis_verdicts(ranked: list[FeatureScore], ablation: list[dict]) -> list[str]:
    lines: list[str] = []
    by_name = {feature.name: feature for feature in ranked}

    old_hypotheses = [
        ("state_scan", "complex composite state explains supply"),
        ("trigger_bundle", "trigger bundle beats singles"),
        ("ma24_slope", "MA24 slope is primary driver"),
        ("volume_ratio", "volume ratio drives selection"),
        ("dominant_trigger", "dominant trigger classification helps"),
        ("late_leader", "top rank + high 24h = late/exhausted"),
        ("compression", "compression predicts breakout"),
        ("momentum_2h", "simple 2h candle momentum"),
        ("rank_bucket", "rank alone predicts forward"),
    ]

    for name, hypothesis in old_hypotheses:
        feature = by_name.get(name)
        if not feature:
            lines.append(f"  {name}: NOT_TESTED")
            continue
        if feature.verdict.startswith("KEEP"):
            lines.append(f"  KEEP: {hypothesis} (MAE={feature.loo_mae_f6:.2f}, spread={feature.spread_f6:.1f})")
        elif feature.verdict.startswith("DISCARD"):
            lines.append(f"  DISCARD: {hypothesis} (MAE={feature.loo_mae_f6:.2f}, spread={feature.spread_f6:.1f})")
        else:
            lines.append(f"  REVISE/HYPOTHESIS: {hypothesis} (MAE={feature.loo_mae_f6:.2f})")

    oracle = by_name.get("state_4h")
    if oracle:
        lines.append(
            f"  ORACLE CEILING: state_4h MAE={oracle.loo_mae_f6:.2f} spread={oracle.spread_f6:.1f} "
            f"(scan-only features cannot match without +4h update)"
        )

    if len(ablation) > 1:
        final = ablation[-2] if ablation[-1].get("step") == "oracle" else ablation[-1]
        lines.append(f"  MINIMAL COMBO: {final.get('selected_combo', final.get('feature'))} MAE={final.get('mae')}")

    return lines


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    records = build_records()
    attach_drawdown(records)
    full_enrich(records)

    catalog_map = {name: fn for _, name, fn in FEATURE_CATALOG}

    ranked: list[FeatureScore] = []
    for tier, name, fn in FEATURE_CATALOG:
        ranked.append(score_feature(records, tier, name, fn))
    ranked.sort(key=lambda item: item.loo_mae_f6)

    baseline = _global_mae(records)
    ablation = greedy_ablation(records, ranked, catalog_map)
    patterns = discover_robust_patterns(records)

    selected = []
    for row in ablation:
        if row.get("step") not in (0, "oracle") and row.get("feature"):
            selected = row.get("selected_combo", row["feature"]).split("+")

    if not selected:
        top = [f.name for f in ranked if f.tier != "oracle" and f.loo_mae_f6 < baseline][:2]
        selected = top or ["distance_ma24"]

    v3_rows = build_v3_scores(records, selected, catalog_map)
    v3_mae, v3_top1, v3_scans = loo_predict(records, lambda r: "|".join(catalog_map[n](r) for n in selected), min_n=2)

    ranking_rows = [
        {
            "tier": feature.tier,
            "feature": feature.name,
            "n_buckets": feature.n_buckets,
            "spread_f6": round(feature.spread_f6, 2),
            "spread_persist4": round(feature.spread_persist4, 1),
            "spread_collapse": round(feature.spread_collapse, 1),
            "loo_mae_f6": round(feature.loo_mae_f6, 2),
            "loo_top1": feature.loo_top1,
            "vs_baseline": round(baseline - feature.loo_mae_f6, 2),
            "verdict": feature.verdict,
        }
        for feature in ranked
    ]

    write_csv(RANKINGS_CSV, ranking_rows)
    write_csv(ABLATION_CSV, ablation)
    write_csv(PATTERNS_CSV, patterns)
    write_csv(SCORES_V3_CSV, v3_rows)

    verdict_lines = hypothesis_verdicts(ranked, ablation)

    lines = [
        "===== SCOUT SEASON2 P3 - EMPIRICAL FEATURE DISCOVERY =====",
        "",
        "Principle: information gain over complexity. No hypothesis defense.",
        f"Records: {len(records)} | Baseline LOO MAE (global median): {baseline:.2f}%",
        "",
        "--- Top predictors (lowest LOO MAE) ---",
    ]
    for feature in ranked[:12]:
        lines.append(
            f"  [{feature.tier}] {feature.name}: MAE={feature.loo_mae_f6:.2f} "
            f"spread_f6={feature.spread_f6:.1f} top1={feature.loo_top1} -> {feature.verdict}"
        )

    lines.extend(["", "--- Simple vs Complex (avg MAE by tier) ---"])
    for tier in ("simple", "complex", "context", "oracle"):
        tier_feats = [feature for feature in ranked if feature.tier == tier]
        if tier_feats:
            avg_mae = statistics.mean([feature.loo_mae_f6 for feature in tier_feats])
            avg_spread = statistics.mean([feature.spread_f6 for feature in tier_feats])
            lines.append(f"  {tier}: avg_MAE={avg_mae:.2f} avg_spread_f6={avg_spread:.1f}")

    lines.extend(["", "--- Greedy ablation (minimal combo) ---"])
    for row in ablation:
        lines.append(f"  step {row.get('step')}: {row.get('feature')} MAE={row.get('mae')} gain={row.get('gain', '')}")

    lines.extend(["", "--- Robust patterns ---"])
    for pattern in patterns[:12]:
        lines.append(
            f"  {pattern['pattern']}: n={pattern['n']} median_f6={pattern['median_f6']}% "
            f"collapse={pattern['collapse_pct']}% [{pattern['confidence']}]"
        )

    lines.extend(["", "--- Hypothesis verdicts ---"])
    lines.extend(verdict_lines)

    lines.extend([
        "",
        f"--- v3 minimal engine (LOO) ---",
        f"  features: {'+'.join(selected)}",
        f"  MAE={v3_mae:.2f}% top1={v3_top1}/{v3_scans}",
        "",
        "--- Research direction ---",
        " 1. +4h state update closes oracle gap (biggest information gain)",
        " 2. Prefer simple momentum/compression/rank over trigger_bundle if validated",
        " 3. Discard features with DISCARD verdict - do not carry to production",
        " 4. Expand dates before trusting n<10 patterns",
        "",
        f"Rankings: {RANKINGS_CSV}",
        f"Ablation: {ABLATION_CSV}",
        f"Patterns: {PATTERNS_CSV}",
        f"V3 scores: {SCORES_V3_CSV}",
        "=" * 56,
    ])
    REPORT_TXT.write_text("\n".join(lines), encoding="utf-8")

    print("===== P3 EMPIRICAL FEATURE DISCOVERY =====")
    print(f"Baseline MAE: {baseline:.2f}%")
    print(f"Best feature: {ranked[0].name} MAE={ranked[0].loo_mae_f6:.2f} ({ranked[0].verdict})")
    print(f"v3 combo: {'+'.join(selected)} MAE={v3_mae:.2f}% top1={v3_top1}/{v3_scans}")
    print(f"Report: {REPORT_TXT}")


if __name__ == "__main__":
    main()
