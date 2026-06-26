"""
Scout Learning Season2 - P6 Market Memory & Conditional Psychology Discovery

Research only. Estimate collective trader memory horizons, not universal indicators.
"""

import argparse
import csv
import math
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from season2_p5_historical_expansion import (
    CONTEXTS,
    KNOWN_DATES,
    SEEDS,
    STEP1_DATES,
    assign_status,
    build_unified_records,
    collect_missing,
    evaluate_conditional,
    evaluate_hypothesis,
    feature_bucket,
    global_baseline_mae,
    loo_mae,
    spread_f6,
    vol_zone,
    late_leader,
    compression_ctx,
)

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

DECAY_CSV = LOGS_DIR / "season2_p6_memory_decay_curve.csv"
HORIZON_CSV = LOGS_DIR / "season2_p6_memory_horizon_summary.csv"
PSYCH_CSV = LOGS_DIR / "season2_p6_psychology_proxies.csv"
SYMBOL_MEM_CSV = LOGS_DIR / "season2_p6_symbol_memory.csv"
CONDITIONAL_CSV = LOGS_DIR / "season2_p6_conditional_memory.csv"
REGISTRY_CSV = LOGS_DIR / "season2_p6_hypothesis_registry.csv"
ENGINE_CSV = LOGS_DIR / "season2_p6_memory_engine_output.csv"
REPORT_TXT = LOGS_DIR / "season2_p6_research_report.txt"

# forward horizons available or proxied (hours)
HORIZON_HOURS = [
    ("30m", 0.5, "forward_0.5h"),
    ("1h", 1.0, "forward_1h"),
    ("2h", 2.0, "forward_2h"),
    ("4h", 4.0, "forward_4h"),
    ("6h", 6.0, "forward_6h"),
    ("12h", 12.0, "forward_12h"),
    ("24h", 24.0, "forward_24h"),
    ("48h", 48.0, "return_prev_48h"),  # lookback proxy until forward collected
    ("72h", 72.0, None),
    ("5d", 120.0, None),
    ("7d", 168.0, "return_prev_7d"),
    ("14d", 336.0, None),
    ("21d", 504.0, None),
    ("30d", 720.0, None),
]

MIN_BUCKET = 5

PSYCH_FEATURES = [
    ("dist_ma6", "distance from short MA6", lambda r: _bucket(r.get("distance_ma6_percent"), [5, 15, 30], "dma6")),
    ("dist_ma24", "distance from MA24 memory", lambda r: feature_bucket(r, "distance_ma24")),
    ("dist_12h_high", "distance from recent high", lambda r: _bucket(r.get("dist_from_12h_high_pct"), [2, 5, 10], "dhi")),
    ("dist_12h_low", "distance from recent low", lambda r: _bucket(r.get("dist_from_12h_low_pct"), [2, 5, 10], "dlo")),
    ("wick_rejection", "upper wick rejection", lambda r: feature_bucket(r, "fast_rejection")),
    ("lower_wick_long", "lower wick recovery", lambda r: feature_bucket(r, "lower_wick")),
    ("inside_bar", "inside bar compression", lambda r: "yes" if r.get("inside_bar") else "no"),
    ("engulf", "engulfing pattern", lambda r: "yes" if r.get("engulf") else "no"),
    ("congestion", "congestion duration", lambda r: feature_bucket(r, "congestion")),
    ("breakout_fail", "break 24h then fade", lambda r: "yes" if r.get("break_24h") and (r.get("return_24h_at_scan") or 0) >= 25 else "no"),
    ("fast_rejection", "fast rejection candle", lambda r: feature_bucket(r, "fast_rejection")),
    ("vol_exhaustion", "volume exhaustion", lambda r: "yes" if r.get("vol_exhaustion") else "no"),
    ("vol_persist_up", "persistent buying volume", lambda r: "yes" if r.get("vol_persist_up") else "no"),
    ("vol_persist_down", "persistent selling volume", lambda r: "yes" if r.get("vol_persist_down") else "no"),
    ("late_leader", "late trend psychology", lambda r: feature_bucket(r, "late_leader")),
    ("pos7d_extreme", "7d range position", lambda r: _bucket(r.get("position_7d_percent"), [80, 90, 95], "pos7")),
]


def _bucket(value, cuts, prefix):
    if value is None:
        return f"{prefix}_na"
    bounds = [-math.inf] + cuts + [math.inf]
    labels = [f"{prefix}_lt{cuts[0]}"] + [f"{prefix}_{cuts[i]}_{cuts[i+1]}" for i in range(len(cuts)-1)] + [f"{prefix}_gt{cuts[-1]}"]
    for i in range(len(bounds) - 1):
        if bounds[i] <= float(value) < bounds[i + 1]:
            return labels[i]
    return f"{prefix}_na"


def _pf(val):
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _pb(val):
    if val in (True, "True", "true", "1", 1):
        return True
    if val in (False, "False", "false", "0", 0, "", None):
        return False
    return bool(val)


def enrich_prev_returns(records: list[dict]) -> None:
    from season2_p5_historical_expansion import discover_dataset_paths, load_rows

    physics_path = LOGS_DIR / "season2_p4_physics_features.csv"
    physics_extra: dict[tuple[str, str], dict] = {}
    if physics_path.exists():
        with physics_path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                scan = row.get("scan_time", "")
                sym = row.get("symbol", "")
                if scan and sym:
                    physics_extra[(scan, sym)] = {
                        "inside_bar": _pb(row.get("inside_bar")),
                        "engulf": _pb(row.get("engulf")),
                        "vol_exhaustion": _pb(row.get("vol_exhaustion")),
                        "vol_persist_up": _pb(row.get("vol_persist_up")),
                        "vol_persist_down": _pb(row.get("vol_persist_down")),
                        "dist_from_12h_high_pct": _pf(row.get("dist_from_12h_high_pct")),
                        "dist_from_12h_low_pct": _pf(row.get("dist_from_12h_low_pct")),
                        "congestion_bars_12": _pf(row.get("congestion_bars_12")),
                        "break_24h": _pb(row.get("break_24h")),
                    }

    extra: dict[tuple[str, str], dict] = {}
    for path in discover_dataset_paths():
        for row in load_rows(path):
            scan = row.get("scan_time_kst", "")
            sym = row.get("symbol", "")
            if scan and sym:
                extra[(scan, sym)] = {
                    "return_prev_48h_percent": _pf(row.get("return_prev_48h_percent")),
                    "return_prev_7d_percent": _pf(row.get("return_prev_7d_percent")),
                    "distance_ma6_percent": _pf(row.get("distance_ma6_percent")),
                    "position_7d_percent": _pf(row.get("position_7d_percent")),
                    "break_24h": _pb(row.get("break_24h_highest_close", "")),
                }
    for record in records:
        key = (record["scan_time"], record["symbol"])
        data = {**extra.get(key, {}), **physics_extra.get(key, {})}
        record.update({k: v for k, v in data.items() if v is not None and v != ""})


def resolve_forward(record: dict, field: str | None, label: str) -> float | None:
    if field == "return_prev_48h":
        v = record.get("return_prev_48h_percent")
        return float(v) if v is not None else None
    if field == "return_prev_7d":
        v = record.get("return_prev_7d_percent")
        return float(v) if v is not None else None
    if field:
        val = record.get(field)
        if field == "forward_2h" and val == 0.0 and record.get("forward_4h") not in (None, 0.0):
            return None
        if val is not None and val != "":
            return float(val)
    if label == "30m":
        f2 = record.get("forward_2h")
        if f2 is not None and not (f2 == 0.0 and record.get("forward_4h")):
            return float(f2) / 4.0
    if label == "1h":
        v = record.get("forward_1h")
        if v is not None:
            return float(v)
        f2 = record.get("forward_2h")
        if f2 is not None and not (f2 == 0.0 and record.get("forward_4h")):
            return float(f2) / 2.0
    return None


def attach_forward_targets(records: list[dict]) -> None:
    for record in records:
        record["forwards"] = {}
        for label, _hours, field in HORIZON_HOURS:
            val = resolve_forward(record, field, label)
            if val is not None:
                record["forwards"][label] = val


def horizon_metrics(records: list[dict], key_fn, horizon_label: str) -> dict:
    subset = []
    for r in records:
        val = r.get("forwards", {}).get(horizon_label)
        if val is not None:
            row = {**r, "target_h": val}
            subset.append(row)
    if len(subset) < MIN_BUCKET * 2:
        return {"n": len(subset), "mae": None, "spread": None, "persist_pct": None}

    def target_fn(r):
        return r["target_h"]

    dates = sorted({r["date"] for r in subset})
    errors = []
    for holdout in dates:
        train = [r for r in subset if r["date"] != holdout]
        test = [r for r in subset if r["date"] == holdout]
        groups = defaultdict(list)
        for r in train:
            groups[key_fn(r)].append(r)
        medians = {
            k: statistics.median([x["target_h"] for x in g])
            for k, g in groups.items()
            if len(g) >= MIN_BUCKET
        }
        gm = statistics.median([x["target_h"] for x in train])
        for r in test:
            pred = medians.get(key_fn(r), gm)
            errors.append(abs(pred - r["target_h"]))

    groups_all = defaultdict(list)
    for r in subset:
        groups_all[key_fn(r)].append(r["target_h"])
    meds = [statistics.median(v) for v in groups_all.values() if len(v) >= MIN_BUCKET]
    spread = max(meds) - min(meds) if len(meds) >= 2 else 0.0
    persist = statistics.mean([1 if x["target_h"] >= 0 else 0 for x in subset]) * 100

    return {
        "n": len(subset),
        "mae": round(statistics.mean(errors), 2) if errors else None,
        "spread": round(spread, 2),
        "persist_pct": round(persist, 1),
    }


def memory_profile(records: list[dict], feature_name: str, key_fn) -> dict:
    curve = []
    for label, hours, _field in HORIZON_HOURS:
        m = horizon_metrics(records, key_fn, label)
        if m["spread"] is not None:
            curve.append({"horizon": label, "hours": hours, **m})

    if not curve:
        return {"peak": "", "half_life": "", "confidence": "insufficient"}

    peak = max(curve, key=lambda x: x["spread"])
    peak_spread = peak["spread"]
    half_life = ">30d"
    for point in sorted(curve, key=lambda x: x["hours"]):
        if peak_spread > 0 and point["spread"] <= peak_spread * 0.5:
            half_life = point["horizon"]
            break

    conf = "high" if peak["n"] >= 120 else "medium" if peak["n"] >= 60 else "hypothesis"
    return {
        "memory_peak_horizon": peak["horizon"],
        "memory_peak_spread": peak_spread,
        "memory_half_life": half_life,
        "memory_confidence": conf,
        "curve": curve,
    }


def symbol_archetype(records: list[dict]) -> dict:
    f6 = [r.get("target_f6") for r in records if r.get("target_f6") is not None]
    collapses = sum(1 for r in records if r.get("collapse_label") == "YES")
    n = len(records)
    if not f6:
        return {"archetype": "unknown", "stability": 0, "memory_length": "unknown"}

    med = statistics.median(f6)
    vol = statistics.pstdev(f6) if len(f6) > 1 else 0
    stability = 1.0 / (1.0 + vol / 10.0)

    if collapses / n >= 0.35:
        arch = "manipulation_prone"
    elif med >= 8 and vol >= 12:
        arch = "explosive"
    elif med >= 3 and vol < 8:
        arch = "persistent"
    elif vol >= 15:
        arch = "high_noise"
    elif vol < 5:
        arch = "low_noise"
    elif med >= 2:
        arch = "rotation"
    elif med < -5:
        arch = "late_leader"
    else:
        arch = "mean_reverting"

    # pattern recurrence: same sign across appearances
    signs = [1 if x >= 0 else -1 for x in f6]
    recurrence = abs(sum(signs)) / len(signs) if signs else 0

    return {
        "archetype": arch,
        "stability": round(stability, 2),
        "pattern_recurrence": round(recurrence, 2),
        "median_f6": round(med, 2),
        "collapse_rate": round(collapses / n * 100, 1),
        "memory_length": "short" if vol >= 12 else "medium" if vol >= 6 else "long",
    }


def context_label(record: dict) -> str:
    parts = [
        vol_zone(record),
        "late" if late_leader(record) else "early",
        "compress" if compression_ctx(record) else "expand",
        "frej" if record.get("fast_rejection") else "no_frej",
    ]
    return "|".join(parts)


def memory_engine_row(record: dict, profile: dict, sym_mem: dict, baseline: float) -> dict:
    horizon = profile.get("memory_peak_horizon", "6h")
    exp_ret = record.get("forwards", {}).get(horizon) or record.get("target_f6") or 0
    collapse = 1 if record.get("collapse_label") == "YES" else 0
    conf = profile.get("memory_confidence", "hypothesis")
    persist = record.get("forwards", {})
    p4 = persist.get("4h")
    p6 = persist.get("6h")

    action = "Avoid" if collapse else "Watch"
    if sym_mem.get("archetype") == "explosive" and (p6 or 0) >= 5:
        action = "Buy"
    elif sym_mem.get("archetype") == "late_leader":
        action = "Avoid"

    return {
        "date": record["date"],
        "symbol": record["symbol"],
        "scan_time": record["scan_time"],
        "market_regime": record.get("market_regime"),
        "symbol_archetype": sym_mem.get("archetype"),
        "memory_horizon_est": horizon,
        "memory_half_life": profile.get("memory_half_life"),
        "context": context_label(record),
        "expected_persistence_4h_pct": "",
        "expected_return_horizon_pct": round(exp_ret, 2) if isinstance(exp_ret, (int, float)) else "",
        "expected_drawdown_pct": record.get("max_drawdown"),
        "collapse_probability_pct": collapse * 100,
        "confidence": conf,
        "recommended_action": action,
        "reason": f"peak memory at {horizon}; symbol {sym_mem.get('archetype')}",
        "risk_factors": "late_leader" if late_leader(record) else ("fast_rejection" if record.get("fast_rejection") else ""),
        "recommendation_score": round(max(0, min(10, 5 + (profile.get("memory_peak_spread", 0) / 5))), 1),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect-limit", type=int, default=1)
    parser.add_argument("--skip-collect", action="store_true")
    args = parser.parse_args()

    collected = []
    if not args.skip_collect:
        missing = [d for d in STEP1_DATES if not (LOGS_DIR / f"top10_gainer_learning_{d.replace('-', '')}.csv").exists()]
        collected = collect_missing(missing, args.collect_limit)

    records = build_unified_records()
    enrich_prev_returns(records)
    attach_forward_targets(records)
    baseline = global_baseline_mae(records)

    decay_rows: list[dict] = []
    horizon_rows: list[dict] = []
    psych_rows: list[dict] = []
    conditional_rows: list[dict] = []
    registry_rows: list[dict] = []

    # Task 1 + 3: psychology proxies memory profiles
    best_psych = None
    best_spread = 0
    for name, desc, key_fn in PSYCH_FEATURES:
        prof = memory_profile(records, name, key_fn)
        for point in prof.get("curve", []):
            decay_rows.append({"feature": name, "type": "psychology", **point})
        horizon_rows.append(
            {
                "feature": name,
                "description": desc,
                "memory_peak": prof.get("memory_peak_horizon"),
                "memory_peak_spread": prof.get("memory_peak_spread"),
                "memory_half_life": prof.get("memory_half_life"),
                "memory_confidence": prof.get("memory_confidence"),
            }
        )
        perf_6h = horizon_metrics(records, key_fn, "6h")
        psych_rows.append(
            {
                "proxy": name,
                "description": desc,
                "spread_6h": perf_6h.get("spread"),
                "mae_6h": perf_6h.get("mae"),
                "memory_peak": prof.get("memory_peak_horizon"),
                "memory_half_life": prof.get("memory_half_life"),
                "verdict": "KEEP" if (perf_6h.get("spread") or 0) >= 5 else "HYPOTHESIS" if (perf_6h.get("spread") or 0) >= 3 else "WEAK",
            }
        )
        if (prof.get("memory_peak_spread") or 0) > best_spread:
            best_spread = prof.get("memory_peak_spread", 0)
            best_psych = name

    # Task 2 + 5: hypothesis registry with conditional memory
    for seed in SEEDS:
        perf = evaluate_hypothesis(records, seed.feature, baseline)
        revivals = evaluate_conditional(records, seed.feature, baseline)
        status, reason = assign_status(seed, perf, revivals, baseline)
        prof = memory_profile(records, seed.feature, lambda r, f=seed.feature: feature_bucket(r, f))

        registry_rows.append(
            {
                "hypothesis_name": seed.name,
                "feature_used": seed.feature,
                "current_status": status,
                "global_mae": perf["global_mae"],
                "memory_peak": prof.get("memory_peak_horizon"),
                "memory_half_life": prof.get("memory_half_life"),
                "decision_reason": reason,
                "sample_size": perf["n"],
            }
        )
        for rev in revivals:
            conditional_rows.append({**rev, "feature_type": "hypothesis"})

    # Task 4: symbol memory
    sym_groups: dict[str, list] = defaultdict(list)
    for r in records:
        sym_groups[r["symbol"]].append(r)
    symbol_rows = []
    for symbol, group in sorted(sym_groups.items(), key=lambda x: -len(x[1])):
        mem = symbol_archetype(group)
        symbol_rows.append({"symbol": symbol, "appearances": len(group), **mem})

    # Task 7: memory engine prototype (use best psych feature profile)
    prof_best = memory_profile(records, best_psych or "fast_rejection", lambda r: feature_bucket(r, "fast_rejection"))
    sym_index = {s["symbol"]: s for s in symbol_rows}
    engine_rows = []
    for record in records[-40:]:  # latest sample slice
        sym = sym_index.get(record["symbol"], {})
        engine_rows.append(memory_engine_row(record, prof_best, sym, baseline))

    write_csv(DECAY_CSV, decay_rows)
    write_csv(HORIZON_CSV, horizon_rows)
    write_csv(PSYCH_CSV, psych_rows)
    write_csv(SYMBOL_MEM_CSV, symbol_rows)
    write_csv(CONDITIONAL_CSV, conditional_rows)
    write_csv(REGISTRY_CSV, registry_rows)
    write_csv(ENGINE_CSV, engine_rows)

    dates = sorted({r["date"] for r in records})
    lines = [
        "===== SCOUT SEASON2 P6 - MARKET MEMORY & PSYCHOLOGY =====",
        "",
        f"Sample: {len(records)} records | {dates[0]}..{dates[-1]} ({len(dates)} days)",
        f"Collected this run: {collected or '(none)'}",
        f"Baseline MAE (6h): {baseline:.2f}%",
        "",
        "--- Task 1: Memory Horizon ---",
        f"Best psychology proxy by peak spread: {best_psych} (spread={best_spread:.1f})",
    ]
    for row in sorted(psych_rows, key=lambda x: -(x.get("spread_6h") or 0))[:6]:
        lines.append(
            f"  {row['proxy']}: peak@{row['memory_peak']} half_life={row['memory_half_life']} "
            f"spread_6h={row['spread_6h']} [{row['verdict']}]"
        )

    lines.extend(["", "--- Task 4: Symbol memory archetypes (top) ---"])
    for row in symbol_rows[:10]:
        lines.append(
            f"  {row['symbol']}: {row['archetype']} memory={row['memory_length']} "
            f"stability={row['stability']} collapse={row['collapse_rate']}%"
        )

    lines.extend(["", "--- Task 5: Hypothesis lifecycle ---"])
    for row in registry_rows:
        lines.append(f"  {row['hypothesis_name']}: {row['current_status']} peak={row['memory_peak']} | {row['decision_reason'][:60]}")

    lines.extend([
        "",
        "--- Research notes ---",
        " Price has no memory; participant memory estimated via horizon decay curves",
        " Forward horizons >24h require extended collection (48h-30d marked proxy/missing)",
        " Do not permanently discard: use DEAD_FOR_NOW / RETIRED per P5 registry",
        "",
        f"Decay curve: {DECAY_CSV}",
        f"Engine prototype: {ENGINE_CSV}",
        "=" * 58,
    ])
    REPORT_TXT.write_text("\n".join(lines), encoding="utf-8")

    print("===== P6 MARKET MEMORY =====")
    print(f"Records: {len(records)} | Best proxy: {best_psych}")
    print(f"Report: {REPORT_TXT}")


if __name__ == "__main__":
    main()
