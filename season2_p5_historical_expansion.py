"""
Scout Learning Season2 - P5 Historical Expansion & Conditional Revival System

Research only. Does NOT optimize v1/v2/v3 engines.
Expands historical data, manages hypothesis lifecycle, finds conditional revival.
"""

import argparse
import csv
import statistics
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from season2_p1_supply_probability_panel import (
    LOGS_DIR,
    classify_regime,
    classify_supply,
    dominant_trigger,
    load_rows,
    normalize,
    pf,
    pb,
    state_at_scan,
    trigger_bundle,
)

EXPANSION_STATUS_CSV = LOGS_DIR / "historical_expansion_status.csv"
REGISTRY_CSV = LOGS_DIR / "hypothesis_registry.csv"
REVIVAL_CSV = LOGS_DIR / "conditional_revival_candidates.csv"
CONTEXT_PERF_CSV = LOGS_DIR / "feature_context_performance.csv"
REGIME_MEMORY_CSV = LOGS_DIR / "market_regime_memory.csv"
REPORT_TXT = LOGS_DIR / "scout_engine_update_report.txt"
PHYSICS_CSV = LOGS_DIR / "season2_p4_physics_features.csv"

STEP1_DATES = [
    "2026-06-12", "2026-06-11", "2026-06-10", "2026-06-09",
    "2026-06-08", "2026-06-07", "2026-06-06",
]
KNOWN_DATES = STEP1_DATES + ["2026-06-13", "2026-06-14", "2026-06-15"]
MIN_BUCKET = 5
MIN_CONTEXT = 8


def discover_dataset_paths() -> list[Path]:
    paths = sorted(LOGS_DIR.glob("top10_gainer_learning_*.csv"))
    paths += [p for p in LOGS_DIR.glob("top3_gainers_*enriched.csv")]
    return paths


def collect_missing(dates: list[str], limit: int) -> list[str]:
    collected: list[str] = []
    for date in dates:
        out = LOGS_DIR / f"top10_gainer_learning_{date.replace('-', '')}.csv"
        if out.exists():
            continue
        if len(collected) >= limit:
            break
        print(f"[collect] {date} ...")
        result = subprocess.run(
            [sys.executable, "season2_historical_collector.py", "--date", date],
            cwd=Path.cwd(),
        )
        if result.returncode == 0 and out.exists():
            collected.append(date)
        else:
            print(f"[collect] failed or empty: {date}")
    return collected


def load_physics_index() -> dict[tuple[str, str], dict]:
    if not PHYSICS_CSV.exists():
        return {}
    index: dict[tuple[str, str], dict] = {}
    for row in csv.DictReader(PHYSICS_CSV.open(encoding="utf-8")):
        key = (row.get("scan_time", ""), row.get("symbol", ""))
        index[key] = row
    return index


def build_unified_records() -> list[dict]:
    physics = load_physics_index()
    records: list[dict] = []
    for path in discover_dataset_paths():
        for row in load_rows(path):
            base = normalize(row, path.name)
            if not base:
                continue
            base["source_file"] = path.name
            phys = physics.get((base["scan_time"], base["symbol"]), {})
            for key in (
                "fast_rejection", "congestion_bars_12", "body_pct", "lower_wick_pct",
                "upper_wick_pct", "vol_ratio_6", "vol_exhaustion", "dist_from_12h_high_pct",
            ):
                if key in phys and phys[key] != "":
                    val = phys[key]
                    if val in ("True", "False"):
                        base[key] = val == "True"
                    else:
                        try:
                            base[key] = float(val)
                        except ValueError:
                            base[key] = val
            records.append(base)

    for record in records:
        record["max_profit_6h"] = max(
            filter(None, [record.get("forward_2h"), record.get("forward_4h"), record.get("forward_6h")]),
            default=None,
        )
        label, collapse, _ = classify_supply({**record, "max_profit_12h": record.get("forward_12h")})
        record["supply_label"] = label
        record["collapse_label"] = collapse
        record["state_scan"] = state_at_scan(record)
        record["trigger_bundle"] = trigger_bundle(record)
        record["dominant_trigger"] = dominant_trigger(record)
        if record.get("forward_6h") is not None:
            record["target_f6"] = record["forward_6h"]
        record["target_persist_4h"] = (
            1 if record.get("forward_4h") is not None and record["forward_4h"] >= 0 else 0
        )

    by_date: dict[str, list] = defaultdict(list)
    for record in records:
        by_date[record["date"]].append(record)
    for date, group in by_date.items():
        regime = classify_regime(group)
        for record in group:
            record["market_regime"] = regime

    return records


def late_leader(record: dict) -> bool:
    return (record.get("rank") or 99) <= 3 and (record.get("return_24h_at_scan") or 0) >= 30


def vol_zone(record: dict) -> str:
    atr = record.get("atr_ratio")
    if atr is None:
        return "vol_unknown"
    if atr <= 1.1:
        return "vol_low"
    if atr >= 1.5:
        return "vol_high"
    return "vol_mid"


def rank_zone(record: dict) -> str:
    rank = record.get("rank") or 99
    if rank <= 3:
        return "rank_top3"
    if rank <= 7:
        return "rank_mid"
    return "rank_tail"


def compression_ctx(record: dict) -> bool:
    return bool(record.get("pre6_tight_range") or record.get("pre6_volatility_compression"))


def feature_bucket(record: dict, feature: str) -> str:
    if feature == "ma24_slope":
        v = record.get("ma24_slope_percent")
        if v is None:
            return "na"
        return "high" if v >= 5 else "low"
    if feature == "volume_ratio_ma24":
        v = record.get("volume_ratio_ma24")
        if v is None:
            return "na"
        return "mid" if 1.2 <= v <= 4 else "extreme"
    if feature == "trigger_bundle":
        return record.get("trigger_bundle") or "na"
    if feature == "dominant_trigger":
        return record.get("dominant_trigger") or "na"
    if feature == "compression":
        return "yes" if compression_ctx(record) else "no"
    if feature == "distance_ma24":
        v = record.get("distance_ma24_percent")
        if v is None:
            return "na"
        if v >= 25:
            return "far"
        if v >= 10:
            return "mid"
        return "near"
    if feature == "fast_rejection":
        return "yes" if record.get("fast_rejection") else "no"
    if feature == "congestion":
        v = record.get("congestion_bars_12")
        if v is None:
            return "na"
        return "high" if float(v) >= 6 else "low"
    if feature == "state_scan":
        return record.get("state_scan") or "na"
    if feature == "late_leader":
        return "yes" if late_leader(record) else "no"
    if feature == "lower_wick":
        v = record.get("lower_wick_pct")
        if v is None:
            return "na"
        return "long" if float(v) >= 45 else "short"
    return "na"


CONTEXTS: list[tuple[str, str, callable]] = [
    ("market_regime", "regime", lambda r: r.get("market_regime", "unknown")),
    ("volatility_zone", "vol", vol_zone),
    ("rank_zone", "rank", rank_zone),
    ("compression", "compress", compression_ctx),
    ("fast_rejection", "frej", lambda r: bool(r.get("fast_rejection"))),
    ("late_leader", "late", late_leader),
    ("scan_hour", "hour", lambda r: r.get("scan_time", "")[11:13]),
]


@dataclass
class HypothesisSeed:
    name: str
    description: str
    feature: str
    prior_status: str


SEEDS = [
    HypothesisSeed("ma24_slope", "MA24 slope predicts forward supply", "ma24_slope", "DEAD_FOR_NOW"),
    HypothesisSeed("volume_ratio_ma24", "Volume vs MA24 predicts continuation", "volume_ratio_ma24", "DEAD_FOR_NOW"),
    HypothesisSeed("trigger_bundle", "Composite trigger bundle", "trigger_bundle", "DEAD_FOR_NOW"),
    HypothesisSeed("dominant_trigger", "Single dominant trigger tag", "dominant_trigger", "DEAD_FOR_NOW"),
    HypothesisSeed("compression", "Pre-scan compression predicts breakout", "compression", "CONDITIONAL"),
    HypothesisSeed("distance_ma24", "Distance from MA24 predicts continuation", "distance_ma24", "OPEN"),
    HypothesisSeed("fast_rejection", "Upper wick rejection predicts fade", "fast_rejection", "CONDITIONAL"),
    HypothesisSeed("congestion", "Local congestion bar count", "congestion", "CONDITIONAL"),
    HypothesisSeed("state_scan", "Composite scan state label", "state_scan", "OPEN"),
    HypothesisSeed("late_leader", "Top rank + high 24h = late trend", "late_leader", "ACTIVE"),
    HypothesisSeed("lower_wick", "Long lower wick recovery signal", "lower_wick", "OPEN"),
]


def loo_mae(records: list[dict], key_fn) -> tuple[float, int, int]:
    dates = sorted({r["date"] for r in records})
    errors: list[float] = []
    top1 = scans = 0
    for holdout in dates:
        train = [r for r in records if r["date"] != holdout]
        test = [r for r in records if r["date"] == holdout]
        groups: dict[str, list] = defaultdict(list)
        for r in train:
            groups[key_fn(r)].append(r)
        medians = {
            k: statistics.median([x["target_f6"] for x in g if x.get("target_f6") is not None])
            for k, g in groups.items()
            if len(g) >= MIN_BUCKET and any(x.get("target_f6") is not None for x in g)
        }
        gvals = [x["target_f6"] for x in train if x.get("target_f6") is not None]
        gm = statistics.median(gvals) if gvals else 0.0
        by_scan: dict[str, list] = defaultdict(list)
        for r in test:
            by_scan[r["scan_time"]].append(r)
        for scan_rows in by_scan.values():
            scans += 1
            preds = []
            for r in scan_rows:
                pred = medians.get(key_fn(r), gm)
                actual = r.get("target_f6")
                if actual is not None:
                    errors.append(abs(pred - actual))
                preds.append((r["symbol"], pred))
            if preds:
                best = max(preds, key=lambda x: x[1])
                actual_best = max(scan_rows, key=lambda x: x.get("target_f6") or -999)
                if best[0] == actual_best["symbol"]:
                    top1 += 1
    return statistics.mean(errors) if errors else 999.0, top1, scans


def spread_f6(records: list[dict], key_fn) -> float:
    groups: dict[str, list] = defaultdict(list)
    for r in records:
        groups[key_fn(r)].append(r)
    meds = []
    for g in groups.values():
        if len(g) < MIN_BUCKET:
            continue
        vals = [x["target_f6"] for x in g if x.get("target_f6") is not None]
        if vals:
            meds.append(statistics.median(vals))
    return max(meds) - min(meds) if len(meds) >= 2 else 0.0


def global_baseline_mae(records: list[dict]) -> float:
    return loo_mae(records, lambda _: "all")[0]


def evaluate_hypothesis(records: list[dict], feature: str, baseline: float) -> dict:
    key_fn = lambda r, f=feature: feature_bucket(r, f)
    mae, top1, scans = loo_mae(records, key_fn)
    spread = spread_f6(records, key_fn)
    return {
        "global_mae": round(mae, 2),
        "global_spread_f6": round(spread, 2),
        "top1": f"{top1}/{scans}",
        "vs_baseline": round(baseline - mae, 2),
        "n": len(records),
    }


def evaluate_conditional(records: list[dict], feature: str, baseline: float) -> list[dict]:
    rows: list[dict] = []
    global_mae = loo_mae(records, lambda r, f=feature: feature_bucket(r, f))[0]
    key_fn = lambda r, f=feature: feature_bucket(r, f)

    for ctx_name, _tag, ctx_fn in CONTEXTS:
        subset = [r for r in records if ctx_fn(r)]
        if len(subset) < MIN_CONTEXT:
            continue
        ctx_mae = loo_mae(subset, key_fn)[0]
        ctx_spread = spread_f6(subset, key_fn)
        gain = global_mae - ctx_mae
        if gain >= 0.25 or (ctx_mae < baseline * 0.99 and ctx_spread >= 3):
            rows.append(
                {
                    "hypothesis": feature,
                    "context": ctx_name,
                    "context_n": len(subset),
                    "context_mae": round(ctx_mae, 2),
                    "global_mae": round(global_mae, 2),
                    "mae_gain_vs_global": round(gain, 2),
                    "context_spread_f6": round(ctx_spread, 2),
                    "revival_candidate": "YES" if gain >= 0.35 or ctx_mae < baseline * 0.98 else "MAYBE",
                }
            )

    # combo contexts for conditional revival search
    combos = [
        ("compression+vol_low", lambda r: compression_ctx(r) and vol_zone(r) == "vol_low"),
        ("compression+vol_high", lambda r: compression_ctx(r) and vol_zone(r) == "vol_high"),
        ("fast_rejection+vol_high", lambda r: bool(r.get("fast_rejection")) and vol_zone(r) == "vol_high"),
        ("late_leader+regime_rotation", lambda r: late_leader(r) and "Rotation" in (r.get("market_regime") or "")),
        ("ma24_high+compression", lambda r: (record_ma24_high(r)) and compression_ctx(r)),
    ]
    for combo_name, combo_fn in combos:
        subset = [r for r in records if combo_fn(r)]
        if len(subset) < MIN_CONTEXT:
            continue
        ctx_mae = loo_mae(subset, key_fn)[0]
        ctx_spread = spread_f6(subset, key_fn)
        gain = global_mae - ctx_mae
        if gain >= 0.25:
            rows.append(
                {
                    "hypothesis": feature,
                    "context": combo_name,
                    "context_n": len(subset),
                    "context_mae": round(ctx_mae, 2),
                    "global_mae": round(global_mae, 2),
                    "mae_gain_vs_global": round(gain, 2),
                    "context_spread_f6": round(ctx_spread, 2),
                    "revival_candidate": "YES" if gain >= 0.35 else "MAYBE",
                }
            )
    return rows


def record_ma24_high(record: dict) -> bool:
    v = record.get("ma24_slope_percent")
    return v is not None and v >= 5


def assign_status(seed: HypothesisSeed, perf: dict, revival_rows: list[dict], baseline: float) -> tuple[str, str]:
    mae = perf["global_mae"]
    gain = perf["vs_baseline"]
    best_revival = max(revival_rows, key=lambda x: x["mae_gain_vs_global"], default=None)

    if mae < baseline * 0.995 and perf["global_spread_f6"] >= 3:
        return "REVIVED" if seed.prior_status == "DEAD_FOR_NOW" else "ACTIVE", f"beats baseline MAE {mae} vs {baseline:.2f}"
    if mae < baseline * 0.97 and perf["global_spread_f6"] >= 5 and perf["n"] >= 80:
        return "ACTIVE", "stable global beat on expanded sample"
    if seed.prior_status == "DEAD_FOR_NOW" and gain >= 0.3:
        return "REVIVED", f"expanded data improved vs baseline by {gain:.2f}pp MAE"
    if best_revival and best_revival["mae_gain_vs_global"] >= 0.5:
        return "CONDITIONAL", f"revives in {best_revival['context']} mae={best_revival['context_mae']}"
    if mae > baseline * 1.02 and not revival_rows:
        return "DEAD_FOR_NOW", "no global or conditional gain on current sample"
    if revival_rows:
        return "CONDITIONAL", "weak global but context pockets exist"
    return "RETIRED", "weak globally; monitor on next expansion batch"


def expansion_status(records: list[dict]) -> list[dict]:
    by_date: dict[str, list] = defaultdict(list)
    for r in records:
        by_date[r["date"]].append(r)
    rows = []
    for date in sorted(by_date.keys()):
        group = by_date[date]
        rows.append(
            {
                "date": date,
                "records": len(group),
                "symbols_unique": len({r["symbol"] for r in group}),
                "scans": len({r["scan_time"] for r in group}),
                "source": Counter_src(group),
                "status": "collected",
            }
        )
    for date in KNOWN_DATES:
        if date not in by_date:
            rows.append({"date": date, "records": 0, "status": "missing"})
    return sorted(rows, key=lambda x: x["date"])


def Counter_src(group: list[dict]) -> str:
    c = defaultdict(int)
    for r in group:
        c[r.get("source_file", "?")] += 1
    return ";".join(f"{k}:{v}" for k, v in sorted(c.items()))


def regime_memory(records: list[dict]) -> list[dict]:
    by_date: dict[str, list] = defaultdict(list)
    for r in records:
        by_date[r["date"]].append(r)
    rows = []
    for date, group in sorted(by_date.items()):
        f6 = [r["target_f6"] for r in group if r.get("target_f6") is not None]
        collapses = sum(1 for r in group if r.get("collapse_label") == "YES")
        rows.append(
            {
                "date": date,
                "market_regime": group[0].get("market_regime", ""),
                "n": len(group),
                "median_forward_6h": round(statistics.median(f6), 2) if f6 else "",
                "collapse_pct": round(collapses / len(group) * 100, 1),
                "unique_symbols": len({r["symbol"] for r in group}),
            }
        )
    return rows


def schedule_next(records: list[dict], revival_rows: list[dict], collected: list[str]) -> list[str]:
    lines = []
    dates_present = {r["date"] for r in records}
    missing_step1 = [d for d in STEP1_DATES if d not in dates_present]
    if missing_step1:
        lines.append(f"NEXT COLLECT: Step1 dates {missing_step1[:3]} (batch of 3)")
    else:
        lines.append("NEXT COLLECT: Step1 complete -> extend to 14-day window (06-01~06-05)")

    if revival_rows:
        top = sorted(revival_rows, key=lambda x: -x["mae_gain_vs_global"])[:3]
        for row in top:
            lines.append(
                f"VALIDATE REVIVAL: {row['hypothesis']} in {row['context']} "
                f"(gain {row['mae_gain_vs_global']}, n={row['context_n']})"
            )

    low_n_dates = [d for d in dates_present if sum(1 for r in records if r["date"] == d) < 40]
    if low_n_dates:
        lines.append(f"EXPAND BREADTH: add TOP20 or extra scans on {low_n_dates}")

    lines.append("HORIZON TEST: collect true forward_30m/1h on next batch")
    return lines


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        with path.open("w", newline="", encoding="utf-8") as f:
            f.write("")
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


def write_report(
    records: list[dict],
    registry: list[dict],
    revival_rows: list[dict],
    collected: list[str],
    baseline: float,
    schedule: list[str],
) -> None:
    dates = sorted({r["date"] for r in records})
    active = [r for r in registry if r["current_status"] == "ACTIVE"]
    conditional = [r for r in registry if r["current_status"] == "CONDITIONAL"]
    revived = [r for r in registry if r["current_status"] == "REVIVED"]
    dead_now = [r for r in registry if r["current_status"] == "DEAD_FOR_NOW"]

    lines = [
        "===== SCOUT SEASON2 P5 - HISTORICAL EXPANSION & REVIVAL =====",
        "",
        f"1. Date range: {dates[0]} -> {dates[-1]} ({len(dates)} days)",
        f"2. Total records: {len(records)}",
        f"3. Newly collected this run: {collected or '(none)'}",
        f"4. Baseline LOO MAE: {baseline:.2f}%",
        "",
        "5. Strong / ACTIVE hypotheses:",
    ]
    for r in active:
        lines.append(f"   - {r['hypothesis_name']}: MAE={r['global_performance']} {r['decision_reason']}")
    lines.append("6. CONDITIONAL (revival contexts):")
    for r in conditional[:8]:
        lines.append(f"   - {r['hypothesis_name']}: {r['revival_conditions']}")
    lines.append("7. REVIVED:")
    for r in revived:
        lines.append(f"   - {r['hypothesis_name']}: {r['decision_reason']}")
    lines.append("8. DEAD_FOR_NOW (not permanent DEAD):")
    for r in dead_now:
        lines.append(f"   - {r['hypothesis_name']}: {r['decision_reason']}")

    lines.extend(["", "9. Top conditional revival candidates:"])
    for row in sorted(revival_rows, key=lambda x: -x["mae_gain_vs_global"])[:8]:
        lines.append(
            f"   {row['hypothesis']} @ {row['context']}: ctx_mae={row['context_mae']} "
            f"gain={row['mae_gain_vs_global']} n={row['context_n']}"
        )

    lines.extend(["", "10. Engine-ready candidates (ACTIVE/REVIVED/strong CONDITIONAL):"])
    engine_ready = [r for r in registry if r["current_status"] in ("ACTIVE", "REVIVED")]
    engine_ready += [r for r in revival_rows if r["revival_candidate"] == "YES"]
    for item in engine_ready[:6]:
        name = item.get("hypothesis_name") or item.get("hypothesis")
        lines.append(f"   - {name}")

    lines.extend(["", "11. NOT for production yet:"])
    for r in dead_now:
        lines.append(f"   - {r['hypothesis_name']} (wait for context or more dates)")

    lines.extend(["", "12. Information gain scheduler:"])
    lines.extend(schedule)

    lines.extend([
        "",
        f"Files: {REGISTRY_CSV}, {REVIVAL_CSV}, {EXPANSION_STATUS_CSV}",
        "=" * 60,
    ])
    REPORT_TXT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect-limit", type=int, default=2, help="max new dates to collect this run")
    parser.add_argument("--skip-collect", action="store_true")
    args = parser.parse_args()

    collected: list[str] = []
    if not args.skip_collect:
        missing = [d for d in STEP1_DATES if not (LOGS_DIR / f"top10_gainer_learning_{d.replace('-', '')}.csv").exists()]
        collected = collect_missing(missing, args.collect_limit)

    records = build_unified_records()
    baseline = global_baseline_mae(records)

    registry_rows: list[dict] = []
    revival_all: list[dict] = []
    context_perf: list[dict] = []

    date_range = f"{min(r['date'] for r in records)}..{max(r['date'] for r in records)}" if records else ""

    for seed in SEEDS:
        perf = evaluate_hypothesis(records, seed.feature, baseline)
        revivals = evaluate_conditional(records, seed.feature, baseline)
        revival_all.extend(revivals)
        status, reason = assign_status(seed, perf, revivals, baseline)

        best_ctx = ""
        if revivals:
            best = max(revivals, key=lambda x: x["mae_gain_vs_global"])
            best_ctx = f"{best['context']}|mae={best['context_mae']}|gain={best['mae_gain_vs_global']}"

        registry_rows.append(
            {
                "hypothesis_name": seed.name,
                "description": seed.description,
                "feature_used": seed.feature,
                "current_status": status,
                "global_performance": f"mae={perf['global_mae']} spread={perf['global_spread_f6']} top1={perf['top1']}",
                "best_context": best_ctx,
                "worst_context": "",
                "sample_size": perf["n"],
                "confidence": "medium" if perf["n"] >= 120 else "low" if perf["n"] >= 60 else "hypothesis",
                "last_tested_date_range": date_range,
                "revival_conditions": best_ctx,
                "counterexamples": "see conditional_revival_candidates.csv",
                "decision_reason": reason,
            }
        )

        for rev in revivals:
            context_perf.append({**rev, "feature": seed.feature})

    exp_status = expansion_status(records)
    regime_rows = regime_memory(records)
    schedule = schedule_next(records, revival_all, collected)

    write_csv(EXPANSION_STATUS_CSV, exp_status)
    write_csv(REGISTRY_CSV, registry_rows)
    write_csv(REVIVAL_CSV, revival_all)
    write_csv(CONTEXT_PERF_CSV, context_perf)
    write_csv(REGIME_MEMORY_CSV, regime_rows)
    write_report(records, registry_rows, revival_all, collected, baseline, schedule)

    print("===== P5 HISTORICAL EXPANSION =====")
    print(f"Records: {len(records)} | Dates: {len(set(r['date'] for r in records))} | Baseline MAE: {baseline:.2f}%")
    print(f"Collected: {collected}")
    print(f"Registry: {REGISTRY_CSV}")
    print(f"Report: {REPORT_TXT}")


if __name__ == "__main__":
    main()
