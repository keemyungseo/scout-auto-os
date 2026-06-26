"""
Scout Season2 - Regime Core (shared infrastructure)

Empirical regime classification, memory banks, applicability scope, regime-change detection.
Governed by Scout Research Constitution: no hidden actors; Unknown over overfitting.
"""

from __future__ import annotations

import csv
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

LOGS_DIR = Path("logs")
MEMORY_BANK_DIR = LOGS_DIR / "memory_bank"

REGIMES = (
    "Bull",
    "Bear",
    "Sideway",
    "Crash",
    "Recovery",
    "Alt_Season",
    "Rotation",
    "Unknown",
)

MIN_REGIME_SAMPLE = 6
MIN_DAY_RECORDS = 20
MIN_SCOPE_N = 8

HISTORY_GLOBS = (
    "top10_gainer_learning_*.csv",
    "top3_gainers_*enriched.csv",
    "top3_gainers_20*.csv",
)


def discover_all_dataset_paths(logs_dir: Path = LOGS_DIR) -> list[Path]:
    seen: set[str] = set()
    paths: list[Path] = []
    for pattern in HISTORY_GLOBS:
        for path in sorted(logs_dir.glob(pattern)):
            if path.name in seen:
                continue
            if "enriched" not in path.name and path.name.startswith("top3_gainers_"):
                enriched = logs_dir / path.name.replace(".csv", "_enriched.csv")
                if enriched.exists():
                    continue
            seen.add(path.name)
            paths.append(path)
    return paths


def generate_date_range(end_date: str, days_back: int) -> list[str]:
    end = datetime.strptime(end_date, "%Y-%m-%d")
    return [(end - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days_back)]


def day_metrics(group: list[dict]) -> dict:
    n = len(group)
    f6 = [r["target_f6"] for r in group if r.get("target_f6") is not None]
    collapses = sum(1 for r in group if r.get("collapse_label") == "YES")
    high = sum(1 for r in group if r.get("supply_label") == "HIGH_SUPPLY")
    collapse_supply = sum(1 for r in group if r.get("supply_label") == "COLLAPSE")
    top1 = [
        r["return_24h_at_scan"]
        for r in group
        if r.get("rank") == 1 and r.get("return_24h_at_scan") is not None
    ]
    unique = len({r["symbol"] for r in group})
    return {
        "n": n,
        "median_f6": statistics.median(f6) if f6 else 0.0,
        "mean_f6": statistics.mean(f6) if f6 else 0.0,
        "collapse_pct": collapses / n * 100 if n else 0.0,
        "high_supply_pct": high / n * 100 if n else 0.0,
        "collapse_supply_pct": collapse_supply / n * 100 if n else 0.0,
        "unique_symbols": unique,
        "avg_top1": statistics.mean(top1) if top1 else 0.0,
        "leader_repeat": n / max(unique, 1),
    }


def classify_ecology_regime(
    group: list[dict],
    prev_metrics: dict | None = None,
    prev_regime: str | None = None,
) -> tuple[str, float, dict]:
    m = day_metrics(group)
    evidence: dict = {k: round(v, 2) if isinstance(v, float) else v for k, v in m.items()}

    if m["n"] < MIN_DAY_RECORDS:
        return "Unknown", 0.0, {**evidence, "reason": "insufficient_day_records"}

    scores: dict[str, float] = {r: 0.0 for r in REGIMES if r != "Unknown"}

    if m["collapse_pct"] >= 20 or m["median_f6"] <= -5 or m["collapse_supply_pct"] >= 12:
        scores["Crash"] += 50 + m["collapse_pct"] * 0.5
    if m["median_f6"] <= -1.5 or (m["collapse_pct"] >= 10 and m["median_f6"] < 1):
        scores["Bear"] += 35 + abs(min(0, m["median_f6"])) * 2
    if m["median_f6"] >= 2.5 and m["collapse_pct"] < 8 and (
        m["high_supply_pct"] >= 20 or m["avg_top1"] >= 25
    ):
        scores["Bull"] += 40 + m["median_f6"] * 2
    if m["unique_symbols"] >= 16 and m["avg_top1"] >= 30 and m["high_supply_pct"] >= 22:
        scores["Alt_Season"] += 45 + (m["unique_symbols"] - 15) * 2
    if m["unique_symbols"] >= 14 and 12 <= m["avg_top1"] <= 38 and m["leader_repeat"] < 2.8:
        scores["Rotation"] += 35 + (m["unique_symbols"] - 13)
    if abs(m["median_f6"]) <= 2 and m["collapse_pct"] < 10 and m["avg_top1"] < 28:
        scores["Sideway"] += 30

    if prev_regime in ("Crash", "Bear") and prev_metrics:
        if m["collapse_pct"] < prev_metrics.get("collapse_pct", 99) - 5:
            scores["Recovery"] += 40
        if m["median_f6"] > prev_metrics.get("median_f6", -99) + 2:
            scores["Recovery"] += 25

    ranked = sorted(scores.items(), key=lambda x: -x[1])
    best, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    gap = best_score - second_score

    if best_score < 25 or gap < 8:
        conf = round(max(0, best_score - gap), 1)
        return "Unknown", conf, {**evidence, "reason": "ambiguous_regime", "scores": scores}

    conf = round(min(95, 40 + gap + best_score * 0.2), 1)
    return best, conf, {**evidence, "runner_up": ranked[1][0], "scores": scores}


def assign_regimes(records: list[dict]) -> dict[str, dict]:
    by_date: dict[str, list] = defaultdict(list)
    for r in records:
        by_date[r["date"]].append(r)

    dates = sorted(by_date.keys())
    day_info: dict[str, dict] = {}
    prev_m: dict | None = None
    prev_r: str | None = None

    for date in dates:
        regime, conf, evidence = classify_ecology_regime(by_date[date], prev_m, prev_r)
        day_info[date] = {
            "ecology_regime": regime,
            "regime_confidence_pct": conf,
            "regime_evidence": evidence,
        }
        for r in by_date[date]:
            r["ecology_regime"] = regime
            r["regime_confidence_pct"] = conf
            r["market_regime"] = regime
        prev_m = day_metrics(by_date[date])
        prev_r = regime

    return day_info


def confidence_tier(n: int, spread: float = 0) -> str:
    if n >= MIN_REGIME_SAMPLE * 4:
        return "high"
    if n >= MIN_REGIME_SAMPLE:
        return "medium"
    return "unknown"


def cohort_law(group: list[dict], law_key: str) -> dict:
    if len(group) < MIN_REGIME_SAMPLE:
        return {
            "law_key": law_key,
            "value": "Unknown",
            "sample_size": len(group),
            "confidence": "unknown",
            "reason": "insufficient_history",
        }
    f6 = [r["target_f6"] for r in group if r.get("target_f6") is not None]
    collapses = sum(1 for r in group if r.get("collapse_label") == "YES")
    n = len(group)
    return {
        "law_key": law_key,
        "value": round(statistics.median(f6), 2) if f6 else "Unknown",
        "median_forward_6h": round(statistics.median(f6), 2) if f6 else "",
        "collapse_rate_pct": round(collapses / n * 100, 1),
        "high_supply_rate_pct": round(
            sum(1 for r in group if r.get("supply_label") == "HIGH_SUPPLY") / n * 100, 1
        ),
        "sample_size": n,
        "confidence": confidence_tier(n),
    }


def law_applicability(
    records: list[dict],
    law_name: str,
    key_fn,
) -> list[dict]:
    """Learn scope before law: which regimes support this pattern."""
    by_regime: dict[str, list] = defaultdict(list)
    global_grp: list[dict] = []

    for r in records:
        k = key_fn(r)
        if not k or k == "unknown":
            continue
        tagged = {**r, "_law_key": k}
        by_regime[r.get("ecology_regime", "Unknown")].append(tagged)
        global_grp.append(tagged)

    if len(global_grp) < MIN_SCOPE_N:
        return [
            {
                "law_name": law_name,
                "law_key": "all",
                "applicable_regime": "Unknown",
                "scope_status": "insufficient_global_history",
                "sample_size": len(global_grp),
                "confidence": "unknown",
            }
        ]

    global_m = cohort_law(global_grp, law_name)
    global_med = global_m.get("median_forward_6h")
    if global_med == "":
        global_med = 0.0
    else:
        global_med = float(global_med)

    rows = []
    for regime in REGIMES:
        grp = [r for r in by_regime.get(regime, []) if r.get("_law_key")]
        if len(grp) < MIN_REGIME_SAMPLE:
            rows.append(
                {
                    "law_name": law_name,
                    "applicable_regime": regime,
                    "scope_status": "unknown_insufficient_n",
                    "sample_size": len(grp),
                    "median_forward_6h": "",
                    "collapse_rate_pct": "",
                    "delta_vs_global": "",
                    "confidence": "unknown",
                }
            )
            continue

        m = cohort_law(grp, law_name)
        med = m.get("median_forward_6h")
        med_f = float(med) if med != "" and med != "Unknown" else None
        delta = round(med_f - global_med, 2) if med_f is not None else ""
        consistent = med_f is not None and abs(med_f - global_med) <= 4
        status = "in_scope" if consistent and m["confidence"] != "unknown" else "out_of_scope"

        rows.append(
            {
                "law_name": law_name,
                "applicable_regime": regime,
                "scope_status": status,
                "sample_size": m["sample_size"],
                "median_forward_6h": m.get("median_forward_6h", ""),
                "collapse_rate_pct": m.get("collapse_rate_pct", ""),
                "delta_vs_global": delta,
                "confidence": m["confidence"],
            }
        )
    return rows


def regime_laws(records: list[dict]) -> list[dict]:
    """All laws computed separately per regime — never pooled."""
    rows = []
    dimensions = [
        ("supply_label", lambda r: r.get("supply_label", "unknown")),
        ("state_scan", lambda r: r.get("state_scan", "unknown")),
        ("dominant_trigger", lambda r: r.get("dominant_trigger", "unknown")),
    ]
    for regime in REGIMES:
        reg_records = [r for r in records if r.get("ecology_regime") == regime]
        if not reg_records:
            continue
        for dim_name, dim_fn in dimensions:
            groups: dict[str, list] = defaultdict(list)
            for r in reg_records:
                groups[dim_fn(r)].append(r)
            for key, grp in groups.items():
                if key in ("unknown", ""):
                    continue
                law = cohort_law(grp, f"{dim_name}={key}")
                rows.append(
                    {
                        "ecology_regime": regime,
                        "law_dimension": dim_name,
                        "law_key": key,
                        "median_forward_6h": law.get("median_forward_6h", "Unknown"),
                        "collapse_rate_pct": law.get("collapse_rate_pct", ""),
                        "high_supply_rate_pct": law.get("high_supply_rate_pct", ""),
                        "sample_size": law["sample_size"],
                        "law_value": law["value"],
                        "confidence": law["confidence"],
                    }
                )
    return rows


def detect_regime_change(
    records: list[dict],
    regime_law_rows: list[dict],
) -> list[dict]:
    """
    If recent regime-period outcomes diverge from historical bank,
    suspect regime change — not a new law.
    """
    by_date: dict[str, list] = defaultdict(list)
    for r in records:
        by_date[r["date"]].append(r)
    dates = sorted(by_date.keys())
    if len(dates) < 4:
        return []

    alerts = []
    for regime in REGIMES:
        reg_dates = [d for d in dates if by_date[d][0].get("ecology_regime") == regime]
        if len(reg_dates) < 2:
            continue

        historical_dates = reg_dates[:-1]
        recent_date = reg_dates[-1]
        hist_records = [r for d in historical_dates for r in by_date[d]]
        recent_records = by_date[recent_date]

        if len(hist_records) < MIN_REGIME_SAMPLE or len(recent_records) < MIN_DAY_RECORDS // 2:
            continue

        hist_m = day_metrics(hist_records)
        recent_m = day_metrics(recent_records)

        for metric, threshold in (
            ("collapse_pct", 12),
            ("median_f6", 3.5),
            ("high_supply_pct", 15),
        ):
            h = hist_m[metric]
            rc = recent_m[metric]
            delta = abs(rc - h)
            if delta >= threshold:
                alerts.append(
                    {
                        "ecology_regime": regime,
                        "recent_date": recent_date,
                        "metric": metric,
                        "historical_value": round(h, 2),
                        "recent_value": round(rc, 2),
                        "delta": round(rc - h, 2),
                        "verdict": "REGIME_CHANGE_SUSPECTED",
                        "action": "Do not mint new law; retest under new regime context",
                        "historical_days": len(historical_dates),
                        "confidence": "medium" if delta >= threshold * 1.5 else "hypothesis",
                    }
                )

        hist_laws = [row for row in regime_law_rows if row["ecology_regime"] == regime]
        for law in hist_laws[:5]:
            if law["confidence"] == "unknown" or law["law_value"] == "Unknown":
                continue
            key_dim = law["law_dimension"]
            key_val = law["law_key"]
            recent_grp = [r for r in recent_records if r.get(key_dim) == key_val]
            if len(recent_grp) < 3:
                continue
            recent_med = statistics.median(
                [r["target_f6"] for r in recent_grp if r.get("target_f6") is not None]
            ) if any(r.get("target_f6") is not None for r in recent_grp) else None
            if recent_med is None:
                continue
            try:
                hist_val = float(law["law_value"])
            except (TypeError, ValueError):
                continue
            if abs(recent_med - hist_val) >= 5:
                alerts.append(
                    {
                        "ecology_regime": regime,
                        "recent_date": recent_date,
                        "metric": f"law:{law['law_key']}",
                        "historical_value": hist_val,
                        "recent_value": round(recent_med, 2),
                        "delta": round(recent_med - hist_val, 2),
                        "verdict": "REGIME_CHANGE_SUSPECTED",
                        "action": "Law drift within regime — verify ecology shift",
                        "historical_days": len(historical_dates),
                        "confidence": "medium",
                    }
                )
    return alerts


def append_memory_bank(
    regime: str,
    bank_type: str,
    payload: dict,
    logs_dir: Path = LOGS_DIR,
) -> Path:
    """Append-only situational memory — old data never discarded."""
    base = logs_dir / "memory_bank" / regime.replace(" ", "_")
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"{bank_type}.jsonl"
    payload = {**payload, "stored_at": datetime.utcnow().isoformat(timespec="seconds") + "Z"}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return path


def snapshot_memory_banks(
    records: list[dict],
    regime_law_rows: list[dict],
    day_info: dict[str, dict],
    logs_dir: Path = LOGS_DIR,
) -> list[dict]:
    """Write regime situational banks from current full history."""
    index_rows = []
    for regime in REGIMES:
        reg_records = [r for r in records if r.get("ecology_regime") == regime]
        if not reg_records:
            continue
        dates = sorted({r["date"] for r in reg_records})
        m = day_metrics(reg_records)
        payload = {
            "regime": regime,
            "date_range": f"{dates[0]}..{dates[-1]}",
            "days": len(dates),
            "records": len(reg_records),
            "metrics": m,
        }
        path = append_memory_bank(regime, "ecology_snapshot", payload, logs_dir)
        index_rows.append({"regime": regime, "bank_file": str(path), "records": len(reg_records), "days": len(dates)})

        laws = [row for row in regime_law_rows if row["ecology_regime"] == regime and row["confidence"] != "unknown"]
        if laws:
            append_memory_bank(regime, "regime_laws", {"regime": regime, "laws": laws[:40]}, logs_dir)

    summary_path = logs_dir / "season2_p14_memory_bank_index.csv"
    if index_rows:
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(index_rows[0].keys()))
            w.writeheader()
            w.writerows(index_rows)
    return index_rows


def cross_regime_structures(regime_law_rows: list[dict]) -> list[dict]:
    """Find laws that repeat across environments vs regime-local."""
    by_key: dict[str, list] = defaultdict(list)
    for row in regime_law_rows:
        if row["confidence"] == "unknown" or row["law_value"] == "Unknown":
            continue
        key = f"{row['law_dimension']}={row['law_key']}"
        by_key[key].append(row)

    structures = []
    for key, rows in by_key.items():
        regimes = {r["ecology_regime"] for r in rows}
        if len(regimes) < 2:
            continue
        meds = [float(r["median_forward_6h"]) for r in rows if r.get("median_forward_6h") not in ("", "Unknown")]
        if len(meds) < 2:
            continue
        spread = max(meds) - min(meds)
        structure = "repeatable_cross_regime" if spread <= 4 else "regime_conditional"
        structures.append(
            {
                "structure_key": key,
                "regimes_present": "|".join(sorted(regimes)),
                "regime_count": len(regimes),
                "median_spread": round(spread, 2),
                "structure_type": structure,
                "total_sample": sum(int(r["sample_size"]) for r in rows),
                "confidence": "high" if len(regimes) >= 3 and spread <= 3 else "medium" if spread <= 5 else "hypothesis",
            }
        )
    return sorted(structures, key=lambda x: (-x["regime_count"], x["median_spread"]))


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
