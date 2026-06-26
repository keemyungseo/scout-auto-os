"""
Scout Learning Season2 - P14 Adaptive Regime Ecology & Memory Bank

Research only. Use maximum Binance history; classify Bull/Bear/Sideway/Crash/etc;
compute all laws per regime; learn applicability scope before laws;
store append-only situational memory banks; suspect regime change before new laws.

Governed by Scout Research Constitution.
"""

import argparse
import csv
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

from season2_p5_historical_expansion import build_unified_records, collect_missing, discover_dataset_paths
from season2_p6_market_memory import attach_forward_targets
from season2_scout_mission import evaluate_convergence, mission_summary_lines
from season2_regime_core import (
    MIN_REGIME_SAMPLE,
    REGIMES,
    append_memory_bank,
    assign_regimes,
    cross_regime_structures,
    day_metrics,
    detect_regime_change,
    discover_all_dataset_paths,
    generate_date_range,
    law_applicability,
    regime_laws,
    snapshot_memory_banks,
    write_csv,
)

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

REGIME_CLASSIFICATION_CSV = LOGS_DIR / "season2_p14_regime_classification.csv"
REGIME_LAWS_CSV = LOGS_DIR / "season2_p14_regime_laws.csv"
APPLICABILITY_CSV = LOGS_DIR / "season2_p14_law_applicability.csv"
REGIME_CHANGE_CSV = LOGS_DIR / "season2_p14_regime_change_alerts.csv"
CROSS_REGIME_CSV = LOGS_DIR / "season2_p14_cross_regime_structures.csv"
HISTORY_STATUS_CSV = LOGS_DIR / "season2_p14_history_status.csv"
ENGINE_CSV = LOGS_DIR / "season2_p14_regime_engine_output.csv"
REGISTRY_CSV = LOGS_DIR / "season2_p14_regime_registry.csv"
CONVERGENCE_CSV = LOGS_DIR / "season2_p14_convergence_tiers.csv"
REPORT_TXT = LOGS_DIR / "season2_p14_research_report.txt"

DEFAULT_END_DATE = "2026-06-15"
DEFAULT_DAYS_BACK = 60


def build_expanded_records() -> list[dict]:
    """All available history — no recency weighting."""
    records = build_unified_records()
    attach_forward_targets(records)
    known_keys = {(r["scan_time"], r["symbol"]) for r in records}

    from season2_p1_supply_probability_panel import load_rows, normalize
    from season2_p5_historical_expansion import load_physics_index

    physics = load_physics_index()
    for path in discover_all_dataset_paths():
        if path.name.startswith("top10_gainer_learning_"):
            continue
        for row in load_rows(path):
            base = normalize(row, path.name)
            if not base:
                continue
            key = (base["scan_time"], base["symbol"])
            if key in known_keys:
                continue
            phys = physics.get(key, {})
            for pk in (
                "fast_rejection", "congestion_bars_12", "body_pct", "vol_ratio_6", "vol_exhaustion",
            ):
                if pk in phys and phys[pk] != "":
                    try:
                        base[pk] = float(phys[pk])
                    except ValueError:
                        base[pk] = phys[pk]
            known_keys.add(key)
            records.append(base)

    from season2_p1_supply_probability_panel import classify_supply, dominant_trigger, state_at_scan, trigger_bundle

    for record in records:
        if record.get("target_f6") is None and record.get("forward_6h") is not None:
            record["target_f6"] = record["forward_6h"]
        if not record.get("supply_label"):
            label, collapse, _ = classify_supply({**record, "max_profit_12h": record.get("forward_12h")})
            record["supply_label"] = label
            record["collapse_label"] = collapse
        if not record.get("state_scan"):
            record["state_scan"] = state_at_scan(record)
            record["trigger_bundle"] = trigger_bundle(record)
            record["dominant_trigger"] = dominant_trigger(record)

    return records


def collect_history(end_date: str, days_back: int, limit: int) -> list[str]:
    targets = generate_date_range(end_date, days_back)
    missing = [
        d for d in targets
        if not (LOGS_DIR / f"top10_gainer_learning_{d.replace('-', '')}.csv").exists()
    ]
    if not missing:
        return []
    print(f"[history] {len(missing)} missing dates in {days_back}-day window; collecting up to {limit}")
    return collect_missing(missing, limit)


def history_status(records: list[dict], end_date: str, days_back: int) -> list[dict]:
    by_date: dict[str, list] = defaultdict(list)
    for r in records:
        by_date[r["date"]].append(r)
    targets = set(generate_date_range(end_date, days_back))
    rows = []
    for date in sorted(targets):
        grp = by_date.get(date, [])
        rows.append(
            {
                "date": date,
                "in_window": "yes",
                "records": len(grp),
                "ecology_regime": grp[0].get("ecology_regime", "") if grp else "",
                "status": "collected" if grp else "missing",
                "source_files": Counter(r.get("source_file", r.get("source", "?")) for r in grp),
            }
        )
    for date, grp in sorted(by_date.items()):
        if date not in targets:
            rows.append(
                {
                    "date": date,
                    "in_window": "extended",
                    "records": len(grp),
                    "ecology_regime": grp[0].get("ecology_regime", ""),
                    "status": "collected",
                    "source_files": Counter(r.get("source_file", r.get("source", "?")) for r in grp),
                }
            )
    for row in rows:
        src = row.pop("source_files")
        row["sources"] = ";".join(f"{k}:{v}" for k, v in sorted(src.items())) if isinstance(src, Counter) else ""
    return sorted(rows, key=lambda x: x["date"])


def task_regime_classification(records: list[dict], day_info: dict[str, dict]) -> list[dict]:
    rows = []
    by_date: dict[str, list] = defaultdict(list)
    for r in records:
        by_date[r["date"]].append(r)
    for date in sorted(by_date.keys()):
        info = day_info[date]
        m = day_metrics(by_date[date])
        rows.append(
            {
                "date": date,
                "ecology_regime": info["ecology_regime"],
                "regime_confidence_pct": info["regime_confidence_pct"],
                "records": m["n"],
                "median_forward_6h": round(m["median_f6"], 2),
                "collapse_pct": round(m["collapse_pct"], 1),
                "high_supply_pct": round(m["high_supply_pct"], 1),
                "unique_symbols": m["unique_symbols"],
                "avg_top1_return": round(m["avg_top1"], 1),
                "runner_up": info["regime_evidence"].get("runner_up", ""),
            }
        )
    return rows


def task_applicability(records: list[dict]) -> list[dict]:
    rows = []
    laws = [
        ("MID_SUPPLY_outcome", lambda r: "MID_SUPPLY" if r.get("supply_label") == "MID_SUPPLY" else ""),
        ("HIGH_SUPPLY_outcome", lambda r: "HIGH_SUPPLY" if r.get("supply_label") == "HIGH_SUPPLY" else ""),
        ("Compression_state", lambda r: r.get("state_scan") if r.get("state_scan") == "Compression" else ""),
        ("Exhaustion_state", lambda r: r.get("state_scan") if r.get("state_scan") == "Exhaustion" else ""),
        ("COLLAPSE_supply", lambda r: "COLLAPSE" if r.get("supply_label") == "COLLAPSE" else ""),
    ]
    for name, fn in laws:
        rows.extend(law_applicability(records, name, fn))
    return rows


def regime_engine_output(records: list[dict], applicability: list[dict], change_alerts: list[dict]) -> list[dict]:
    scope_index: dict[tuple[str, str], dict] = {}
    for row in applicability:
        scope_index[(row["law_name"], row["applicable_regime"])] = row

    alert_regimes = {a["ecology_regime"] for a in change_alerts if a.get("verdict") == "REGIME_CHANGE_SUSPECTED"}

    rows = []
    for record in records[-40:]:
        regime = record.get("ecology_regime", "Unknown")
        supply = record.get("supply_label", "unknown")
        state = record.get("state_scan", "unknown")

        def scope_for(law_name: str) -> dict:
            return scope_index.get((law_name, regime), {})

        mid = scope_for("MID_SUPPLY_outcome")
        high = scope_for("HIGH_SUPPLY_outcome")
        coll = scope_for("COLLAPSE_supply")

        supporting = []
        conflicting = []
        if mid.get("scope_status") == "in_scope" and supply == "MID_SUPPLY":
            supporting.append(f"MID_SUPPLY in_scope med={mid.get('median_forward_6h')}")
        elif supply == "MID_SUPPLY" and mid.get("scope_status") == "out_of_scope":
            conflicting.append("MID_SUPPLY out_of_scope in this regime")

        if coll.get("scope_status") == "in_scope" and supply == "COLLAPSE":
            supporting.append("COLLAPSE pattern known in regime")
        if regime in alert_regimes:
            conflicting.append("REGIME_CHANGE_SUSPECTED")

        unknown_prob = 100.0 - float(record.get("regime_confidence_pct") or 0)
        if regime == "Unknown":
            unknown_prob = max(unknown_prob, 60)
        if not supporting:
            unknown_prob = min(95, unknown_prob + 15)

        if regime == "Unknown" or unknown_prob >= 55:
            stance = "Wait"
            reason = f"regime={regime} unknown_prob={unknown_prob:.0f}%"
        elif regime == "Crash" or supply == "COLLAPSE":
            stance = "Avoid"
            reason = f"Crash ecology or COLLAPSE supply"
        elif regime == "Recovery" and supporting:
            stance = "Watch"
            reason = "Recovery regime — scope validated, monitor"
        elif regime == "Bull" and high.get("scope_status") == "in_scope":
            stance = "Buy"
            reason = f"Bull regime HIGH_SUPPLY in_scope"
        elif regime in alert_regimes:
            stance = "Wait"
            reason = "Regime change suspected — do not apply stale law"
        elif supporting:
            stance = "Watch"
            reason = f"in_scope evidence in {regime}"
        else:
            stance = "Wait"
            reason = "insufficient scoped evidence"

        rows.append(
            {
                "date": record["date"],
                "symbol": record["symbol"],
                "scan_time": record["scan_time"],
                "ecology_regime": regime,
                "regime_confidence_pct": record.get("regime_confidence_pct"),
                "supply_label": supply,
                "state_scan": state,
                "supporting_evidence": "|".join(supporting) or "none",
                "conflicting_evidence": "|".join(conflicting) or "none",
                "unknown_probability_pct": round(unknown_prob, 1),
                "regime_change_flag": "yes" if regime in alert_regimes else "no",
                "recommended_stance": stance,
                "reason": reason,
                "confidence_tier": "unknown" if unknown_prob >= 50 else "medium" if supporting else "hypothesis",
            }
        )
    return rows


def build_convergence_tiers(structures: list[dict], applicability: list[dict]) -> list[dict]:
    """Tag P14 outputs as core / background / hypothesis per Scout mission."""
    rows = []
    for st in structures:
        improves = []
        if st["structure_type"] == "repeatable_cross_regime":
            improves = ["trend_persistence_estimation", "real_vs_fake_trend_discrimination"]
        elif st["structure_type"] == "regime_conditional":
            improves = ["real_vs_fake_trend_discrimination"]
        ev = evaluate_convergence(
            "ecology_regime",
            improves=improves,
            sample_size=int(st.get("total_sample", 0)),
            confidence=st.get("confidence", "hypothesis"),
        )
        rows.append(
            {
                "finding": st["structure_key"],
                "finding_type": "cross_regime_structure",
                "tier": ev["tier"],
                "operational": ev["operational"],
                "convergence_criteria_met": "|".join(ev["convergence_criteria_met"]),
                "promotion_note": ev["promotion_note"],
            }
        )
    for row in applicability:
        if row.get("scope_status") != "in_scope":
            continue
        ev = evaluate_convergence(
            "supply_label" if "SUPPLY" in row["law_name"] else "ecology_regime",
            improves=["trend_persistence_estimation"],
            sample_size=int(row.get("sample_size", 0)),
            confidence=row.get("confidence", "hypothesis"),
        )
        rows.append(
            {
                "finding": f"{row['law_name']}@{row['applicable_regime']}",
                "finding_type": "law_applicability",
                "tier": ev["tier"],
                "operational": ev["operational"],
                "convergence_criteria_met": "|".join(ev["convergence_criteria_met"]),
                "promotion_note": ev["promotion_note"],
            }
        )
    ev_regime = evaluate_convergence(
        "ecology_regime",
        improves=["real_vs_fake_trend_discrimination"],
        sample_size=sum(int(s.get("total_sample", 0)) for s in structures),
        confidence="medium" if structures else "unknown",
    )
    rows.append(
        {
            "finding": "regime_classification_layer",
            "finding_type": "p14_regime",
            "tier": ev_regime["tier"],
            "operational": False,
            "convergence_criteria_met": "|".join(ev_regime["convergence_criteria_met"]),
            "promotion_note": "background until Bull/Bear/Sideway history sufficient",
        }
    )
    return rows


def build_registry(change_alerts: list[dict], structures: list[dict]) -> list[dict]:
    rows = []
    for alert in change_alerts[:10]:
        rows.append(
            {
                "hypothesis_kind": "regime_change",
                "hypothesis_name": f"{alert['ecology_regime']}:{alert['metric']}",
                "prior_status": "OPEN",
                "current_status": "CONDITIONAL",
                "note": alert["action"],
            }
        )
    for st in structures:
        if st["structure_type"] == "repeatable_cross_regime":
            rows.append(
                {
                    "hypothesis_kind": "cross_regime_structure",
                    "hypothesis_name": st["structure_key"],
                    "prior_status": "OPEN",
                    "current_status": "ACTIVE" if st["confidence"] == "high" else "CONDITIONAL",
                    "note": f"regimes={st['regimes_present']} spread={st['median_spread']}",
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--days-back", type=int, default=DEFAULT_DAYS_BACK)
    parser.add_argument("--collect-limit", type=int, default=5, help="Max new dates to collect this run")
    parser.add_argument("--skip-collect", action="store_true")
    args = parser.parse_args()

    collected = []
    if not args.skip_collect:
        collected = collect_history(args.end_date, args.days_back, args.collect_limit)

    records = build_expanded_records()
    day_info = assign_regimes(records)

    regime_rows = task_regime_classification(records, day_info)
    law_rows = regime_laws(records)
    applicability = task_applicability(records)
    change_alerts = detect_regime_change(records, law_rows)
    structures = cross_regime_structures(law_rows)
    bank_index = snapshot_memory_banks(records, law_rows, day_info)
    hist_status = history_status(records, args.end_date, args.days_back)
    engine = regime_engine_output(records, applicability, change_alerts)
    convergence = build_convergence_tiers(structures, applicability)
    registry = build_registry(change_alerts, structures)

    write_csv(REGIME_CLASSIFICATION_CSV, regime_rows)
    write_csv(REGIME_LAWS_CSV, law_rows)
    write_csv(APPLICABILITY_CSV, applicability)
    write_csv(REGIME_CHANGE_CSV, change_alerts)
    write_csv(CROSS_REGIME_CSV, structures)
    write_csv(HISTORY_STATUS_CSV, hist_status)
    write_csv(ENGINE_CSV, engine)
    write_csv(REGISTRY_CSV, registry)
    write_csv(CONVERGENCE_CSV, convergence)

    dates = sorted({r["date"] for r in records})
    regime_counts = Counter(r.get("ecology_regime", "Unknown") for r in records)
    in_scope = sum(1 for r in applicability if r.get("scope_status") == "in_scope")
    unknown_laws = sum(1 for r in law_rows if r.get("law_value") == "Unknown" or r.get("confidence") == "unknown")

    lines = [
        "===== SCOUT SEASON2 P14 - REGIME ECOLOGY & MEMORY BANK =====",
        "",
        f"History: {len(records)} records | {dates[0]}..{dates[-1]} ({len(dates)} days)",
        f"Window target: {args.days_back} days ending {args.end_date}",
        f"Collected this run: {collected or '(none)'} | Dataset files: {len(discover_all_dataset_paths())}",
        f"Memory banks: {len(bank_index)} regime snapshots (append-only, never discarded)",
        "",
        "Principles:",
        "  - No special treatment for recent 3 days",
        "  - All laws computed per regime separately",
        "  - Unknown when sample insufficient",
        "  - Regime change suspected before new laws",
        "  - Applicability scope learned before law values",
        "",
        "--- Regime classification ---",
    ]
    for regime in REGIMES:
        n = regime_counts.get(regime, 0)
        if n:
            days = len({r["date"] for r in records if r.get("ecology_regime") == regime})
            lines.append(f"  {regime}: {n} records across {days} days")

    lines.extend(["", "--- Regime days ---"])
    for row in regime_rows:
        lines.append(
            f"  {row['date']}: {row['ecology_regime']} ({row['regime_confidence_pct']}%) "
            f"med_f6={row['median_forward_6h']} collapse={row['collapse_pct']}%"
        )

    lines.extend(["", "--- Cross-regime structures ---"])
    for row in structures[:6]:
        lines.append(
            f"  {row['structure_key']}: {row['structure_type']} "
            f"regimes={row['regime_count']} spread={row['median_spread']}"
        )

    lines.extend(["", "--- Regime change alerts ---"])
    if change_alerts:
        for row in change_alerts[:6]:
            lines.append(
                f"  {row['ecology_regime']} {row['metric']}: {row['historical_value']} -> "
                f"{row['recent_value']} [{row['verdict']}]"
            )
    else:
        lines.append("  (none above threshold — or insufficient history)")

    core_n = sum(1 for r in convergence if r.get("tier") == "core")
    bg_n = sum(1 for r in convergence if r.get("tier") == "background")

    lines.extend([
        "",
        f"Applicability: {in_scope} in_scope / {len(applicability)} scope rows",
        f"Laws: {len(law_rows)} regime-local rows | {unknown_laws} Unknown (insufficient n)",
        f"Convergence tiers: {core_n} core / {bg_n} background / {len(convergence)} total",
        "",
    ])
    lines.extend(mission_summary_lines())
    lines.extend([
        "",
        f"Engine: {ENGINE_CSV}",
        f"Memory bank: {LOGS_DIR / 'memory_bank'}",
        "=" * 58,
    ])
    REPORT_TXT.write_text("\n".join(lines), encoding="utf-8")

    print("===== P14 REGIME ECOLOGY & MEMORY BANK =====")
    print(f"Records: {len(records)} | Days: {len(dates)} | Regimes: {sum(1 for c in regime_counts.values() if c)}")
    print(f"Collected: {collected or '(none)'}")
    print(f"Report: {REPORT_TXT}")


if __name__ == "__main__":
    main()
