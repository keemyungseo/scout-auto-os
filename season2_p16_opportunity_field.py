"""
Scout Learning Season2 - P16 Opportunity Field & Seedbed Discovery

Evaluates opportunity fields (scan-level environments), not isolated symbols.
Builds on P15 individual evaluations. Does NOT predict price.

Governed by Scout Research Constitution and Scout Mission.
"""

import argparse
import csv
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from season2_p14_regime_memory_bank import build_expanded_records
from season2_p15_situation_output import (
    build_persistence_tables,
    enrich_record_stack,
    evaluate_record,
    load_active_interactions,
    load_core_patterns,
    load_engine_index,
    rank_within_scans,
)
from season2_p6_market_memory import attach_forward_targets
from season2_p7_behaviour_grammar import enrich_physics
from season2_p8_participant_state import enrich_panel_fields
from season2_p9_conditional_interaction_mining import prepare_records
from season2_regime_core import assign_regimes
from season2_scout_mission import mission_summary_lines

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

FIELDS_CSV = LOGS_DIR / "season2_p16_opportunity_fields.csv"
SEEDBED_CSV = LOGS_DIR / "season2_p16_seedbed_quality.csv"
CLUSTERS_CSV = LOGS_DIR / "season2_p16_field_clusters.csv"
EXPANSION_CSV = LOGS_DIR / "season2_p16_pre_expansion_signals.csv"
FAKE_CLUSTERS_CSV = LOGS_DIR / "season2_p16_fake_trend_clusters.csv"
COLLAPSE_PROP_CSV = LOGS_DIR / "season2_p16_collapse_propagation.csv"
FIELD_RANKING_CSV = LOGS_DIR / "season2_p16_field_ranking.csv"
WATCHLIST_CSV = LOGS_DIR / "season2_p16_watchlist.csv"
REPORT_TXT = LOGS_DIR / "season2_p16_research_report.txt"

MIN_CLUSTER = 2
MIN_FIELD = 3
EXPANSION_F6_THRESHOLD = 3.0

EARLY_SITUATIONS = {"Accumulation", "Early Trend", "Healthy Trend"}
LATE_SITUATIONS = {"Late Trend", "Distribution", "Recovery"}
SEEDBED_LEVELS = ("Very fertile", "Fertile", "Neutral", "Exhausted", "Dangerous", "Unknown")


def clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


def pf(val, default=None):
    if val is None or val == "":
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


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


def load_p15_evaluations() -> list[dict]:
    path = LOGS_DIR / "season2_p15_operational_scores.csv"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def structure_signature(row: dict) -> str:
    ps = (row.get("participant_state") or "unknown").split("_")[0][:12]
    return "|".join(
        [
            row.get("situation", "Unknown"),
            row.get("health_class", "?"),
            row.get("pressure_band", "?"),
            row.get("supply_context", "?"),
            ps,
        ]
    )


def cluster_field(members: list[dict]) -> dict[str, list[dict]]:
    clusters: dict[str, list[dict]] = defaultdict(list)
    for m in members:
        clusters[structure_signature(m)].append(m)
    return dict(clusters)


def avg_field(members: list[dict], key: str, numeric: bool = True):
    vals = [pf(m.get(key)) for m in members]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None if numeric else ""
    return round(statistics.mean(vals), 1) if numeric else Counter(m.get(key) for m in members).most_common(1)[0][0]


def persist_avg(members: list[dict], hz: str = "6h"):
    key = f"persist_{hz}_pct"
    vals = [pf(m.get(key)) for m in members if m.get(key) not in ("Unknown", "", None)]
    return round(statistics.mean(vals), 1) if vals else "Unknown"


def field_coherence(members: list[dict]) -> tuple[str, float, list[str]]:
    """Supportive vs conflicting candidate environment."""
    notes = []
    situations = Counter(m.get("situation") for m in members)
    early = sum(situations.get(s, 0) for s in EARLY_SITUATIONS)
    late = sum(situations.get(s, 0) for s in LATE_SITUATIONS)
    n = len(members)

    real_scores = [pf(m.get("real_trend_score"), 50) for m in members]
    fake_scores = [pf(m.get("fake_trend_score"), 50) for m in members]
    real_spread = max(real_scores) - min(real_scores) if real_scores else 0
    fake_spread = max(fake_scores) - min(fake_scores) if fake_scores else 0

    if early >= 2 and late >= 2:
        notes.append("early_late_split")
    if real_spread >= 25:
        notes.append("real_score_divergence")
    if fake_spread >= 30:
        notes.append("fake_score_divergence")

    supplies = {m.get("supply_context") for m in members}
    if "COLLAPSE" in supplies and early >= 1:
        notes.append("collapse_among_growth")

    if not notes and early >= n * 0.4:
        return "supportive", clamp(70 + early / n * 20), notes
    if len(notes) >= 2:
        return "conflicting", clamp(30 - len(notes) * 8), notes
    if notes:
        return "mixed", 45.0, notes
    return "neutral", 50.0, notes


def classify_seedbed(cluster: list[dict], cluster_id: str) -> dict:
    n = len(cluster)
    if n < MIN_CLUSTER:
        return {
            "cluster_id": cluster_id,
            "seedbed_quality": "Unknown",
            "sample_size": n,
            "reason": "insufficient_cluster_size",
        }

    early_n = sum(1 for m in cluster if m.get("situation") in EARLY_SITUATIONS)
    late_n = sum(1 for m in cluster if m.get("situation") in LATE_SITUATIONS)
    collapse_avg = avg_field(cluster, "collapse_risk_pct") or 0
    fake_avg = avg_field(cluster, "fake_trend_score") or 50
    real_avg = avg_field(cluster, "real_trend_score") or 50
    collapse_supply = sum(1 for m in cluster if m.get("supply_context") == "COLLAPSE")

    if collapse_avg >= 55 or collapse_supply >= 1:
        quality = "Dangerous"
        reason = f"collapse_risk={collapse_avg}% supply_collapse={collapse_supply}"
    elif late_n >= n * 0.6:
        quality = "Exhausted"
        reason = f"late_distribution_dominant n={late_n}/{n}"
    elif early_n >= n * 0.5 and fake_avg < 60 and collapse_avg < 25:
        quality = "Very fertile" if real_avg >= 62 and fake_avg < 50 else "Fertile"
        reason = f"early_cluster={early_n}/{n} real={real_avg} fake={fake_avg}"
    elif early_n >= 1 and fake_avg < 75:
        quality = "Fertile"
        reason = f"partial_early={early_n}/{n}"
    elif n >= MIN_CLUSTER and fake_avg < 80:
        quality = "Neutral"
        reason = "mixed_maturity_no_dominant_signal"
    else:
        quality = "Unknown"
        reason = "ambiguous_ecology"

    return {
        "cluster_id": cluster_id,
        "seedbed_quality": quality,
        "sample_size": n,
        "symbols": "|".join(m["symbol"] for m in cluster),
        "dominant_situation": Counter(m.get("situation") for m in cluster).most_common(1)[0][0],
        "field_health_avg": avg_field(cluster, "health_score"),
        "field_pressure_avg": avg_field(cluster, "pressure_score"),
        "field_persistence_6h": persist_avg(cluster, "6h"),
        "real_avg": real_avg,
        "fake_avg": fake_avg,
        "collapse_avg": collapse_avg,
        "reason": reason,
    }


def build_evaluations(records: list[dict], use_cache: bool) -> list[dict]:
    if use_cache:
        cached = load_p15_evaluations()
        if len(cached) >= len(records) * 0.9:
            key_index = {(r["scan_time"], r["symbol"]): r for r in cached}
            merged = []
            for rec in records:
                key = (rec["scan_time"], rec["symbol"])
                if key in key_index:
                    merged.append(key_index[key])
            if len(merged) >= len(records) * 0.85:
                return merged

    interactions = load_active_interactions()
    persist_tables = build_persistence_tables(records)
    health_idx = load_engine_index(LOGS_DIR / "season2_p11_health_engine_output.csv")
    pressure_idx = load_engine_index(LOGS_DIR / "season2_p12_pressure_engine_output.csv")
    regime_idx = load_engine_index(LOGS_DIR / "season2_p14_regime_engine_output.csv")
    role_idx = load_engine_index(LOGS_DIR / "season2_p13_role_engine_output.csv")
    evolution_idx = load_engine_index(LOGS_DIR / "season2_p10_evolution_engine_output.csv")

    evaluations = [
        evaluate_record(r, persist_tables, interactions, health_idx, pressure_idx, regime_idx, role_idx, evolution_idx)
        for r in records
    ]
    return rank_within_scans(evaluations)


def attach_forward(records: list[dict], evaluations: list[dict]) -> None:
    fwd = {(r["scan_time"], r["symbol"]): r.get("target_f6") or r.get("forward_6h") for r in records}
    for ev in evaluations:
        ev["forward_6h"] = fwd.get((ev["scan_time"], ev["symbol"]))


def analyze_fields(
    evaluations: list[dict],
) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict], list[dict]]:
    by_scan: dict[str, list[dict]] = defaultdict(list)
    for ev in evaluations:
        by_scan[ev["scan_time"]].append(ev)

    field_rows = []
    seedbed_rows = []
    cluster_rows = []
    fake_rows = []
    collapse_rows = []

    for scan_time, members in sorted(by_scan.items()):
        if len(members) < MIN_FIELD:
            continue

        field_id = f"field_{scan_time.replace(' ', '_').replace(':', '')}"
        date = members[0]["date"]
        regime = Counter(m.get("regime_context") for m in members).most_common(1)[0][0]

        f6_vals = [pf(m.get("forward_6h")) for m in members if pf(m.get("forward_6h")) is not None]
        med_f6 = round(statistics.median(f6_vals), 2) if f6_vals else None
        expansion = med_f6 is not None and med_f6 >= EXPANSION_F6_THRESHOLD

        early_density = sum(1 for m in members if m.get("situation") in EARLY_SITUATIONS) / len(members)
        fake_avg = avg_field(members, "fake_trend_score") or 50
        real_avg = avg_field(members, "real_trend_score") or 50
        coherence, coherence_score, coherence_notes = field_coherence(members)

        clusters = cluster_field(members)
        fertile_clusters = 0
        dangerous_clusters = 0

        for cid, (sig, cluster) in enumerate(sorted(clusters.items(), key=lambda x: -len(x[1]))):
            cluster_id = f"{field_id}_c{cid + 1}"
            sb = classify_seedbed(cluster, cluster_id)
            seedbed_rows.append(
                {
                    "field_id": field_id,
                    "scan_time": scan_time,
                    "date": date,
                    **sb,
                    "structure_signature": sig,
                }
            )
            if sb["seedbed_quality"] in ("Very fertile", "Fertile"):
                fertile_clusters += 1
            if sb["seedbed_quality"] == "Dangerous":
                dangerous_clusters += 1

            for m in cluster:
                cluster_rows.append(
                    {
                        "field_id": field_id,
                        "cluster_id": cluster_id,
                        "scan_time": scan_time,
                        "symbol": m["symbol"],
                        "structure_signature": sig,
                        "situation": m.get("situation"),
                        "seedbed_quality": sb["seedbed_quality"],
                        "relative_rank": m.get("relative_rank", ""),
                        "relative_strength": m.get("relative_strength"),
                    }
                )

        # Early trend cluster detection (task 1)
        early_cluster = [m for m in members if m.get("situation") in EARLY_SITUATIONS]
        early_clustered = len(early_cluster) >= MIN_CLUSTER

        # Fake trend cluster (task 4)
        fake_cluster = [m for m in members if pf(m.get("fake_trend_score"), 0) >= 70]
        if len(fake_cluster) >= MIN_CLUSTER:
            fake_rows.append(
                {
                    "field_id": field_id,
                    "scan_time": scan_time,
                    "date": date,
                    "fake_cluster_size": len(fake_cluster),
                    "symbols": "|".join(m["symbol"] for m in fake_cluster),
                    "avg_fake_score": round(statistics.mean(pf(m["fake_trend_score"], 0) for m in fake_cluster), 1),
                    "dominant_situation": Counter(m.get("situation") for m in fake_cluster).most_common(1)[0][0],
                    "coherence": "fake_cluster_detected",
                }
            )

        # Collapse propagation (task 5)
        high_coll = [m for m in members if pf(m.get("collapse_risk_pct"), 0) >= 35]
        if len(high_coll) >= MIN_CLUSTER:
            sigs = Counter(structure_signature(m) for m in high_coll)
            collapse_rows.append(
                {
                    "field_id": field_id,
                    "scan_time": scan_time,
                    "high_risk_count": len(high_coll),
                    "shared_structures": "|".join(f"{k}:{v}" for k, v in sigs.most_common(3)),
                    "collapse_supply_count": sum(1 for m in high_coll if m.get("supply_context") == "COLLAPSE"),
                    "propagation": "localized" if len(sigs) <= 2 else "diffuse",
                }
            )

        watchlist_conf = "Unknown"
        if fertile_clusters >= 1 and coherence == "supportive" and fake_avg < 65:
            watchlist_conf = "high" if early_density >= 0.35 else "medium"
        elif dangerous_clusters >= 1 or fake_avg >= 75:
            watchlist_conf = "low"
        elif len(members) < MIN_FIELD + 2:
            watchlist_conf = "Unknown"

        field_rows.append(
            {
                "field_id": field_id,
                "date": date,
                "scan_time": scan_time,
                "opportunity_field": field_id,
                "candidate_density": len(members),
                "early_trend_density_pct": round(early_density * 100, 1),
                "early_trend_clustered": "yes" if early_clustered else "no",
                "cluster_count": len(clusters),
                "fertile_seedbed_count": fertile_clusters,
                "field_health": avg_field(members, "health_score"),
                "field_pressure": avg_field(members, "pressure_score"),
                "field_persistence_6h": persist_avg(members, "6h"),
                "field_uncertainty": avg_field(members, "uncertainty_pct"),
                "field_maturity": Counter(m.get("trend_maturity") for m in members).most_common(1)[0][0],
                "dominant_situation": Counter(m.get("situation") for m in members).most_common(1)[0][0],
                "real_environment_score": real_avg,
                "fake_environment_score": fake_avg,
                "environment_verdict": "genuine_bias" if real_avg > fake_avg + 5 else "fake_bias" if fake_avg > real_avg + 10 else "mixed",
                "coherence": coherence,
                "coherence_score": coherence_score,
                "coherence_notes": "|".join(coherence_notes) if coherence_notes else "",
                "regime_context": regime,
                "median_forward_6h": med_f6 if med_f6 is not None else "Unknown",
                "expansion_field": "yes" if expansion else "no" if med_f6 is not None else "Unknown",
                "watchlist_confidence": watchlist_conf,
            }
        )

    return field_rows, seedbed_rows, cluster_rows, fake_rows, collapse_rows, by_scan


def pre_expansion_analysis(field_rows: list[dict]) -> list[dict]:
    """Task 3: fertile fields before large expansions."""
    expansion_fields = [f for f in field_rows if f.get("expansion_field") == "yes"]
    non_expansion = [f for f in field_rows if f.get("expansion_field") == "no"]

    def avg_key(rows, key):
        vals = [pf(r.get(key)) for r in rows if pf(r.get(key)) is not None]
        return round(statistics.mean(vals), 1) if vals else None

    rows = [
        {
            "signal": "early_trend_density_before_expansion",
            "expansion_fields_n": len(expansion_fields),
            "non_expansion_n": len(non_expansion),
            "expansion_avg_early_density": avg_key(expansion_fields, "early_trend_density_pct"),
            "non_expansion_avg_early_density": avg_key(non_expansion, "early_trend_density_pct"),
            "expansion_avg_fertile_seedbeds": avg_key(expansion_fields, "fertile_seedbed_count"),
            "expansion_avg_fake_env": avg_key(expansion_fields, "fake_environment_score"),
            "verdict": "Unknown" if len(expansion_fields) < 3 else "conditional",
            "note": "fertile seedbeds may precede expansion when early_density elevated",
        }
    ]

    for f in expansion_fields:
        if pf(f.get("early_trend_density_pct"), 0) >= 25 or f.get("fertile_seedbed_count", 0) >= 1:
            rows.append(
                {
                    "signal": "field_pre_expansion_match",
                    "field_id": f["field_id"],
                    "scan_time": f["scan_time"],
                    "early_trend_density_pct": f["early_trend_density_pct"],
                    "fertile_seedbed_count": f["fertile_seedbed_count"],
                    "median_forward_6h": f["median_forward_6h"],
                    "verdict": "pre_expansion_pattern",
                }
            )
    return rows


def rank_fields(field_rows: list[dict]) -> list[dict]:
    """Relative field ranking — not absolute."""
    if not field_rows:
        return []

    scored = []
    for f in field_rows:
        score = 50.0
        score += pf(f.get("early_trend_density_pct"), 0) * 0.25
        score += (pf(f.get("real_environment_score"), 50) - pf(f.get("fake_environment_score"), 50)) * 0.3
        score -= pf(f.get("field_uncertainty"), 30) * 0.2
        score += pf(f.get("coherence_score"), 50) * 0.15
        score += pf(f.get("fertile_seedbed_count"), 0) * 5
        if f.get("environment_verdict") == "fake_bias":
            score -= 15
        if f.get("watchlist_confidence") == "Unknown":
            score -= 10
        scored.append({**f, "field_relative_score": round(clamp(score), 1)})

    scored.sort(key=lambda x: -x["field_relative_score"])
    n = len(scored)
    for i, row in enumerate(scored, 1):
        row["field_relative_rank"] = i
        row["field_rank_percentile"] = round((n - i) / max(n - 1, 1) * 100, 1) if n > 1 else 100.0
    return scored


def build_watchlist(ranked_fields: list[dict], seedbed_rows: list[dict]) -> list[dict]:
    field_index = {f["field_id"]: f for f in ranked_fields}
    rows = []

    for sb in seedbed_rows:
        if sb["seedbed_quality"] not in ("Very fertile", "Fertile"):
            continue
        if sb.get("sample_size", 0) < MIN_CLUSTER:
            continue
        f = field_index.get(sb["field_id"], {})
        collapse_avg = pf(sb.get("collapse_avg"), 99)
        if collapse_avg is not None and collapse_avg >= 40:
            continue

        if sb["seedbed_quality"] == "Very fertile" and f.get("coherence") == "supportive":
            conf = "high"
        elif sb["seedbed_quality"] in ("Very fertile", "Fertile") and pf(f.get("field_relative_score"), 0) >= 45:
            conf = "medium"
        elif sb["seedbed_quality"] == "Fertile":
            conf = "hypothesis"
        else:
            conf = "Unknown"

        rows.append(
            {
                "field_id": sb["field_id"],
                "scan_time": sb["scan_time"],
                "date": sb.get("date", f.get("date", "")),
                "cluster_id": sb["cluster_id"],
                "seedbed_quality": sb["seedbed_quality"],
                "symbols": sb.get("symbols", ""),
                "dominant_situation": sb.get("dominant_situation", ""),
                "field_relative_rank": f.get("field_relative_rank", ""),
                "field_relative_score": f.get("field_relative_score", ""),
                "watchlist_confidence": conf,
                "field_uncertainty": f.get("field_uncertainty", ""),
                "environment_verdict": f.get("environment_verdict", ""),
                "coherence": f.get("coherence", ""),
                "estimated_persistence_6h": sb.get("field_persistence_6h", "Unknown"),
                "reason": sb.get("reason", ""),
            }
        )

    return sorted(rows, key=lambda x: (-pf(x.get("field_relative_score"), 0), x.get("watchlist_confidence") != "high"))[:40]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-p15", action="store_true", help="Force recompute P15 evaluations")
    args = parser.parse_args()

    records = build_expanded_records()
    enrich_physics(records)
    enrich_panel_fields(records)
    attach_forward_targets(records)
    prepare_records(records)
    enrich_record_stack(records)
    assign_regimes(records)

    evaluations = build_evaluations(records, use_cache=not args.refresh_p15)
    attach_forward(records, evaluations)

    field_rows, seedbed_rows, cluster_rows, fake_rows, collapse_rows, _ = analyze_fields(evaluations)
    expansion_rows = pre_expansion_analysis(field_rows)
    ranked_fields = rank_fields(field_rows)
    watchlist = build_watchlist(ranked_fields, seedbed_rows)

    write_csv(FIELDS_CSV, ranked_fields)
    write_csv(SEEDBED_CSV, seedbed_rows)
    write_csv(CLUSTERS_CSV, cluster_rows)
    write_csv(EXPANSION_CSV, expansion_rows)
    write_csv(FAKE_CLUSTERS_CSV, fake_rows)
    write_csv(COLLAPSE_PROP_CSV, collapse_rows)
    write_csv(FIELD_RANKING_CSV, ranked_fields)
    write_csv(WATCHLIST_CSV, watchlist)

    seed_dist = Counter(s["seedbed_quality"] for s in seedbed_rows)
    lines = [
        "===== SCOUT SEASON2 P16 - OPPORTUNITY FIELD & SEEDBED =====",
        "",
        f"Evaluations: {len(evaluations)} | Opportunity fields: {len(field_rows)}",
        f"Seedbed clusters: {len(seedbed_rows)} | Watchlist entries: {len(watchlist)}",
        "",
        "--- Seedbed quality distribution ---",
    ]
    for q, n in seed_dist.most_common():
        lines.append(f"  {q}: {n}")

    lines.extend(["", "--- Top opportunity fields (relative rank) ---"])
    for row in ranked_fields[:6]:
        lines.append(
            f"  #{row.get('field_relative_rank')} {row['scan_time']}: "
            f"early_density={row['early_trend_density_pct']}% "
            f"env={row['environment_verdict']} watch={row['watchlist_confidence']}"
        )

    lines.extend(["", "--- Pre-expansion signals ---"])
    if expansion_rows:
        sig = expansion_rows[0]
        lines.append(
            f"  Expansion fields n={sig.get('expansion_fields_n')} "
            f"avg_early_density={sig.get('expansion_avg_early_density')}%"
        )

    lines.extend(["", "--- Fake trend clusters ---", f"  Fields with fake clusters: {len(fake_rows)}"])
    lines.extend(["", "--- Collapse propagation ---", f"  Fields with collapse propagation: {len(collapse_rows)}"])
    lines.extend([
        "",
        "Principles: environment not individual | no price prediction | Unknown valid",
    ])
    lines.extend(mission_summary_lines())
    lines.extend(["", f"Watchlist: {WATCHLIST_CSV}", "=" * 58])
    REPORT_TXT.write_text("\n".join(lines), encoding="utf-8")

    print("===== P16 OPPORTUNITY FIELD & SEEDBED =====")
    print(f"Fields: {len(field_rows)} | Seedbeds: {len(seedbed_rows)} | Watchlist: {len(watchlist)}")
    print(f"Report: {REPORT_TXT}")


if __name__ == "__main__":
    main()
