"""
Scout Learning Season2 - P15 Situation Evaluation Output & Practical Scout Layer

Converts P6-P14 core research into operational situation evaluation.
Does NOT predict price. Research separated from operation.

Governed by Scout Research Constitution and Scout Mission convergence gates.
"""

import argparse
import csv
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from season2_p5_historical_expansion import STEP1_DATES, build_unified_records, collect_missing
from season2_p6_market_memory import attach_forward_targets
from season2_p7_behaviour_grammar import enrich_physics
from season2_p8_participant_state import enrich_panel_fields
from season2_p9_conditional_interaction_mining import prepare_records
from season2_p10_situation_evolution import build_edges, label_records, load_interactions
from season2_p11_situation_health import (
    annotate_record_momentum,
    cohort_metrics,
    record_health_class,
    task1_health_scores,
)
from season2_p12_situation_pressure import (
    load_csv_index,
    load_p10_situation_transitions,
    record_pressure,
    stress_level,
)
from season2_p14_regime_memory_bank import build_expanded_records
from season2_regime_core import assign_regimes
from season2_scout_mission import PERSISTENCE_HORIZONS, mission_summary_lines

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

OPERATIONAL_CSV = LOGS_DIR / "season2_p15_operational_scores.csv"
CONFIDENCE_CSV = LOGS_DIR / "season2_p15_confidence.csv"
RANKING_CSV = LOGS_DIR / "season2_p15_relative_ranking.csv"
ACTION_CSV = LOGS_DIR / "season2_p15_action_engine.csv"
UNCERTAINTY_CSV = LOGS_DIR / "season2_p15_uncertainty.csv"
LATEST_CSV = LOGS_DIR / "season2_p15_latest_scans.csv"
REPORT_TXT = LOGS_DIR / "season2_p15_research_report.txt"

CONVERGENCE_CSV = LOGS_DIR / "season2_p14_convergence_tiers.csv"
P9_INTERACTIONS = LOGS_DIR / "season2_p9_interactions.csv"

OPERATIONAL_SITUATIONS = (
    "Unknown",
    "Accumulation",
    "Early Trend",
    "Healthy Trend",
    "Late Trend",
    "Distribution",
    "Recovery",
    "Transition",
)

# Core layer max influence (background cannot dominate)
CORE_WEIGHT = {
    "health": 0.22,
    "pressure": 0.18,
    "participant": 0.15,
    "supply": 0.15,
    "evolution": 0.12,
    "interaction": 0.10,
    "memory": 0.08,
}
CONTEXT_MAX = 0.10  # regime + role combined cap


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


def load_engine_index(path: Path, key_cols: tuple[str, ...] = ("symbol", "scan_time")) -> dict[tuple, dict]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return {tuple(row.get(c, "") for c in key_cols): row for row in csv.DictReader(f)}


def load_core_patterns() -> set[str]:
    if not CONVERGENCE_CSV.exists():
        return set()
    core = set()
    with CONVERGENCE_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("tier") == "core" and row.get("operational") == "True":
                core.add(row.get("finding", ""))
    return core


def load_active_interactions() -> list[dict]:
    if not P9_INTERACTIONS.exists():
        return []
    with P9_INTERACTIONS.open(encoding="utf-8") as f:
        return [
            row for row in csv.DictReader(f)
            if row.get("status") in ("ACTIVE", "CONDITIONAL") and row.get("confidence") in ("high", "medium")
        ]


def map_operational_situation(record: dict) -> str:
    sit = record.get("situation_archetype", "")
    ps = record.get("participant_state") or ""
    supply = record.get("supply_label") or ""

    if supply == "COLLAPSE" or sit == "forced_liquidation_zone":
        return "Recovery"
    if sit == "early_accumulation" or "compression" in ps or "accumulation" in ps:
        return "Accumulation"
    if sit == "late_chasing_zone" or "late_chasing" in ps or "late_markup" in ps:
        return "Late Trend"
    if sit == "distribution" or "exhaustion" in ps:
        return "Distribution"
    if sit == "healthy_trend":
        if "warm" in ps or "compression" in ps:
            return "Early Trend"
        return "Healthy Trend"
    if sit in ("rotation_chop", "unclassified_situation") or "transition" in ps or "rotation" in ps:
        return "Transition"
    if not sit or sit == "unknown":
        return "Unknown"
    return "Transition"


def trend_maturity(record: dict, op_sit: str) -> str:
    mapping = {
        "Accumulation": "birth",
        "Early Trend": "early_growth",
        "Healthy Trend": "growth",
        "Late Trend": "late",
        "Distribution": "exhaustion",
        "Recovery": "repair",
        "Transition": "ambiguous",
        "Unknown": "unknown",
    }
    return mapping.get(op_sit, "unknown")


def build_persistence_tables(records: list[dict]) -> dict[tuple[str, str], dict]:
    """Empirical persistence by operational_situation x supply."""
    groups: dict[tuple[str, str], list] = defaultdict(list)
    for r in records:
        key = (map_operational_situation(r), r.get("supply_label", "unknown"))
        groups[key].append(r)

    tables = {}
    for key, grp in groups.items():
        if len(grp) < 4:
            continue
        row = {"sample_size": len(grp)}
        for hz, field in [
            ("1h", None),
            ("2h", "forward_2h"),
            ("6h", "forward_6h"),
            ("12h", "forward_12h"),
            ("24h", "forward_12h"),
        ]:
            if hz == "1h":
                vals = [r["forward_2h"] / 2 for r in grp if r.get("forward_2h") is not None]
            elif hz == "24h":
                vals = [r.get("forward_12h") for r in grp if r.get("forward_12h") is not None]
            else:
                vals = [r.get(field) for r in grp if r.get(field) is not None]
            if vals:
                row[f"persist_{hz}_pct"] = round(sum(1 for v in vals if v >= 0) / len(vals) * 100, 1)
            else:
                row[f"persist_{hz}_pct"] = None
        tables[key] = row
    return tables


def estimate_persistence(record: dict, tables: dict[tuple, dict]) -> dict[str, float | str]:
    key = (map_operational_situation(record), record.get("supply_label", "unknown"))
    tbl = tables.get(key)
    if not tbl or tbl["sample_size"] < 6:
        return {hz: "Unknown" for hz in PERSISTENCE_HORIZONS}

    out = {}
    for hz in PERSISTENCE_HORIZONS:
        val = tbl.get(f"persist_{hz}_pct")
        out[hz] = val if val is not None else "Unknown"
    return out


def real_fake_scores(record: dict, health_row: dict | None, pressure_row: dict | None) -> tuple[float, float, list[str]]:
    """Core layers only: health, pressure, participant, supply, evolution."""
    evidence: list[str] = []
    real = 50.0
    fake = 50.0

    hclass = (health_row or {}).get("health_class") or record.get("health_class", "")
    hscore = pf((health_row or {}).get("health_score") or record.get("health_score"), 50)
    ps = record.get("participant_state") or ""
    supply = record.get("supply_label") or ""
    sit = record.get("situation_archetype", "")
    pressure = pf((pressure_row or {}).get("pressure_score") or record.get("pressure_score"), 40)
    evo_conflict = pf(record.get("evolution_conflict"), 0)

    if hclass in ("IMPROVING", "STABLE", "RECOVERING"):
        real += 15
        evidence.append(f"health_class={hclass}")
    if hclass == "FRAGILE":
        fake += 20
        evidence.append("health=FRAGILE")
    if sit == "healthy_trend" and hclass == "FRAGILE":
        fake += 15
        evidence.append("label_health_mismatch")

    if supply == "MID_SUPPLY":
        real += 12
        evidence.append("supply=MID_SUPPLY")
    elif supply == "HIGH_SUPPLY":
        real += 8
        evidence.append("supply=HIGH_SUPPLY")
    elif supply == "COLLAPSE":
        fake += 30
        evidence.append("supply=COLLAPSE")

    if "late_chasing" in ps or "exhaustion" in ps:
        fake += 12
        evidence.append(f"participant={ps}")
    if "warm" in ps or "accumulation" in ps:
        real += 10
        evidence.append(f"participant={ps}")

    if pressure >= 50:
        fake += min(20, (pressure - 45) * 0.8)
        evidence.append(f"pressure={pressure:.0f}")
    elif pressure < 30:
        real += 8

    if evo_conflict >= 35:
        fake += 12
        evidence.append("evolution_conflict")

    gram = record.get("grammar") or ""
    if "vol_increase" in gram and "exhaustion" not in gram:
        real += 8
    if "vol_exhaustion" in gram or "upper_wick" in gram:
        fake += 8

    return clamp(real), clamp(fake), evidence[:6]


def collapse_risk(record: dict, health_row: dict | None, pressure_row: dict | None, interact: dict | None) -> tuple[float, list[str]]:
    evidence = []
    risk = 10.0

    if record.get("collapse_label") == "YES":
        risk = 85.0
        evidence.append("historical_collapse_label")

    coll_h = pf((health_row or {}).get("collapse_probability_pct"), None)
    if coll_h is not None:
        risk = max(risk, coll_h * 0.6)
        evidence.append(f"health_collapse_p={coll_h}")

    coll_p = pf((pressure_row or {}).get("collapse_probability_pct"), None)
    if coll_p is not None:
        risk = max(risk, coll_p * 0.5)
        evidence.append(f"pressure_collapse_p={coll_p}")

    if interact:
        cp = pf(interact.get("collapse_probability_pct"), 0)
        if cp:
            risk = (risk + cp) / 2
            evidence.append(f"interaction_collapse_p={cp}")

    if record.get("supply_label") == "COLLAPSE":
        risk = max(risk, 75)
        evidence.append("COLLAPSE_supply")

    op_sit = map_operational_situation(record)
    if op_sit in ("Late Trend", "Distribution"):
        risk += 8
        evidence.append(f"situation={op_sit}")

    return clamp(risk), evidence[:5]


def match_interaction(record: dict, interactions: list[dict]) -> dict | None:
    supply_map = {"MID_SUPPLY": "supply_mid_supply", "HIGH_SUPPLY": "supply_high", "LOW_SUPPLY": "supply_low", "COLLAPSE": "supply_collapse"}
    collapse_type = supply_map.get(record.get("supply_label", ""), "")
    ps = record.get("participant_state", "")
    sit = record.get("situation_archetype", "")

    best = None
    best_score = -1.0
    for row in interactions[:120]:
        key = row.get("situation_key", "")
        if collapse_type and collapse_type in key and sit in (row.get("situation_description", ""), "healthy_trend", "early_accumulation"):
            score = pf(row.get("information_score"), 0)
            if score > best_score:
                best_score = score
                best = row
        elif ps and ps in key:
            score = pf(row.get("information_score"), 0)
            if score > best_score:
                best_score = score
                best = row
    return best


def relative_strength_score(
    real: float,
    fake: float,
    persist_6h: float | str,
    collapse: float,
    interact: dict | None,
) -> float:
    base = real - fake * 0.5 + 50
    if isinstance(persist_6h, (int, float)):
        base += (persist_6h - 50) * 0.3
    base -= collapse * 0.4
    if interact:
        base += pf(interact.get("persistence_pct"), 50) * 0.1
        base += pf(interact.get("mae_improvement"), 0) * 0.5
    return clamp(base)


def compute_uncertainty(
    record: dict,
    regime_row: dict | None,
    role_row: dict | None,
    persist: dict,
    evidence_count: int,
) -> tuple[float, list[str]]:
    reasons = []
    u = 20.0

    if map_operational_situation(record) == "Unknown":
        u += 25
        reasons.append("situation_unknown")
    if record.get("ecology_regime") == "Unknown":
        u += 15
        reasons.append("regime_unknown")
    if regime_row and pf(regime_row.get("unknown_probability_pct"), 0) >= 40:
        u += 10
        reasons.append("regime_low_confidence")
    if role_row and pf(role_row.get("unknown_probability_pct"), 0) >= 45:
        u += 8
        reasons.append("role_ambiguous")  # background context only

    unknown_persist = sum(1 for v in persist.values() if v == "Unknown")
    u += unknown_persist * 8
    if unknown_persist >= 3:
        reasons.append("persistence_unknown")

    if evidence_count < 3:
        u += 15
        reasons.append("thin_core_evidence")

    return clamp(u), reasons[:5]


def confidence_tier(uncertainty: float, evidence_count: int, sample_backing: bool) -> str:
    if uncertainty >= 55 or evidence_count < 2:
        return "unknown"
    if uncertainty >= 35 or not sample_backing:
        return "hypothesis"
    if uncertainty >= 20:
        return "medium"
    return "high"


def recommend_action(
    uncertainty: float,
    real: float,
    fake: float,
    collapse: float,
    op_sit: str,
    maturity: str,
    supply: str,
    regime: str,
) -> tuple[str, str]:
    if uncertainty >= 60:
        return "Unknown", "uncertainty too high — insufficient converged evidence"
    if collapse >= 70 or supply == "COLLAPSE":
        return "Avoid", f"collapse_risk={collapse:.0f}% supply={supply}"
    if fake >= 65 and real < 45:
        return "Reduce", f"fake_trend={fake:.0f} real={real:.0f}"
    if op_sit == "Late Trend" or maturity == "exhaustion":
        return "Reduce", f"maturity={maturity} situation={op_sit}"
    if op_sit == "Distribution":
        return "Watch", "distribution phase — monitor release"
    if real >= 62 and fake < 40 and collapse < 25 and op_sit in ("Early Trend", "Healthy Trend", "Accumulation"):
        if real >= 72 and uncertainty < 30:
            return "Strong Buy", f"real={real:.0f} persist_context favorable"
        return "Buy", f"real={real:.0f} early/healthy trend signals"
    if op_sit == "Recovery" and collapse < 40 and uncertainty < 45:
        return "Watch", "recovery ecology — confirm before engagement"
    if op_sit == "Transition" or regime == "Unknown":
        return "Hold", "transition/unknown — no forced classification"
    if uncertainty >= 40:
        return "Watch", f"uncertainty={uncertainty:.0f}%"
    return "Watch", f"situation={op_sit} real={real:.0f} fake={fake:.0f}"


def enrich_record_stack(records: list[dict]) -> None:
    interactions = load_interactions()
    label_records(records, interactions)
    edges = build_edges(records)
    annotate_record_momentum(records, edges)
    sit_trans = load_p10_situation_transitions()
    health_scores = task1_health_scores(records, edges, sit_trans)
    health_index = {r["situation_archetype"]: r for r in health_scores}
    grammar_vitality = load_csv_index(LOGS_DIR / "season2_p11_grammar_vitality.csv", "grammar")

    for r in records:
        r["health_class"] = record_health_class(r, health_index)
        h = health_index.get(r.get("situation_archetype", ""), {})
        r["health_score"] = h.get("health_score", "")
        r.update(record_pressure(r, health_index, grammar_vitality, sit_trans))


def evaluate_record(
    record: dict,
    persist_tables: dict,
    interactions: list[dict],
    health_idx: dict,
    pressure_idx: dict,
    regime_idx: dict,
    role_idx: dict,
    evolution_idx: dict,
) -> dict:
    key = (record.get("symbol", ""), record.get("scan_time", ""))
    health_row = health_idx.get(key, {})
    pressure_row = pressure_idx.get(key, {})
    regime_row = regime_idx.get(key, {})
    role_row = role_idx.get(key, {})  # background context
    evo_row = evolution_idx.get(key, {})

    op_sit = map_operational_situation(record)
    maturity = trend_maturity(record, op_sit)
    persist = estimate_persistence(record, persist_tables)
    interact = match_interaction(record, interactions)

    real, fake, real_ev = real_fake_scores(record, health_row, pressure_row)
    collapse, collapse_ev = collapse_risk(record, health_row, pressure_row, interact)
    persist_6h = persist.get("6h", "Unknown")
    strength = relative_strength_score(real, fake, persist_6h, collapse, interact)

    core_evidence = real_ev + collapse_ev
    if interact:
        core_evidence.append(f"interaction_persist={interact.get('persistence_pct')}%")
    if health_row.get("health_score"):
        core_evidence.append(f"health={health_row['health_score']}")

    uncertainty, unc_reasons = compute_uncertainty(
        record, regime_row, role_row, persist,
        evidence_count=len(core_evidence),
    )
    conf = confidence_tier(uncertainty, len(core_evidence), isinstance(persist_6h, (int, float)))

    regime_ctx = record.get("ecology_regime", "Unknown")
    if regime_row:
        regime_ctx = regime_row.get("ecology_regime", regime_ctx)
    role_ctx = role_row.get("primary_function", "") if role_row else ""
    if role_row and pf(role_row.get("unknown_probability_pct"), 0) >= 50:
        role_ctx = f"{role_ctx}(ambiguous)" if role_ctx else ""

    action, reason = recommend_action(
        uncertainty, real, fake, collapse, op_sit, maturity,
        record.get("supply_label", ""), regime_ctx,
    )

    return {
        "date": record["date"],
        "symbol": record["symbol"],
        "scan_time": record["scan_time"],
        "situation": op_sit,
        "research_situation": record.get("situation_archetype", ""),
        "trend_maturity": maturity,
        "persist_1h_pct": persist.get("1h", "Unknown"),
        "persist_2h_pct": persist.get("2h", "Unknown"),
        "persist_6h_pct": persist.get("6h", "Unknown"),
        "persist_12h_pct": persist.get("12h", "Unknown"),
        "persist_24h_pct": persist.get("24h", "Unknown"),
        "real_trend_score": round(real, 1),
        "fake_trend_score": round(fake, 1),
        "collapse_risk_pct": round(collapse, 1),
        "health_score": health_row.get("health_score") or record.get("health_score", ""),
        "health_class": health_row.get("health_class") or record.get("health_class", ""),
        "pressure_score": pressure_row.get("pressure_score") or record.get("pressure_score", ""),
        "pressure_band": pressure_row.get("pressure_band") or record.get("pressure_band", ""),
        "supply_context": record.get("supply_label", "unknown"),
        "regime_context": regime_ctx,
        "role_context": role_ctx or "n/a",
        "participant_state": record.get("participant_state", ""),
        "grammar": (record.get("grammar") or "")[:40],
        "relative_strength": round(strength, 1),
        "confidence": conf,
        "uncertainty_pct": round(uncertainty, 1),
        "recommended_action": action,
        "reason": reason,
        "empirical_evidence": "|".join(core_evidence[:8]),
        "uncertainty_reasons": "|".join(unc_reasons),
        "evolution_next": evo_row.get("most_likely_next_situation", ""),
    }


def rank_within_scans(evaluations: list[dict]) -> list[dict]:
    by_scan: dict[str, list] = defaultdict(list)
    for row in evaluations:
        by_scan[row["scan_time"]].append(row)

    ranked = []
    for scan_time, group in sorted(by_scan.items()):
        ordered = sorted(group, key=lambda x: -x["relative_strength"])
        n = len(ordered)
        for i, row in enumerate(ordered, 1):
            ranked.append(
                {
                    **row,
                    "scan_time": scan_time,
                    "relative_rank": i,
                    "scan_candidate_count": n,
                    "rank_percentile": round((n - i) / max(n - 1, 1) * 100, 1) if n > 1 else 100.0,
                    "rank_tier": "top" if i <= max(1, n // 3) else "mid" if i <= max(2, 2 * n // 3) else "bottom",
                }
            )
    return ranked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect-limit", type=int, default=0)
    parser.add_argument("--skip-collect", action="store_true", default=True)
    parser.add_argument("--latest-only", action="store_true", help="Evaluate only most recent scan batch")
    args = parser.parse_args()

    if not args.skip_collect and args.collect_limit > 0:
        missing = [d for d in STEP1_DATES if not (LOGS_DIR / f"top10_gainer_learning_{d.replace('-', '')}.csv").exists()]
        collect_missing(missing, args.collect_limit)

    records = build_expanded_records()
    enrich_physics(records)
    enrich_panel_fields(records)
    attach_forward_targets(records)
    prepare_records(records)
    enrich_record_stack(records)
    assign_regimes(records)

    core_patterns = load_core_patterns()
    interactions = load_active_interactions()
    persist_tables = build_persistence_tables(records)

    health_idx = load_engine_index(LOGS_DIR / "season2_p11_health_engine_output.csv")
    pressure_idx = load_engine_index(LOGS_DIR / "season2_p12_pressure_engine_output.csv")
    regime_idx = load_engine_index(LOGS_DIR / "season2_p14_regime_engine_output.csv")
    role_idx = load_engine_index(LOGS_DIR / "season2_p13_role_engine_output.csv")
    evolution_idx = load_engine_index(LOGS_DIR / "season2_p10_evolution_engine_output.csv")

    target_records = records
    if args.latest_only:
        latest_scan = max(r["scan_time"] for r in records)
        target_records = [r for r in records if r["scan_time"] == latest_scan]

    evaluations = [
        evaluate_record(r, persist_tables, interactions, health_idx, pressure_idx, regime_idx, role_idx, evolution_idx)
        for r in target_records
    ]
    ranked = rank_within_scans(evaluations)

    latest_date = max(r["date"] for r in records)
    latest_scans = [r for r in ranked if r["date"] == latest_date]

    operational_rows = [{k: v for k, v in r.items() if k not in ("relative_rank", "scan_candidate_count", "rank_percentile", "rank_tier")} for r in evaluations]

    confidence_rows = [
        {
            "symbol": r["symbol"],
            "scan_time": r["scan_time"],
            "confidence": r["confidence"],
            "uncertainty_pct": r["uncertainty_pct"],
            "core_evidence_count": len((r.get("empirical_evidence") or "").split("|")) if r.get("empirical_evidence") else 0,
            "convergence_layers": "health|pressure|participant|supply|interaction|memory",
            "background_context": f"regime={r['regime_context']};role={r['role_context']}",
        }
        for r in evaluations
    ]

    uncertainty_rows = [
        {
            "symbol": r["symbol"],
            "scan_time": r["scan_time"],
            "uncertainty_pct": r["uncertainty_pct"],
            "uncertainty_reasons": r["uncertainty_reasons"],
            "situation": r["situation"],
            "recommended_action": r["recommended_action"],
            "verdict": "prefer_unknown" if r["uncertainty_pct"] >= 55 else "evaluated",
        }
        for r in evaluations
    ]

    action_rows = [
        {
            "date": r["date"],
            "symbol": r["symbol"],
            "scan_time": r["scan_time"],
            "situation": r["situation"],
            "recommended_action": r["recommended_action"],
            "reason": r["reason"],
            "empirical_evidence": r["empirical_evidence"],
            "real_trend_score": r["real_trend_score"],
            "fake_trend_score": r["fake_trend_score"],
            "collapse_risk_pct": r["collapse_risk_pct"],
            "confidence": r["confidence"],
        }
        for r in evaluations
    ]

    write_csv(OPERATIONAL_CSV, operational_rows)
    write_csv(CONFIDENCE_CSV, confidence_rows)
    write_csv(RANKING_CSV, ranked)
    write_csv(ACTION_CSV, action_rows)
    write_csv(UNCERTAINTY_CSV, uncertainty_rows)
    write_csv(LATEST_CSV, latest_scans)

    action_dist = Counter(r["recommended_action"] for r in evaluations)
    sit_dist = Counter(r["situation"] for r in evaluations)
    avg_unc = statistics.mean(r["uncertainty_pct"] for r in evaluations) if evaluations else 0

    lines = [
        "===== SCOUT SEASON2 P15 - SITUATION EVALUATION OUTPUT =====",
        "",
        f"Records evaluated: {len(evaluations)} | History pool: {len(records)}",
        f"Core convergence patterns loaded: {len(core_patterns)}",
        f"Active interactions (P9 core): {len(interactions)}",
        f"Latest date scans: {len(latest_scans)}",
        "",
        "--- Operational situation distribution ---",
    ]
    for sit, n in sit_dist.most_common():
        lines.append(f"  {sit}: {n}")

    lines.extend(["", "--- Recommended actions ---"])
    for act, n in action_dist.most_common():
        lines.append(f"  {act}: {n}")

    lines.extend(["", "--- Latest scan top candidates ---"])
    for row in sorted(latest_scans, key=lambda x: x["relative_rank"])[:8]:
        lines.append(
            f"  #{row['relative_rank']} {row['symbol']}: {row['situation']} "
            f"real={row['real_trend_score']} fake={row['fake_trend_score']} "
            f"action={row['recommended_action']} unc={row['uncertainty_pct']}%"
        )

    lines.extend([
        "",
        f"Avg uncertainty: {avg_unc:.1f}%",
        "",
        "Principles:",
        "  - P6-P14 core findings only drive scores",
        "  - P13 roles + P14 regime = context, not dominant",
        "  - No price targets | No hidden actors | Unknown valid",
        "",
    ])
    lines.extend(mission_summary_lines())
    lines.extend([
        "",
        f"Action engine: {ACTION_CSV}",
        f"Latest scans: {LATEST_CSV}",
        "=" * 58,
    ])
    REPORT_TXT.write_text("\n".join(lines), encoding="utf-8")

    print("===== P15 SITUATION EVALUATION OUTPUT =====")
    print(f"Evaluated: {len(evaluations)} | Latest scans: {len(latest_scans)}")
    print(f"Report: {REPORT_TXT}")


if __name__ == "__main__":
    main()
