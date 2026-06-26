"""
Scout Learning Season2 - P27 Scout Market Regime & Adaptive Personality Engine

Classifies market ecology and adapts Scout behavior per regime.
No price forecasting. No Buy/Sell. P25/P26 principles protected.
"""

import argparse
import csv
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from season2_scout_mission import mission_summary_lines

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

REGIMES_CSV = LOGS_DIR / "season2_p27_market_regimes.csv"
PROFILES_CSV = LOGS_DIR / "season2_p27_regime_profiles.csv"
PERSONALITY_CSV = LOGS_DIR / "season2_p27_personality.csv"
TRANSITIONS_CSV = LOGS_DIR / "season2_p27_regime_transitions.csv"
CONF_REGIME_CSV = LOGS_DIR / "season2_p27_confidence_by_regime.csv"
PLAYBOOK_ECO_CSV = LOGS_DIR / "season2_p27_playbook_ecology.csv"
DIARY_CSV = LOGS_DIR / "season2_p27_market_diary.csv"
COUNTERFACTUAL_CSV = LOGS_DIR / "season2_p27_counterfactual.csv"
PROTECTED_CSV = LOGS_DIR / "season2_p27_protected_principles.csv"
REPORT_TXT = LOGS_DIR / "season2_p27_research_report.txt"

REGIMES = (
    "Healthy Expansion", "Rotation", "Compression", "Conflict",
    "Recovery", "Panic", "Mixed",
)
POSITIVE_OUTCOMES = {"favorable", "mixed"}


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


def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pf(val, default=None):
    if val is None or val == "":
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def pi(val, default=0) -> int:
    v = pf(val, default)
    return int(v) if v is not None else default


def pbool(val) -> bool:
    return val in (True, "True", "true", "1", 1)


def classify_scan_regime(scan_rows: list[dict], field: dict | None) -> tuple[str, str, list[str]]:
    """Classify scan-level market ecology from aggregate observation ecology."""
    if not scan_rows:
        return "Mixed", "sparse", []

    n = len(scan_rows)
    collapse_avg = statistics.mean(pf(r.get("collapse_risk_pct"), 10) or 10 for r in scan_rows)
    collapse_high = sum(1 for r in scan_rows if pf(r.get("collapse_risk_pct"), 0) and pf(r.get("collapse_risk_pct"), 0) >= 30)
    false_conv = sum(1 for r in scan_rows if pbool(r.get("false_convergence_flagged")))
    conflict_coherence = sum(1 for r in scan_rows if r.get("field_coherence") == "conflicting")
    fertile = sum(1 for r in scan_rows if r.get("seedbed_quality") in ("Fertile", "Very fertile"))
    persist2 = sum(1 for r in scan_rows if pi(r.get("convergence_persist_scans")) >= 2)
    low_supply = sum(1 for r in scan_rows if r.get("supply_context") == "LOW_SUPPLY")
    high_supply = sum(1 for r in scan_rows if r.get("supply_context") == "HIGH_SUPPLY")
    collapse_supply = sum(1 for r in scan_rows if r.get("supply_context") == "COLLAPSE")
    recovery_sit = sum(1 for r in scan_rows if r.get("situation") == "Recovery")
    accum = sum(1 for r in scan_rows if r.get("situation") == "Accumulation")
    tier_x = sum(1 for r in scan_rows if r.get("priority_tier") == "X")
    growing = sum(1 for r in scan_rows if r.get("convergence_state") == "growing_convergence")

    evidence = []
    if collapse_supply >= 2 or collapse_high >= n * 0.25 or collapse_avg >= 35:
        evidence.append("collapse_elevated")
        return "Panic", "collapse_supply_or_risk", evidence
    if fertile >= 2 and persist2 >= max(1, n * 0.15) and collapse_avg < 22:
        evidence.append("fertile_persist_building")
        return "Healthy Expansion", "fertile_seedbed_building", evidence
    if recovery_sit >= 2 and collapse_avg < 25:
        evidence.append("recovery_situations")
        return "Recovery", "post_stress_improvement", evidence
    if low_supply >= n * 0.35 and accum >= n * 0.2 and collapse_avg < 22:
        evidence.append("compression_supply")
        return "Compression", "accumulation_low_supply", evidence
    if field and pf(field.get("early_trend_density_pct"), 0) and pf(field.get("early_trend_density_pct"), 0) >= 35:
        evidence.append("early_trend_rotation")
        return "Rotation", "multi_symbol_early_activity", evidence
    if high_supply >= n * 0.45 and growing >= n * 0.2:
        evidence.append("supply_rotation")
        return "Rotation", "supply_rotation_ambiguous", evidence
    if false_conv >= n * 0.4 or (conflict_coherence >= n * 0.75 and fertile == 0):
        evidence.append("false_convergence_or_conflict")
        return "Conflict", "cosmetic_agreement_ecology", evidence
    if conflict_coherence >= n * 0.5:
        evidence.append("field_conflict_mixed")
        return "Mixed", "conflicting_but_not_panic", evidence
    evidence.append("no_dominant_signature")
    return "Mixed", "ambiguous_ecology", evidence


def personality_for_regime(regime: str) -> dict:
    """Adaptive Scout personality — never violates protected principles."""
    profiles = {
        "Healthy Expansion": {
            "personality": "attentive_watch",
            "caution_level": "moderate",
            "unknown_bias": "low",
            "watch_bias": "high",
            "attention_increase": "conditional",
            "late_recognition_penalty": "reduced",
            "guidance": "Watch closely; allow attention increase only with persist>=2 and no field conflict",
        },
        "Rotation": {
            "personality": "field_first_watch",
            "caution_level": "moderate",
            "unknown_bias": "medium",
            "watch_bias": "high",
            "attention_increase": "rare",
            "late_recognition_penalty": "normal",
            "guidance": "Inspect field before symbol; rotation ecology favors patience over chasing",
        },
        "Compression": {
            "personality": "patient_unknown",
            "caution_level": "high",
            "unknown_bias": "high",
            "watch_bias": "medium",
            "attention_increase": "rare",
            "late_recognition_penalty": "reduced",
            "guidance": "Compression rewards patience; Unknown valid until persistence confirms",
        },
        "Conflict": {
            "personality": "defensive_unknown",
            "caution_level": "very_high",
            "unknown_bias": "very_high",
            "watch_bias": "low",
            "attention_increase": "never",
            "late_recognition_penalty": "normal",
            "guidance": "False convergence ecology — Unknown or Ignore; never force clarity",
        },
        "Recovery": {
            "personality": "cautious_watch",
            "caution_level": "high",
            "unknown_bias": "medium",
            "watch_bias": "high",
            "attention_increase": "conditional",
            "late_recognition_penalty": "reduced",
            "guidance": "Recovery paths need persistence confirmation; Watch default",
        },
        "Panic": {
            "personality": "minimal_attention",
            "caution_level": "maximum",
            "unknown_bias": "high",
            "watch_bias": "minimal",
            "attention_increase": "never",
            "late_recognition_penalty": "normal",
            "guidance": "Collapse ecology — Ignore unnecessary attention; protect discipline",
        },
        "Mixed": {
            "personality": "default_watch",
            "caution_level": "moderate",
            "unknown_bias": "medium",
            "watch_bias": "high",
            "attention_increase": "rare",
            "late_recognition_penalty": "normal",
            "guidance": "Ambiguous ecology — Watch default; Unknown when noisy",
        },
    }
    return profiles.get(regime, profiles["Mixed"])


def stance_for_regime(regime: str, conf: dict, case: dict) -> str:
    pers = personality_for_regime(regime)
    false_conv = pbool(case.get("false_convergence_flagged"))
    tier = case.get("priority_tier", "D")
    bucket = conf.get("confidence_bucket", "medium")

    if regime == "Panic" or tier == "X" or case.get("supply_context") == "COLLAPSE":
        return "Ignore"
    if regime == "Conflict" or false_conv:
        return "Unknown"
    if pers["unknown_bias"] in ("very_high", "high") and bucket == "low":
        return "Unknown"
    if pers["attention_increase"] == "conditional" and bucket == "medium" and pi(case.get("convergence_persist_scans")) >= 2 and not false_conv:
        return "Increase Attention"
    if pers["watch_bias"] in ("high", "medium"):
        return "Watch"
    return "Unknown"


def build_regime_map(library: list[dict], fields: list[dict], confidence: list[dict]) -> tuple[dict, list[dict]]:
    field_by_scan = {f["scan_time"]: f for f in fields}
    conf_by_key = {(c["scan_time"], c["symbol"]): c for c in confidence}

    by_scan: dict[str, list] = defaultdict(list)
    for row in library:
        by_scan[row["scan_time"]].append(row)

    scan_regimes: dict[str, dict] = {}
    obs_regimes: list[dict] = []

    for scan_time, rows in sorted(by_scan.items()):
        regime, signature, evidence = classify_scan_regime(rows, field_by_scan.get(scan_time))
        scan_regimes[scan_time] = {
            "scan_time": scan_time,
            "date": rows[0]["date"],
            "market_regime": regime,
            "regime_signature": signature,
            "regime_evidence": "|".join(evidence),
            "observation_count": len(rows),
        }
        for case in rows:
            conf = conf_by_key.get((scan_time, case["symbol"]), {})
            pers = personality_for_regime(regime)
            stance = stance_for_regime(regime, conf, case)
            support = conf.get("largest_support") or case.get("situation", "")
            conflict = conf.get("largest_risk") or case.get("field_coherence", "")
            obs_regimes.append({
                "date": case["date"],
                "symbol": case["symbol"],
                "scan_time": scan_time,
                "market_regime": regime,
                "regime_signature": signature,
                "regime_evidence": "|".join(evidence),
                "situation": case.get("situation"),
                "priority_tier": case.get("priority_tier"),
                "playbook": case.get("playbook_match"),
                "field_coherence": case.get("field_coherence"),
                "empirical_outcome": case.get("empirical_outcome"),
                "confidence_score": conf.get("confidence_score"),
                "confidence_bucket": conf.get("confidence_bucket"),
                "persistence_scans": case.get("convergence_persist_scans"),
                "scout_personality": pers["personality"],
                "recommended_stance": stance,
                "largest_support": support,
                "largest_conflict": conflict,
            })
    return scan_regimes, obs_regimes


def regime_profiles(obs: list[dict], library: list[dict]) -> list[dict]:
    lib_idx = {(c["scan_time"], c["symbol"]): c for c in library}
    by_regime: dict[str, list] = defaultdict(list)
    for o in obs:
        by_regime[o["market_regime"]].append(o)

    rows = []
    for regime in REGIMES:
        items = by_regime.get(regime, [])
        if not items:
            continue
        cases = [lib_idx.get((i["scan_time"], i["symbol"]), {}) for i in items]
        coh = Counter(c.get("field_coherence") for c in cases if c)
        pb = Counter(i.get("playbook") for i in items)
        tier = Counter(i.get("priority_tier") for i in items)
        unknown = sum(1 for i in items if i.get("recommended_stance") == "Unknown")
        conf = [pf(i.get("confidence_score"), 0) or 0 for i in items if i.get("confidence_score")]

        promo = sum(pi(c.get("promotion_count")) for c in cases if c)
        demo = sum(pi(c.get("demotion_count")) for c in cases if c)
        false_c = sum(1 for c in cases if pbool(c.get("false_convergence_flagged")))
        interact = sum(1 for c in cases if c.get("interaction") == "supported" or "interaction" in (c.get("support_families") or ""))
        supply_high = sum(1 for c in cases if c.get("supply_context") == "HIGH_SUPPLY")
        persist2 = sum(1 for c in cases if pi(c.get("convergence_persist_scans")) >= 2)
        collapse = statistics.mean(pf(c.get("collapse_risk_pct"), 10) or 10 for c in cases if c) if cases else 0

        rows.append({
            "market_regime": regime,
            "observation_count": len(items),
            "field_coherence_mode": coh.most_common(1)[0][0] if coh else "",
            "conflicting_pct": round(100 * coh.get("conflicting", 0) / len(items), 1),
            "avg_persistence_scans": round(statistics.mean(pi(c.get("convergence_persist_scans")) for c in cases if c), 2) if cases else 0,
            "persist_2plus_pct": round(100 * persist2 / len(items), 1),
            "high_supply_pct": round(100 * supply_high / len(items), 1),
            "interaction_supported_pct": round(100 * interact / len(items), 1),
            "avg_collapse_risk": round(collapse, 1),
            "false_convergence_pct": round(100 * false_c / len(items), 1),
            "promotion_events": promo,
            "demotion_events": demo,
            "playbook_distribution": "|".join(f"{k}({v})" for k, v in pb.most_common(4)),
            "unknown_stance_pct": round(100 * unknown / len(items), 1),
            "avg_confidence": round(statistics.mean(conf), 1) if conf else 0,
            "confidence_low_pct": round(100 * sum(1 for i in items if i.get("confidence_bucket") == "low") / len(items), 1),
            "confidence_medium_pct": round(100 * sum(1 for i in items if i.get("confidence_bucket") == "medium") / len(items), 1),
        })
    return rows


def regime_transitions(scan_regimes: dict) -> list[dict]:
    by_date: dict[str, list] = defaultdict(list)
    for scan_time, info in scan_regimes.items():
        by_date[info["date"]].append(info)
    for d in by_date:
        by_date[d].sort(key=lambda x: x["scan_time"])

    transitions: Counter = Counter()
    paths: list[dict] = []
    dates = sorted(by_date.keys())
    for d in dates:
        scans = by_date[d]
        for i in range(1, len(scans)):
            fr, to = scans[i - 1]["market_regime"], scans[i]["market_regime"]
            key = f"{fr}->{to}"
            transitions[key] += 1
            paths.append({
                "date": d,
                "from_scan": scans[i - 1]["scan_time"],
                "to_scan": scans[i]["scan_time"],
                "from_regime": fr,
                "to_regime": to,
                "transition": key,
            })

    rows = []
    for trans, freq in transitions.most_common():
        fr, to = trans.split("->", 1)
        rows.append({
            "transition": trans,
            "from_regime": fr,
            "to_regime": to,
            "frequency": freq,
            "path_type": _path_type(fr, to),
        })
    return rows, paths


def _path_type(fr: str, to: str) -> str:
    known = {
        ("Healthy Expansion", "Rotation"): "expansion_to_rotation",
        ("Rotation", "Conflict"): "rotation_to_conflict",
        ("Conflict", "Panic"): "conflict_to_panic",
        ("Panic", "Recovery"): "panic_to_recovery",
        ("Recovery", "Healthy Expansion"): "recovery_to_expansion",
        ("Compression", "Healthy Expansion"): "compression_to_expansion",
        ("Mixed", "Rotation"): "mixed_to_rotation",
    }
    return known.get((fr, to), "empirical_path")


def confidence_by_regime(obs: list[dict]) -> list[dict]:
    by_regime: dict[str, list] = defaultdict(list)
    for o in obs:
        by_regime[o["market_regime"]].append(o)

    rows = []
    for regime in REGIMES:
        items = by_regime.get(regime, [])
        if not items:
            continue
        n = len(items)
        fav = sum(1 for i in items if i.get("empirical_outcome") == "favorable")
        conf_scores = [pf(i.get("confidence_score"), 0) or 0 for i in items if i.get("confidence_score")]
        over = sum(1 for i in items if i.get("confidence_bucket") == "medium" and i.get("empirical_outcome") == "unfavorable" and pf(i.get("confidence_score"), 0) and pf(i.get("confidence_score"), 0) >= 50)
        under = sum(1 for i in items if i.get("confidence_bucket") == "low" and i.get("empirical_outcome") == "favorable")
        unknown_stance = sum(1 for i in items if i.get("recommended_stance") == "Unknown")
        watch_ok = sum(1 for i in items if i.get("recommended_stance") == "Watch" and i.get("empirical_outcome") in POSITIVE_OUTCOMES)

        rows.append({
            "market_regime": regime,
            "count": n,
            "avg_confidence": round(statistics.mean(conf_scores), 1) if conf_scores else 0,
            "overconfidence_count": over,
            "underconfidence_count": under,
            "calibration_quality": round(100 * (fav + sum(1 for i in items if i.get("empirical_outcome") == "mixed") * 0.5) / n, 1),
            "attention_success_pct": round(100 * watch_ok / max(sum(1 for i in items if i.get("recommended_stance") == "Watch"), 1), 1),
            "unknown_honesty_pct": round(100 * unknown_stance / n, 1),
            "confidence_note": _conf_note(regime, over, under, n),
        })
    return rows


def _conf_note(regime: str, over: int, under: int, n: int) -> str:
    if regime in ("Conflict", "Panic") and under > over:
        return "Appropriately low confidence ecology"
    if regime == "Healthy Expansion" and under > over:
        return "Slight underconfidence acceptable — persistence gate preserved"
    if over > under:
        return "Watch for overconfidence in this ecology"
    return "Calibrated"


def playbook_ecology(obs: list[dict]) -> list[dict]:
    rows = []
    by_regime_pb: dict[tuple, list] = defaultdict(list)
    for o in obs:
        pb = o.get("playbook") or "none"
        by_regime_pb[(o["market_regime"], pb)].append(o)

    for (regime, pb), items in sorted(by_regime_pb.items(), key=lambda x: (-len(x[1]), x[0][0])):
        n = len(items)
        fav = sum(1 for i in items if i.get("empirical_outcome") == "favorable")
        rows.append({
            "market_regime": regime,
            "playbook": pb,
            "count": n,
            "favorable_pct": round(100 * fav / n, 1),
            "dominant_stance": Counter(i.get("recommended_stance") for i in items).most_common(1)[0][0],
            "ecology_verdict": _playbook_verdict(regime, pb, fav / n),
            "scout_guidance": _playbook_guidance(regime, pb),
        })
    return rows


def _playbook_verdict(regime: str, pb: str, fav_rate: float) -> str:
    if regime == "Conflict" and pb in ("C", "B"):
        return "background_caution"
    if regime == "Healthy Expansion" and pb == "A":
        return "watch_closely"
    if regime == "Panic":
        return "background"
    if fav_rate >= 0.5:
        return "compatible"
    if fav_rate < 0.3:
        return "caution"
    return "neutral"


def _playbook_guidance(regime: str, pb: str) -> str:
    if pb == "C" or regime == "Conflict":
        return "Unknown — cosmetic agreement ecology"
    if pb == "A" and regime in ("Healthy Expansion", "Compression"):
        return "Watch with persistence gate"
    if pb == "E":
        return "Patience — late recognition ecology"
    return "Watch default"


def counterfactual_regime(obs: list[dict]) -> list[dict]:
    proposals = [
        {"id": "RF1", "regime": "Healthy Expansion", "change": "Reduce late_recognition penalty (allow medium confidence Watch)", "test": lambda o, s: s == "Watch" if o["market_regime"] == "Healthy Expansion" and o.get("confidence_bucket") == "low" and o.get("empirical_outcome") == "favorable" else None},
        {"id": "RF2", "regime": "Conflict", "change": "Force Unknown stance in Conflict regime", "test": lambda o, s: "Unknown" if o["market_regime"] == "Conflict" and s in ("Watch", "Increase Attention") else None},
        {"id": "RF3", "regime": "Panic", "change": "Force Ignore in Panic regime for Tier C/D", "test": lambda o, s: "Ignore" if o["market_regime"] == "Panic" and o.get("priority_tier") in ("C", "D", "X") else None},
        {"id": "RF4", "regime": "Compression", "change": "Increase Unknown bias in Compression", "test": lambda o, s: "Unknown" if o["market_regime"] == "Compression" and s == "Watch" and pi(o.get("persistence_scans")) == 0 else None},
    ]
    rows = []
    for p in proposals:
        improved = harmed = unchanged = 0
        subset = [o for o in obs if o["market_regime"] == p["regime"]]
        for o in subset:
            old = o.get("recommended_stance", "Watch")
            new = p["test"](o, old)
            if new is None or new == old:
                unchanged += 1
                continue
            outcome = o.get("empirical_outcome", "")
            if outcome == "unfavorable" and new in ("Unknown", "Ignore"):
                improved += 1
            elif outcome == "favorable" and new in ("Unknown", "Ignore") and old == "Watch":
                harmed += 1
            elif outcome in POSITIVE_OUTCOMES and new == "Watch" and old == "Unknown":
                improved += 1
            else:
                unchanged += 1
        net = improved - harmed
        rows.append({
            "proposal_id": p["id"],
            "regime": p["regime"],
            "change": p["change"],
            "observations_tested": len(subset),
            "would_improve": improved,
            "would_harm": harmed,
            "net_discipline_gain": net,
            "recommendation": "ACCEPT" if net > 0 and harmed <= improved else "REJECT",
        })
    return rows


def load_protected() -> list[dict]:
    rows = []
    for src, path in [
        ("P25", LOGS_DIR / "season2_p25_protected_principles.csv"),
        ("P26", LOGS_DIR / "season2_p26_protected_principles.csv"),
    ]:
        for r in load_csv(path):
            rows.append({**r, "layer": src})
    if not rows:
        rows = [{"principle": "Honest Unknown", "never_change": "yes", "layer": "P27"}]
    rows.append({"principle": "P26 calibrated confidence", "never_change": "yes", "layer": "P27"})
    return rows


def build_report(scan_regimes, obs_regimes, profiles, personalities, transitions, conf_regime, playbook, counterfactual, protected) -> str:
    regime_counts = Counter(o["market_regime"] for o in obs_regimes)
    accepted = [c for c in counterfactual if c["recommendation"] == "ACCEPT"]

    lines = [
        "===== SCOUT SEASON2 P27 - MARKET REGIME & ADAPTIVE PERSONALITY =====",
        "",
        f"Scans: {len(scan_regimes)} | Observations: {len(obs_regimes)}",
        "",
        "=== Final Answers ===",
        "",
        "1. What market regimes exist?",
    ]
    for r, n in regime_counts.most_common():
        lines.append(f"   - {r}: {n} observations")

    lines.extend(["", "2. How should Scout behave in each?"])
    for p in personalities:
        lines.append(f"   - {p['market_regime']}: {p['personality']} — {p['guidance'][:80]}")

    lines.extend(["", "3. Which regimes increase caution?"])
    for p in personalities:
        if p["caution_level"] in ("high", "very_high", "maximum"):
            lines.append(f"   - {p['market_regime']} ({p['caution_level']})")

    lines.extend(["", "4. Which reduce late recognition?"])
    for p in personalities:
        if p.get("late_recognition_penalty") == "reduced":
            lines.append(f"   - {p['market_regime']}: reduced late recognition penalty — persistence still required")

    lines.extend(["", "5. How does confidence change by regime?"])
    for c in conf_regime:
        lines.append(f"   - {c['market_regime']}: avg={c['avg_confidence']} cal={c['calibration_quality']}% {c['confidence_note']}")

    lines.extend(["", "6. Which playbooks fit each ecology?"])
    for pb in sorted(playbook, key=lambda x: (x["market_regime"], -x["count"]))[:10]:
        lines.append(f"   - {pb['market_regime']} + playbook {pb['playbook']}: {pb['ecology_verdict']}")

    lines.extend(["", "7. Which protected principles remain unchanged?"])
    seen = set()
    for p in protected:
        name = p.get("principle", "")
        if name not in seen:
            lines.append(f"   - {name}")
            seen.add(name)

    lines.extend([
        "",
        "8. How did Scout evolve P15->P27?",
        "   P15 situation -> P16 field -> P21 attention -> P23 memory -> P24 replay",
        "   -> P25 bias correction -> P26 confidence -> P27 regime-adaptive personality",
        "",
        "--- Top regime transitions ---",
    ])
    for t in transitions[:8]:
        lines.append(f"  {t['transition']}: n={t['frequency']} ({t['path_type']})")

    lines.extend(["", "--- Regime counterfactual ---"])
    for c in counterfactual:
        lines.append(f"  {c['proposal_id']}: {c['recommendation']} net={c['net_discipline_gain']} ({c['change'][:60]})")

    lines.extend([
        "",
        "A great Scout understands the environment before allocating attention.",
        "Never forecast. Never Buy/Sell.",
    ])
    lines.extend(mission_summary_lines())
    lines.append("=" * 58)
    return "\n".join(lines)


def ensure_deps() -> None:
    for path, mod in [
        ("season2_p23_case_library.csv", "season2_p23_scout_memory"),
        ("season2_p26_confidence_scores.csv", "season2_p26_scout_confidence_calibration"),
    ]:
        if not (LOGS_DIR / path).exists():
            __import__(mod).main()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-deps", action="store_true")
    args = parser.parse_args()

    if args.rebuild_deps:
        import season2_p23_scout_memory
        import season2_p26_scout_confidence_calibration
        season2_p23_scout_memory.main()
        season2_p26_scout_confidence_calibration.main()
    else:
        ensure_deps()

    library = load_csv(LOGS_DIR / "season2_p23_case_library.csv")
    confidence = load_csv(LOGS_DIR / "season2_p26_confidence_scores.csv")
    fields = load_csv(LOGS_DIR / "season2_p16_opportunity_fields.csv")
    if not library:
        print("Run P23 first")
        return

    scan_regimes, obs_regimes = build_regime_map(library, fields, confidence)
    profiles = regime_profiles(obs_regimes, library)
    transition_summary, transition_paths = regime_transitions(scan_regimes)
    conf_regime = confidence_by_regime(obs_regimes)
    playbook = playbook_ecology(obs_regimes)
    counterfactual = counterfactual_regime(obs_regimes)
    protected = load_protected()

    personalities = []
    for regime in REGIMES:
        pers = personality_for_regime(regime)
        count = sum(1 for o in obs_regimes if o["market_regime"] == regime)
        if count == 0:
            continue
        personalities.append({"market_regime": regime, "observation_count": count, **pers})

    diary = [{
        "date": o["date"],
        "symbol": o["symbol"],
        "scan_time": o["scan_time"],
        "current_regime": o["market_regime"],
        "regime_evidence": o["regime_evidence"],
        "largest_support": o["largest_support"],
        "largest_conflict": o["largest_conflict"],
        "scout_personality": o["scout_personality"],
        "recommended_stance": o["recommended_stance"],
        "confidence_bucket": o.get("confidence_bucket"),
        "playbook": o.get("playbook"),
    } for o in obs_regimes]

    report = build_report(scan_regimes, obs_regimes, profiles, personalities, transition_summary, conf_regime, playbook, counterfactual, protected)

    write_csv(REGIMES_CSV, list(scan_regimes.values()))
    write_csv(PROFILES_CSV, profiles)
    write_csv(PERSONALITY_CSV, personalities)
    path_rows = [{**p, "record_type": "path"} for p in transition_paths]
    summary_rows = [{**t, "record_type": "summary"} for t in transition_summary]
    write_csv(TRANSITIONS_CSV, summary_rows + path_rows)
    write_csv(CONF_REGIME_CSV, conf_regime)
    write_csv(PLAYBOOK_ECO_CSV, playbook)
    write_csv(DIARY_CSV, diary)
    write_csv(COUNTERFACTUAL_CSV, counterfactual)
    write_csv(PROTECTED_CSV, protected)
    REPORT_TXT.write_text(report, encoding="utf-8")

    print("===== P27 MARKET REGIME & PERSONALITY =====")
    print(f"Scans: {len(scan_regimes)} | Observations: {len(obs_regimes)} | Regimes: {len(profiles)}")
    print(f"Report: {REPORT_TXT}")


if __name__ == "__main__":
    main()
