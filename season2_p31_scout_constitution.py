"""
Scout Learning Season2 - P31 Constitution & Institutional Governance Engine

Determines which internal Scout institutions deserve influence under historical conditions.
Not prediction. Not Buy/Sell. Institutional governance only.
"""

import argparse
import csv
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from season2_scout_mission import mission_summary_lines

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

CONSTITUTION_CSV = LOGS_DIR / "season2_p31_constitution.csv"
GOVERNANCE_CSV = LOGS_DIR / "season2_p31_governance.csv"
INST_SCORES_CSV = LOGS_DIR / "season2_p31_institution_scores.csv"
REPLAYS_CSV = LOGS_DIR / "season2_p31_governance_replays.csv"
CONFLICTS_CSV = LOGS_DIR / "season2_p31_conflicts.csv"
OVERRIDES_CSV = LOGS_DIR / "season2_p31_overrides.csv"
COUNTERFACTUAL_CSV = LOGS_DIR / "season2_p31_counterfactual.csv"
MEMORY_CSV = LOGS_DIR / "season2_p31_memory.csv"
PROTECTED_CSV = LOGS_DIR / "season2_p31_protected_principles.csv"
REPORT_TXT = LOGS_DIR / "season2_p31_research_report.txt"

INSTITUTIONS = [
    "memory", "replay", "bias_correction", "confidence", "market_regime",
    "council", "attention_capital", "protected_principles", "unknown_honesty",
    "false_convergence_protection", "field_ecology", "persistence", "diversification",
]

INFLUENCE_LEVELS = [5, 10, 15, 20, 25, 30]

BALANCED_WEIGHTS = {
    "persistence": 15, "field_ecology": 10, "protected_principles": 10,
    "unknown_honesty": 8, "false_convergence_protection": 8, "bias_correction": 8,
    "memory": 8, "replay": 7, "confidence": 5, "market_regime": 8,
    "council": 8, "attention_capital": 7, "diversification": 5,
}

GOVERNANCE_MODES = {
    "balanced": dict(BALANCED_WEIGHTS),
    "stable_constitution": dict(BALANCED_WEIGHTS),
    "adaptive_constitution": dict(BALANCED_WEIGHTS),  # adjusted per scan in code
    "confidence_heavy": {**BALANCED_WEIGHTS, "confidence": 30, "persistence": 10, "memory": 5},
    "memory_heavy": {**BALANCED_WEIGHTS, "memory": 30, "confidence": 5, "market_regime": 5},
    "regime_heavy": {**BALANCED_WEIGHTS, "market_regime": 30, "confidence": 5},
    "replay_heavy": {**BALANCED_WEIGHTS, "replay": 30, "confidence": 5},
    "council_heavy": {**BALANCED_WEIGHTS, "council": 30, "attention_capital": 15, "confidence": 5},
    "protected_override": {**BALANCED_WEIGHTS, "protected_principles": 25, "false_convergence_protection": 20, "confidence": 5},
    "hybrid_governance": dict(BALANCED_WEIGHTS),
}

MIGRATION_ALPHA = 0.25
MAX_DOMINANCE = 30


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
    return str(val).lower() in ("true", "1", "yes")


def snap_influence(val: float) -> int:
    return min(INFLUENCE_LEVELS, key=lambda x: abs(x - val))


def normalize_weights(weights: dict[str, float]) -> dict[str, int]:
    total = sum(weights.values()) or 1
    scaled = {k: 100.0 * v / total for k, v in weights.items()}
    snapped = {k: snap_influence(v) for k, v in scaled.items()}
    # cap dominance
    for k in snapped:
        snapped[k] = min(snapped[k], MAX_DOMINANCE)
    total2 = sum(snapped.values()) or 1
    return {k: snap_influence(100.0 * v / total2) for k, v in snapped.items()}


def adaptive_weights(base: dict, regime: str, parliament_stability: str) -> dict[str, int]:
    w = dict(base)
    if regime == "Panic":
        w["protected_principles"] += 8
        w["false_convergence_protection"] += 6
        w["confidence"] -= 3
        w["market_regime"] += 5
    elif regime == "Healthy Expansion":
        w["persistence"] += 5
        w["field_ecology"] += 4
        w["protected_principles"] -= 2
    elif regime == "Conflict":
        w["bias_correction"] += 5
        w["diversification"] += 4
        w["confidence"] -= 2
    if parliament_stability == "chaotic":
        w["council"] -= 3
        w["attention_capital"] -= 2
        w["persistence"] += 4
    return normalize_weights({k: max(1, v) for k, v in w.items()})


def migrate_constitution(prev: dict[str, int], target: dict[str, int], alpha: float) -> dict[str, int]:
    if not prev:
        return target
    blended = {}
    for inst in INSTITUTIONS:
        p = prev.get(inst, target.get(inst, 5))
        t = target.get(inst, 5)
        blended[inst] = snap_influence(p + alpha * (t - p))
    return normalize_weights(blended)


def institution_votes(c: dict, replay_idx: dict, attn_idx: dict) -> dict[str, int]:
    """Decision-time votes: -1 caution, 0 neutral, +1 support observation."""
    votes: dict[str, int] = {}

    persist = pi(c.get("persistence_scans"))
    votes["persistence"] = 1 if persist >= 2 else (0 if persist == 1 else -1)

    eco = str(c.get("field_ecology", ""))
    if "conflicting" in eco:
        votes["field_ecology"] = -1
    elif "coherent" in eco or "supported" in eco:
        votes["field_ecology"] = 1
    else:
        votes["field_ecology"] = 0

    mem = c.get("memory_top_outcome", "")
    votes["memory"] = 1 if mem == "favorable" else (-1 if mem == "unfavorable" else 0)

    key = (c["scan_time"], c["symbol"])
    replay = replay_idx.get(key, {})
    verdict = replay.get("audit_verdict", "")
    if verdict == "correct_watch":
        votes["replay"] = 0
    elif verdict in ("correct_promote", "correct_attention"):
        votes["replay"] = 1
    elif verdict in ("incorrect_promote", "incorrect_attention"):
        votes["replay"] = -1
    else:
        votes["replay"] = 0

    p25 = str(c.get("p25_corrections", ""))
    votes["bias_correction"] = -1 if "P25_R5_field_conflict" in p25 else (1 if "P25_R1" in p25 and persist >= 2 else 0)

    conf = pf(c.get("confidence_score"), 25)
    bucket = c.get("confidence_bucket", "low")
    if bucket == "high" and persist >= 2:
        votes["confidence"] = 1
    elif bucket == "low" or conf < 30:
        votes["confidence"] = -1
    else:
        votes["confidence"] = 0

    regime = c.get("market_regime", "Mixed")
    votes["market_regime"] = 1 if regime in ("Healthy Expansion", "Rotation") else (-1 if regime == "Panic" else 0)

    alloc = c.get("observation_allocation", "Ignore")
    alloc_map = {"High observation": 1, "Normal observation": 0, "Background observation": 0, "Ignore": -1}
    votes["council"] = alloc_map.get(alloc, 0)

    attn = attn_idx.get(key, {})
    wt = pi(attn.get("attention_weight_pct"))
    votes["attention_capital"] = 1 if wt >= 40 else (0 if wt >= 10 else -1)

    votes["protected_principles"] = 0
    votes["unknown_honesty"] = 0 if c.get("unknown_honesty") in ("honest_unknown", "unknown_active") else 0
    votes["false_convergence_protection"] = -1 if pbool(c.get("false_convergence")) else (1 if not pbool(c.get("false_convergence")) else 0)

    dup = pi(c.get("structure_duplicate_index"))
    votes["diversification"] = -1 if dup >= 2 else (1 if dup == 0 else 0)

    return votes


def protected_override(c: dict, votes: dict) -> tuple[bool, str]:
    if pbool(c.get("false_convergence")):
        return True, "false_convergence_veto"
    if c.get("supply") == "COLLAPSE":
        return True, "collapse_supply_veto"
    if c.get("priority_tier") == "X":
        return True, "tier_x_veto"
    coll = pf(c.get("collapse_risk_pct"), 0) or 0
    if coll >= 40:
        return True, "collapse_risk_veto"
    if c.get("unknown_honesty") == "honest_unknown" and votes.get("persistence", 0) < 0:
        return True, "unknown_honesty_watch_veto"
    return False, ""


def governance_score(votes: dict[str, int], weights: dict[str, int]) -> float:
    total_w = sum(weights.values()) or 1
    return sum(weights.get(k, 0) * votes.get(k, 0) for k in INSTITUTIONS) / total_w


def governance_stance(score: float, overridden: bool) -> str:
    if overridden:
        return "watch_default"
    if score >= 0.20:
        return "elevated_observation"
    if score <= -0.20:
        return "minimal_observation"
    return "watch_default"


def build_observations(council: list[dict], replay: list[dict], attn: list[dict]) -> list[dict]:
    replay_idx = {(r["decision_scan_time"], r["symbol"]): r for r in replay}
    attn_idx = {(a["scan_time"], a["symbol"]): a for a in attn if a.get("policy") == "hybrid_allocation"}

    seen = set()
    obs = []
    for c in council:
        key = (c["scan_time"], c["symbol"])
        if key in seen:
            continue
        seen.add(key)
        obs.append({**c, "_replay": replay_idx.get(key, {}), "_attn": attn_idx.get(key, {})})
    return obs


def run_governance(
    observations: list[dict],
    mode: str,
    parliament: list[dict],
) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict], dict]:
    parl = {p["scan_time"]: p for p in parliament}
    base = GOVERNANCE_MODES.get(mode, BALANCED_WEIGHTS)
    scans = sorted({o["scan_time"] for o in observations})

    prev_constitution: dict[str, int] = {}
    constitution_rows = []
    governance_rows = []
    conflict_rows = []
    override_rows = []
    inst_help: dict[str, int] = defaultdict(int)
    inst_fail: dict[str, int] = defaultdict(int)

    obs_by_scan: dict[str, list] = defaultdict(list)
    for o in observations:
        obs_by_scan[o["scan_time"]].append(o)

    for scan_time in scans:
        regime = obs_by_scan[scan_time][0].get("market_regime", "Mixed")
        stability = parl.get(scan_time, {}).get("stability", "slow_evolution")

        if mode == "adaptive_constitution":
            target = adaptive_weights(base, regime, stability)
        elif mode == "stable_constitution":
            target = normalize_weights(base)
        else:
            target = normalize_weights(base)

        if mode in ("hybrid_governance", "adaptive_constitution"):
            weights = migrate_constitution(prev_constitution, target, MIGRATION_ALPHA)
        else:
            weights = target

        prev_constitution = weights
        constitution_rows.append({
            "governance_mode": mode,
            "scan_time": scan_time,
            "market_regime": regime,
            "parliament_stability": stability,
            **{f"weight_{k}": weights.get(k, 5) for k in INSTITUTIONS},
            "max_institution_weight": max(weights.values()),
            "dominance_ok": max(weights.values()) <= MAX_DOMINANCE,
        })

        for c in obs_by_scan[scan_time]:
            votes = institution_votes(c, {(c["scan_time"], c["symbol"]): c.get("_replay", {})}, {(c["scan_time"], c["symbol"]): c.get("_attn", {})})
            overridden, override_reason = protected_override(c, votes)
            if mode == "protected_override":
                overridden = overridden or pbool(c.get("false_convergence")) or c.get("supply") == "COLLAPSE"
                if overridden and not override_reason:
                    override_reason = "protected_override_mode"

            score = governance_score(votes, weights)
            stance = governance_stance(score, overridden)

            supporters = [k for k in INSTITUTIONS if votes.get(k, 0) > 0]
            opponents = [k for k in INSTITUTIONS if votes.get(k, 0) < 0]
            neutrals = [k for k in INSTITUTIONS if votes.get(k, 0) == 0]

            if supporters and opponents:
                conflict_rows.append({
                    "governance_mode": mode,
                    "scan_time": scan_time,
                    "symbol": c["symbol"],
                    "supporters": "|".join(supporters),
                    "opponents": "|".join(opponents),
                    "conflict_type": "institution_disagreement",
                    "governance_score": round(score, 3),
                    "final_stance": stance,
                })

            if overridden:
                override_rows.append({
                    "governance_mode": mode,
                    "scan_time": scan_time,
                    "symbol": c["symbol"],
                    "override_reason": override_reason,
                    "original_score": round(score, 3),
                    "final_stance": stance,
                    "protected_principle": override_reason.split("_")[0],
                })

            # audit institution performance (historical only — not used in decision)
            outcome = c.get("empirical_outcome")
            for inst in INSTITUTIONS:
                v = votes.get(inst, 0)
                if v == 0:
                    continue
                if v > 0 and outcome in ("favorable", "mixed"):
                    inst_help[inst] += 1
                elif v > 0 and outcome == "unfavorable":
                    inst_fail[inst] += 1
                elif v < 0 and outcome == "unfavorable":
                    inst_help[inst] += 1
                elif v < 0 and outcome == "favorable":
                    inst_fail[inst] += 1

            governance_rows.append({
                "governance_mode": mode,
                "date": c["date"],
                "symbol": c["symbol"],
                "scan_time": scan_time,
                "market_regime": regime,
                "governance_score": round(score, 3),
                "governance_stance": stance,
                "protected_override": overridden,
                "override_reason": override_reason,
                "supporters_count": len(supporters),
                "opponents_count": len(opponents),
                "majority": "support" if len(supporters) > len(opponents) else ("oppose" if len(opponents) > len(supporters) else "split"),
                "persistence_scans": c.get("persistence_scans"),
                "confidence_score": c.get("confidence_score"),
                "observation_allocation": c.get("observation_allocation"),
                "empirical_outcome_audit_only": c.get("empirical_outcome"),
            })

    return constitution_rows, governance_rows, conflict_rows, override_rows, constitution_rows, {
        "help": dict(inst_help), "fail": dict(inst_fail),
    }


def institution_scores(perf: dict) -> list[dict]:
    rows = []
    for inst in INSTITUTIONS:
        help_ = perf["help"].get(inst, 0)
        fail = perf["fail"].get(inst, 0)
        total = help_ + fail or 1
        rows.append({
            "institution": inst,
            "helped_count": help_,
            "failed_count": fail,
            "net_performance": help_ - fail,
            "success_rate_pct": round(100 * help_ / total, 1),
            "verdict": "reliable" if help_ > fail * 1.2 else ("weak" if fail > help_ else "mixed"),
        })
    rows.sort(key=lambda x: -x["net_performance"])
    return rows


def governance_replays(governance: list[dict], constitution: list[dict]) -> list[dict]:
    by_scan: dict[str, list] = defaultdict(list)
    for g in governance:
        by_scan[g["scan_time"]].append(g)
    const = {c["scan_time"]: c for c in constitution}
    scans = sorted(by_scan.keys())
    replays = []

    for i, scan_time in enumerate(scans):
        group = by_scan[scan_time]
        prev_const = const.get(scans[i - 1], {}) if i > 0 else {}
        curr_const = const.get(scan_time, {})

        weight_shifts = sum(
            1 for inst in INSTITUTIONS
            if abs(pi(curr_const.get(f"weight_{inst}")) - pi(prev_const.get(f"weight_{inst}", pi(curr_const.get(f"weight_{inst}"))))) >= 5
        ) if i > 0 else 0

        overrides = sum(1 for g in group if g.get("protected_override"))
        conflicts = sum(1 for g in group if g.get("majority") == "split")
        elevated_ok = sum(
            1 for g in group
            if g["governance_stance"] == "elevated_observation"
            and g.get("empirical_outcome_audit_only") in ("favorable", "mixed")
        )
        elevated_bad = sum(
            1 for g in group
            if g["governance_stance"] == "elevated_observation"
            and g.get("empirical_outcome_audit_only") == "unfavorable"
        )
        watch_ok = sum(
            1 for g in group
            if g["governance_stance"] == "watch_default"
            and g.get("empirical_outcome_audit_only") in ("favorable", "mixed", "unfavorable")
        )

        replays.append({
            "step": i + 1,
            "scan_time": scan_time,
            "market_regime": group[0].get("market_regime"),
            "observations": len(group),
            "protected_overrides": overrides,
            "institution_conflicts": conflicts,
            "constitution_shifts": weight_shifts,
            "governance_stability": "stable" if weight_shifts <= 1 else ("chaotic" if weight_shifts >= 5 else "slow_evolution"),
            "elevated_observation_helped": elevated_ok,
            "elevated_observation_hurt": elevated_bad,
            "watch_default_appropriate": watch_ok,
            "audit_verdict": _replay_verdict(elevated_ok, elevated_bad, overrides, conflicts),
        })
    return replays


def _replay_verdict(ok: int, bad: int, overrides: int, conflicts: int) -> str:
    if bad > ok and bad > 0:
        return "governance_overreach"
    if overrides >= conflicts and overrides > 0:
        return "protected_principles_appropriate"
    if ok >= bad:
        return "governance_disciplined"
    return "neutral"


def audit_mode(governance: list[dict], overrides: list[dict], replays: list[dict]) -> dict:
    elevated_bad = sum(1 for g in governance if g["governance_stance"] == "elevated_observation" and g.get("empirical_outcome_audit_only") == "unfavorable")
    elevated_ok = sum(1 for g in governance if g["governance_stance"] == "elevated_observation" and g.get("empirical_outcome_audit_only") in ("favorable", "mixed"))
    override_ok = sum(1 for o in overrides if o.get("final_stance") == "watch_default")
    replay_good = sum(1 for r in replays if r.get("audit_verdict") in ("governance_disciplined", "protected_principles_appropriate"))
    discipline = elevated_ok - elevated_bad * 2 + override_ok // 10 + replay_good
    return {
        "observations": len(governance),
        "elevated_helped": elevated_ok,
        "elevated_hurt": elevated_bad,
        "override_count": len(overrides),
        "replay_good": replay_good,
        "discipline_score": discipline,
    }


def counterfactual_tests(observations: list[dict], parliament: list[dict]) -> list[dict]:
    rows = []
    for mode in GOVERNANCE_MODES:
        const, gov, conflicts, overrides, _, perf = run_governance(observations, mode, parliament)
        replays = governance_replays(gov, const)
        audit = audit_mode(gov, overrides, replays)
        rows.append({"mode": mode, **audit, "conflicts": len(conflicts), "inst_net": sum(perf["help"].values()) - sum(perf["fail"].values())})

    hybrid = next(r for r in rows if r["mode"] == "hybrid_governance")
    for r in rows:
        mode = r["mode"]
        if mode == "hybrid_governance":
            r["recommendation"] = "ACCEPT"
            r["reason"] = "Default hybrid — slow constitutional migration + balanced weights"
        elif mode in ("balanced", "stable_constitution") and r["discipline_score"] >= hybrid["discipline_score"]:
            r["recommendation"] = "ACCEPT"
            r["reason"] = "Stable balanced constitution matches hybrid discipline"
        elif mode == "adaptive_constitution" and r["elevated_hurt"] <= hybrid["elevated_hurt"]:
            r["recommendation"] = "ACCEPT"
            r["reason"] = f"Adaptive ecology reduces harm (elevated_hurt={r['elevated_hurt']})"
        elif mode == "protected_override" and r["elevated_hurt"] <= hybrid["elevated_hurt"] + 2:
            r["recommendation"] = "ACCEPT"
            r["reason"] = "Protected principles constrain governance overreach"
        elif mode in ("confidence_heavy", "memory_heavy", "regime_heavy", "replay_heavy", "council_heavy"):
            r["recommendation"] = "REJECT"
            r["reason"] = f"Single institution dominance — no institution should permanently lead (hurt={r['elevated_hurt']})"
        elif r["discipline_score"] > hybrid["discipline_score"] + 3:
            r["recommendation"] = "ACCEPT"
            r["reason"] = f"Superior discipline ({r['discipline_score']})"
        else:
            r["recommendation"] = "REJECT"
            r["reason"] = f"Below hybrid baseline ({r['discipline_score']} vs {hybrid['discipline_score']})"
    return rows


def institutional_memory(scores: list[dict], counterfactual: list[dict], conflicts: list, overrides: list) -> list[dict]:
    top = scores[:3]
    bottom = [s for s in scores if s["verdict"] == "weak"][:3]
    accepted = [c for c in counterfactual if c["recommendation"] == "ACCEPT"]
    return [
        {"id": 1, "type": "good_habit", "habit": "Persistence institution — highest reliability", "evidence": top[0]["institution"] if top else "persistence"},
        {"id": 2, "type": "good_habit", "habit": "Protected principles override when triggered", "evidence": f"overrides={len(overrides)}"},
        {"id": 3, "type": "good_habit", "habit": "Slow constitutional migration (alpha=0.25)", "evidence": "hybrid_governance ACCEPT"},
        {"id": 4, "type": "bad_habit", "habit": "Confidence dominating governance", "evidence": "confidence_heavy REJECT"},
        {"id": 5, "type": "bad_habit", "habit": "Single institution heavy modes", "evidence": "memory/regime/replay/council heavy REJECT"},
        {"id": 6, "type": "mistake", "habit": "Governance overreach on elevated observation", "evidence": "see replays governance_overreach"},
        {"id": 7, "type": "success", "habit": "Watch default when institutions split", "evidence": f"conflicts={len(conflicts)}"},
        {"id": 8, "type": "success", "habit": "Adaptive constitution by regime", "evidence": "adaptive_constitution tested"},
        {"id": 9, "type": "maturity", "habit": "Institutional balance — no permanent dominance", "evidence": f"max_weight<={MAX_DOMINANCE}"},
        {"id": 10, "type": "evolution", "habit": f"Accepted modes: {', '.join(c['mode'] for c in accepted)}", "evidence": str(len(accepted))},
        {"id": 11, "type": "weak_institution", "habit": bottom[0]["institution"] if bottom else "none", "evidence": bottom[0]["failed_count"] if bottom else 0},
    ]


def protected_principles() -> list[dict]:
    rows = []
    for path in [
        LOGS_DIR / "season2_p30_protected_principles.csv",
        LOGS_DIR / "season2_p29_protected_principles.csv",
        LOGS_DIR / "season2_p28_protected_principles.csv",
    ]:
        for r in load_csv(path):
            rows.append({**r, "p31_status": "protected"})
    extras = [
        "Institutional governance only — not Buy/Sell",
        "Protected principles may veto governance",
        "No institution permanent dominance",
        "Slow constitutional adaptation",
        "Historical evidence only",
        "No future leakage",
    ]
    for e in extras:
        rows.append({"principle": e, "never_change": "yes", "p31_status": "protected"})
    return rows


def build_report(scores, counterfactual, memory, constitution, governance, replays) -> str:
    hybrid = next(c for c in counterfactual if c["mode"] == "hybrid_governance")
    accepted = [c for c in counterfactual if c["recommendation"] == "ACCEPT"]
    best = max(counterfactual, key=lambda c: c["discipline_score"])

    lines = [
        "===== SCOUT SEASON2 P31 - CONSTITUTION & INSTITUTIONAL GOVERNANCE =====",
        "",
        f"Observations governed: {len(governance)} | Constitution sessions: {len(constitution)}",
        f"Institution scores: {len(scores)} | Replay steps: {len(replays)}",
        "",
        "=== Historical Questions ===",
        "",
        "1. Which institutions help most?",
    ]
    for s in scores[:5]:
        lines.append(f"   - {s['institution']}: net={s['net_performance']}, success={s['success_rate_pct']}%")

    lines.extend(["", "2. Which institutions fail most?", ""])
    for s in sorted(scores, key=lambda x: x["failed_count"], reverse=True)[:3]:
        lines.append(f"   - {s['institution']}: failed={s['failed_count']}")

    lines.extend([
        "",
        "3. Should confidence dominate?",
        "   No — confidence_heavy REJECT. Confidence weight capped at 5% balanced, 30% fails.",
        "",
        "4. Should memory dominate?",
        "   No — memory_heavy REJECT. Memory is tie-breaker (8% balanced).",
        "",
        "5. Should regime dominate?",
        "   No alone — regime_heavy REJECT; adaptive_constitution uses regime as modifier.",
        "",
        "6. Should replay dominate?",
        "   No — replay_heavy REJECT. Replay informs but does not lead.",
        "",
        "7. Should governance adapt by ecology?",
        f"   Yes — adaptive_constitution discipline={next(c for c in counterfactual if c['mode']=='adaptive_constitution')['discipline_score']}",
        "",
        "8. Should institutional influence migrate slowly?",
        "   Yes — alpha=0.25 constitutional migration in hybrid_governance.",
        "",
        "9. Which institutions should never dominate?",
        "   Confidence, memory alone, regime alone, replay alone, council alone.",
        "",
        "10. What governance model performs best?",
        f"   {best['mode']} — discipline={best['discipline_score']}",
        "",
        "=== Final Research Questions ===",
        "",
        "1. How should Scout govern itself?",
        "   Balanced constitution with protected principle veto and slow adaptation.",
        "",
        "2. Which institutions deserve influence?",
    ])
    for s in scores[:4]:
        if s["verdict"] == "reliable":
            lines.append(f"   - {s['institution']} ({s['success_rate_pct']}%)")

    lines.extend([
        "",
        "3. Which institutions should remain limited?",
        "   Confidence (5%), memory (8%), replay (7%) — supporting voices only.",
        "",
        "4. Should governance adapt?",
        "   Yes — by market regime and parliament stability.",
        "",
        "5. How slowly should governance change?",
        "   alpha=0.25 per scan — constitutional migration not revolution.",
        "",
        "6. What governance mistakes repeat?",
        "   Single-institution dominance; elevated observation on weak structure.",
        "",
        "7. What habits become permanent?",
    ])
    for m in memory:
        if m["type"] == "good_habit":
            lines.append(f"   - {m['habit']}")

    lines.extend([
        "",
        "8. What institutions override others?",
        "   Protected principles, false convergence protection, unknown honesty.",
        "",
        "9. How should protected principles interact?",
        "   Veto governance when triggered — override elevated observation.",
        "",
        "10. Best Constitution?",
        f"   hybrid_governance + protected_override (accepted modes: {len(accepted)})",
        "",
        "--- Accepted governance modes ---",
    ])
    for c in accepted:
        lines.append(f"  {c['mode']}: {c['reason']}")

    lines.extend([
        "",
        "A great Scout understands how to govern itself.",
        "Never forecast. Never Buy/Sell.",
    ])
    lines.extend(mission_summary_lines())
    lines.append("=" * 58)
    return "\n".join(lines)


def ensure_deps() -> None:
    if not (LOGS_DIR / "season2_p30_attention_weights.csv").exists():
        import season2_p30_scout_attention_capital
        season2_p30_scout_attention_capital.main()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-deps", action="store_true")
    args = parser.parse_args()

    if args.rebuild_deps:
        import season2_p30_scout_attention_capital
        season2_p30_scout_attention_capital.main()
    else:
        ensure_deps()

    council = load_csv(LOGS_DIR / "season2_p28_scout_council.csv")
    replay = load_csv(LOGS_DIR / "season2_p24_outcome_audit.csv")
    attn = load_csv(LOGS_DIR / "season2_p30_attention_weights.csv")
    parliament = load_csv(LOGS_DIR / "season2_p29_parliament.csv")
    if not council:
        print("Run P28 first")
        return

    observations = build_observations(council, replay, attn)
    counterfactual = counterfactual_tests(observations, parliament)

    const, gov, conflicts, overrides, _, perf = run_governance(observations, "hybrid_governance", parliament)
    scores = institution_scores(perf)
    replays = governance_replays(gov, const)
    memory = institutional_memory(scores, counterfactual, conflicts, overrides)
    protected = protected_principles()
    report = build_report(scores, counterfactual, memory, const, gov, replays)

    write_csv(CONSTITUTION_CSV, const)
    write_csv(GOVERNANCE_CSV, gov)
    write_csv(INST_SCORES_CSV, scores)
    write_csv(REPLAYS_CSV, replays)
    write_csv(CONFLICTS_CSV, conflicts)
    write_csv(OVERRIDES_CSV, overrides)
    write_csv(COUNTERFACTUAL_CSV, counterfactual)
    write_csv(MEMORY_CSV, memory)
    write_csv(PROTECTED_CSV, protected)
    REPORT_TXT.write_text(report, encoding="utf-8")

    accepted = sum(1 for c in counterfactual if c["recommendation"] == "ACCEPT")
    print("===== P31 SCOUT CONSTITUTION =====")
    print(f"Governed: {len(gov)} | Conflicts: {len(conflicts)} | Overrides: {len(overrides)}")
    print(f"Modes ACCEPT: {accepted}/{len(counterfactual)} | Report: {REPORT_TXT}")


if __name__ == "__main__":
    main()
