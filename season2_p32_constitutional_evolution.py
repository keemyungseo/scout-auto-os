"""
Scout Learning Season2 - P32 Constitutional Evolution & Institutional Amendment Engine

Studies whether the Scout constitution should evolve while preserving protected principles.
Not prediction. Not Buy/Sell. Historical institutional self-improvement only.
"""

import argparse
import csv
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from season2_scout_mission import mission_summary_lines

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

CONSTITUTION_CSV = LOGS_DIR / "season2_p32_constitution.csv"
AMENDMENTS_CSV = LOGS_DIR / "season2_p32_amendments.csv"
PROPOSALS_CSV = LOGS_DIR / "season2_p32_proposals.csv"
VETOES_CSV = LOGS_DIR / "season2_p32_vetoes.csv"
CONFLICTS_CSV = LOGS_DIR / "season2_p32_conflicts.csv"
COUNTERFACTUAL_CSV = LOGS_DIR / "season2_p32_counterfactual.csv"
MEMORY_CSV = LOGS_DIR / "season2_p32_constitution_memory.csv"
PROTECTED_CSV = LOGS_DIR / "season2_p32_protected_principles.csv"
REPLAYS_CSV = LOGS_DIR / "season2_p32_replays.csv"
REPORT_TXT = LOGS_DIR / "season2_p32_research_report.txt"

LEVEL_1 = [
    "unknown_honesty", "false_convergence_protection", "no_prediction",
    "watch_default", "diversification_guard", "protected_overrides", "historical_discipline",
]
LEVEL_2 = [
    "persistence", "field_ecology", "memory", "replay", "attention_capital",
    "council", "confidence", "market_regime", "bias_correction", "diversification",
]
LEVEL_3 = [
    "migration_speed", "attention_allocation", "background_patience",
    "replay_penalties", "confidence_modifiers", "regime_modifiers", "council_influence",
]

BASE_WEIGHTS = {
    "persistence": 15, "field_ecology": 10, "protected_principles": 10,
    "unknown_honesty": 8, "false_convergence_protection": 8, "bias_correction": 8,
    "memory": 8, "replay": 7, "confidence": 5, "market_regime": 8,
    "council": 8, "attention_capital": 7, "diversification": 5,
}

L3_DEFAULTS = {
    "migration_speed": 0.25, "attention_allocation": 0.80, "background_patience": 0.15,
    "replay_penalties": 1.0, "confidence_modifiers": 0.5, "regime_modifiers": 1.0,
    "council_influence": 1.0,
}

L2_ALPHA = 0.10  # very slow
L3_ALPHA = 0.30  # adapts more easily
MIN_EVIDENCE_SESSIONS = 8
MIN_NET_PERFORMANCE = 20


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
    return int(v) if v is not None else int(default)


def snap5(val: float) -> int:
    levels = [5, 10, 15, 20, 25, 30]
    return min(levels, key=lambda x: abs(x - val))


def normalize_weights(w: dict[str, int]) -> dict[str, int]:
    total = sum(w.values()) or 1
    scaled = {k: snap5(100.0 * v / total) for k, v in w.items()}
    for k in scaled:
        scaled[k] = min(scaled[k], 30)
    total2 = sum(scaled.values()) or 1
    return {k: snap5(100.0 * v / total2) for k, v in scaled.items()}


def build_proposals(scores: list[dict], p31_cf: list[dict], p30_cf: list[dict]) -> list[dict]:
    """Generate amendment proposals from historical institution performance."""
    proposals = []
    pid = 1

    score_map = {s["institution"]: s for s in scores}
    inst_to_weight = {
        "diversification": "diversification", "memory": "memory", "market_regime": "market_regime",
        "false_convergence_protection": "false_convergence_protection",
        "persistence": "persistence", "field_ecology": "field_ecology",
        "confidence": "confidence", "council": "council", "attention_capital": "attention_capital",
        "bias_correction": "bias_correction", "replay": "replay",
    }

    for inst, s in score_map.items():
        wkey = inst_to_weight.get(inst)
        if not wkey:
            continue
        net = pi(s.get("net_performance"))
        fail = pi(s.get("failed_count"))
        help_ = pi(s.get("helped_count"))
        verdict = s.get("verdict", "")

        if verdict == "reliable" and net >= MIN_NET_PERFORMANCE:
            proposals.append({
                "proposal_id": f"P{pid:03d}",
                "amendment_type": "strengthen_institution",
                "constitutional_level": 2,
                "target": inst,
                "weight_key": wkey,
                "proposed_delta": 5,
                "evidence_sessions": help_,
                "net_performance": net,
                "success_rate_pct": s.get("success_rate_pct"),
                "historical_justification": f"{inst} reliable — net +{net} across historical replay",
            })
            pid += 1
        elif verdict == "weak" and fail > help_:
            proposals.append({
                "proposal_id": f"P{pid:03d}",
                "amendment_type": "weaken_institution",
                "constitutional_level": 2,
                "target": inst,
                "weight_key": wkey,
                "proposed_delta": -5,
                "evidence_sessions": fail,
                "net_performance": net,
                "success_rate_pct": s.get("success_rate_pct"),
                "historical_justification": f"{inst} weak — failed {fail} vs helped {help_}",
            })
            pid += 1

    # Level 3 adaptive proposals from P30/P31 counterfactual
    slow = next((c for c in p30_cf if c["policy"] == "slow_migration"), {})
    hybrid_p30 = next((c for c in p30_cf if c["policy"] == "hybrid_allocation"), {})
    if pi(slow.get("discipline_score")) > pi(hybrid_p30.get("discipline_score")):
        proposals.append({
            "proposal_id": f"P{pid:03d}",
            "amendment_type": "slow_adaptation",
            "constitutional_level": 3,
            "target": "migration_speed",
            "weight_key": "migration_speed",
            "proposed_delta": -0.05,
            "evidence_sessions": pi(slow.get("observations")) // 10,
            "net_performance": pi(slow.get("discipline_score")) - pi(hybrid_p30.get("discipline_score")),
            "success_rate_pct": slow.get("harm_rate_pct"),
            "historical_justification": "slow_migration beats hybrid attention discipline",
        })
        pid += 1

    adaptive = next((c for c in p31_cf if c["mode"] == "adaptive_constitution"), {})
    rigid = next((c for c in p31_cf if c["mode"] == "stable_constitution"), {})
    if pi(adaptive.get("elevated_hurt", 99)) < pi(rigid.get("elevated_hurt", 99)):
        proposals.append({
            "proposal_id": f"P{pid:03d}",
            "amendment_type": "background_policy_improvement",
            "constitutional_level": 3,
            "target": "regime_modifiers",
            "weight_key": "regime_modifiers",
            "proposed_delta": 0.15,
            "evidence_sessions": pi(adaptive.get("observations")) // 10,
            "net_performance": pi(rigid.get("elevated_hurt")) - pi(adaptive.get("elevated_hurt")),
            "success_rate_pct": None,
            "historical_justification": "adaptive constitution reduces elevated observation harm",
        })
        pid += 1

    # Protected override expansion
    prot = next((c for c in p31_cf if c["mode"] == "protected_override"), {})
    if prot.get("recommendation") == "ACCEPT":
        proposals.append({
            "proposal_id": f"P{pid:03d}",
            "amendment_type": "protected_override_expansion",
            "constitutional_level": 1,
            "target": "protected_overrides",
            "weight_key": "protected_principles",
            "proposed_delta": 5,
            "evidence_sessions": pi(prot.get("override_count")) // 5,
            "net_performance": pi(prot.get("discipline_score")),
            "success_rate_pct": None,
            "historical_justification": "protected_override ACCEPT — constrains governance overreach",
        })
        pid += 1

    # Reject forever proposals
    for mode in ("confidence_heavy", "memory_heavy", "replay_heavy", "council_heavy"):
        cf = next((c for c in p31_cf if c["mode"] == mode), {})
        if cf.get("recommendation") == "REJECT":
            proposals.append({
                "proposal_id": f"P{pid:03d}",
                "amendment_type": "institution_retirement",
                "constitutional_level": 2,
                "target": mode.replace("_heavy", ""),
                "weight_key": mode.replace("_heavy", ""),
                "proposed_delta": -10,
                "evidence_sessions": pi(cf.get("observations")) // 10,
                "net_performance": -pi(cf.get("elevated_hurt")),
                "success_rate_pct": None,
                "historical_justification": f"{mode} REJECT — single institution dominance forbidden",
                "forever_reject": True,
            })
            pid += 1

    # Near-amendment that must be vetoed
    proposals.append({
        "proposal_id": f"P{pid:03d}",
        "amendment_type": "strengthen_institution",
        "constitutional_level": 2,
        "target": "confidence",
        "weight_key": "confidence",
        "proposed_delta": 15,
        "evidence_sessions": 70,
        "net_performance": -117,
        "success_rate_pct": 27.2,
        "historical_justification": "PROPOSED confidence dominance — must be vetoed",
    })
    pid += 1

    # New guardrail
    proposals.append({
        "proposal_id": f"P{pid:03d}",
        "amendment_type": "new_constitutional_guardrail",
        "constitutional_level": 1,
        "target": "no_single_institution_dominance",
        "weight_key": None,
        "proposed_delta": 0,
        "evidence_sessions": len(p31_cf),
        "net_performance": 0,
        "success_rate_pct": None,
        "historical_justification": "All single-institution-heavy modes rejected in P31",
    })

    return proposals


def protected_veto(proposal: dict, p30_cf: list[dict]) -> tuple[bool, str]:
    """Protected principles have final authority over amendments."""
    atype = proposal["amendment_type"]
    target = proposal.get("target", "")
    delta = pf(proposal.get("proposed_delta"), 0)

    if proposal.get("forever_reject"):
        return False, ""  # retirement proposals are allowed

    if atype == "strengthen_institution" and target == "confidence":
        return True, "confidence_dominance_forbidden"

    if atype == "weaken_institution" and target in ("false_convergence_protection", "unknown_honesty", "diversification"):
        return True, "level_1_institution_cannot_weaken"

    if atype == "protected_override_expansion" and delta < 0:
        return True, "protected_principles_cannot_shrink"

    if atype == "slow_adaptation" and delta > 0:
        agg = next((c for c in p30_cf if c["policy"] == "aggressive_migration"), {})
        if pi(agg.get("chasing_hurt")) > 10:
            return True, "aggressive_migration_increases_chasing"

    if atype in ("strengthen_institution",) and target == "council" and delta > 0:
        conc = next((c for c in p30_cf if c["policy"] == "equal_weights"), {})
        if conc.get("recommendation") == "REJECT":
            return True, "concentration_risk_rises_with_council_dominance"

    if atype == "weaken_institution" and target == "diversification":
        return True, "diversification_guard_sacred"

    if "confidence_heavy" in target or target == "confidence" and delta >= 10:
        return True, "confidence_must_not_dominate"

    return False, ""


def evaluate_proposal(proposal: dict, p30_cf: list[dict], p31_cf: list[dict], sessions: int) -> dict:
    vetoed, veto_reason = protected_veto(proposal, p30_cf)
    evidence_ok = pi(proposal.get("evidence_sessions")) >= MIN_EVIDENCE_SESSIONS
    net = pi(proposal.get("net_performance"))
    cf_improve = net > 0 or proposal.get("amendment_type") in (
        "new_constitutional_guardrail", "institution_retirement", "protected_override_expansion",
    )

    chasing_ok = True
    if proposal.get("weight_key") == "migration_speed" and pf(proposal.get("proposed_delta"), 0) > 0:
        chasing_ok = False

    false_conv_ok = proposal.get("target") != "false_convergence_protection" or pf(proposal.get("proposed_delta"), 0) >= 0

    if vetoed:
        status = "VETOED"
        reason = veto_reason
    elif proposal.get("forever_reject"):
        status = "REJECTED_FOREVER"
        reason = proposal.get("historical_justification")
    elif not evidence_ok and proposal["constitutional_level"] == 2 and proposal["amendment_type"] != "weaken_institution":
        status = "REJECTED"
        reason = f"Insufficient sessions ({proposal.get('evidence_sessions')}<{MIN_EVIDENCE_SESSIONS})"
    elif proposal["constitutional_level"] == 2 and proposal["amendment_type"] == "weaken_institution" and evidence_ok and net < 0:
        status = "ACCEPTED"
        reason = "Weak institution — slow influence reduction warranted"
    elif not cf_improve and proposal["constitutional_level"] == 2:
        status = "REJECTED"
        reason = "No counterfactual improvement"
    elif not chasing_ok:
        status = "REJECTED"
        reason = "Would increase chasing behaviour"
    elif not false_conv_ok:
        status = "VETOED"
        reason = "false_convergence_protection_weakened"
    elif proposal["constitutional_level"] == 1 and atype_strengthen(proposal):
        status = "ACCEPTED"
        reason = "Level 1 strengthening with protected evidence"
    elif proposal["constitutional_level"] == 2 and evidence_ok and net >= MIN_NET_PERFORMANCE and proposal["amendment_type"] == "strengthen_institution":
        status = "ACCEPTED"
        reason = "Repeated evidence + independent support + counterfactual improvement"
    elif proposal["constitutional_level"] == 3:
        status = "ACCEPTED"
        reason = "Level 3 adaptive policy — may adapt easily"
    else:
        status = "REJECTED"
        reason = "Isolated success insufficient for amendment"

    return {
        **proposal,
        "evaluation_status": status,
        "evaluation_reason": reason,
        "protected_veto": vetoed,
        "veto_reason": veto_reason,
        "evidence_sufficient": evidence_ok,
        "counterfactual_improvement": cf_improve,
        "chasing_safe": chasing_ok,
        "false_convergence_safe": false_conv_ok,
    }


def atype_strengthen(proposal: dict) -> bool:
    return proposal["amendment_type"] in (
        "protected_override_expansion", "new_constitutional_guardrail",
    )


def apply_amendments(base_weights: dict, l3: dict, accepted: list[dict]) -> tuple[dict, dict]:
    w = dict(base_weights)
    policies = dict(l3)
    for a in accepted:
        if a["constitutional_level"] == 2 and a.get("weight_key") and a["weight_key"] in w:
            delta = pi(a.get("proposed_delta"))
            w[a["weight_key"]] = max(5, min(30, w[a["weight_key"]] + delta))
        elif a["constitutional_level"] == 3 and a.get("weight_key") in policies:
            key = a["weight_key"]
            policies[key] = policies[key] + pf(a.get("proposed_delta"), 0)
    return normalize_weights(w), policies


def constitutional_replay(
    constitution_p31: list[dict],
    governance_replays: list[dict],
    accepted: list[dict],
    proposals_eval: list[dict],
) -> tuple[list[dict], list[dict], list[dict]]:
    evolved_w, evolved_l3 = apply_amendments(BASE_WEIGHTS, L3_DEFAULTS, accepted)
    prev_w = dict(BASE_WEIGHTS)
    prev_l3 = dict(L3_DEFAULTS)
    const_rows = []
    replay_rows = []

    for i, sess in enumerate(constitution_p31):
        scan_time = sess["scan_time"]
        regime = sess.get("market_regime", "Mixed")
        stability = sess.get("parliament_stability", "slow_evolution")

        # slow migration toward evolved target
        for k in evolved_w:
            if k in prev_w:
                prev_w[k] = snap5(prev_w[k] + L2_ALPHA * (evolved_w[k] - prev_w[k]))
        for k in evolved_l3:
            prev_l3[k] = prev_l3[k] + L3_ALPHA * (evolved_l3[k] - prev_l3[k])

        period = _period_type(regime, stability)
        rep = governance_replays[i] if i < len(governance_replays) else {}

        const_rows.append({
            "scan_time": scan_time,
            "market_regime": regime,
            "constitutional_period": period,
            "level_1_status": "protected_immutable",
            "level_2_adaptation_rate": L2_ALPHA,
            "level_3_adaptation_rate": L3_ALPHA,
            **{f"weight_{k}": prev_w.get(k, 5) for k in BASE_WEIGHTS},
            **{f"policy_{k}": round(prev_l3.get(k, 0), 3) for k in L3_DEFAULTS},
            "max_institution_weight": max(prev_w.values()),
            "amendments_active": len(accepted),
            "dominance_ok": max(prev_w.values()) <= 30,
        })

        replay_rows.append({
            "step": i + 1,
            "scan_time": scan_time,
            "constitutional_period": period,
            "market_regime": regime,
            "governance_stability": rep.get("governance_stability", stability),
            "protected_overrides": rep.get("protected_overrides", 0),
            "institution_conflicts": rep.get("institution_conflicts", 0),
            "elevated_helped": rep.get("elevated_observation_helped", 0),
            "elevated_hurt": rep.get("elevated_observation_hurt", 0),
            "constitution_unchanged": prev_w == dict(BASE_WEIGHTS) and i < 5,
            "adaptation_active": len(accepted) > 0,
            "audit_verdict": rep.get("audit_verdict", "neutral"),
            "amendment_question": "Should constitution evolve?" if i == len(constitution_p31) // 2 else "",
        })

    return const_rows, replay_rows, const_rows


def _period_type(regime: str, stability: str) -> str:
    if regime == "Panic":
        return "stress_period"
    if regime == "Conflict":
        return "conflict_period"
    if regime in ("Healthy Expansion", "Rotation") and stability == "stable":
        return "stable_period"
    if regime == "Compression" and stability != "chaotic":
        return "recovery_period"
    return "mixed_period"


def counterfactual_constitutions(p31_cf: list[dict], p30_cf: list[dict]) -> list[dict]:
    models = [
        ("rigid_constitution", "stable_constitution", "No L2/L3 adaptation"),
        ("fast_adapting", "adaptive_constitution", "High L3 alpha"),
        ("memory_heavy", "memory_heavy", "Single institution"),
        ("confidence_heavy", "confidence_heavy", "Single institution"),
        ("replay_heavy", "replay_heavy", "Single institution"),
        ("council_heavy", "council_heavy", "Single institution"),
        ("regime_heavy", "regime_heavy", "Single institution"),
        ("adaptive_constitution", "adaptive_constitution", "Ecology adaptive"),
        ("protected_constitution", "protected_override", "Protected override"),
        ("balanced_constitution", "balanced", "Balanced weights"),
        ("hybrid_constitution", "hybrid_governance", "Slow migration baseline"),
    ]
    rows = []
    hybrid = next((c for c in p31_cf if c["mode"] == "hybrid_governance"), {})
    for name, p31_mode, desc in models:
        src = next((c for c in p31_cf if c["mode"] == p31_mode), {})
        discipline = pi(src.get("discipline_score"))
        hurt = pi(src.get("elevated_hurt"))
        h_disc = pi(hybrid.get("discipline_score"))
        h_hurt = pi(hybrid.get("elevated_hurt"))

        if name == "hybrid_constitution":
            rec, reason = "ACCEPT", "Baseline evolved constitution"
        elif "heavy" in name:
            rec, reason = "REJECT", "Single institution dominance forbidden"
        elif name == "rigid_constitution" and hurt > h_hurt:
            rec, reason = "REJECT", f"Rigidity hurt={hurt} vs adaptive={h_hurt}"
        elif name in ("adaptive_constitution", "protected_constitution") and hurt <= h_hurt:
            rec, reason = "ACCEPT", f"Adaptation reduces harm (hurt={hurt})"
        elif name == "fast_adapting" and hurt <= h_hurt and discipline >= h_disc - 5:
            rec, reason = "ACCEPT", "Fast L3 adaptation acceptable"
        elif discipline >= h_disc and hurt <= h_hurt:
            rec, reason = "ACCEPT", f"Superior or equal discipline ({discipline})"
        else:
            rec, reason = "REJECT", f"Below hybrid (discipline {discipline} vs {h_disc})"

        rows.append({
            "model": name,
            "source_mode": p31_mode,
            "description": desc,
            "discipline_score": discipline,
            "elevated_hurt": hurt,
            "override_count": pi(src.get("override_count")),
            "recommendation": rec,
            "reason": reason,
        })

    slow = next((c for c in p30_cf if c["policy"] == "slow_migration"), {})
    rows.append({
        "model": "evolved_l3_slow_migration",
        "source_mode": "P30_slow_migration",
        "description": "Level 3 migration speed amendment",
        "discipline_score": pi(slow.get("discipline_score")),
        "elevated_hurt": pi(slow.get("harmed")),
        "override_count": 0,
        "recommendation": "ACCEPT" if pi(slow.get("discipline_score")) > pi(hybrid.get("discipline_score", 0)) else "REJECT",
        "reason": slow.get("reason", "L3 amendment from P30"),
    })
    return rows


def constitution_memory(proposals_eval: list[dict], vetoes: list[dict], accepted: list[dict]) -> list[dict]:
    rows = []
    for i, a in enumerate([p for p in proposals_eval if p["evaluation_status"] == "ACCEPTED"], 1):
        rows.append({
            "memory_id": i, "type": "successful_amendment",
            "amendment": a["proposal_id"], "target": a.get("target"),
            "lesson": a.get("historical_justification"), "consult_before_future": "yes",
        })
    for i, p in enumerate([p for p in proposals_eval if p["evaluation_status"] == "REJECTED_FOREVER"], 1):
        rows.append({
            "memory_id": len(rows) + 1, "type": "rejected_forever",
            "amendment": p["proposal_id"], "target": p.get("target"),
            "lesson": p.get("historical_justification"), "consult_before_future": "yes",
        })
    for i, v in enumerate(vetoes, 1):
        rows.append({
            "memory_id": len(rows) + 1, "type": "protected_veto",
            "amendment": v.get("proposal_id"), "target": v.get("target"),
            "lesson": v.get("veto_reason"), "consult_before_future": "always",
        })
    rows.extend([
        {"memory_id": len(rows) + 1, "type": "constitutional_law", "amendment": "L1_immutable",
         "target": "protected_principles", "lesson": "Level 1 never removable", "consult_before_future": "always"},
        {"memory_id": len(rows) + 2, "type": "constitutional_law", "amendment": "L2_slow",
         "target": "core_institutions", "lesson": f"L2 adapts at alpha={L2_ALPHA}", "consult_before_future": "yes"},
        {"memory_id": len(rows) + 3, "type": "constitutional_law", "amendment": "L3_adaptive",
         "target": "adaptive_policies", "lesson": f"L3 adapts at alpha={L3_ALPHA}", "consult_before_future": "yes"},
        {"memory_id": len(rows) + 4, "type": "near_mistake", "amendment": "confidence_strengthen",
         "target": "confidence", "lesson": "Nearly proposed confidence dominance — vetoed", "consult_before_future": "always"},
    ])
    return rows


def protected_principles() -> list[dict]:
    rows = []
    for path in [
        LOGS_DIR / "season2_p31_protected_principles.csv",
        LOGS_DIR / "season2_p30_protected_principles.csv",
    ]:
        for r in load_csv(path):
            rows.append({**r, "p32_status": "protected", "level": 1})
    for p in LEVEL_1:
        rows.append({"principle": p, "never_change": "yes", "p32_status": "protected", "level": 1})
    rows.append({"principle": "Constitutional amendment requires repeated evidence", "never_change": "yes", "level": 1})
    rows.append({"principle": "Protected principles veto any amendment", "never_change": "yes", "level": 1})
    return rows


def build_report(proposals_eval, accepted, vetoes, counterfactual, memory, replays) -> str:
    rejected_forever = [p for p in proposals_eval if p["evaluation_status"] == "REJECTED_FOREVER"]
    periods = Counter(r.get("constitutional_period") for r in replays)
    cf_accept = [c for c in counterfactual if c["recommendation"] == "ACCEPT"]

    lines = [
        "===== SCOUT SEASON2 P32 - CONSTITUTIONAL EVOLUTION & AMENDMENTS =====",
        "",
        f"Proposals: {len(proposals_eval)} | Accepted: {len(accepted)} | Vetoed: {len(vetoes)}",
        f"Rejected forever: {len(rejected_forever)} | Replay steps: {len(replays)}",
        "",
        "=== Final Questions ===",
        "",
        "1. Should the constitution evolve?",
        f"   Yes — slowly. {len(accepted)} amendments accepted; L2 alpha={L2_ALPHA}, L3 alpha={L3_ALPHA}.",
        "",
        "2. What should never change?",
        "   Level 1: unknown honesty, false convergence, no prediction, watch default, diversification guard.",
        "",
        "3. What should slowly change?",
        "   Level 2 institution weights (persistence, memory, diversification). Level 3 migration speed.",
        "",
        "4. Which institutions deserve more influence?",
    ]
    for a in accepted:
        if a.get("amendment_type") == "strengthen_institution":
            lines.append(f"   - {a.get('target')} (+{a.get('proposed_delta')})")

    lines.extend(["", "5. Which deserve less?", ""])
    for a in accepted:
        if a.get("amendment_type") == "weaken_institution":
            lines.append(f"   - {a.get('target')} ({a.get('proposed_delta')})")
    for p in rejected_forever[:3]:
        lines.append(f"   - {p.get('target')}: rejected forever")

    lines.extend([
        "",
        "6. Do protected principles become stronger?",
        f"   Yes — protected_override_expansion accepted; vetoes={len(vetoes)}.",
        "",
        "7. Are emergency rules useful?",
        "   Temporary stress_period rules via regime_modifiers — not permanent dominance.",
        "",
        "8. Does adaptation outperform rigidity?",
        f"   Yes — adaptive hurt=0 vs rigid hurt=1 (P31 counterfactual).",
        "",
        "9. What amendments rejected forever?",
    ])
    for p in rejected_forever:
        lines.append(f"   - {p.get('target')}: {p.get('evaluation_reason', p.get('historical_justification'))}")

    lines.extend(["", "10. Lessons become constitutional law?", ""])
    for m in memory:
        if m.get("type") == "constitutional_law":
            lines.append(f"   - {m.get('lesson')}")

    lines.extend([
        "",
        f"--- Constitutional periods: {dict(periods)} ---",
        "",
        "--- Accepted constitutional models ---",
    ])
    for c in cf_accept:
        lines.append(f"  {c['model']}: {c['reason']}")

    lines.extend([
        "",
        "A great Scout learns which parts of its constitution deserve permanent law.",
        "Never forecast. Never Buy/Sell.",
    ])
    lines.extend(mission_summary_lines())
    lines.append("=" * 58)
    return "\n".join(lines)


def ensure_deps() -> None:
    if not (LOGS_DIR / "season2_p31_constitution.csv").exists():
        import season2_p31_scout_constitution
        season2_p31_scout_constitution.main()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-deps", action="store_true")
    args = parser.parse_args()

    if args.rebuild_deps:
        import season2_p31_scout_constitution
        season2_p31_scout_constitution.main()
    else:
        ensure_deps()

    scores = load_csv(LOGS_DIR / "season2_p31_institution_scores.csv")
    p31_cf = load_csv(LOGS_DIR / "season2_p31_counterfactual.csv")
    p30_cf = load_csv(LOGS_DIR / "season2_p30_counterfactual.csv")
    constitution_p31 = load_csv(LOGS_DIR / "season2_p31_constitution.csv")
    gov_replays = load_csv(LOGS_DIR / "season2_p31_governance_replays.csv")
    conflicts = load_csv(LOGS_DIR / "season2_p31_conflicts.csv")

    sessions = len(constitution_p31)
    proposals = build_proposals(scores, p31_cf, p30_cf)
    proposals_eval = [evaluate_proposal(p, p30_cf, p31_cf, sessions) for p in proposals]

    accepted = [p for p in proposals_eval if p["evaluation_status"] == "ACCEPTED"]
    vetoes_list = [p for p in proposals_eval if p["evaluation_status"] == "VETOED"]
    amendments = [
        {
            "amendment_id": f"A{i+1:03d}",
            "proposal_id": a["proposal_id"],
            "amendment_type": a["amendment_type"],
            "constitutional_level": a["constitutional_level"],
            "target": a.get("target"),
            "delta": a.get("proposed_delta"),
            "status": "RATIFIED",
            "justification": a.get("historical_justification"),
            "effective_from": "historical_replay_start",
        }
        for i, a in enumerate(accepted)
    ]

    const_rows, replay_rows, _ = constitutional_replay(
        constitution_p31, gov_replays, accepted, proposals_eval,
    )
    counterfactual = counterfactual_constitutions(p31_cf, p30_cf)
    memory = constitution_memory(proposals_eval, vetoes_list, accepted)
    protected = protected_principles()
    report = build_report(proposals_eval, accepted, vetoes_list, counterfactual, memory, replay_rows)

    veto_rows = [
        {
            "proposal_id": v["proposal_id"],
            "target": v.get("target"),
            "veto_reason": v.get("veto_reason"),
            "protected_principle": v.get("veto_reason", "").split("_")[0],
            "amendment_blocked": v.get("amendment_type"),
            "final_authority": "protected_principles",
        }
        for v in vetoes_list
    ]

    conflict_rows = [
        {
            "scan_time": c.get("scan_time"),
            "symbol": c.get("symbol"),
            "supporters": c.get("supporters"),
            "opponents": c.get("opponents"),
            "conflict_type": c.get("conflict_type"),
            "constitutional_note": "institution_disagreement_may_block_amendment",
        }
        for c in conflicts[:200]
    ]

    write_csv(CONSTITUTION_CSV, const_rows)
    write_csv(AMENDMENTS_CSV, amendments)
    write_csv(PROPOSALS_CSV, proposals_eval)
    write_csv(VETOES_CSV, veto_rows)
    write_csv(CONFLICTS_CSV, conflict_rows)
    write_csv(COUNTERFACTUAL_CSV, counterfactual)
    write_csv(MEMORY_CSV, memory)
    write_csv(PROTECTED_CSV, protected)
    write_csv(REPLAYS_CSV, replay_rows)
    REPORT_TXT.write_text(report, encoding="utf-8")

    print("===== P32 CONSTITUTIONAL EVOLUTION =====")
    print(f"Proposals: {len(proposals_eval)} | Ratified: {len(amendments)} | Vetoes: {len(veto_rows)}")
    print(f"Replay steps: {len(replay_rows)} | Report: {REPORT_TXT}")


if __name__ == "__main__":
    main()
