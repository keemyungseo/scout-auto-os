"""
Scout Learning Season2 - Validation 001 Blind Future Simulation

Scores symbol universe using ONLY discovered P39-P62 laws and pre-freeze data.
STRICT NO_ACTION | NO_API | NO_PRICE | NO_LOOKAHEAD.
"""

from __future__ import annotations

import argparse
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

from season2_p37_scout_decision_hierarchy import load_csv, pf, pi, write_csv
from season2_p60_scout_attention_field import infer_attention_score
from season2_scout_mission import mission_summary_lines

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

OBSERVATION_TIME_KST = "2026-06-19 11:00:00"
KST = timezone(timedelta(hours=9))

SCORES_CSV = LOGS_DIR / "validation001_scores.csv"
TOP2_CSV = LOGS_DIR / "validation001_top2.csv"
REASONING_TXT = LOGS_DIR / "validation001_reasoning.txt"
STATE_PRED_CSV = LOGS_DIR / "validation001_state_prediction.csv"
PROB_CSV = LOGS_DIR / "validation001_probability.csv"

# --- Discovered law coefficients (P59-P62, frozen) ---
CRITICAL_MASS = 31.45
CRITICAL_BELIEF = 40.81
CRITICAL_ATTENTION = 4.49
SYNC_BELIEF = 0.4416
SYNC_MEMORY = 2.5737
NARR_MEMORY = 1.920
NARR_POTENTIAL = 0.124
NARR_API = 0.143
NARR_ECOLOGY = -3.487
COLLAPSE_ATTENTION = -0.0160
COLLAPSE_ENTROPY = 0.4629
TREND_REPLACEMENT = 0.3138
TREND_FLOW = 0.0031
MEMORY_KERNEL = 0.5


def clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


def t0_only(rows: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in rows:
        if pi(r.get("checkpoint_hour")) != 0 and r.get("checkpoint") != "T+0h":
            continue
        sym = r["symbol"]
        out[sym] = r
    return out


def load_t0_process() -> dict[str, dict]:
    """Load T+0 checkpoint only — no future checkpoints."""
    order = t0_only(load_csv(LOGS_DIR / "season2_p49_order_parameter.csv"))
    belief = t0_only(load_csv(LOGS_DIR / "season2_p58_belief_field.csv"))
    narrative = t0_only(load_csv(LOGS_DIR / "season2_p59_narrative_field.csv"))
    attention = t0_only(load_csv(LOGS_DIR / "season2_p60_attention_field.csv"))
    sync = t0_only(load_csv(LOGS_DIR / "season2_p62_sync_field.csv"))
    phase = t0_only(load_csv(LOGS_DIR / "season2_p62_phase_transition.csv"))
    future = t0_only(load_csv(LOGS_DIR / "season2_p52_future_distribution.csv"))
    ecology = t0_only(load_csv(LOGS_DIR / "season2_p57_ecology_entropy.csv"))
    dynamics = t0_only(load_csv(LOGS_DIR / "season2_p57_population_dynamics.csv"))
    migration = t0_only(load_csv(LOGS_DIR / "season2_p61_attention_migration.csv"))
    epr = t0_only(load_csv(LOGS_DIR / "season2_p54_epr.csv"))
    goals = t0_only(load_csv(LOGS_DIR / "season2_p54_goal_distribution.csv"))

    symbols = set(order) | set(belief) | set(narrative)
    process: dict[str, dict] = {}
    for sym in symbols:
        o = order.get(sym, {})
        b = belief.get(sym, {})
        n = narrative.get(sym, {})
        a = attention.get(sym, {})
        s = sync.get(sym, {})
        p = phase.get(sym, {})
        f = future.get(sym, {})
        e = ecology.get(sym, {})
        d = dynamics.get(sym, {})
        m = migration.get(sym, {})
        ep = epr.get(sym, {})
        g = goals.get(sym, {})
        rec = {
            "symbol": sym,
            "data_source": "process_t0",
            "p39_state": o.get("p39_state", "Observation"),
            "Potential": pf(o.get("var_Potential")) or 0.0,
            "API": pf(o.get("var_API")) or 0.0,
            "Quality": pf(o.get("var_Quality")) or 0.0,
            "Flow": pf(o.get("var_FlowVelocity")) or 0.0,
            "Participation": (pf(o.get("var_FlowVelocity")) or 0) * (pf(o.get("var_Persistence")) or 0) / 100.0,
            "OrderParameter": pf(o.get("order_parameter_score")) or 0.0,
            "BeliefConsensus": pf(b.get("belief_consensus")) or 0.0,
            "NarrativeScore": pf(n.get("narrative_score")) or 0.0,
            "AttentionScore": pf(a.get("attention_score")) or 0.0,
            "SynchronizationScore": pf(s.get("synchronization_score")) or 0.0,
            "PhaseLabel": p.get("phase_label", "NearCritical"),
            "CollapseRisk": pf(f.get("prob_collapse")) or 0.0,
            "Entropy": pf(f.get("future_entropy")) or 0.0,
            "EcologyEntropy": pf(e.get("ecology_entropy")) or 0.0,
            "ReplacementRate": pf(d.get("replacement_rate")) or 1.0,
            "MigrationRate": pf(m.get("migration_rate")) or 0.0,
            "NetMigration": pf(m.get("net_attention")) or 0.0,
            "EPR": pf(ep.get("EPR")) or 0.0,
            "GoalConcentration": pf(g.get("goal_concentration")) or 0.5,
            "Memory": MEMORY_KERNEL,
        }
        if rec["AttentionScore"] <= 0:
            rec["AttentionScore"] = infer_attention_score(rec, 1.0)
        if rec["SynchronizationScore"] <= 0:
            rec["SynchronizationScore"] = clamp(
                SYNC_BELIEF * rec["BeliefConsensus"] + SYNC_MEMORY * rec["Memory"] * 10
            )
        process[sym] = rec
    return process


def proxy_from_institution(row: dict) -> dict:
    """Map P37 institution scan to latent process proxies (no future checkpoints)."""
    memory = pf(row.get("memory_score")) or 35.0
    belief = (pf(row.get("false_convergence_protection_score")) or 50) * 0.6 + (
        pf(row.get("unknown_honesty_score")) or 50
    ) * 0.4
    diversification = pf(row.get("diversification_score")) or 50.0
    discipline = pf(row.get("discipline_score")) or 0.0
    hierarchy = pf(row.get("hierarchy_score")) or 0.0
    ecology = (100.0 - diversification) / 25.0
    entropy = (100.0 - (pf(row.get("unknown_honesty_score")) or 50)) / 100.0 * 2.0
    attention = clamp(max(0.0, discipline + 20.0))
    participation = clamp(max(0.0, discipline + 15.0))
    flow = clamp(50.0 + hierarchy * 80.0)
    narrative = clamp((memory + belief) / 2.0)
    sync = clamp(SYNC_BELIEF * belief + SYNC_MEMORY * (memory / 10.0))
    collapse = 0.35 if row.get("vetoed") == "True" else 0.12
    return {
        "symbol": row["symbol"],
        "data_source": "institution_proxy",
        "p39_state": "Observation",
        "Potential": memory * 0.8,
        "API": belief * 0.7,
        "Quality": diversification * 0.6,
        "Flow": flow,
        "Participation": participation,
        "OrderParameter": clamp(50 + hierarchy * 40),
        "BeliefConsensus": belief,
        "NarrativeScore": narrative,
        "AttentionScore": attention,
        "SynchronizationScore": sync,
        "PhaseLabel": "NearCritical" if sync < CRITICAL_MASS else "LocalAlignment",
        "CollapseRisk": collapse,
        "Entropy": entropy,
        "EcologyEntropy": ecology,
        "ReplacementRate": 1.0,
        "MigrationRate": 0.0,
        "NetMigration": 0.0,
        "EPR": clamp(100 - abs(discipline)),
        "GoalConcentration": 0.45,
        "Memory": memory / 100.0,
        "institution_grade": row.get("institution_grade", ""),
        "hierarchy_stance": row.get("hierarchy_stance", ""),
    }


def apply_discovered_laws(r: dict) -> dict:
    """Combine ONLY discovered law relationships into components."""
    memory_scaled = r["Memory"] * 10.0 if r["Memory"] <= 1 else r["Memory"]
    one_minus_entropy = max(0.0, 1.0 - r["Entropy"] * 0.1)

    sync_law = SYNC_BELIEF * r["BeliefConsensus"] + SYNC_MEMORY * (memory_scaled / 10.0)
    narr_law = (
        NARR_MEMORY * memory_scaled
        + NARR_POTENTIAL * r["Potential"]
        + NARR_API * r["API"]
        + NARR_ECOLOGY * r["EcologyEntropy"]
    )
    collapse_law = COLLAPSE_ATTENTION * r["AttentionScore"] + COLLAPSE_ENTROPY * one_minus_entropy
    persist_law = TREND_REPLACEMENT * r["ReplacementRate"] * 100 + TREND_FLOW * r["Flow"]

    return {
        "sync_component": sync_law,
        "narrative_component": narr_law,
        "collapse_component": collapse_law,
        "persistence_component": persist_law,
    }


def compute_scout_score(r: dict) -> tuple[float, float, dict]:
    laws = apply_discovered_laws(r)
    raw = (
        0.30 * laws["sync_component"]
        + 0.25 * laws["narrative_component"]
        + 0.25 * laws["persistence_component"]
        - 0.10 * laws["collapse_component"] * 100
        + 0.10 * r["AttentionScore"]
    )
    score = clamp(raw)

    # Confidence from agreement among law components
    comps = [laws["sync_component"], laws["narrative_component"], laws["persistence_component"]]
    mean_c = sum(comps) / len(comps)
    spread = sum(abs(c - mean_c) for c in comps) / max(mean_c, 1.0)
    confidence = clamp(100 - spread * 30, 20, 95)

    # Penalty for high collapse / veto
    if r["CollapseRisk"] > 0.25:
        score *= 0.85
        confidence *= 0.9
    return score, confidence, laws


def lifecycle_label(r: dict) -> str:
    sync = r["SynchronizationScore"]
    if sync >= CRITICAL_MASS * 1.2:
        return "peak"
    if sync >= CRITICAL_MASS * 0.8:
        return "growth"
    if r["BeliefConsensus"] >= CRITICAL_BELIEF:
        return "maintenance"
    if r["CollapseRisk"] > 0.2:
        return "decay"
    return "birth"


def predict_states(r: dict, laws: dict) -> dict[str, str]:
    sync = r["SynchronizationScore"]
    belief = r["BeliefConsensus"]
    d_sync = laws["sync_component"] - sync
    near_critical = abs(sync - CRITICAL_MASS) < 8
    rising = d_sync > 0 or belief >= CRITICAL_BELIEF * 0.9

    def label_2h() -> str:
        if sync >= CRITICAL_MASS and belief >= CRITICAL_BELIEF:
            return "TrendBirth"
        if near_critical and rising:
            return "NearCritical"
        if r["CollapseRisk"] > 0.2:
            return "CollapseRiskIncrease"
        if r["NetMigration"] > 0:
            return "Migration"
        return "Neutral"

    def label_4h() -> str:
        base = label_2h()
        if base == "TrendBirth":
            return "Expansion"
        if sync >= CRITICAL_MASS:
            return "Persistence"
        if r["EcologyEntropy"] > 2.0:
            return "Fragmentation"
        if base == "NearCritical":
            return "LockedTrend"
        return base

    def label_6h() -> str:
        if sync >= CRITICAL_MASS * 1.1:
            return "LockedTrend"
        if r["CollapseRisk"] > 0.25:
            return "Fragmentation"
        if r["MigrationRate"] > 0.05:
            return "Rotation"
        if label_4h() == "Persistence":
            return "Persistence"
        if sync < CRITICAL_MASS * 0.7:
            return "Recovery"
        return "Neutral"

    return {"2h": label_2h(), "4h": label_4h(), "6h": label_6h()}


def transition_probabilities(r: dict, horizon: str) -> dict[str, float]:
    sync = r["SynchronizationScore"]
    belief = r["BeliefConsensus"]
    collapse = r["CollapseRisk"]
    mig = r["MigrationRate"]

    if horizon == "2h":
        weights = {
            "Persistence": max(0, sync - 20) + belief * 0.3,
            "TrendBirth": max(0, CRITICAL_MASS - sync + 5) if belief > 35 else 0,
            "NearCritical": max(0, 40 - abs(sync - CRITICAL_MASS)),
            "Fragmentation": collapse * 80 + max(0, r["EcologyEntropy"] - 1) * 10,
            "Recovery": max(0, 30 - sync),
            "Migration": mig * 100,
            "CollapseRiskIncrease": collapse * 60,
            "Neutral": 15,
        }
    elif horizon == "4h":
        weights = {
            "Persistence": sync * 0.8 + belief * 0.2,
            "Expansion": max(0, sync - CRITICAL_MASS + 10),
            "LockedTrend": max(0, sync - CRITICAL_MASS) * 0.5,
            "Fragmentation": collapse * 100 + r["EcologyEntropy"] * 5,
            "Recovery": max(0, 35 - sync),
            "Rotation": mig * 120,
            "CollapseRiskIncrease": collapse * 50,
            "Neutral": 10,
        }
    else:
        weights = {
            "Persistence": sync * 0.6,
            "LockedTrend": max(0, sync - CRITICAL_MASS),
            "Fragmentation": collapse * 90 + max(0, sync - 50) * 0.3,
            "Recovery": max(0, 40 - sync),
            "Rotation": mig * 100 + 5,
            "CollapseRiskIncrease": collapse * 70,
            "Neutral": 12,
        }

    total = sum(weights.values()) or 1.0
    return {k: round(v / total * 100, 1) for k, v in weights.items()}


def build_reasoning(top: list[dict]) -> str:
    lines = [
        "===== VALIDATION 001 - BLIND FUTURE SIMULATION =====",
        "",
        f"Observation freeze: {OBSERVATION_TIME_KST} KST",
        "STRICT NO_ACTION | NO_LOOKAHEAD | NO_PRICE | NO_API",
        "",
        "Method: ScoutScore combines ONLY discovered P39-P62 law relationships.",
        "Data cutoff: T+0 checkpoint for process-tracked symbols;",
        "institution-proxy mapping for remainder of observation universe.",
        "",
        "=== Symbol selection reasoning ===",
        "",
    ]
    for i, r in enumerate(top, 1):
        lines.extend([
            f"--- Rank {i}: {r['symbol']} (ScoutScore={r['scout_score']}) ---",
            "",
            f"This symbol is selected because:",
        ])
        reasons: list[str] = []
        if r["belief_consensus"] >= CRITICAL_BELIEF * 0.85:
            reasons.append("BeliefConsensus is near or above the discovered critical threshold")
        elif r["belief_consensus"] > 50:
            reasons.append("BeliefConsensus shows moderate alignment")
        if r["synchronization_score"] >= CRITICAL_MASS * 0.8:
            reasons.append("SynchronizationScore approaches discovered CriticalMass")
        if r["attention_score"] >= CRITICAL_ATTENTION:
            reasons.append("AttentionScore exceeds discovered CriticalAttention floor")
        if r["net_migration"] > 0:
            reasons.append("Migration inflow is positive")
        elif r["migration_rate"] == 0:
            reasons.append("Migration is stable (no outflow at freeze)")
        if r["ecology_entropy"] < 2.0:
            reasons.append("Ecology entropy is relatively low")
        if r["replacement_rate"] >= 0.8:
            reasons.append("Replacement remains stable")
        if r["collapse_risk"] < 0.15:
            reasons.append("CollapseRisk remains suppressed")
        if r["persistence_expectation"] > 50:
            reasons.append("Persistence expectation from discovered TrendPersistence law is elevated")
        if not reasons:
            reasons.append("Combined law components rank highest in observation universe")
        for reason in reasons:
            lines.append(f"  - {reason}")
        lines.append("")

    lines.extend([
        "=== Future Evaluation ===",
        "",
        "(EMPTY — to be filled after +2h / +4h / +6h process data becomes available)",
        "",
        *mission_summary_lines(),
    ])
    return "\n".join(lines)


def run() -> None:
    freeze_dt = datetime(2026, 6, 19, 11, 0, 0, tzinfo=KST)
    print(f"Validation 001 | freeze={OBSERVATION_TIME_KST} KST | NO_LOOKAHEAD")

    universe = load_csv(LOGS_DIR / "season2_p37_observation_log.csv")
    if not universe:
        raise SystemExit("Observation universe required.")

    process_t0 = load_t0_process()
    scored: list[dict] = []

    for row in universe:
        sym = row["symbol"]
        if sym in process_t0:
            r = process_t0[sym]
        else:
            r = proxy_from_institution(row)

        score, confidence, laws = compute_scout_score(r)
        lifecycle = lifecycle_label(r)
        states = predict_states(r, laws)

        scored.append({
            "symbol": sym,
            "scout_score": round(score, 2),
            "confidence": round(confidence, 2),
            "data_source": r["data_source"],
            "lifecycle": lifecycle,
            "phase": r["PhaseLabel"],
            "synchronization_score": round(r["SynchronizationScore"], 2),
            "belief_consensus": round(r["BeliefConsensus"], 2),
            "narrative_score": round(r["NarrativeScore"], 2),
            "attention_score": round(r["AttentionScore"], 2),
            "ecology_entropy": round(r["EcologyEntropy"], 4),
            "migration_rate": round(r["MigrationRate"], 4),
            "net_migration": round(r["NetMigration"], 4),
            "collapse_risk": round(r["CollapseRisk"], 4),
            "replacement_rate": round(r["ReplacementRate"], 4),
            "persistence_expectation": round(laws["persistence_component"], 2),
            "sync_law_value": round(laws["sync_component"], 2),
            "narrative_law_value": round(laws["narrative_component"], 2),
            "collapse_law_value": round(laws["collapse_component"], 4),
            "p39_state": r["p39_state"],
            "predicted_2h": states["2h"],
            "predicted_4h": states["4h"],
            "predicted_6h": states["6h"],
        })

    scored.sort(key=lambda x: (-x["scout_score"], -x["confidence"]))
    for i, row in enumerate(scored, 1):
        row["rank"] = i

    top2 = scored[:2]

    score_rows = [{
        "observation_freeze_kst": OBSERVATION_TIME_KST,
        "observation_freeze_utc": freeze_dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        **{k: v for k, v in r.items()},
        "learning_recommendation": "NO_ACTION",
    } for r in scored]

    top2_rows = [{
        "observation_freeze_kst": OBSERVATION_TIME_KST,
        **{k: v for k, v in r.items()},
        "learning_recommendation": "NO_ACTION",
    } for r in top2]

    state_rows: list[dict] = []
    prob_rows: list[dict] = []
    for r in scored:
        sym = r["symbol"]
        src = process_t0.get(sym) or proxy_from_institution(
            next(x for x in universe if x["symbol"] == sym)
        )
        laws = apply_discovered_laws(src)
        for horizon in ("2h", "4h", "6h"):
            state_rows.append({
                "observation_freeze_kst": OBSERVATION_TIME_KST,
                "symbol": sym,
                "horizon": horizon,
                "expected_state": r[f"predicted_{horizon}"],
                "scout_score": r["scout_score"],
                "rank": r["rank"],
                "learning_recommendation": "NO_ACTION",
            })
            probs = transition_probabilities(src, horizon)
            for state, pct in probs.items():
                prob_rows.append({
                    "observation_freeze_kst": OBSERVATION_TIME_KST,
                    "symbol": sym,
                    "horizon": horizon,
                    "state": state,
                    "probability_pct": pct,
                    "learning_recommendation": "NO_ACTION",
                })

    reasoning = build_reasoning(top2)

    write_csv(SCORES_CSV, score_rows)
    write_csv(TOP2_CSV, top2_rows)
    write_csv(STATE_PRED_CSV, state_rows)
    write_csv(PROB_CSV, prob_rows)
    REASONING_TXT.write_text(reasoning, encoding="utf-8")

    print(f"Scored {len(scored)} symbols | Top2: {top2[0]['symbol']} ({top2[0]['scout_score']}), "
          f"{top2[1]['symbol']} ({top2[1]['scout_score']})")
    print(f"Saved validation001 outputs to {LOGS_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation 001 Blind Future Simulation")
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
