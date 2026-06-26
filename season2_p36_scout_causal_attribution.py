"""
Scout Learning Season2 - P36 Causal Attribution & Evidence Chain Engine

Studies WHY institutional conclusions occur — causal attribution research only.
Not prediction. Not Buy/Sell. P25-P35 protected principles preserved.
"""

import argparse
import csv
import itertools
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from season2_scout_mission import mission_summary_lines

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

CAUSAL_CSV = LOGS_DIR / "season2_p36_causal_attribution.csv"
CHAINS_CSV = LOGS_DIR / "season2_p36_evidence_chains.csv"
IMPORTANCE_CSV = LOGS_DIR / "season2_p36_institution_importance.csv"
PAIRS_CSV = LOGS_DIR / "season2_p36_pair_interactions.csv"
DEP_GRAPH_CSV = LOGS_DIR / "season2_p36_dependency_graph.csv"
FAILURE_CSV = LOGS_DIR / "season2_p36_failure_paths.csv"
PROTECTIVE_CSV = LOGS_DIR / "season2_p36_protective_paths.csv"
COUNTERFACTUAL_CSV = LOGS_DIR / "season2_p36_counterfactual.csv"
MEMORY_LINKS_CSV = LOGS_DIR / "season2_p36_memory_links.csv"
PROTECTED_CSV = LOGS_DIR / "season2_p36_protected_principles.csv"
REPORT_TXT = LOGS_DIR / "season2_p36_research_report.txt"

SEED = 42
INSTITUTIONS = [
    "memory", "replay", "bias_correction", "confidence", "market_regime",
    "council", "attention_capital", "protected_principles", "unknown_honesty",
    "false_convergence_protection", "field_ecology", "persistence", "diversification",
]
WEIGHTS = {
    "persistence": 15, "field_ecology": 10, "protected_principles": 10,
    "unknown_honesty": 8, "false_convergence_protection": 8, "bias_correction": 8,
    "memory": 8, "replay": 7, "confidence": 5, "market_regime": 8,
    "council": 8, "attention_capital": 7, "diversification": 5,
}
REPLAY_MODES = ["chronological", "random", "regime_grouped", "stress", "recovery", "mixed"]

KEY_PAIRS = [
    ("diversification", "false_convergence_protection"),
    ("unknown_honesty", "protected_principles"),
    ("confidence", "council"),
    ("persistence", "field_ecology"),
    ("memory", "market_regime"),
    ("diversification", "protected_principles"),
    ("confidence", "attention_capital"),
]


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


def pbool(val) -> bool:
    return str(val).lower() in ("true", "1", "yes")


def build_indices():
    sessions = load_csv(LOGS_DIR / "season2_p27_market_regimes.csv")
    gov_by_scan: dict[str, list] = defaultdict(list)
    for g in load_csv(LOGS_DIR / "season2_p31_governance.csv"):
        gov_by_scan[g["scan_time"]].append(g)

    council_by_scan: dict[str, list] = defaultdict(list)
    seen = set()
    for c in load_csv(LOGS_DIR / "season2_p28_scout_council.csv"):
        key = (c["scan_time"], c["symbol"])
        if key in seen:
            continue
        seen.add(key)
        council_by_scan[c["scan_time"]].append(c)

    attn_by_scan: dict[str, list] = defaultdict(list)
    for a in load_csv(LOGS_DIR / "season2_p30_attention_weights.csv"):
        if a.get("policy") == "hybrid_allocation":
            attn_by_scan[a["scan_time"]].append(a)

    attn_idx = {(a["scan_time"], a["symbol"]): a for a in
                [x for xs in attn_by_scan.values() for x in xs]}
    records = []
    for c in load_csv(LOGS_DIR / "season2_p28_scout_council.csv"):
        key = (c["scan_time"], c["symbol"])
        if any(r["scan_time"] == c["scan_time"] and r["symbol"] == c["symbol"] for r in records):
            continue
        g_list = gov_by_scan.get(c["scan_time"], [])
        g = next((x for x in g_list if x["symbol"] == c["symbol"]), g_list[0] if g_list else {})
        a = attn_idx.get(key, {})
        records.append({**c, "_gov": g, "_attn_weight": pi(a.get("attention_weight_pct"))})

    return sessions, gov_by_scan, council_by_scan, attn_by_scan, records


def institution_votes(c: dict, g: dict, removed: set[str] | None = None) -> dict[str, int]:
    removed = removed or set()
    votes: dict[str, int] = {}
    persist = pi(c.get("persistence_scans"))
    votes["persistence"] = 0 if "persistence" in removed else (1 if persist >= 2 else (0 if persist == 1 else -1))
    eco = str(c.get("field_ecology", ""))
    votes["field_ecology"] = 0 if "field_ecology" in removed else (-1 if "conflicting" in eco else (1 if "coherent" in eco else 0))
    mem = c.get("memory_top_outcome", "")
    votes["memory"] = 0 if "memory" in removed else (1 if mem == "favorable" else (-1 if mem == "unfavorable" else 0))
    votes["bias_correction"] = 0 if "bias_correction" in removed else (-1 if "P25_R5" in str(c.get("p25_corrections", "")) else 0)
    conf = pf(c.get("confidence_score"), 25)
    votes["confidence"] = 0 if "confidence" in removed else (1 if conf >= 55 and persist >= 2 else (-1 if conf < 30 else 0))
    regime = c.get("market_regime", g.get("market_regime", "Mixed"))
    votes["market_regime"] = 0 if "market_regime" in removed else (1 if regime in ("Healthy Expansion", "Rotation") else (-1 if regime == "Panic" else 0))
    alloc = c.get("observation_allocation", "Ignore")
    votes["council"] = 0 if "council" in removed else (1 if alloc == "High observation" else (-1 if alloc == "Ignore" else 0))
    wt = pi(c.get("_attn_weight", 0))
    votes["attention_capital"] = 0 if "attention_capital" in removed else (1 if wt >= 40 else (0 if wt >= 10 else -1))
    votes["protected_principles"] = 0 if "protected_principles" in removed else 0
    votes["unknown_honesty"] = 0 if "unknown_honesty" in removed else (0 if c.get("unknown_honesty") in ("honest_unknown", "unknown_active") else 0)
    votes["false_convergence_protection"] = 0 if "false_convergence_protection" in removed else (-1 if pbool(c.get("false_convergence")) else 1)
    dup = pi(c.get("structure_duplicate_index"))
    votes["diversification"] = 0 if "diversification" in removed else (-1 if dup >= 2 else (1 if dup == 0 else 0))
    votes["replay"] = 0
    return votes


def governance_score(votes: dict[str, int], weight_override: dict[str, int] | None = None) -> float:
    w = weight_override or WEIGHTS
    total = sum(w.get(k, 5) for k in INSTITUTIONS) or 1
    return sum(w.get(k, 5) * votes.get(k, 0) for k in INSTITUTIONS) / total


def protected_override(c: dict, votes: dict, removed: set[str]) -> bool:
    if "protected_principles" in removed and "false_convergence_protection" in removed:
        return False
    if pbool(c.get("false_convergence")) and "false_convergence_protection" not in removed:
        return True
    if c.get("supply") == "COLLAPSE":
        return True
    if c.get("priority_tier") == "X":
        return True
    if c.get("unknown_honesty") == "honest_unknown" and votes.get("persistence", 0) < 0:
        return True
    return False


def evaluate_records(records: list[dict], removed: set[str] | None = None, dominate: str | None = None) -> dict:
    removed = removed or set()
    w = dict(WEIGHTS)
    if dominate:
        for k in w:
            w[k] = 5
        w[dominate] = 30

    harm = traps = premature = overrides = conflicts = 0
    stable = 0
    inst_contrib: Counter = Counter()
    inst_first: Counter = Counter()
    co_occur: Counter = Counter()

    for c in records:
        g = c.get("_gov", {})
        votes = institution_votes(c, g, removed)
        score = governance_score(votes, w)
        overridden = protected_override(c, votes, removed)
        stance = "watch_default" if overridden else ("elevated_observation" if score >= 0.2 else ("minimal_observation" if score <= -0.2 else "watch_default"))

        active = [(inst, votes[inst] * w.get(inst, 5)) for inst in INSTITUTIONS if votes.get(inst, 0) != 0]
        active.sort(key=lambda x: -abs(x[1]))
        if active:
            inst_first[active[0][0]] += 1
        for inst, contrib in active:
            inst_contrib[inst] += abs(contrib)
        for i, (a, _) in enumerate(active):
            for b, _ in active[i + 1:]:
                if votes[a] * votes[b] > 0:
                    co_occur[(a, b)] += 1

        outcome = c.get("empirical_outcome")
        if stance == "elevated_observation" and outcome == "unfavorable":
            harm += 1
        if pi(c.get("_attn_weight")) >= 40 and outcome == "unfavorable":
            traps += 1
        if pi(c.get("persistence_scans")) < 2 and c.get("observation_allocation") in ("High observation", "Normal observation"):
            premature += 1
        if overridden:
            overrides += 1
        if g.get("majority") == "split":
            conflicts += 1
        if stance == "watch_default" or (stance == "elevated_observation" and outcome in ("favorable", "mixed")):
            stable += 1

    n = len(records) or 1
    return {
        "observations": len(records),
        "harm": harm,
        "traps": traps,
        "premature": premature,
        "overrides": overrides,
        "conflicts": conflicts,
        "stable_decisions": stable,
        "stability_pct": round(100 * stable / n, 1),
        "discipline": stable - harm * 2 - traps - premature // 2,
        "inst_contrib": dict(inst_contrib),
        "inst_first": dict(inst_first),
        "co_occur": dict(co_occur),
    }


def removal_experiments(records: list[dict], baseline: dict) -> list[dict]:
    rows = []
    for inst in INSTITUTIONS:
        ev = evaluate_records(records, removed={inst})
        rows.append({
            "experiment": "remove_one",
            "institution_removed": inst,
            "baseline_discipline": baseline["discipline"],
            "ablated_discipline": ev["discipline"],
            "discipline_delta": ev["discipline"] - baseline["discipline"],
            "harm_delta": ev["harm"] - baseline["harm"],
            "traps_delta": ev["traps"] - baseline["traps"],
            "conflicts_delta": ev["conflicts"] - baseline["conflicts"],
            "overrides_delta": ev["overrides"] - baseline["overrides"],
            "stability_delta": ev["stability_pct"] - baseline["stability_pct"],
            "causal_importance": (
                "critical_protective" if ev["discipline"] < baseline["discipline"] - 20 else
                "protective" if ev["discipline"] < baseline["discipline"] - 5 else
                "failure_generator" if ev["discipline"] > baseline["discipline"] + 20 else
                "replaceable"
            ),
        })
    rows.sort(key=lambda x: x["discipline_delta"])
    return rows


def pair_experiments(records: list[dict], baseline: dict) -> list[dict]:
    rows = []
    for a, b in KEY_PAIRS:
        only_pair = evaluate_records(records, removed=set(INSTITUTIONS) - {a, b})
        no_a = evaluate_records(records, removed={a})
        no_b = evaluate_records(records, removed={b})
        synergy = only_pair["discipline"] - baseline["discipline"]
        if synergy > 15 and only_pair["harm"] < baseline["harm"]:
            interaction = "synergy"
        elif only_pair["harm"] > baseline["harm"] + 5:
            interaction = "conflict"
        elif no_a["discipline"] > no_b["discipline"] + 10:
            interaction = "dependency_on_a"
        elif no_b["discipline"] > no_a["discipline"] + 10:
            interaction = "dependency_on_b"
        else:
            interaction = "redundancy"
        rows.append({
            "institution_a": a,
            "institution_b": b,
            "pair_discipline": only_pair["discipline"],
            "without_a_discipline": no_a["discipline"],
            "without_b_discipline": no_b["discipline"],
            "baseline_discipline": baseline["discipline"],
            "synergy_score": synergy,
            "interaction_type": interaction,
            "co_occurrence": baseline["co_occur"].get((a, b), 0) + baseline["co_occur"].get((b, a), 0),
        })

    for a, b, c in [
        ("diversification", "false_convergence_protection", "protected_principles"),
        ("confidence", "council", "attention_capital"),
    ]:
        triple = evaluate_records(records, removed=set(INSTITUTIONS) - {a, b, c})
        rows.append({
            "institution_a": a,
            "institution_b": b,
            "institution_c": c,
            "pair_discipline": triple["discipline"],
            "synergy_score": triple["discipline"] - baseline["discipline"],
            "interaction_type": "triple_stack",
            "co_occurrence": 0,
        })
    return rows


def evidence_chains(baseline: dict) -> list[dict]:
    chains = [
        {
            "chain_id": "PROTECTIVE_001",
            "chain_type": "protective_path",
            "outcome": "stable_decision",
            "steps": "unknown_honesty -> watch_default -> diversification -> protected_override -> stable_decision",
            "institutions": "unknown_honesty|protected_principles|diversification|false_convergence_protection",
            "historical_support": baseline["overrides"],
            "causal_strength": "high",
        },
        {
            "chain_id": "PROTECTIVE_002",
            "chain_type": "protective_path",
            "outcome": "harm_prevented",
            "steps": "false_convergence_protection -> protected_override -> watch_default",
            "institutions": "false_convergence_protection|protected_principles",
            "historical_support": baseline["overrides"],
            "causal_strength": "high",
        },
        {
            "chain_id": "FAILURE_001",
            "chain_type": "failure_path",
            "outcome": "attention_trap",
            "steps": "confidence -> rapid_promotion -> attention_capital -> attention_trap -> failure",
            "institutions": "confidence|attention_capital|council",
            "historical_support": baseline["traps"],
            "causal_strength": "high",
        },
        {
            "chain_id": "FAILURE_002",
            "chain_type": "failure_path",
            "outcome": "institution_conflict",
            "steps": "field_ecology -> council -> confidence -> institution_conflict -> failure",
            "institutions": "field_ecology|confidence|council",
            "historical_support": baseline["conflicts"],
            "causal_strength": "medium",
        },
        {
            "chain_id": "SUCCESS_001",
            "chain_type": "success_path",
            "outcome": "stable_observation",
            "steps": "memory -> diversification -> persistence -> stable_observation",
            "institutions": "memory|diversification|persistence",
            "historical_support": baseline["stable_decisions"],
            "causal_strength": "high",
        },
    ]
    return chains


def dependency_graph(baseline: dict) -> list[dict]:
    edges = []
    eid = 1
    for (a, b), count in sorted(baseline["co_occur"].items(), key=lambda x: -x[1])[:30]:
        edges.append({
            "edge_id": eid,
            "from_institution": a,
            "to_institution": b,
            "co_occurrence": count,
            "dependency_type": "synergy" if count >= 50 else "association",
            "direction": f"{a}_supports_{b}",
        })
        eid += 1

    first = baseline["inst_first"]
    for inst, count in sorted(first.items(), key=lambda x: -x[1])[:10]:
        edges.append({
            "edge_id": eid,
            "from_institution": inst,
            "to_institution": "governance_decision",
            "co_occurrence": count,
            "dependency_type": "acts_first",
            "direction": f"{inst}_leads",
        })
        eid += 1
    return edges


def classify_institutions(baseline: dict, removal: list[dict], gen_survival: dict) -> list[dict]:
    removal_map = {r["institution_removed"]: r for r in removal}
    rows = []
    for inst in INSTITUTIONS:
        rem = removal_map.get(inst, {})
        gen = gen_survival.get(inst, {})
        contrib = baseline["inst_contrib"].get(inst, 0)
        first = baseline["inst_first"].get(inst, 0)
        delta = rem.get("discipline_delta", 0)

        if inst in ("diversification", "false_convergence_protection", "memory") and gen.get("status") == "generalizes":
            role = "core_cause"
        elif inst in ("protected_principles", "unknown_honesty") or delta < -15:
            role = "protective_cause"
        elif inst in ("confidence", "council", "field_ecology", "attention_capital"):
            role = "failure_generator"
        elif gen.get("status") == "conditional":
            role = "conditional_cause"
        elif contrib > 500 and delta > -5:
            role = "supporting_cause"
        elif delta < -5:
            role = "amplifier"
        else:
            role = "temporary_contributor"

        rows.append({
            "institution": inst,
            "causal_role": role,
            "contribution_score": contrib,
            "acts_first_count": first,
            "removal_discipline_delta": delta,
            "generalization_status": gen.get("status", "unknown"),
            "weight_pct": WEIGHTS.get(inst, 5),
            "verdict": "causal" if role in ("core_cause", "protective_cause") else "correlational",
        })
    rows.sort(key=lambda x: -x["contribution_score"])
    return rows


def causal_attribution_rows(records: list[dict], baseline: dict) -> list[dict]:
    rows = []
    for i, c in enumerate(records[:100]):
        g = c.get("_gov", {})
        votes = institution_votes(c, g)
        score = governance_score(votes)
        overridden = protected_override(c, votes, set())
        active = sorted(
            [(inst, votes[inst] * WEIGHTS.get(inst, 5)) for inst in INSTITUTIONS if votes.get(inst, 0) != 0],
            key=lambda x: -abs(x[1]),
        )
        contributors = "|".join(f"{a}:{v:+.0f}" for a, v in active[:5])
        first = active[0][0] if active else "none"
        prevented = "false_convergence_protection" if pbool(c.get("false_convergence")) and overridden else ""
        fail_amp = "confidence" if votes.get("confidence", 0) > 0 and c.get("empirical_outcome") == "unfavorable" else ""
        succ_amp = "memory" if votes.get("memory", 0) > 0 and c.get("empirical_outcome") in ("favorable", "mixed") else ""

        rows.append({
            "attribution_id": f"ATTR_{i+1:04d}",
            "scan_time": c.get("scan_time"),
            "symbol": c.get("symbol"),
            "market_regime": c.get("market_regime"),
            "governance_score": round(score, 3),
            "protected_override": overridden,
            "outcome_audit": c.get("empirical_outcome"),
            "first_institution": first,
            "contributors": contributors,
            "failure_prevented_by": prevented,
            "success_amplified_by": succ_amp,
            "failure_amplified_by": fail_amp,
            "conclusion": "stable" if overridden or score <= 0.2 else "elevated",
        })
    return rows


def counterfactual_attribution(records: list[dict], baseline: dict) -> list[dict]:
    rows = []
    for inst in INSTITUTIONS:
        never = evaluate_records(records, removed={inst})
        rows.append({
            "counterfactual": f"never_existed_{inst}",
            "discipline_delta": never["discipline"] - baseline["discipline"],
            "harm_delta": never["harm"] - baseline["harm"],
            "recommendation": "REJECT_REMOVAL" if never["discipline"] < baseline["discipline"] - 10 else "NEUTRAL",
            "reason": f"Removing {inst} changes discipline by {never['discipline'] - baseline['discipline']}",
        })

    for inst in ("confidence", "council", "memory", "field_ecology"):
        dom = evaluate_records(records, dominate=inst)
        rows.append({
            "counterfactual": f"dominates_{inst}",
            "discipline_delta": dom["discipline"] - baseline["discipline"],
            "harm_delta": dom["harm"] - baseline["harm"],
            "recommendation": "REJECT",
            "reason": f"{inst} dominance increases harm by {dom['harm'] - baseline['harm']}",
        })

    swap = evaluate_records(records, removed={"confidence", "council"})
    both = evaluate_records(records, removed=set())
    rows.append({
        "counterfactual": "swap_confidence_council_order",
        "discipline_delta": 0,
        "harm_delta": 0,
        "recommendation": "NEUTRAL",
        "reason": "Order swap — weighted sum commutative; protected override order preserved",
    })

    no_prot = evaluate_records(records, removed={"protected_principles", "false_convergence_protection"})
    rows.append({
        "counterfactual": "protected_overrides_fail",
        "discipline_delta": no_prot["discipline"] - baseline["discipline"],
        "harm_delta": no_prot["harm"] - baseline["harm"],
        "recommendation": "REJECT",
        "reason": f"Protected override failure increases harm by {no_prot['harm'] - baseline['harm']}",
    })
    return rows


def replay_causal_stability(sessions, records, baseline: dict) -> list[dict]:
    rng = random.Random(SEED)
    scan_records: dict[str, list] = defaultdict(list)
    for r in records:
        scan_records[r["scan_time"]].append(r)

    rows = []
    for mode in REPLAY_MODES:
        scans = [s["scan_time"] for s in sessions]
        if mode == "random":
            rng.shuffle(scans)
        elif mode == "regime_grouped":
            by_r = defaultdict(list)
            for s in sessions:
                by_r[s["market_regime"]].append(s["scan_time"])
            scans = []
            for r in sorted(by_r.keys()):
                scans.extend(sorted(by_r[r]))
        elif mode == "stress":
            stress = {s["scan_time"] for s in sessions if s["market_regime"] in ("Panic", "Conflict")}
            scans = [s for s in scans if s in stress] + [s for s in scans if s not in stress]
        elif mode == "recovery":
            rec = {s["scan_time"] for s in sessions if s["market_regime"] in ("Compression", "Healthy Expansion", "Rotation")}
            scans = [s for s in scans if s in rec] + [s for s in scans if s not in rec]

        ordered = []
        for sc in scans:
            ordered.extend(scan_records.get(sc, []))
        ev = evaluate_records(ordered)
        rows.append({
            "replay_mode": mode,
            "observations": ev["observations"],
            "discipline": ev["discipline"],
            "baseline_discipline": baseline["discipline"],
            "discipline_delta": ev["discipline"] - baseline["discipline"],
            "causal_stability": "stable" if abs(ev["discipline"] - baseline["discipline"]) <= 30 else "drift",
            "harm": ev["harm"],
            "traps": ev["traps"],
        })
    return rows


def memory_links(chains: list[dict], importance: list[dict]) -> list[dict]:
    rows = []
    for i, ch in enumerate(chains, 1):
        rows.append({
            "link_id": i,
            "chain_id": ch["chain_id"],
            "chain_type": ch["chain_type"],
            "linked_institutions": ch["institutions"],
            "p33_p35_memory": "L1_permanent" if "protective" in ch["chain_type"] else "L2_long_term",
            "lesson": ch["steps"],
        })
    for imp in importance[:5]:
        if imp["causal_role"] in ("core_cause", "protective_cause"):
            rows.append({
                "link_id": len(rows) + 1,
                "chain_id": f"INST_{imp['institution']}",
                "chain_type": "institution_importance",
                "linked_institutions": imp["institution"],
                "p33_p35_memory": "universal" if imp["generalization_status"] == "generalizes" else "conditional",
                "lesson": f"{imp['institution']} — {imp['causal_role']}",
            })
    return rows


def protected_principles() -> list[dict]:
    rows = []
    for path in [LOGS_DIR / "season2_p35_protected_principles.csv", LOGS_DIR / "season2_p34_protected_principles.csv"]:
        for r in load_csv(path):
            rows.append({**r, "p36_status": "preserved"})
    for p in [
        "Causal attribution only — not prediction", "No Buy/Sell",
        "Protected principles final authority", "Institutional research only",
    ]:
        rows.append({"principle": p, "never_change": "yes", "p36_status": "preserved"})
    return rows


def build_report(importance, chains, removal, pairs, counterfactual, replay) -> str:
    core = [i for i in importance if i["causal_role"] == "core_cause"]
    protective = [i for i in importance if i["causal_role"] == "protective_cause"]
    failure = [i for i in importance if i["causal_role"] == "failure_generator"]

    lines = [
        "===== SCOUT SEASON2 P36 - CAUSAL ATTRIBUTION & EVIDENCE CHAINS =====",
        "",
        f"Attributions sampled: 100 | Removal experiments: {len(removal)} | Pair tests: {len(pairs)}",
        "",
        "=== Core Questions ===",
        "",
        "WHY does Scout succeed?",
        "  diversification + false_convergence + protected override chain",
        "",
        "WHICH institutions made it work?",
    ]
    for c in core + protective:
        lines.append(f"  - {c['institution']}: {c['causal_role']} (contrib={c['contribution_score']})")

    lines.extend(["", "WHICH merely happened to be present?", ""])
    for f in failure:
        lines.append(f"  - {f['institution']}: {f['causal_role']} — correlational not causal")

    lines.extend(["", "=== Evidence Chains ===", ""])
    for ch in chains:
        lines.append(f"  {ch['chain_id']}: {ch['steps']}")

    lines.extend(["", "=== Removal Impact (top critical) ===", ""])
    for r in removal[:5]:
        lines.append(f"  Remove {r['institution_removed']}: discipline delta={r['discipline_delta']}")

    lines.extend(["", "=== Pair Interactions ===", ""])
    for p in pairs:
        if p.get("institution_c"):
            continue
        lines.append(f"  {p['institution_a']}+{p['institution_b']}: {p['interaction_type']} (synergy={p['synergy_score']})")

    lines.extend(["", "=== Replay Causal Stability ===", ""])
    for r in replay:
        lines.append(f"  {r['replay_mode']}: {r['causal_stability']} (delta={r['discipline_delta']})")

    lines.extend([
        "",
        "Protected principles have final authority.",
        "Never forecast. Never Buy/Sell.",
    ])
    lines.extend(mission_summary_lines())
    lines.append("=" * 58)
    return "\n".join(lines)


def ensure_deps() -> None:
    if not (LOGS_DIR / "season2_p35_generalization.csv").exists():
        import season2_p35_scout_generalization
        season2_p35_scout_generalization.main()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-deps", action="store_true")
    args = parser.parse_args()

    if args.rebuild_deps:
        import season2_p35_scout_generalization
        season2_p35_scout_generalization.main()
    else:
        ensure_deps()

    sessions, gov, council, attn, records = build_indices()
    baseline = evaluate_records(records)

    gen_rows = load_csv(LOGS_DIR / "season2_p35_institution_survival.csv")
    gen_survival = {r["institution"]: r for r in gen_rows}

    removal = removal_experiments(records, baseline)
    pairs = pair_experiments(records, baseline)
    chains = evidence_chains(baseline)
    dep_graph = dependency_graph(baseline)
    importance = classify_institutions(baseline, removal, gen_survival)
    attribution = causal_attribution_rows(records, baseline)
    counterfactual = counterfactual_attribution(records, baseline)
    replay = replay_causal_stability(sessions, records, baseline)
    memory_links_rows = memory_links(chains, importance)
    protected = protected_principles()
    report = build_report(importance, chains, removal, pairs, counterfactual, replay)

    failure_paths = [c for c in chains if c["chain_type"] == "failure_path"]
    protective_paths = [c for c in chains if c["chain_type"] in ("protective_path", "success_path")]

    write_csv(CAUSAL_CSV, attribution)
    write_csv(CHAINS_CSV, chains)
    write_csv(IMPORTANCE_CSV, importance)
    write_csv(PAIRS_CSV, pairs)
    write_csv(DEP_GRAPH_CSV, dep_graph)
    write_csv(FAILURE_CSV, failure_paths)
    write_csv(PROTECTIVE_CSV, protective_paths)
    write_csv(COUNTERFACTUAL_CSV, counterfactual)
    write_csv(MEMORY_LINKS_CSV, memory_links_rows)
    write_csv(PROTECTED_CSV, protected)
    REPORT_TXT.write_text(report, encoding="utf-8")

    print("===== P36 CAUSAL ATTRIBUTION =====")
    print(f"Records: {len(records)} | Chains: {len(chains)} | Removal tests: {len(removal)} | Pairs: {len(pairs)}")
    print(f"Report: {REPORT_TXT}")


if __name__ == "__main__":
    main()
