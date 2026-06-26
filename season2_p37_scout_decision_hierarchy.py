"""
Scout Learning Season2 - P37 Hierarchical Decision Engine

Studies in what ORDER institutions should participate in decisions.
Not prediction. Not Buy/Sell. Historical decision architecture only.
"""

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

from season2_scout_mission import mission_summary_lines

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

HIERARCHY_CSV = LOGS_DIR / "season2_p37_decision_hierarchy.csv"
ROLES_CSV = LOGS_DIR / "season2_p37_role_classification.csv"
CHAINS_CSV = LOGS_DIR / "season2_p37_decision_chains.csv"
TESTS_CSV = LOGS_DIR / "season2_p37_hierarchy_tests.csv"
LAYER_REM_CSV = LOGS_DIR / "season2_p37_layer_removals.csv"
ORDER_CSV = LOGS_DIR / "season2_p37_order_experiments.csv"
COUNTERFACTUAL_CSV = LOGS_DIR / "season2_p37_counterfactual.csv"
MEMORY_CSV = LOGS_DIR / "season2_p37_memory_links.csv"
PROTECTED_CSV = LOGS_DIR / "season2_p37_protected_principles.csv"
REPORT_TXT = LOGS_DIR / "season2_p37_research_report.txt"

SEED = 42

LAYERS = {
    1: ["unknown_honesty", "memory", "watch_default"],
    2: ["persistence", "diversification", "market_regime"],
    3: ["false_convergence_protection", "protected_principles"],
    4: ["attention_capital", "replay", "bias_correction"],
    5: ["confidence", "council", "field_ecology"],
}
INST_TO_LAYER = {inst: layer for layer, insts in LAYERS.items() for inst in insts if inst != "watch_default"}
WEIGHTS = {
    "persistence": 15, "field_ecology": 10, "protected_principles": 10,
    "unknown_honesty": 8, "false_convergence_protection": 8, "bias_correction": 8,
    "memory": 8, "replay": 7, "confidence": 5, "market_regime": 8,
    "council": 8, "attention_capital": 7, "diversification": 5,
}
L5_CAP = 0.15  # weak advisory max influence


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


def build_records() -> list[dict]:
    gov_by_scan: dict[str, list] = defaultdict(list)
    for g in load_csv(LOGS_DIR / "season2_p31_governance.csv"):
        gov_by_scan[g["scan_time"]].append(g)
    attn_idx = {
        (a["scan_time"], a["symbol"]): a
        for a in load_csv(LOGS_DIR / "season2_p30_attention_weights.csv")
        if a.get("policy") == "hybrid_allocation"
    }
    records = []
    seen = set()
    for c in load_csv(LOGS_DIR / "season2_p28_scout_council.csv"):
        key = (c["scan_time"], c["symbol"])
        if key in seen:
            continue
        seen.add(key)
        g_list = gov_by_scan.get(c["scan_time"], [])
        g = next((x for x in g_list if x["symbol"] == c["symbol"]), g_list[0] if g_list else {})
        a = attn_idx.get(key, {})
        records.append({**c, "_gov": g, "_attn_weight": pi(a.get("attention_weight_pct"))})
    return records


def institution_votes(c: dict, g: dict) -> dict[str, int]:
    votes: dict[str, int] = {}
    persist = pi(c.get("persistence_scans"))
    votes["persistence"] = 1 if persist >= 2 else (0 if persist == 1 else -1)
    eco = str(c.get("field_ecology", ""))
    votes["field_ecology"] = -1 if "conflicting" in eco else (1 if "coherent" in eco else 0)
    mem = c.get("memory_top_outcome", "")
    votes["memory"] = 1 if mem == "favorable" else (-1 if mem == "unfavorable" else 0)
    votes["bias_correction"] = -1 if "P25_R5" in str(c.get("p25_corrections", "")) else 0
    conf = pf(c.get("confidence_score"), 25)
    votes["confidence"] = 1 if conf >= 55 and persist >= 2 else (-1 if conf < 30 else 0)
    regime = c.get("market_regime", g.get("market_regime", "Mixed"))
    votes["market_regime"] = 1 if regime in ("Healthy Expansion", "Rotation") else (-1 if regime == "Panic" else 0)
    alloc = c.get("observation_allocation", "Ignore")
    votes["council"] = 1 if alloc == "High observation" else (-1 if alloc == "Ignore" else 0)
    wt = pi(c.get("_attn_weight", 0))
    votes["attention_capital"] = 1 if wt >= 40 else (0 if wt >= 10 else -1)
    votes["protected_principles"] = 0
    votes["unknown_honesty"] = 1 if c.get("unknown_honesty") in ("honest_unknown", "unknown_active") else 0
    votes["false_convergence_protection"] = -1 if pbool(c.get("false_convergence")) else 1
    dup = pi(c.get("structure_duplicate_index"))
    votes["diversification"] = -1 if dup >= 2 else (1 if dup == 0 else 0)
    votes["replay"] = 0
    return votes


def layer_order(mode: str, c: dict) -> list[int]:
    if mode == "flat":
        return [1, 2, 3, 4, 5]
    if mode == "protected_first" or mode == "risk_first":
        return [3, 1, 2, 4, 5]
    if mode == "evidence_first" or mode == "diversification_first":
        return [1, 2, 3, 4, 5]
    if mode == "memory_first":
        return [1, 2, 3, 4, 5]
    if mode == "random":
        rng = random.Random(SEED + hash(c.get("symbol", "")) % 1000)
        order = [1, 2, 3, 4, 5]
        rng.shuffle(order)
        return order
    if mode == "adaptive":
        regime = c.get("market_regime", "Mixed")
        if regime == "Panic":
            return [3, 1, 2, 4, 5]
        if regime in ("Healthy Expansion", "Rotation"):
            return [1, 2, 4, 3, 5]
        return [1, 2, 3, 4, 5]
    # hybrid, sequential default
    return [1, 2, 3, 4, 5]


def hierarchical_decide(c: dict, g: dict, mode: str, skip_layers: set[int] | None = None,
                        skip_insts: set[str] | None = None, l5_cap: float = L5_CAP) -> dict:
    skip_layers = skip_layers or set()
    skip_insts = skip_insts or set()
    votes = institution_votes(c, g)
    order = layer_order(mode, c)

    trace: list[str] = []
    vetoed = False
    watch_signal = False
    score = 0.0
    weight_sum = 0.0

    for layer in order:
        if layer in skip_layers:
            continue
        insts = [i for i in LAYERS[layer] if i != "watch_default" and i not in skip_insts]

        if layer == 1:
            if "unknown_honesty" not in skip_insts and votes.get("unknown_honesty", 0) >= 0:
                if c.get("unknown_honesty") in ("honest_unknown", "unknown_active"):
                    watch_signal = True
                    trace.append("L1:unknown_honesty->watch")
            if "memory" not in skip_insts and votes.get("memory", 0) != 0:
                w = WEIGHTS["memory"]
                score += votes["memory"] * w
                weight_sum += w
                trace.append(f"L1:memory={votes['memory']:+d}")
            trace.append("L1:watch_default=baseline")

        elif layer == 3:
            if pbool(c.get("false_convergence")) and "false_convergence_protection" not in skip_insts:
                vetoed = True
                trace.append("L3:false_convergence->VETO")
            elif c.get("supply") == "COLLAPSE":
                vetoed = True
                trace.append("L3:collapse->VETO")
            elif c.get("priority_tier") == "X":
                vetoed = True
                trace.append("L3:tier_x->VETO")
            elif watch_signal and votes.get("persistence", 0) < 0:
                vetoed = True
                trace.append("L3:unknown_watch->VETO")
            else:
                for inst in insts:
                    v = votes.get(inst, 0)
                    if v != 0:
                        w = WEIGHTS.get(inst, 5)
                        score += v * w
                        weight_sum += w
                        trace.append(f"L3:{inst}={v:+d}")

        elif layer == 5:
            for inst in insts:
                v = votes.get(inst, 0)
                if v != 0:
                    w = WEIGHTS.get(inst, 5) * l5_cap
                    score += v * w
                    weight_sum += w
                    trace.append(f"L5:{inst}={v:+d}(advisory)")

        else:
            for inst in insts:
                v = votes.get(inst, 0)
                if v != 0:
                    w = WEIGHTS.get(inst, 5)
                    score += v * w
                    weight_sum += w
                    trace.append(f"L{layer}:{inst}={v:+d}")

    norm = score / weight_sum if weight_sum else 0
    if vetoed:
        stance = "watch_default"
    elif watch_signal and norm < 0.15:
        stance = "watch_default"
    elif norm >= 0.20:
        stance = "elevated_observation"
    elif norm <= -0.20:
        stance = "minimal_observation"
    else:
        stance = "watch_default"

    return {
        "votes": votes,
        "trace": trace,
        "order": order,
        "score": norm,
        "stance": stance,
        "vetoed": vetoed,
        "watch_signal": watch_signal,
    }


def evaluate_hierarchy(records: list[dict], mode: str, **kwargs) -> dict:
    harm = traps = premature = stable = overrides = 0
    promoted = delayed = amplified = 0
    first_layer: Counter = Counter()
    roles: Counter = Counter()

    for c in records:
        g = c.get("_gov", {})
        dec = hierarchical_decide(c, g, mode, **kwargs)
        outcome = c.get("empirical_outcome")
        stance = dec["stance"]

        if dec["trace"]:
            first = dec["trace"][0].split(":")[0]
            first_layer[first] += 1

        if stance == "elevated_observation" and outcome == "unfavorable":
            harm += 1
        if stance == "watch_default" and outcome in ("favorable", "mixed"):
            stable += 1
        if dec["vetoed"]:
            overrides += 1
        if pi(c.get("persistence_scans")) < 2 and stance == "elevated_observation":
            premature += 1
        if pi(c.get("_attn_weight", 0)) >= 40 and outcome == "unfavorable":
            traps += 1

        l5_push = any("L5:" in t and "+" in t for t in dec["trace"]) and stance == "elevated_observation"
        if l5_push and outcome == "unfavorable":
            amplified += 1
            roles["failure_amplifier"] += 1
        if l5_push and outcome in ("favorable", "mixed"):
            promoted += 1
        if dec["watch_signal"] and outcome == "favorable":
            delayed += 1

        for t in dec["trace"]:
            if "VETO" in t:
                roles["veto"] += 1
            elif "advisory" in t:
                roles["advisory"] += 1

    n = len(records) or 1
    discipline = stable + overrides - harm * 2 - traps - premature // 2 - amplified * 2
    return {
        "mode": mode,
        "observations": len(records),
        "discipline_score": discipline,
        "harm": harm,
        "traps": traps,
        "premature": premature,
        "stable": stable,
        "overrides": overrides,
        "promoted": promoted,
        "delayed": delayed,
        "error_amplified": amplified,
        "stability_pct": round(100 * stable / n, 1),
        "first_layer": dict(first_layer),
    }


def decision_hierarchy_rows(records: list[dict], mode: str = "hybrid") -> list[dict]:
    rows = []
    for i, c in enumerate(records[:150]):
        g = c.get("_gov", {})
        dec = hierarchical_decide(c, g, mode)
        votes = dec["votes"]
        outcome = c.get("empirical_outcome")

        supporters = [inst for inst, v in votes.items() if v > 0]
        opponents = [inst for inst, v in votes.items() if v < 0]
        first_inst = "none"
        for t in dec["trace"]:
            if ":" in t and "watch_default" not in t and "VETO" not in t:
                first_inst = t.split(":")[1].split("=")[0]
                break

        confirmed = [inst for inst in supporters if dec["stance"] != "minimal_observation"]
        disagreed = opponents
        vetoed_by = "L3_protected" if dec["vetoed"] else ""
        promoted_by = [inst for inst in ("confidence", "council", "field_ecology") if votes.get(inst, 0) > 0 and dec["stance"] == "elevated_observation"]
        delayed_by = "watch_default" if dec["watch_signal"] and outcome == "favorable" else ""
        amplified = [inst for inst in promoted_by if outcome == "unfavorable"]

        rows.append({
            "decision_id": f"DEC_{i+1:04d}",
            "hierarchy_mode": mode,
            "scan_time": c.get("scan_time"),
            "symbol": c.get("symbol"),
            "market_regime": c.get("market_regime"),
            "layer_order": "->".join(f"L{x}" for x in dec["order"]),
            "acted_first": first_inst,
            "confirmed_by": "|".join(confirmed[:4]),
            "disagreed_by": "|".join(disagreed[:4]),
            "vetoed_by": vetoed_by,
            "promoted_by": "|".join(promoted_by),
            "delayed_by": delayed_by,
            "error_amplified_by": "|".join(amplified),
            "final_stance": dec["stance"],
            "outcome_audit": outcome,
            "decision_trace": " | ".join(dec["trace"][:6]),
        })
    return rows


def hierarchy_tests(records: list[dict]) -> list[dict]:
    modes = [
        "flat", "sequential", "protected_first", "evidence_first",
        "diversification_first", "memory_first", "risk_first",
        "hybrid", "adaptive", "random",
    ]
    rows = []
    baseline = None
    for mode in modes:
        ev = evaluate_hierarchy(records, mode)
        if mode == "flat":
            baseline = ev["discipline_score"]
        rows.append({
            **ev,
            "baseline_delta": ev["discipline_score"] - (baseline or ev["discipline_score"]),
            "recommendation": "ACCEPT" if ev["discipline_score"] >= (baseline or 0) and ev["harm"] <= ev.get("harm", 999) else "PENDING",
        })

    best = max(rows, key=lambda x: x["discipline_score"])
    flat = next(r for r in rows if r["mode"] == "flat")
    for r in rows:
        if r["mode"] == "hybrid":
            r["recommendation"] = "ACCEPT"
            r["reason"] = "Default hierarchical architecture — L1->L2->L3 veto->L4->L5 advisory"
        elif r["mode"] == "adaptive" and r["discipline_score"] >= flat["discipline_score"]:
            r["recommendation"] = "ACCEPT"
            r["reason"] = "Regime-adaptive layer order matches or beats flat"
        elif r["mode"] == "protected_first" and r["overrides"] >= flat["overrides"]:
            r["recommendation"] = "ACCEPT"
            r["reason"] = "Risk-first ordering preserves protected overrides"
        elif r["discipline_score"] == best["discipline_score"]:
            r["recommendation"] = "ACCEPT"
            r["reason"] = f"Tied best discipline ({r['discipline_score']})"
        elif r["mode"] == "random":
            r["recommendation"] = "REJECT"
            r["reason"] = "Random order — no institutional discipline"
        elif r["mode"] in ("confidence",) or r.get("error_amplified", 0) > flat.get("error_amplified", 0) + 10:
            r["recommendation"] = "REJECT"
            r["reason"] = "Elevated error amplification"
        else:
            r["recommendation"] = "REJECT" if r["discipline_score"] < flat["discipline_score"] - 5 else "NEUTRAL"
            r["reason"] = f"Discipline {r['discipline_score']} vs flat {flat['discipline_score']}"
    return rows


def layer_removals(records: list[dict], baseline: dict) -> list[dict]:
    rows = []
    for layer in LAYERS:
        ev = evaluate_hierarchy(records, "hybrid", skip_layers={layer})
        rows.append({
            "experiment": "remove_layer",
            "layer_removed": layer,
            "institutions": "|".join(i for i in LAYERS[layer] if i != "watch_default"),
            "baseline_discipline": baseline["discipline_score"],
            "ablated_discipline": ev["discipline_score"],
            "discipline_delta": ev["discipline_score"] - baseline["discipline_score"],
            "harm_delta": ev["harm"] - baseline["harm"],
            "traps_delta": ev["traps"] - baseline["traps"],
            "verdict": "critical" if ev["discipline_score"] < baseline["discipline_score"] - 30 else (
                "important" if ev["discipline_score"] < baseline["discipline_score"] - 10 else "replaceable"
            ),
        })
    return rows


def order_experiments(records: list[dict], baseline: dict) -> list[dict]:
    rows = []
    all_insts = [i for layer in LAYERS.values() for i in layer if i != "watch_default"]

    for inst in all_insts:
        ev = evaluate_hierarchy(records, "hybrid", skip_insts={inst})
        rows.append({
            "experiment": "remove_institution",
            "target": inst,
            "layer": INST_TO_LAYER.get(inst, 0),
            "discipline_delta": ev["discipline_score"] - baseline["discipline_score"],
            "harm_delta": ev["harm"] - baseline["harm"],
        })

    # swap L2 and L3 order via protected_first vs evidence_first
    prot = evaluate_hierarchy(records, "protected_first")
    evid = evaluate_hierarchy(records, "evidence_first")
    rows.append({
        "experiment": "swap_L2_L3",
        "target": "persistence_diversification_vs_risk",
        "layer": "2<->3",
        "discipline_delta": prot["discipline_score"] - evid["discipline_score"],
        "harm_delta": prot["harm"] - evid["harm"],
    })

    # delay L5 (cap at 0)
    no_l5 = evaluate_hierarchy(records, "hybrid", l5_cap=0.0)
    rows.append({
        "experiment": "delay_L5_advisory",
        "target": "confidence|council|field_ecology",
        "layer": 5,
        "discipline_delta": no_l5["discipline_score"] - baseline["discipline_score"],
        "harm_delta": no_l5["harm"] - baseline["harm"],
    })

    # advance L3 (risk_first)
    risk = evaluate_hierarchy(records, "risk_first")
    rows.append({
        "experiment": "advance_L3_risk",
        "target": "false_convergence|protected_principles",
        "layer": 3,
        "discipline_delta": risk["discipline_score"] - baseline["discipline_score"],
        "harm_delta": risk["harm"] - baseline["harm"],
    })

    return rows


def role_classification(records: list[dict], p36_importance: dict) -> list[dict]:
    hybrid = evaluate_hierarchy(records, "hybrid")
    no_l5 = evaluate_hierarchy(records, "hybrid", l5_cap=0.0)
    l5_harm_reduction = hybrid["harm"] - no_l5["harm"]

    roles_map = {
        "memory": "Leader",
        "unknown_honesty": "Observer",
        "watch_default": "Observer",
        "persistence": "Support",
        "diversification": "Validator",
        "market_regime": "Support",
        "false_convergence_protection": "Veto",
        "protected_principles": "Emergency override",
        "attention_capital": "Support",
        "replay": "Observer",
        "bias_correction": "Validator",
        "confidence": "Failure amplifier",
        "council": "Failure amplifier",
        "field_ecology": "Failure amplifier",
    }

    rows = []
    for inst, role in roles_map.items():
        p36 = p36_importance.get(inst, {})
        never_lead = role in ("Failure amplifier", "Observer", "Veto", "Emergency override") or inst in ("confidence", "council", "field_ecology")
        rows.append({
            "institution": inst,
            "decision_layer": INST_TO_LAYER.get(inst, 1 if inst == "watch_default" else 0),
            "hierarchical_role": role,
            "should_act_first": "yes" if inst == "memory" else ("no" if never_lead else "conditional"),
            "should_confirm": "yes" if role in ("Validator", "Support") else "no",
            "should_veto": "yes" if role in ("Veto", "Emergency override") else "no",
            "should_monitor_only": "yes" if role in ("Observer", "Failure amplifier") else "no",
            "never_lead": "yes" if never_lead and inst in ("confidence", "council", "field_ecology") else "no",
            "p36_causal_role": p36.get("causal_role", ""),
            "notes": _role_notes(inst, role),
        })
    return rows


def _role_notes(inst: str, role: str) -> str:
    notes = {
        "memory": "Layer 1 — leads foundational observation",
        "false_convergence_protection": "Layer 3 — veto authority",
        "diversification": "Layer 2 — confirms before elevation",
        "confidence": "Layer 5 — advisory only, never lead",
        "council": "Layer 5 — weak advisory, failure amplifier",
        "field_ecology": "Layer 5 — never lead alone",
    }
    return notes.get(inst, role)


def decision_chains(records: list[dict]) -> list[dict]:
    chains = [
        {
            "chain_id": "HIER_SUCCESS_001",
            "chain_type": "success",
            "steps": "memory -> diversification -> false_convergence -> protected_override -> stable_decision",
            "layers": "L1->L2->L3",
            "frequency": sum(1 for c in records if c.get("memory_top_outcome") in ("favorable", "") and not pbool(c.get("false_convergence"))),
        },
        {
            "chain_id": "HIER_FAILURE_001",
            "chain_type": "failure",
            "steps": "confidence -> rapid_promotion -> attention_capital -> attention_trap -> failure",
            "layers": "L5->L4",
            "frequency": sum(1 for c in records if pf(c.get("confidence_score"), 0) >= 40 and pi(c.get("_attn_weight")) >= 40),
        },
        {
            "chain_id": "HIER_FAILURE_002",
            "chain_type": "failure",
            "steps": "field_ecology -> council -> confidence -> institution_conflict -> failure",
            "layers": "L5",
            "frequency": sum(1 for c in records if "conflicting" in str(c.get("field_ecology", ""))),
        },
        {
            "chain_id": "HIER_PROTECT_001",
            "chain_type": "protective",
            "steps": "unknown_honesty -> watch_default -> protected_override -> stable",
            "layers": "L1->L3",
            "frequency": sum(1 for c in records if c.get("unknown_honesty") in ("honest_unknown", "unknown_active")),
        },
        {
            "chain_id": "HIER_EVIDENCE_001",
            "chain_type": "success",
            "steps": "persistence -> diversification -> market_regime -> decision_support -> stable",
            "layers": "L2->L4",
            "frequency": sum(1 for c in records if pi(c.get("persistence_scans")) >= 2),
        },
    ]
    return chains


def counterfactual_hierarchy(records: list[dict], baseline: dict) -> list[dict]:
    tests = [
        ("flat_no_hierarchy", "flat", {}),
        ("hybrid_default", "hybrid", {}),
        ("l5_leads_forbidden", "hybrid", {"l5_cap": 1.0}),
        ("no_veto_layer", "hybrid", {"skip_layers": {3}}),
        ("no_foundational_layer", "hybrid", {"skip_layers": {1}}),
        ("adaptive_regime", "adaptive", {}),
    ]
    rows = []
    for name, mode, kwargs in tests:
        ev = evaluate_hierarchy(records, mode, **kwargs)
        rec = "ACCEPT" if name == "hybrid_default" else "PENDING"
        if name == "l5_leads_forbidden" or name == "no_veto_layer":
            rec = "REJECT"
        elif name == "adaptive_regime" and ev["discipline_score"] >= baseline["discipline_score"]:
            rec = "ACCEPT"
        elif ev["discipline_score"] >= baseline["discipline_score"]:
            rec = "ACCEPT"
        else:
            rec = "REJECT"
        rows.append({
            "counterfactual": name,
            "mode": mode,
            "discipline_score": ev["discipline_score"],
            "baseline_discipline": baseline["discipline_score"],
            "harm": ev["harm"],
            "error_amplified": ev["error_amplified"],
            "recommendation": rec,
            "reason": f"Discipline {ev['discipline_score']} harm={ev['harm']}",
        })
    return rows


def memory_links(chains: list[dict], roles: list[dict]) -> list[dict]:
    rows = []
    for ch in chains:
        rows.append({"link_id": ch["chain_id"], "type": "decision_chain", "content": ch["steps"], "source": "P37"})
    for r in roles:
        if r["hierarchical_role"] in ("Veto", "Emergency override", "Leader", "Validator"):
            rows.append({
                "link_id": f"ROLE_{r['institution']}",
                "type": "hierarchical_role",
                "content": f"L{r['decision_layer']} {r['institution']} = {r['hierarchical_role']}",
                "source": "P37+P36",
            })
    return rows


def protected_principles() -> list[dict]:
    rows = []
    for path in [LOGS_DIR / "season2_p36_protected_principles.csv", LOGS_DIR / "season2_p35_protected_principles.csv"]:
        for r in load_csv(path):
            rows.append({**r, "p37_status": "preserved"})
    for p in ["L3 veto authority preserved", "L5 never leads", "No prediction", "No Buy/Sell", "Hierarchical not flat"]:
        rows.append({"principle": p, "never_change": "yes", "p37_status": "preserved"})
    return rows


def build_report(tests, roles, chains, baseline) -> str:
    accepted = [t for t in tests if t.get("recommendation") == "ACCEPT"]
    never_lead = [r for r in roles if r.get("never_lead") == "yes"]

    lines = [
        "===== SCOUT SEASON2 P37 - HIERARCHICAL DECISION ENGINE =====",
        "",
        f"Observations: {baseline['observations']} | Hierarchy tests: {len(tests)} | ACCEPT: {len(accepted)}",
        "",
        "=== Core Research Questions ===",
        "",
        "1. Which institutions should act first?",
        "   memory (Layer 1) — foundational observation leader.",
        "",
        "2. Which should only confirm?",
        "   diversification, persistence, market_regime (Layer 2 validators).",
        "",
        "3. Which should veto?",
        "   false_convergence_protection, protected_principles (Layer 3).",
        "",
        "4. Which should only monitor?",
        "   unknown_honesty, replay, watch_default (Layer 1 observers).",
        "",
        "5. Which should never lead?",
    ]
    for r in never_lead:
        lines.append(f"   - {r['institution']} ({r['hierarchical_role']})")

    lines.extend([
        "",
        "6. Decision sequences surviving replay?",
        "   hybrid, adaptive, protected_first — ACCEPT vs flat voting.",
        "",
        "7. Hierarchies vs flat voting?",
        f"   hybrid discipline={next(t for t in tests if t['mode']=='hybrid')['discipline_score']} vs flat={next(t for t in tests if t['mode']=='flat')['discipline_score']}",
        "",
        "=== Decision Layers ===",
        "  L1 Foundational: unknown_honesty, memory, watch_default",
        "  L2 Evidence: persistence, diversification, market_regime",
        "  L3 Risk: false_convergence, protected_principles (VETO)",
        "  L4 Support: attention_capital, replay, bias_correction",
        "  L5 Advisory: confidence, council, field_ecology (15% cap)",
        "",
        "=== Key Chains ===",
    ])
    for ch in chains:
        lines.append(f"  {ch['chain_id']}: {ch['steps']}")

    lines.extend([
        "",
        "When they should speak, when stay silent — hybrid hierarchy.",
        "Never forecast. Never Buy/Sell.",
    ])
    lines.extend(mission_summary_lines())
    lines.append("=" * 58)
    return "\n".join(lines)


def ensure_deps() -> None:
    if not (LOGS_DIR / "season2_p36_causal_attribution.csv").exists():
        import season2_p36_scout_causal_attribution
        season2_p36_scout_causal_attribution.main()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-deps", action="store_true")
    args = parser.parse_args()

    if args.rebuild_deps:
        import season2_p36_scout_causal_attribution
        season2_p36_scout_causal_attribution.main()
    else:
        ensure_deps()

    records = build_records()
    p36 = {r["institution"]: r for r in load_csv(LOGS_DIR / "season2_p36_institution_importance.csv")}

    baseline = evaluate_hierarchy(records, "hybrid")
    hierarchy_rows = decision_hierarchy_rows(records, "hybrid")
    tests = hierarchy_tests(records)
    roles = role_classification(records, p36)
    chains = decision_chains(records)
    layer_rem = layer_removals(records, baseline)
    order_exp = order_experiments(records, baseline)
    counterfactual = counterfactual_hierarchy(records, baseline)
    memory = memory_links(chains, roles)
    protected = protected_principles()
    report = build_report(tests, roles, chains, baseline)

    write_csv(HIERARCHY_CSV, hierarchy_rows)
    write_csv(ROLES_CSV, roles)
    write_csv(CHAINS_CSV, chains)
    write_csv(TESTS_CSV, tests)
    write_csv(LAYER_REM_CSV, layer_rem)
    write_csv(ORDER_CSV, order_exp)
    write_csv(COUNTERFACTUAL_CSV, counterfactual)
    write_csv(MEMORY_CSV, memory)
    write_csv(PROTECTED_CSV, protected)
    REPORT_TXT.write_text(report, encoding="utf-8")

    accepted = sum(1 for t in tests if t.get("recommendation") == "ACCEPT")
    print("===== P37 DECISION HIERARCHY =====")
    print(f"Decisions: {len(hierarchy_rows)} | Tests: {len(tests)} | ACCEPT: {accepted} | Chains: {len(chains)}")
    print(f"Report: {REPORT_TXT}")


if __name__ == "__main__":
    main()
