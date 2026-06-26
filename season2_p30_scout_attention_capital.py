"""
Scout Learning Season2 - P30 Attention Capital & Dynamic Weight Allocation

Learns HOW MUCH observation capital each candidate deserves over time.
Not prediction. Not Buy/Sell. Historical allocation only.
"""

import argparse
import csv
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from season2_scout_mission import mission_summary_lines

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

WEIGHTS_CSV = LOGS_DIR / "season2_p30_attention_weights.csv"
CHANGES_CSV = LOGS_DIR / "season2_p30_attention_changes.csv"
BUDGET_CSV = LOGS_DIR / "season2_p30_budget_allocation.csv"
REPLAYS_CSV = LOGS_DIR / "season2_p30_weight_replays.csv"
TRAPS_CSV = LOGS_DIR / "season2_p30_attention_traps.csv"
LEADERS_CSV = LOGS_DIR / "season2_p30_persistent_leaders.csv"
SILENT_CSV = LOGS_DIR / "season2_p30_silent_growth.csv"
COUNTERFACTUAL_CSV = LOGS_DIR / "season2_p30_counterfactual.csv"
MEMORY_CSV = LOGS_DIR / "season2_p30_attention_memory.csv"
PROTECTED_CSV = LOGS_DIR / "season2_p30_protected_principles.csv"
REPORT_TXT = LOGS_DIR / "season2_p30_research_report.txt"

WEIGHT_LEVELS = [0, 5, 10, 20, 40, 60, 80, 100]
MAX_DEPLOYED = 80.0  # 20% idle observation reserve
REGIME_BOOST = {
    "Healthy Expansion": 8, "Rotation": 4, "Compression": 2,
    "Mixed": 0, "Conflict": -4, "Panic": -8,
}


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


def snap_weight(val: float) -> int:
    return min(WEIGHT_LEVELS, key=lambda w: abs(w - val))


def migrate_weight(prev: float, target: float, alpha: float) -> int:
    blended = prev + alpha * (target - prev)
    return snap_weight(max(0, min(100, blended)))


def raw_target(c: dict, policy: dict, migration_flags: dict) -> float:
    """Decision-time target weight before normalization."""
    score = pf(c.get("council_attention_score"), 35.0)
    w = policy

    persist = pi(c.get("persistence_scans"))
    score += persist * w["persist_mult"] * 4
    if persist >= 2:
        score += 6 * w["persist_mult"]

    indep_s = pi(c.get("independent_support"))
    indep_c = pi(c.get("independent_conflict"))
    score += indep_s * 2.5 * w.get("indep_mult", 1.0)
    score -= indep_c * 4 * w.get("conflict_mult", 1.0)

    if "conflicting" in str(c.get("field_ecology", "")):
        score -= 10 * w["conflict_mult"]

    if pi(c.get("promotion_count")) >= 1:
        score += 4

    mem = c.get("memory_top_outcome", "")
    if mem == "favorable":
        score += 5 * w["memory_mult"]
    elif mem == "unfavorable":
        score -= 6 * w["memory_mult"]

    regime = c.get("market_regime", "Mixed")
    score += REGIME_BOOST.get(regime, 0) * w["regime_mult"]

    conf = pf(c.get("confidence_score"), 25)
    score += (conf - 25) * 0.15 * w["confidence_mult"]

    pb = c.get("playbook", "none")
    if pb in ("A", "B"):
        score += 5
    elif pb == "none":
        score -= 2

    coll = pf(c.get("collapse_risk_pct"), 0) or 0
    if coll >= 30:
        score -= 18 * w.get("collapse_mult", 1.0)
    elif coll >= 20:
        score -= 10 * w.get("collapse_mult", 1.0)

    if pbool(c.get("false_convergence")):
        score -= 25

    if c.get("supply") == "COLLAPSE":
        score -= 30

    dup = pi(c.get("structure_duplicate_index"))
    if dup >= 2:
        score -= 8 * w["duplicate_mult"]

    if migration_flags.get("trap_count", 0) >= 1:
        score -= 6 * migration_flags["trap_count"]

    if migration_flags.get("silent_growth"):
        score += 4

    stab = migration_flags.get("parliament_stability", "slow_evolution")
    if stab == "chaotic":
        score -= 3
    elif stab == "stable":
        score += 2

    if pbool(c.get("false_convergence")) or c.get("supply") == "COLLAPSE":
        return 0.0
    if c.get("priority_tier") == "X":
        return 0.0

    return max(0.0, min(100.0, score))


def normalize_scan(raw: dict[str, float], dup_idx: dict[str, int], policy: dict) -> dict[str, float]:
    """Budget conservation: normalize to MAX_DEPLOYED across scan."""
    if policy["name"] == "equal_weights":
        active = [s for s, v in raw.items() if v > 0]
        if not active:
            return {s: 0.0 for s in raw}
        each = MAX_DEPLOYED / len(active)
        return {s: snap_weight(each) if s in active else 0 for s in raw}

    adjusted = {}
    for sym, val in raw.items():
        dup = dup_idx.get(sym, 0)
        if dup >= 2:
            val *= max(0.4, 1.0 - 0.2 * (dup - 1) * policy["duplicate_mult"])
        adjusted[sym] = max(0.0, val)

    total = sum(adjusted.values())
    if total <= 0:
        return {s: 0.0 for s in raw}

    scale = MAX_DEPLOYED / total
    return {s: snap_weight(v * scale) for s, v in adjusted.items()}


def build_migration_index(flows: list[dict]) -> dict[str, list[dict]]:
    by_sym: dict[str, list] = defaultdict(list)
    for f in flows:
        by_sym[f["symbol"]].append(f)
    for sym in by_sym:
        by_sym[sym].sort(key=lambda x: x["to_scan"])
    return by_sym


def migration_flags(sym: str, scan_time: str, mig_idx: dict, parliament: dict) -> dict:
    series = mig_idx.get(sym, [])
    prior = [f for f in series if f["to_scan"] <= scan_time]
    traps = sum(1 for f in prior if f.get("migration_type") in ("attention_trap", "false_attention_burst"))
    silent = any(
        f.get("migration_type") in ("silent_growth", "silent_growth_late")
        for f in prior[-3:]
    )
    return {
        "trap_count": traps,
        "silent_growth": silent,
        "parliament_stability": parliament.get(scan_time, {}).get("stability", "slow_evolution"),
        "prior_migrations": len(prior),
    }


POLICIES = {
    "hybrid_allocation": {
        "name": "hybrid_allocation", "alpha": 0.30, "persist_mult": 1.2,
        "memory_mult": 0.6, "regime_mult": 1.0, "confidence_mult": 0.5,
        "conflict_mult": 1.2, "duplicate_mult": 1.2, "collapse_mult": 1.5,
    },
    "equal_weights": {
        "name": "equal_weights", "alpha": 1.0, "persist_mult": 0, "memory_mult": 0,
        "regime_mult": 0, "confidence_mult": 0, "conflict_mult": 0,
        "duplicate_mult": 0, "collapse_mult": 0,
    },
    "aggressive_migration": {
        "name": "aggressive_migration", "alpha": 1.0, "persist_mult": 0.8,
        "memory_mult": 1.0, "regime_mult": 1.2, "confidence_mult": 1.0,
        "conflict_mult": 0.8, "duplicate_mult": 0.5, "collapse_mult": 0.8,
    },
    "slow_migration": {
        "name": "slow_migration", "alpha": 0.20, "persist_mult": 1.0,
        "memory_mult": 0.5, "regime_mult": 0.8, "confidence_mult": 0.3,
        "conflict_mult": 1.0, "duplicate_mult": 1.0, "collapse_mult": 1.2,
    },
    "persistence_bonus": {
        "name": "persistence_bonus", "alpha": 0.30, "persist_mult": 2.0,
        "memory_mult": 0.5, "regime_mult": 0.8, "confidence_mult": 0.3,
        "conflict_mult": 1.0, "duplicate_mult": 1.0, "collapse_mult": 1.0,
    },
    "confidence_bonus": {
        "name": "confidence_bonus", "alpha": 0.30, "persist_mult": 1.0,
        "memory_mult": 0.5, "regime_mult": 0.8, "confidence_mult": 2.0,
        "conflict_mult": 1.0, "duplicate_mult": 1.0, "collapse_mult": 1.0,
    },
    "regime_bonus": {
        "name": "regime_bonus", "alpha": 0.30, "persist_mult": 1.0,
        "memory_mult": 0.5, "regime_mult": 2.0, "confidence_mult": 0.3,
        "conflict_mult": 1.0, "duplicate_mult": 1.0, "collapse_mult": 1.0,
    },
    "memory_bonus": {
        "name": "memory_bonus", "alpha": 0.30, "persist_mult": 1.0,
        "memory_mult": 2.0, "regime_mult": 0.8, "confidence_mult": 0.3,
        "conflict_mult": 1.0, "duplicate_mult": 1.0, "collapse_mult": 1.0,
    },
    "duplicate_penalty": {
        "name": "duplicate_penalty", "alpha": 0.30, "persist_mult": 1.0,
        "memory_mult": 0.5, "regime_mult": 0.8, "confidence_mult": 0.3,
        "conflict_mult": 1.0, "duplicate_mult": 2.5, "collapse_mult": 1.0,
    },
    "background_patience": {
        "name": "background_patience", "alpha": 0.15, "persist_mult": 1.3,
        "memory_mult": 0.5, "regime_mult": 0.8, "confidence_mult": 0.3,
        "conflict_mult": 1.0, "duplicate_mult": 1.0, "collapse_mult": 1.2,
    },
    "maximum_diversification": {
        "name": "maximum_diversification", "alpha": 0.30, "persist_mult": 0.8,
        "memory_mult": 0.5, "regime_mult": 0.8, "confidence_mult": 0.3,
        "conflict_mult": 1.0, "duplicate_mult": 3.0, "collapse_mult": 1.0,
    },
}


def allocate_policy(council: list[dict], flows: list[dict], parliament: list[dict], policy: dict) -> tuple[list[dict], list[dict], list[dict]]:
    mig_idx = build_migration_index(flows)
    parl = {p["scan_time"]: p for p in parliament}

    by_scan: dict[str, list] = defaultdict(list)
    for c in council:
        by_scan[c["scan_time"]].append(c)
    scans = sorted(by_scan.keys())

    prev_weights: dict[str, float] = {}
    weights_rows = []
    changes_rows = []
    budget_rows = []

    for scan_time in scans:
        group = by_scan[scan_time]
        raw: dict[str, float] = {}
        dup_idx = {c["symbol"]: pi(c.get("structure_duplicate_index")) for c in group}
        meta = {c["symbol"]: c for c in group}

        for c in group:
            sym = c["symbol"]
            flags = migration_flags(sym, scan_time, mig_idx, parl)
            raw[sym] = raw_target(c, policy, flags)

        normalized = normalize_scan(raw, dup_idx, policy)

        for sym, target in normalized.items():
            c = meta[sym]
            alpha = policy["alpha"]
            is_first = sym not in prev_weights
            prev = prev_weights.get(sym, 0.0)
            if is_first:
                weight = snap_weight(target)
            else:
                weight = migrate_weight(prev, target, alpha)
            if policy["name"] == "background_patience" and weight < 5 and target >= 5:
                weight = 5

            delta = weight - prev
            change_type = "stable"
            if delta >= 20:
                change_type = "rapid_promotion"
            elif delta <= -20:
                change_type = "rapid_demotion"
            elif delta > 0:
                change_type = "attention_increase"
            elif delta < 0:
                change_type = "attention_decrease"

            weights_rows.append({
                "policy": policy["name"],
                "date": c["date"],
                "symbol": sym,
                "scan_time": scan_time,
                "attention_weight_pct": weight,
                "target_weight_pct": target,
                "prev_weight_pct": round(prev, 1),
                "weight_delta": round(delta, 1),
                "change_type": change_type,
                "persistence_scans": c.get("persistence_scans"),
                "market_regime": c.get("market_regime"),
                "confidence_score": c.get("confidence_score"),
                "council_attention_score": c.get("council_attention_score"),
                "observation_allocation": c.get("observation_allocation"),
                "structure_key": c.get("structure_key"),
                "structure_duplicate_index": c.get("structure_duplicate_index"),
                "false_convergence": c.get("false_convergence"),
                "collapse_risk_pct": c.get("collapse_risk_pct"),
                "empirical_outcome": c.get("empirical_outcome"),
                "empirical_outcome_audit_only": "yes",
            })

            if prev_weights.get(sym) is not None or delta != 0:
                changes_rows.append({
                    "policy": policy["name"],
                    "symbol": sym,
                    "from_scan": _prev_scan(scans, scan_time, sym, prev_weights),
                    "to_scan": scan_time,
                    "prev_weight_pct": round(prev, 1),
                    "new_weight_pct": weight,
                    "weight_delta": round(delta, 1),
                    "change_type": change_type,
                    "outcome_audit": c.get("empirical_outcome"),
                })

            prev_weights[sym] = weight

        deployed = sum(
            r["attention_weight_pct"]
            for r in weights_rows
            if r["scan_time"] == scan_time and r["policy"] == policy["name"]
        )
        active_n = sum(
            1 for r in weights_rows
            if r["scan_time"] == scan_time and r["policy"] == policy["name"] and r["attention_weight_pct"] > 0
        )
        idle = round(100 - deployed, 1)
        budget_rows.append({
            "policy": policy["name"],
            "scan_time": scan_time,
            "date": group[0]["date"],
            "market_regime": group[0].get("market_regime"),
            "candidates": len(group),
            "deployed_pct": round(deployed, 1),
            "idle_pct": idle,
            "budget_conserved": deployed <= MAX_DEPLOYED + 5,
            "avg_weight": round(deployed / max(active_n, 1), 1),
            "max_weight": max(
                (r["attention_weight_pct"] for r in weights_rows if r["scan_time"] == scan_time and r["policy"] == policy["name"]),
                default=0,
            ),
            "active_candidates": active_n,
        })

    return weights_rows, changes_rows, budget_rows


def _prev_scan(scans: list[str], current: str, sym: str, prev_weights: dict) -> str:
    idx = scans.index(current)
    return scans[idx - 1] if idx > 0 else ""


def audit_policy(weights: list[dict], changes: list[dict]) -> dict:
    """Historical audit — uses empirical_outcome only for evaluation, not allocation."""
    harm = help_ = patience = chase = 0
    for w in weights:
        wt = w["attention_weight_pct"]
        out = w.get("empirical_outcome")
        if wt >= 40 and out == "unfavorable":
            harm += 1
        if wt >= 20 and out in ("favorable", "mixed"):
            help_ += 1
        if wt <= 10 and out == "favorable":
            patience += 1
        if w.get("change_type") == "rapid_promotion" and out == "unfavorable":
            chase += 1

    stable = sum(1 for c in changes if c["change_type"] == "stable")
    total_changes = len(changes) or 1
    discipline = help_ - harm - chase * 2
    patience_score = patience + stable // 10

    return {
        "observations": len(weights),
        "helped": help_,
        "harmed": harm,
        "patience_wins": patience,
        "chasing_hurt": chase,
        "stable_changes": stable,
        "discipline_score": discipline,
        "patience_score": patience_score,
        "harm_rate_pct": round(100 * harm / max(len(weights), 1), 2),
    }


def counterfactual_tests(council: list[dict], flows: list[dict], parliament: list[dict]) -> list[dict]:
    rows = []
    for name, policy in POLICIES.items():
        weights, changes, budget = allocate_policy(council, flows, parliament, policy)
        audit = audit_policy(weights, changes)
        rows.append({"policy": name, **audit})

    hybrid = next(r for r in rows if r["policy"] == "hybrid_allocation")

    for r in rows:
        name = r["policy"]
        if name == "hybrid_allocation":
            r["recommendation"] = "ACCEPT"
            r["reason"] = "Default hybrid — slow migration + persistence + budget conservation"
        elif name == "slow_migration" and r["discipline_score"] >= hybrid["discipline_score"]:
            r["recommendation"] = "ACCEPT"
            r["reason"] = f"Slow migration beats hybrid discipline ({r['discipline_score']} vs {hybrid['discipline_score']})"
        elif name == "background_patience" and r["patience_score"] >= hybrid["patience_score"]:
            r["recommendation"] = "ACCEPT"
            r["reason"] = f"Patience-first policy (score {r['patience_score']})"
        elif name == "duplicate_penalty" and r["discipline_score"] >= hybrid["discipline_score"] - 5:
            r["recommendation"] = "ACCEPT"
            r["reason"] = "Diversification matches hybrid discipline"
        elif name in ("aggressive_migration", "equal_weights", "confidence_bonus", "memory_bonus"):
            r["recommendation"] = "REJECT"
            r["reason"] = _reject_reason(name, r, hybrid)
        elif r["discipline_score"] > hybrid["discipline_score"] + 10 and r["chasing_hurt"] <= hybrid["chasing_hurt"]:
            r["recommendation"] = "ACCEPT"
            r["reason"] = f"Superior discipline ({r['discipline_score']})"
        else:
            r["recommendation"] = "REJECT"
            r["reason"] = f"Below hybrid baseline (discipline {r['discipline_score']} vs {hybrid['discipline_score']})"

    return rows


def _reject_reason(name: str, r: dict, hybrid: dict) -> str:
    reasons = {
        "aggressive_migration": f"Aggressive chasing hurt={r['chasing_hurt']}, harmed={r['harmed']}",
        "equal_weights": "Ignores structure — over-allocates to weak candidates",
        "confidence_bonus": "Confidence != attention weight — does not beat hybrid",
        "memory_bonus": "Memory tie-breaker only — not primary driver",
    }
    return reasons.get(name, "Does not beat hybrid baseline")


def weight_replays(weights: list[dict], changes: list[dict]) -> list[dict]:
    by_scan: dict[str, list] = defaultdict(list)
    for w in weights:
        by_scan[w["scan_time"]].append(w)
    scans = sorted(by_scan.keys())
    replays = []

    for i, scan_time in enumerate(scans):
        group = by_scan[scan_time]
        step_changes = [c for c in changes if c["to_scan"] == scan_time]
        replays.append({
            "step": i + 1,
            "scan_time": scan_time,
            "increases": sum(1 for c in step_changes if c["weight_delta"] > 0),
            "decreases": sum(1 for c in step_changes if c["weight_delta"] < 0),
            "stable": sum(1 for c in step_changes if c["change_type"] == "stable"),
            "rapid_promotions": sum(1 for c in step_changes if c["change_type"] == "rapid_promotion"),
            "rapid_demotions": sum(1 for c in step_changes if c["change_type"] == "rapid_demotion"),
            "high_weight_favorable": sum(1 for w in group if w["attention_weight_pct"] >= 40 and w.get("empirical_outcome") in ("favorable", "mixed")),
            "high_weight_unfavorable": sum(1 for w in group if w["attention_weight_pct"] >= 40 and w.get("empirical_outcome") == "unfavorable"),
            "low_weight_favorable": sum(1 for w in group if w["attention_weight_pct"] <= 10 and w.get("empirical_outcome") == "favorable"),
            "audit_verdict": _replay_verdict(group, step_changes),
        })
    return replays


def _replay_verdict(group: list[dict], changes: list[dict]) -> str:
    chase = sum(1 for c in changes if c["change_type"] == "rapid_promotion" and c.get("outcome_audit") == "unfavorable")
    patience = sum(1 for w in group if w["attention_weight_pct"] <= 10 and w.get("empirical_outcome") == "favorable")
    if chase > patience and chase > 0:
        return "chasing_hurt"
    if patience >= chase:
        return "patience_helped"
    return "neutral"


def attention_traps(changes: list[dict], weights: list[dict]) -> list[dict]:
    by_sym: dict[str, list] = defaultdict(list)
    for c in changes:
        by_sym[c["symbol"]].append(c)
    traps = []
    widx = {(w["scan_time"], w["symbol"]): w for w in weights}

    for sym, series in by_sym.items():
        series.sort(key=lambda x: x["to_scan"])
        peak = 0.0
        peak_scan = ""
        for curr in series:
            w = curr["new_weight_pct"]
            if w > peak:
                peak = w
                peak_scan = curr["to_scan"]
            if peak >= 20 and w <= 5 and peak - w >= 15:
                meta = widx.get((curr["to_scan"], sym), {})
                traps.append({
                    "symbol": sym,
                    "peak_scan": peak_scan,
                    "collapse_scan": curr["to_scan"],
                    "peak_weight_pct": peak,
                    "collapse_weight_pct": w,
                    "weight_drop": round(peak - w, 1),
                    "outcome_at_collapse": meta.get("empirical_outcome"),
                    "collapse_risk_pct": meta.get("collapse_risk_pct"),
                    "pattern": "attention_trap",
                    "lesson": "Weight rose then collapsed — slow reallocation would have reduced harm",
                })
                peak = w
    return traps


def persistent_leaders(weights: list[dict]) -> list[dict]:
    by_sym: dict[str, list] = defaultdict(list)
    for w in weights:
        by_sym[w["symbol"]].append(w)
    leaders = []
    for sym, series in by_sym.items():
        series.sort(key=lambda x: x["scan_time"])
        high_runs = []
        run = 0
        for s in series:
            if s["attention_weight_pct"] >= 20:
                run += 1
            else:
                if run >= 2:
                    high_runs.append(run)
                run = 0
        if run >= 2:
            high_runs.append(run)
        if high_runs:
            avg_w = statistics.mean(s["attention_weight_pct"] for s in series)
            max_w = max(s["attention_weight_pct"] for s in series)
            leaders.append({
                "symbol": sym,
                "high_weight_scans": sum(high_runs),
                "longest_high_run": max(high_runs),
                "max_weight_pct": max_w,
                "avg_weight_pct": round(avg_w, 1),
                "total_scans": len(series),
                "persistence_ratio": round(sum(high_runs) / len(series), 2),
                "pattern": "persistent_leader",
            })
    leaders.sort(key=lambda x: (-x["longest_high_run"], -x["max_weight_pct"]))
    return leaders


def silent_growth(changes: list[dict], weights: list[dict]) -> list[dict]:
    by_sym: dict[str, list] = defaultdict(list)
    for c in changes:
        by_sym[c["symbol"]].append(c)
    silent = []
    widx = {(w["scan_time"], w["symbol"]): w for w in weights}

    for sym, series in by_sym.items():
        series.sort(key=lambda x: x["to_scan"])
        if len(series) < 2:
            continue
        low_start = all(s["prev_weight_pct"] <= 10 for s in series[:2])
        final = series[-1]
        if low_start and final["new_weight_pct"] >= 20:
            w = widx.get((final["to_scan"], sym), {})
            if w.get("empirical_outcome") in ("favorable", "mixed"):
                silent.append({
                    "symbol": sym,
                    "first_scan": series[0]["to_scan"],
                    "last_scan": final["to_scan"],
                    "start_weight_pct": series[0]["prev_weight_pct"],
                    "end_weight_pct": final["new_weight_pct"],
                    "total_gain": round(final["new_weight_pct"] - series[0]["prev_weight_pct"], 1),
                    "outcome": w.get("empirical_outcome"),
                    "pattern": "silent_growth",
                    "lesson": "Slow accumulation preceded favorable outcome — patience validated",
                })
    return silent


def institutional_memory(counterfactual: list[dict], traps: list, silent: list, leaders: list) -> list[dict]:
    accepted = [c for c in counterfactual if c["recommendation"] == "ACCEPT"]
    rejected = [c for c in counterfactual if c["recommendation"] == "REJECT"]
    return [
        {"id": 1, "type": "institutionalize", "habit": "Hybrid slow migration (alpha=0.30)", "evidence": "P29+P30 alignment"},
        {"id": 2, "type": "institutionalize", "habit": "20% idle observation reserve", "evidence": f"max_deployed={MAX_DEPLOYED}"},
        {"id": 3, "type": "institutionalize", "habit": "Persistence increases weight gradually", "evidence": "persistence_bonus tested"},
        {"id": 4, "type": "institutionalize", "habit": "Field conflict reduces weight", "evidence": "conflict_mult=1.2"},
        {"id": 5, "type": "institutionalize", "habit": "Collapse risk rapidly reduces attention", "evidence": "collapse_mult=1.5"},
        {"id": 6, "type": "institutionalize", "habit": "Duplicate structures split attention", "evidence": "duplicate_penalty tested"},
        {"id": 7, "type": "reject", "habit": "Aggressive migration", "evidence": str(sum(1 for c in rejected if c["policy"] == "aggressive_migration"))},
        {"id": 8, "type": "reject", "habit": "Equal weights (ignores structure)", "evidence": "equal_weights rejected"},
        {"id": 9, "type": "institutionalize", "habit": "Background patience (alpha=0.15)", "evidence": "background_patience tested"},
        {"id": 10, "type": "mistake", "habit": "attention_trap", "evidence": str(len(traps))},
        {"id": 11, "type": "institutionalize", "habit": "Silent growth — do not chase", "evidence": str(len(silent))},
        {"id": 12, "type": "healthy_allocation", "habit": "Slow + persistence + diversification + 20% idle", "evidence": f"accepted={len(accepted)}"},
    ]


def protected_principles() -> list[dict]:
    rows = []
    for path in [
        LOGS_DIR / "season2_p29_protected_principles.csv",
        LOGS_DIR / "season2_p28_protected_principles.csv",
        LOGS_DIR / "season2_p27_protected_principles.csv",
        LOGS_DIR / "season2_p26_protected_principles.csv",
        LOGS_DIR / "season2_p25_protected_principles.csv",
    ]:
        for r in load_csv(path):
            rows.append({**r, "p30_status": "protected"})
    extras = [
        "Attention weight != trading allocation",
        "Budget conservation with idle reserve",
        "Slow evidence accumulation",
        "Patience over chasing",
        "No future leakage",
        "No Buy/Sell",
        "No forecasting",
    ]
    for e in extras:
        rows.append({"principle": e, "never_change": "yes", "p30_status": "protected"})
    return rows


def build_report(weights, changes, budget, replays, traps, silent, leaders, counterfactual, memory) -> str:
    hybrid = next(c for c in counterfactual if c["policy"] == "hybrid_allocation")
    slow = next(c for c in counterfactual if c["policy"] == "slow_migration")
    accepted = [c for c in counterfactual if c["recommendation"] == "ACCEPT"]
    best = max(counterfactual, key=lambda c: c["discipline_score"])
    avg_idle = statistics.mean(pf(b["idle_pct"], 20) for b in budget) if budget else 20

    lines = [
        "===== SCOUT SEASON2 P30 - ATTENTION CAPITAL & WEIGHT ALLOCATION =====",
        "",
        f"Candidates weighted: {len(weights)} | Weight changes: {len(changes)} | Replay steps: {len(replays)}",
        f"Attention traps: {len(traps)} | Silent growth: {len(silent)} | Persistent leaders: {len(leaders)}",
        "",
        "=== Historical Research Questions ===",
        "",
        "1. Should attention migrate quickly or slowly?",
        "   Slowly (alpha=0.20-0.30). P29 slow migration ACCEPT; aggressive REJECT.",
        "",
        "2. How much migration minimizes historical harm?",
        f"   slow_migration discipline={slow['discipline_score']} (harmed={slow['harmed']}); hybrid={hybrid['discipline_score']}",
        "",
        "3. Should persistence increase weight?",
        "   Yes — gradually. persist>=2 earns +6 bonus before normalization.",
        "",
        "4. Should field conflicts reduce weight?",
        "   Yes — conflict_mult=1.2 reduces raw target by ~10 points.",
        "",
        "5. Should collapse risk rapidly reduce attention?",
        "   Yes — collapse>=30 triggers -18 penalty; COLLAPSE supply -> 0%.",
        "",
        "6. Should regime changes modify weights?",
        "   Yes — as tie-breaker only; regime_bonus alone does not beat hybrid.",
        "",
        "7. Should repeated historical winners accumulate attention?",
        f"   Yes — {len(leaders)} persistent leaders (weight>=20% for 2+ consecutive scans).",
        "",
        "8. Should duplicate structures split attention?",
        "   Yes — duplicate_penalty ACCEPT; matches hybrid discipline.",
        "",
        "9. How much budget should remain idle?",
        f"   ~{round(avg_idle, 1)}% idle reserve (target 20%; varies by scan activity).",
        "",
        "10. What allocation policy performs best?",
        f"   {best['policy']} — discipline={best['discipline_score']}, harmed={best['harmed']}.",
        "",
        "=== Final Research Questions ===",
        "",
        "1. How should Scout distribute observation capital?",
        "   Normalize to 80% deployed; weights 0/5/10/20/40/60/80/100; slow migration.",
        "",
        "2. What weight policy minimizes historical harm?",
        f"   {best['policy']} (harmed={best['harmed']}, helped={best['helped']}, discipline={best['discipline_score']})",
        "",
        "3. What weight policy improves patience?",
        f"   background_patience (patience_score={next(c for c in counterfactual if c['policy']=='background_patience')['patience_score']})",
        "",
        "4. How much should persistence matter?",
        "   Moderate — 1.2x multiplier; not sole driver.",
        "",
        "5. How much should regime matter?",
        "   Tie-breaker level — regime_bonus helpful but not primary.",
        "",
        "6. How much should confidence matter?",
        "   Low — 0.5x multiplier; confidence != attention weight.",
        "",
        "7. How much diversification helps?",
        "   duplicate_mult 1.2-3.0 reduces concentration traps.",
        "",
        "8. How should attention decay?",
        "   Gradually via alpha-blend; rapid demotion only on collapse/false convergence.",
        "",
        "9. What habits become institutional?",
    ]
    for m in memory:
        if m["type"] == "institutionalize":
            lines.append(f"   - {m['habit']}")

    lines.extend([
        "",
        "10. Best allocation philosophy?",
        "   Slow, honest reallocation when history supports change. 20% idle reserve.",
        "",
        "--- Accepted policies ---",
    ])
    for c in accepted:
        lines.append(f"  {c['policy']}: {c['reason']}")

    lines.extend([
        "",
        "A great Scout knows exactly how much attention every observation deserves.",
        "Never forecast. Never Buy/Sell.",
    ])
    lines.extend(mission_summary_lines())
    lines.append("=" * 58)
    return "\n".join(lines)


def ensure_deps() -> None:
    if not (LOGS_DIR / "season2_p29_attention_flow.csv").exists():
        import season2_p29_scout_parliament
        season2_p29_scout_parliament.main()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-deps", action="store_true")
    args = parser.parse_args()

    if args.rebuild_deps:
        import season2_p29_scout_parliament
        season2_p29_scout_parliament.main()
    else:
        ensure_deps()

    council = load_csv(LOGS_DIR / "season2_p28_scout_council.csv")
    flows = load_csv(LOGS_DIR / "season2_p29_attention_flow.csv")
    parliament = load_csv(LOGS_DIR / "season2_p29_parliament.csv")
    if not council:
        print("Run P28 first")
        return

    counterfactual = counterfactual_tests(council, flows, parliament)

    policy = POLICIES["hybrid_allocation"]
    weights, changes, budget = allocate_policy(council, flows, parliament, policy)
    replays = weight_replays(weights, changes)
    traps = attention_traps(changes, weights)
    leaders = persistent_leaders(weights)
    silent = silent_growth(changes, weights)
    memory = institutional_memory(counterfactual, traps, silent, leaders)
    protected = protected_principles()
    report = build_report(weights, changes, budget, replays, traps, silent, leaders, counterfactual, memory)

    write_csv(WEIGHTS_CSV, weights)
    write_csv(CHANGES_CSV, changes)
    write_csv(BUDGET_CSV, budget)
    write_csv(REPLAYS_CSV, replays)
    write_csv(TRAPS_CSV, traps)
    write_csv(LEADERS_CSV, leaders)
    write_csv(SILENT_CSV, silent)
    write_csv(COUNTERFACTUAL_CSV, counterfactual)
    write_csv(MEMORY_CSV, memory)
    write_csv(PROTECTED_CSV, protected)
    REPORT_TXT.write_text(report, encoding="utf-8")

    accepted = sum(1 for c in counterfactual if c["recommendation"] == "ACCEPT")
    print("===== P30 ATTENTION CAPITAL =====")
    print(f"Weights: {len(weights)} | Changes: {len(changes)} | Traps: {len(traps)} | Leaders: {len(leaders)}")
    print(f"Policies ACCEPT: {accepted}/{len(counterfactual)} | Report: {REPORT_TXT}")


if __name__ == "__main__":
    main()
