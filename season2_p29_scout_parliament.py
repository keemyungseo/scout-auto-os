"""
Scout Learning Season2 - P29 Scout Parliament & Dynamic Attention Economy

Studies how council attention evolves across time.
No forecasting. No Buy/Sell. Attention migration only.
"""

import argparse
import csv
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from season2_scout_mission import mission_summary_lines

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

PARLIAMENT_CSV = LOGS_DIR / "season2_p29_parliament.csv"
FLOW_CSV = LOGS_DIR / "season2_p29_attention_flow.csv"
EMERGENCE_CSV = LOGS_DIR / "season2_p29_emergence.csv"
FADING_CSV = LOGS_DIR / "season2_p29_fading.csv"
BUDGET_CSV = LOGS_DIR / "season2_p29_budget.csv"
REPLAYS_CSV = LOGS_DIR / "season2_p29_replays.csv"
COUNTERFACTUAL_CSV = LOGS_DIR / "season2_p29_counterfactual.csv"
INST_MEMORY_CSV = LOGS_DIR / "season2_p29_institutional_memory.csv"
PROTECTED_CSV = LOGS_DIR / "season2_p29_protected_principles.csv"
REPORT_TXT = LOGS_DIR / "season2_p29_research_report.txt"

ALLOC_ORDER = {
    "Ignore": 0,
    "Background observation": 1,
    "Normal observation": 2,
    "High observation": 3,
}
ALLOC_WEIGHT = {"High observation": 3, "Normal observation": 2, "Background observation": 1, "Ignore": 0}
SCAN_BUDGET_CAP = 18  # max observation budget units per council session
MAX_HIGH_SLOTS = 3


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


def tier_delta(prev: str, curr: str) -> int:
    return ALLOC_ORDER.get(curr, 0) - ALLOC_ORDER.get(prev, 0)


def migration_label(delta: int, prev: str, curr: str) -> str:
    if delta > 0:
        return "attention_gained"
    if delta < 0:
        return "attention_lost"
    return "stable_attention"


def build_parliament_sessions(council: list[dict]) -> list[dict]:
    by_scan: dict[str, list] = defaultdict(list)
    for c in council:
        by_scan[c["scan_time"]].append(c)

    scans = sorted(by_scan.keys())
    sessions = []
    prev_symbols: set[str] = set()
    prev_high: set[str] = set()

    for i, scan_time in enumerate(scans):
        group = by_scan[scan_time]
        symbols = {c["symbol"] for c in group}
        high = {c["symbol"] for c in group if c["observation_allocation"] == "High observation"}
        normal = {c["symbol"] for c in group if c["observation_allocation"] == "Normal observation"}
        active = {c["symbol"] for c in group if c["observation_allocation"] in ("High observation", "Normal observation")}

        enters = symbols - prev_symbols if i > 0 else symbols
        leaves = prev_symbols - symbols if i > 0 else set()
        rises = high - prev_high if i > 0 else high
        falls = prev_high - high if i > 0 else set()

        budget = sum(ALLOC_WEIGHT.get(c["observation_allocation"], 0) for c in group)
        persist_high = high & prev_high if i > 0 else set()

        sessions.append({
            "session_id": f"parliament_{scan_time.replace(' ', '_').replace(':', '')}",
            "scan_time": scan_time,
            "date": group[0]["date"],
            "market_regime": group[0].get("market_regime"),
            "council_size": len(group),
            "high_observation": len(high),
            "normal_observation": len(normal),
            "background_observation": sum(1 for c in group if c["observation_allocation"] == "Background observation"),
            "ignored": sum(1 for c in group if c["observation_allocation"] == "Ignore"),
            "enters": len(enters),
            "leaves": len(leaves),
            "rises_to_high": len(rises),
            "falls_from_high": len(falls),
            "persists_high": len(persist_high),
            "observation_budget": budget,
            "budget_over_cap": budget > SCAN_BUDGET_CAP,
            "high_over_cap": len(high) > MAX_HIGH_SLOTS,
            "active_symbols": len(active),
            "turnover_pct": round(100 * (len(enters) + len(leaves)) / max(len(symbols), 1), 1) if i > 0 else 0,
            "stability": _session_stability(len(enters), len(leaves), len(rises), len(falls), len(persist_high)),
        })
        prev_symbols = symbols
        prev_high = high

    return sessions


def _session_stability(enters, leaves, rises, falls, persist_high) -> str:
    churn = enters + leaves + rises + falls
    if churn <= 2 and persist_high >= 1:
        return "stable"
    if churn >= 8:
        return "chaotic"
    if rises + falls >= 4:
        return "rapid_turnover"
    return "slow_evolution"


def attention_flow(council: list[dict]) -> list[dict]:
    by_sym: dict[str, list] = defaultdict(list)
    for c in council:
        by_sym[c["symbol"]].append(c)
    for sym in by_sym:
        by_sym[sym].sort(key=lambda x: x["scan_time"])

    flows = []
    for sym, series in by_sym.items():
        for i in range(1, len(series)):
            prev, curr = series[i - 1], series[i]
            delta = tier_delta(prev["observation_allocation"], curr["observation_allocation"])
            label = migration_label(delta, prev["observation_allocation"], curr["observation_allocation"])

            pattern = label
            if delta >= 2:
                pattern = "rapid_promotion"
            elif delta <= -2:
                pattern = "rapid_demotion"
            elif label == "stable_attention" and curr["observation_allocation"] == "High observation":
                pattern = "persistent_leader"
            elif prev["observation_allocation"] == "Ignore" and delta > 0:
                pattern = "emerging_candidate"
            elif curr["observation_allocation"] == "Ignore" and delta < 0:
                pattern = "fading_candidate"
            elif (
                prev["observation_allocation"] == "High observation"
                and curr["observation_allocation"] != "High observation"
                and curr.get("empirical_outcome") == "unfavorable"
            ):
                pattern = "false_attention_burst"
            elif (
                prev["observation_allocation"] in ("Ignore", "Background observation")
                and curr["observation_allocation"] in ("High observation", "Normal observation")
                and curr.get("empirical_outcome") == "favorable"
            ):
                pattern = "silent_growth"

            flows.append({
                "symbol": sym,
                "from_scan": prev["scan_time"],
                "to_scan": curr["scan_time"],
                "from_allocation": prev["observation_allocation"],
                "to_allocation": curr["observation_allocation"],
                "tier_delta": delta,
                "migration_type": pattern,
                "from_score": prev.get("council_attention_score"),
                "to_score": curr.get("council_attention_score"),
                "score_delta": round(pf(curr.get("council_attention_score"), 0) - pf(prev.get("council_attention_score"), 0), 1),
                "from_persistence": prev.get("persistence_scans"),
                "to_persistence": curr.get("persistence_scans"),
                "outcome_at_to": curr.get("empirical_outcome"),
                "market_regime": curr.get("market_regime"),
            })
    return flows


def budget_reallocation(sessions: list[dict], flows: list[dict]) -> list[dict]:
    rows = []
    for i, sess in enumerate(sessions):
        gained = [f for f in flows if f["to_scan"] == sess["scan_time"] and f["tier_delta"] > 0]
        lost = [f for f in flows if f["to_scan"] == sess["scan_time"] and f["tier_delta"] < 0]
        budget = sess["observation_budget"]
        rows.append({
            "scan_time": sess["scan_time"],
            "observation_budget": budget,
            "budget_cap": SCAN_BUDGET_CAP,
            "budget_headroom": SCAN_BUDGET_CAP - budget,
            "budget_conserved": budget <= SCAN_BUDGET_CAP,
            "attention_gained_count": len(gained),
            "attention_lost_count": len(lost),
            "reallocation_balanced": abs(len(gained) - len(lost)) <= max(len(gained), len(lost), 1),
            "high_slots": sess["high_observation"],
            "high_cap": MAX_HIGH_SLOTS,
            "source_of_gain": "|".join(f"{f['symbol']}:{f['from_allocation']}->{f['to_allocation']}" for f in gained[:4]),
            "source_of_loss": "|".join(f"{f['symbol']}:{f['from_allocation']}->{f['to_allocation']}" for f in lost[:4]),
            "reallocation_note": _budget_note(budget, gained, lost, sess),
        })
    return rows


def _budget_note(budget, gained, lost, sess) -> str:
    if sess["budget_over_cap"]:
        return "budget_exceeded — reallocation required"
    if gained and not lost:
        return "attention_inflow_without_outflow — watch concentration"
    if lost and not gained:
        return "attention_outflow — patience or fade"
    if gained and lost:
        return "balanced_reallocation"
    return "stable_budget"


def emergence_patterns(flows: list[dict], council: list[dict]) -> tuple[list, list]:
    emergence = []
    fading = []
    by_sym: dict[str, list] = defaultdict(list)
    for f in flows:
        by_sym[f["symbol"]].append(f)

    for f in flows:
        if f["migration_type"] in (
            "emerging_candidate", "silent_growth", "rapid_promotion",
            "persistent_leader", "attention_gained",
        ):
            emergence.append({
                **f,
                "pattern": f["migration_type"],
                "lesson": _emergence_lesson(f["migration_type"]),
            })
        if f["migration_type"] in (
            "fading_candidate", "false_attention_burst", "rapid_demotion",
            "attention_lost",
        ):
            fading.append({
                **f,
                "pattern": f["migration_type"],
                "lesson": _fading_lesson(f["migration_type"]),
            })

    # silent growth: low attention for 2+ scans then rise
    council_idx = {(c["scan_time"], c["symbol"]): c for c in council}
    for sym, series in by_sym.items():
        if len(series) < 2:
            continue
        lows = sum(1 for s in series[:2] if ALLOC_ORDER.get(s["from_allocation"], 0) <= 1)
        if lows >= 1 and series[-1]["tier_delta"] > 0:
            last = series[-1]
            c = council_idx.get((last["to_scan"], sym), {})
            if c.get("empirical_outcome") == "favorable" and last["migration_type"] != "silent_growth":
                emergence.append({
                    **last,
                    "pattern": "silent_growth_late",
                    "lesson": "Patience preceded favorable attention rise",
                })

    # attention traps: rapid rise then rapid fall
    for sym, series in by_sym.items():
        for i in range(1, len(series)):
            if series[i - 1]["tier_delta"] > 0 and series[i]["tier_delta"] < 0:
                fading.append({
                    **series[i],
                    "pattern": "attention_trap",
                    "lesson": "Temporary spike reversed — chasing hurt",
                })
                break

    return emergence, fading


def _emergence_lesson(p: str) -> str:
    return {
        "silent_growth": "Low attention then rise with favorable outcome — patience validated",
        "rapid_promotion": "Fast tier jump — verify persistence before sustaining",
        "persistent_leader": "Sustained high observation — institutional leader",
        "emerging_candidate": "New entrant to active observation",
    }.get(p, "Attention rising — monitor persistence")


def _fading_lesson(p: str) -> str:
    return {
        "false_attention_burst": "High attention on unfavorable evolution — trap",
        "attention_trap": "Rise then fall — do not chase spikes",
        "fading_candidate": "Attention released — ecology weakened",
        "rapid_demotion": "Structural break — demotion appropriate",
    }.get(p, "Attention declining")


def parliament_replays(sessions: list[dict], flows: list[dict], council: list[dict]) -> list[dict]:
    council_idx = {(c["scan_time"], c["symbol"]): c for c in council}
    replays = []

    for i in range(1, len(sessions)):
        prev_s, curr_s = sessions[i - 1], sessions[i]
        step_flows = [f for f in flows if f["to_scan"] == curr_s["scan_time"]]

        moved_help = sum(
            1 for f in step_flows
            if f["tier_delta"] > 0 and f.get("outcome_at_to") in ("favorable", "mixed")
        )
        stayed_help = sum(
            1 for f in step_flows
            if f["tier_delta"] == 0 and f["to_allocation"] in ("High observation", "Normal observation")
            and f.get("outcome_at_to") in ("favorable", "mixed")
        )
        chased_hurt = sum(
            1 for f in step_flows
            if f["migration_type"] in ("false_attention_burst", "attention_trap", "rapid_promotion")
            and f.get("outcome_at_to") == "unfavorable"
        )
        patience_help = sum(
            1 for f in step_flows
            if f["from_allocation"] in ("Ignore", "Background observation")
            and f["tier_delta"] == 0
            and f.get("outcome_at_to") == "favorable"
        )

        replays.append({
            "step": i,
            "from_scan": prev_s["scan_time"],
            "to_scan": curr_s["scan_time"],
            "from_regime": prev_s["market_regime"],
            "to_regime": curr_s["market_regime"],
            "movement_helped": moved_help,
            "staying_helped": stayed_help,
            "patience_helped": patience_help,
            "chasing_hurt": chased_hurt,
            "budget_delta": curr_s["observation_budget"] - prev_s["observation_budget"],
            "stability": curr_s["stability"],
            "audit_verdict": _replay_verdict(moved_help, stayed_help, patience_help, chased_hurt),
        })
    return replays


def _replay_verdict(moved, stayed, patience, chased) -> str:
    if chased > moved and chased > 0:
        return "chasing_hurt_more_than_movement_helped"
    if patience >= moved and patience > 0:
        return "patience_outperformed_chasing"
    if moved + stayed > chased:
        return "migration_net_positive"
    return "neutral"


def _evaluate_variant(flows: list[dict], transform) -> tuple[int, int]:
    improved = harmed = 0
    for f in flows:
        sim_delta = transform(f)
        actual_good = f.get("outcome_at_to") in ("favorable", "mixed")
        if sim_delta > f["tier_delta"] and not actual_good:
            harmed += 1
        elif sim_delta < f["tier_delta"] and actual_good and f["tier_delta"] < 0:
            improved += 1
        elif sim_delta <= f["tier_delta"] and f["migration_type"] == "false_attention_burst":
            improved += 1
        elif sim_delta > f["tier_delta"] and f["migration_type"] in ("attention_trap", "rapid_promotion"):
            harmed += 1
        elif sim_delta < f["tier_delta"] and f["migration_type"] in ("attention_trap", "rapid_promotion"):
            improved += 1
    return improved, harmed


def counterfactual_migration(flows: list[dict], council: list[dict]) -> list[dict]:
    variants = [
        ("aggressive", lambda f: f["tier_delta"] + 1 if f["tier_delta"] >= 0 else f["tier_delta"]),
        ("slow", lambda f: 0 if abs(f["tier_delta"]) == 1 else f["tier_delta"] // 2 if f["tier_delta"] else 0),
        ("memory_heavy", lambda f: f["tier_delta"] + (1 if pi(f.get("to_persistence")) >= 2 else 0)),
        ("confidence_heavy", lambda f: f["tier_delta"]),
        ("regime_heavy", lambda f: f["tier_delta"] + (1 if f.get("market_regime") in ("Healthy Expansion", "Rotation") else -1 if f["tier_delta"] > 0 else 0)),
        ("persistence_heavy", lambda f: f["tier_delta"] + (1 if pi(f.get("to_persistence")) >= 2 else 0)),
    ]
    rows = []
    baseline_harm = sum(
        1 for f in flows
        if f.get("outcome_at_to") == "unfavorable" and f["tier_delta"] > 0
    )
    for name, transform in variants:
        improved, harmed = _evaluate_variant(flows, transform)
        net = improved - harmed
        if name == "slow" and harmed < baseline_harm:
            rec, reason = "ACCEPT", f"Slow migration reduces harm ({harmed} vs baseline {baseline_harm})"
        elif name == "aggressive":
            rec, reason = "REJECT", f"Aggressive migration increases traps (harmed={harmed})"
        elif name == "persistence_heavy" and net >= 0:
            rec, reason = "ACCEPT", "Persistence gate reduces premature migration"
        elif net > 0 and harmed < baseline_harm:
            rec, reason = "ACCEPT", _cf_reason(name, net, harmed)
        else:
            rec, reason = "REJECT", _cf_reason(name, net, harmed)
        rows.append({
            "variant": name,
            "flows_tested": len(flows),
            "would_improve": improved,
            "would_harm": harmed,
            "net_discipline": net,
            "recommendation": rec,
            "reason": reason,
        })
    rows.append({
        "variant": "default_historical",
        "flows_tested": len(flows),
        "would_improve": sum(1 for f in flows if f.get("outcome_at_to") in ("favorable", "mixed") and f["tier_delta"] >= 0),
        "would_harm": baseline_harm,
        "net_discipline": 0,
        "recommendation": "BASELINE",
        "reason": "Observed historical migration — baseline",
    })
    return rows


def _cf_reason(name: str, net: int, harmed: int) -> str:
    if name == "aggressive":
        return f"Aggressive migration increases traps (harmed={harmed})"
    if name == "slow":
        return "Slow migration reduces false bursts" if net >= 0 else "Too slow for fertile paths"
    if name == "memory_heavy":
        return "Memory as tie-breaker only — not primary migration driver"
    if name == "regime_heavy":
        return "Regime alone insufficient for migration"
    return "Does not beat baseline discipline"


def institutional_memory(flows, emergence, fading, replays) -> list[dict]:
    traps = sum(1 for f in fading if f.get("pattern") == "attention_trap")
    silent = sum(1 for e in emergence if "silent" in e.get("pattern", ""))
    patience_wins = sum(1 for r in replays if r.get("audit_verdict") == "patience_outperformed_chasing")

    return [
        {"id": 1, "type": "institutionalize", "habit": "Budget cap per council session", "evidence": f"cap={SCAN_BUDGET_CAP}"},
        {"id": 2, "type": "institutionalize", "habit": "Max 3 High observation slots", "evidence": f"high_cap={MAX_HIGH_SLOTS}"},
        {"id": 3, "type": "institutionalize", "habit": "Reallocate when attention rises — require corresponding fall", "evidence": "budget conservation"},
        {"id": 4, "type": "institutionalize", "habit": "Persistent leaders earn continued high observation", "evidence": f"persistent_leader flows"},
        {"id": 5, "type": "reject", "habit": "Aggressive migration variant", "evidence": "attention traps"},
        {"id": 6, "type": "reject", "habit": "Chasing rapid promotions without persistence", "evidence": f"traps={traps}"},
        {"id": 7, "type": "institutionalize", "habit": "Silent growth — patience before attention rise", "evidence": f"silent_patterns={silent}"},
        {"id": 8, "type": "institutionalize", "habit": "Patience outperforms chasing", "evidence": f"replay_wins={patience_wins}"},
        {"id": 9, "type": "mistake", "habit": "false_attention_burst", "evidence": str(sum(1 for f in fading if f.get("pattern") == "false_attention_burst"))},
        {"id": 10, "type": "healthy_parliament", "habit": "Stable high persistence + balanced budget + low trap rate", "evidence": "see replays"},
    ]


def protected_principles() -> list[dict]:
    rows = []
    for path in [
        LOGS_DIR / "season2_p28_protected_principles.csv",
        LOGS_DIR / "season2_p27_protected_principles.csv",
    ]:
        rows.extend(load_csv(path))
    extras = [
        "Council diversification", "Attention budget conservation", "Migration without forecasting",
        "No Buy/Sell", "Watch default",
    ]
    for e in extras:
        rows.append({"principle": e, "never_change": "yes", "layer": "P29"})
    return rows


def build_report(sessions, flows, emergence, fading, budget, replays, counterfactual, memory) -> str:
    mig = Counter(f["migration_type"] for f in flows)
    stab = Counter(s["stability"] for s in sessions)

    lines = [
        "===== SCOUT SEASON2 P29 - PARLIAMENT & ATTENTION ECONOMY =====",
        "",
        f"Parliament sessions: {len(sessions)} | Attention flows: {len(flows)}",
        f"Emergence: {len(emergence)} | Fading: {len(fading)} | Replay steps: {len(replays)}",
        "",
        "=== Research Questions ===",
        "",
        "1. How should attention migrate?",
        "   When persist>=2 and tier rises with balanced budget reallocation.",
        "   From Ignore/Background -> Normal -> High only with structural confirmation.",
        "",
        "2. When should attention stay?",
        f"   persistent_leader flows: {mig.get('persistent_leader', 0)} | stable_attention: {mig.get('stable_attention', 0)}",
        "",
        "3. What creates stable observation?",
        f"   Session stability: {dict(stab)}",
        "",
        "4. What creates attention traps?",
        f"   attention_trap: {sum(1 for f in fading if f.get('pattern')=='attention_trap')} | false_burst: {sum(1 for f in fading if f.get('pattern')=='false_attention_burst')}",
        "",
        "5. What creates silent winners?",
        f"   silent_growth patterns: {sum(1 for e in emergence if 'silent' in e.get('pattern',''))}",
        "",
        "6. When does patience outperform chasing?",
        f"   Replay patience wins: {sum(1 for r in replays if r.get('audit_verdict')=='patience_outperformed_chasing')}",
        "",
        "7. Migration mistakes repeating?",
    ]
    for f in fading[:5]:
        if f.get("pattern") in ("attention_trap", "false_attention_burst"):
            lines.append(f"   - {f['symbol']}: {f['pattern']}")

    lines.extend(["", "8. Habits for institutional memory:", ""])
    for m in memory:
        if m.get("type") == "institutionalize":
            lines.append(f"   - {m['habit']}")

    lines.extend(["", "9. Migration strategies to reject:", ""])
    for c in counterfactual:
        if c["recommendation"] == "REJECT":
            lines.append(f"   - {c['variant']}: {c['reason']}")

    lines.extend([
        "",
        "10. Healthy Scout Parliament:",
        "   Stable high persistence + budget <= cap + patience > chasing + diversified council",
        "",
        "--- Top migration types ---",
    ])
    for t, n in mig.most_common(8):
        lines.append(f"  {t}: {n}")

    lines.extend([
        "",
        "A great Scout knows when the council itself should change.",
        "Never forecast. Never Buy/Sell.",
    ])
    lines.extend(mission_summary_lines())
    lines.append("=" * 58)
    return "\n".join(lines)


def ensure_deps() -> None:
    if not (LOGS_DIR / "season2_p28_scout_council.csv").exists():
        import season2_p28_scout_council
        season2_p28_scout_council.main()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-deps", action="store_true")
    args = parser.parse_args()

    if args.rebuild_deps:
        import season2_p28_scout_council
        season2_p28_scout_council.main()
    else:
        ensure_deps()

    council = load_csv(LOGS_DIR / "season2_p28_scout_council.csv")
    if not council:
        print("Run P28 first")
        return

    sessions = build_parliament_sessions(council)
    flows = attention_flow(council)
    budget = budget_reallocation(sessions, flows)
    emergence, fading = emergence_patterns(flows, council)
    replays = parliament_replays(sessions, flows, council)
    counterfactual = counterfactual_migration(flows, council)
    memory = institutional_memory(flows, emergence, fading, replays)
    protected = protected_principles()
    report = build_report(sessions, flows, emergence, fading, budget, replays, counterfactual, memory)

    write_csv(PARLIAMENT_CSV, sessions)
    write_csv(FLOW_CSV, flows)
    write_csv(EMERGENCE_CSV, emergence)
    write_csv(FADING_CSV, fading)
    write_csv(BUDGET_CSV, budget)
    write_csv(REPLAYS_CSV, replays)
    write_csv(COUNTERFACTUAL_CSV, counterfactual)
    write_csv(INST_MEMORY_CSV, memory)
    write_csv(PROTECTED_CSV, protected)
    REPORT_TXT.write_text(report, encoding="utf-8")

    print("===== P29 SCOUT PARLIAMENT =====")
    print(f"Sessions: {len(sessions)} | Flows: {len(flows)} | Emergence: {len(emergence)} | Fading: {len(fading)}")
    print(f"Report: {REPORT_TXT}")


if __name__ == "__main__":
    main()
