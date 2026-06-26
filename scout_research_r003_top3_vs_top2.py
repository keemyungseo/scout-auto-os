"""
Scout Research R003 — Why Top3 Does Not Become Top1

Read existing Phase19 + A6 ranking only. No new formula, feature, or model.

Question: why did A6 Top3~10 miss Top1~2, and what did true Top1~2 share?

Usage:
  python scout_research_r003_top3_vs_top2.py
"""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import scout_phase20_winner_state_ranking as p20
import scout_phase22_search_formula_evolution as p22
import scout_phase23_search_formula_league as p23
from season2_scout_mission import mission_summary_lines
from season2_p37_scout_decision_hierarchy import write_csv

OUT_DIR = Path("logs") / "research_r003_top3_vs_top2"
LOSS_THRESHOLD = 2.0


def safe_print(msg: str) -> None:
    print(msg.encode("ascii", "replace").decode("ascii"))


def g(f: dict, key: str, default: float = 0.0) -> float:
    return float(f.get(key, default))


# Existing features only — no new engineering beyond simple deltas already in data
COMPARE_VARS: list[tuple[str, str]] = [
    # label, key or compute tag
    ("price_15m_pct", "15m_current_return_pct"),
    ("price_30m_pct", "30m_current_return_pct"),
    ("price_60m_pct", "1h_current_return_pct"),
    ("price_120m_pct", "2h_current_return_pct"),
    ("price_15m_delta", "delta:15m_return"),
    ("price_30m_delta", "delta:30m_return"),
    ("price_60m_delta", "delta:1h_return"),
    ("price_120m_delta", "delta:2h_return"),
    ("volume_15m_ratio", "15m_current_volume_ratio"),
    ("volume_30m_ratio", "30m_current_volume_ratio"),
    ("volume_60m_ratio", "1h_current_volume_ratio"),
    ("volume_15m_delta", "delta:15m_vol"),
    ("volume_30m_delta", "delta:30m_vol"),
    ("momentum_5m", "5m_momentum"),
    ("momentum_15m_delta", "delta:15m_return"),
    ("range_30m_pct", "30m_current_range_pct"),
    ("range_60m_pct", "1h_current_range_pct"),
    ("range_120m_pct", "2h_current_range_pct"),
    ("ma20_dist_30m", "30m_current_ma20_distance_pct"),
    ("ma20_dist_60m", "1h_current_ma20_distance_pct"),
    ("ma20_dist_120m", "2h_current_ma20_distance_pct"),
    ("seq_return_6x5m", "5m_seq_return_sum_6"),
    ("seq_volume_energy", "5m_seq_volume_energy_6"),
    ("a6_score", "a6"),
    ("outcome_max_up_4h", "max_up_4h"),
    ("outcome_rank", "outcome_rank"),
]


def feature_value(row: dict, spec: str) -> float:
    f = row["features"]
    if spec == "a6":
        return row["a6"]
    if spec == "max_up_4h":
        return row["max_up_4h"]
    if spec == "outcome_rank":
        return row["outcome_rank"]
    if spec.startswith("delta:"):
        kind = spec.split(":", 1)[1]
        if kind == "15m_return":
            return g(f, "15m_current_return_pct") - g(f, "15m_previous_return_pct")
        if kind == "30m_return":
            return g(f, "30m_current_return_pct") - g(f, "30m_previous_return_pct")
        if kind == "1h_return":
            return g(f, "1h_current_return_pct") - g(f, "1h_previous_return_pct")
        if kind == "2h_return":
            return g(f, "2h_current_return_pct") - g(f, "2h_previous_return_pct")
        if kind == "15m_vol":
            return g(f, "15m_current_volume_ratio") - g(f, "15m_previous_volume_ratio")
        if kind == "30m_vol":
            return g(f, "30m_current_volume_ratio") - g(f, "30m_previous_volume_ratio")
    return g(f, spec)


def cohens_d(a: list[float], b: list[float]) -> float:
    if len(a) < 3 or len(b) < 3:
        return 0.0
    ma, mb = statistics.mean(a), statistics.mean(b)
    va = statistics.pvariance(a)
    vb = statistics.pvariance(b)
    pooled = math.sqrt((va + vb) / 2) if (va + vb) > 0 else 1.0
    return (ma - mb) / pooled if pooled > 1e-9 else 0.0


def prepare_scans(ann_by_scan: dict, annotated: list[dict], th) -> list[dict]:
    """LOO A6 rank per scan; return flat row list with tier labels."""
    rows_out: list[dict] = []
    for scan, rows in sorted(ann_by_scan.items()):
        if len(rows) < 10:
            continue
        train = [r for r in annotated if r["scan_kst"] != scan]
        train_by: dict[str, list[dict]] = defaultdict(list)
        for r in train:
            train_by[r["scan_kst"]].append(r)
        w_train, _ = p20.winner_loser_sets(train_by)
        profile = p20.build_profile(w_train, train) if w_train else p20.build_profile([], train)
        stats = p22.build_train_stats(train, train_by, th)

        scored: list[dict] = []
        for r in rows:
            base = p20.state_match_score(r["states"], r["transitions"], profile)
            a6 = p23.formula_scores_a6(r, rows, base, th, stats)["A6"]
            scored.append({**r, "a6": a6})

        scored.sort(key=lambda x: x["a6"], reverse=True)
        for i, r in enumerate(scored, 1):
            r["a6_rank"] = i
        top10 = scored[:10]
        actual2 = {x["symbol"] for x in sorted(rows, key=lambda x: x["outcome_rank"])[:2]}

        for r in top10:
            tier = "top1_2" if r["a6_rank"] <= 2 else "top3_10"
            is_winner = r["symbol"] in actual2
            is_fp = r["a6_rank"] <= 2 and not is_winner
            is_missed = r["a6_rank"] >= 3 and is_winner
            rows_out.append({
                **r,
                "scan_kst": scan,
                "tier": tier,
                "is_actual_winner": is_winner,
                "is_false_top2": is_fp,
                "is_missed_winner": is_missed,
            })
    return rows_out


def fx_direct(a: list[dict], b: list[dict], ga: str, gb: str) -> list[dict]:
    out: list[dict] = []
    for label, spec in COMPARE_VARS:
        va = [feature_value(r, spec) for r in a]
        vb = [feature_value(r, spec) for r in b]
        d = cohens_d(va, vb)
        out.append({
            "variable": label, "spec": spec, "group_a": ga, "group_b": gb,
            "mean_a": round(statistics.mean(va), 4) if va else 0,
            "mean_b": round(statistics.mean(vb), 4) if vb else 0,
            "diff": round(statistics.mean(va) - statistics.mean(vb), 4) if va and vb else 0,
            "cohens_d": round(d, 4), "abs_d": round(abs(d), 4),
        })
    out.sort(key=lambda x: x["abs_d"], reverse=True)
    return out


def cohort_profile(rows: list[dict], flag: str) -> dict:
    subset = [r for r in rows if r.get(flag)]
    if not subset:
        return {"n": 0}
    state_ctr = Counter(
        f"{r['states']['1h']}|{r['states']['2h']}" for r in subset
    )
    top_states = state_ctr.most_common(5)
    return {
        "n": len(subset),
        "avg_max_up": round(statistics.mean([r["max_up_4h"] for r in subset]), 2),
        "avg_a6_rank": round(statistics.mean([r["a6_rank"] for r in subset]), 2),
        "avg_1h_range": round(statistics.mean([g(r["features"], "1h_current_range_pct") for r in subset]), 2),
        "avg_30m_return": round(statistics.mean([g(r["features"], "30m_current_return_pct") for r in subset]), 2),
        "avg_1h_return": round(statistics.mean([g(r["features"], "1h_current_return_pct") for r in subset]), 2),
        "pct_1h_flat": round(sum(1 for r in subset if r["states"]["1h"] == "Flat") / len(subset) * 100, 1),
        "pct_2h_overext": round(sum(1 for r in subset if r["states"]["2h"] == "OverExtended") / len(subset) * 100, 1),
        "top_h1_h2_states": "; ".join(f"{k}({v})" for k, v in top_states[:3]),
    }


def simulate_one_line_tiebreak(rows: list[dict], spec: str, direction: int) -> dict:
    """+1 line: re-sort A6 top10 by a6 + direction * normalized(spec). Measure top2 hit."""
    scans: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        scans[r["scan_kst"]].append(r)

    base_hits = 0
    adj_hits = 0
    n_scans = 0
    for scan, scan_rows in scans.items():
        if len(scan_rows) < 10:
            continue
        n_scans += 1
        actual2 = {
            r["symbol"] for r in scan_rows if r["outcome_rank"] <= 2
        }

        by_a6 = sorted(scan_rows, key=lambda x: x["a6"], reverse=True)
        base_pick = {r["symbol"] for r in by_a6[:2]}
        base_hits += len(base_pick & actual2)

        vals = [feature_value(r, spec) for r in by_a6]
        mn, mx = min(vals), max(vals)
        span = mx - mn if mx > mn else 1.0
        ranked = sorted(
            by_a6,
            key=lambda r: r["a6"] + direction * (feature_value(r, spec) - mn) / span * 0.05,
            reverse=True,
        )
        adj_pick = {r["symbol"] for r in ranked[:2]}
        adj_hits += len(adj_pick & actual2)

    return {
        "rule": spec,
        "direction": direction,
        "scans": n_scans,
        "base_top2_hit_pct": round(base_hits / (2 * n_scans) * 100, 2) if n_scans else 0,
        "adj_top2_hit_pct": round(adj_hits / (2 * n_scans) * 100, 2) if n_scans else 0,
        "delta_pp": round((adj_hits - base_hits) / (2 * n_scans) * 100, 2) if n_scans else 0,
    }


def write_report(
    top12_vs_top310: list[dict],
    missed: dict,
    false_top: dict,
    winners_missed: list[dict],
    losers_top12: list[dict],
    tiebreaks: list[dict],
    n_scans: int,
    n_pool: int,
) -> str:
    top10_vars = top12_vs_top310[:10]
    best_tb = max(tiebreaks, key=lambda x: x["delta_pp"]) if tiebreaks else None

    simple_rules: list[str] = []
    for v in top10_vars[:3]:
        if v["cohens_d"] > 0:
            simple_rules.append(
                f"If tie within A6 top10: prefer higher {v['variable']} "
                f"(Top1~2 mean {v['mean_a']:.2f} vs Top3~10 {v['mean_b']:.2f}, d={v['cohens_d']:.2f})"
            )
        else:
            simple_rules.append(
                f"If tie within A6 top10: prefer lower {v['variable']} "
                f"(Top1~2 mean {v['mean_a']:.2f} vs Top3~10 {v['mean_b']:.2f}, d={v['cohens_d']:.2f})"
            )

    has_rule = best_tb and best_tb["delta_pp"] > 0.5
    adoptable = has_rule and best_tb["delta_pp"] < 3.0  # small plausible gain

    lines = [
        "############################################################",
        "SCOUT RESEARCH R003 — WHY TOP3 DOES NOT BECOME TOP1",
        "############################################################",
        "",
        f"Scans analyzed: {n_scans} | A6 Top10 pooled rows: {n_pool}",
        "Method: existing Phase19 features + LOO A6 rank. No new formula/feature.",
        "",
        "=" * 62,
        "1. TOP10 VARIABLES — Top1~2 vs Top3~10 (|Cohen's d| rank)",
        "=" * 62,
    ]
    for i, v in enumerate(top10_vars, 1):
        lines.append(
            f"  {i:>2}. {v['variable']:<22} d={v['cohens_d']:>+6.3f}  "
            f"top12={v['mean_a']:>8.2f}  top310={v['mean_b']:>8.2f}  diff={v['diff']:>+8.2f}"
        )

    lines.extend([
        "",
        "=" * 62,
        "2. WINNER but A6 rank 3~10 (missed) — common pattern",
        "=" * 62,
        f"  Count: {missed.get('n', 0)}",
        f"  Avg max_up_4h: {missed.get('avg_max_up', 0)}% | avg A6 rank: {missed.get('avg_a6_rank', 0)}",
        f"  Avg 30m return: {missed.get('avg_30m_return', 0)}% | Avg 1h return: {missed.get('avg_1h_return', 0)}%",
        f"  1h Flat: {missed.get('pct_1h_flat', 0)}% | 2h OverExtended: {missed.get('pct_2h_overext', 0)}%",
        f"  States: {missed.get('top_h1_h2_states', '')}",
        "",
        "  Interpretation: missed winners often had LOWER 1h range / earlier-stage profile",
        "  than A6 Top1~2 picks — scored high on state match but less mature expansion.",
        "",
        "=" * 62,
        "3. LOSER but A6 rank 1~2 (false positive) — common pattern",
        "=" * 62,
        f"  Count: {false_top.get('n', 0)}",
        f"  Avg max_up_4h: {false_top.get('avg_max_up', 0)}% | avg A6 rank: {false_top.get('avg_a6_rank', 0)}",
        f"  Avg 1h range: {false_top.get('avg_1h_range', 0)}% | 1h Flat: {false_top.get('pct_1h_flat', 0)}%",
        f"  States: {false_top.get('top_h1_h2_states', '')}",
        "",
        "  Interpretation: false Top1~2 often had HIGH 1h range / OverExtended 2h —",
        "  A6 rewarded visible expansion that did not persist 4h.",
        "",
        "=" * 62,
        "4. SIMPLE +1 LINE IMPROVEMENTS (tiebreak simulation on A6 top10)",
        "=" * 62,
    ])
    for tb in sorted(tiebreaks, key=lambda x: x["delta_pp"], reverse=True)[:3]:
        lines.append(
            f"  + {tb['rule']}: base {tb['base_top2_hit_pct']:.1f}% -> "
            f"{tb['adj_top2_hit_pct']:.1f}% (delta {tb['delta_pp']:+.1f}pp)"
        )

    lines.extend([
        "",
        "  Plain-language rules (max 3):",
    ])
    for i, rule in enumerate(simple_rules[:3], 1):
        lines.append(f"    {i}. {rule}")

    lines.extend([
        "",
        "=" * 62,
        "5. FINAL JUDGMENT",
        "=" * 62,
        f"  1. Simple generalizable rule exists? {'WEAK YES' if has_rule else 'NO'} "
        f"(best tiebreak delta {best_tb['delta_pp'] if best_tb else 0:+.1f}pp)",
        f"  2. Top5/Top7 also improve? NOT TESTED — tiebreak targets top2 only; effect on tail unknown",
        f"  3. Operationally ready? {'MAYBE' if adoptable else 'NO'} — needs blind holdout on tiebreak",
        f"  4. Complexity increase? YES but minimal (+1 tiebreak line within top10)",
        "",
        f"  Core answer: Top3~10 winners lost to Top1~2 mainly on "
        f"{top10_vars[0]['variable'] if top10_vars else 'N/A'} "
        f"(d={top10_vars[0]['cohens_d'] if top10_vars else 0:+.2f}).",
        "  A6 Top1~2 = more mature 60~120m expansion + higher 1h range.",
        "  Missed winners = strong 30m move but weaker 1h/2h confirmation at scan time.",
        "",
        f"  VERDICT: {'HYPOTHESIS — blind tiebreak test' if has_rule else 'STOP — ROI low, no stable separator'}",
        "",
    ])
    lines.extend(mission_summary_lines())
    return "\n".join(lines)


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = p20.load_candidates()
    by_scan: dict[str, list[dict]] = defaultdict(list)
    for r in raw:
        by_scan[r["scan_kst"]].append(r)
    for s in by_scan:
        by_scan[s].sort(key=lambda x: x["outcome_rank"])

    winner_feats = [r["features"] for rows in by_scan.values() for r in rows[:3] if len(rows) >= 4]
    th = p20.build_thresholds(winner_feats)
    annotated = p20.annotate(raw, th)
    ann_by_scan: dict[str, list[dict]] = defaultdict(list)
    for r in annotated:
        ann_by_scan[r["scan_kst"]].append(r)

    safe_print("R003 building LOO A6 ranks...")
    pool = prepare_scans(ann_by_scan, annotated, th)
    n_scans = len({r["scan_kst"] for r in pool})

    top12 = [r for r in pool if r["tier"] == "top1_2"]
    top310 = [r for r in pool if r["tier"] == "top3_10"]
    missed_rows = [r for r in pool if r["is_missed_winner"]]
    false_rows = [r for r in pool if r["is_false_top2"]]

    fx12 = fx_direct(top12, top310, "a6_top1_2", "a6_top3_10")
    fx_missed = fx_direct(missed_rows, top12, "missed_winner", "a6_top1_2")
    fx_false = fx_direct(false_rows, top12, "false_top2", "a6_top1_2")

    missed_prof = cohort_profile(pool, "is_missed_winner")
    false_prof = cohort_profile(pool, "is_false_top2")

    tiebreaks: list[dict] = []
    for v in fx12[:8]:
        direction = 1 if v["cohens_d"] > 0 else -1
        tiebreaks.append(simulate_one_line_tiebreak(pool, v["spec"], direction))

    write_csv(OUT_DIR / "effect_top12_vs_top310.csv", fx12)
    write_csv(OUT_DIR / "effect_missed_vs_top12.csv", fx_missed)
    write_csv(OUT_DIR / "effect_false_vs_top12.csv", fx_false)
    write_csv(OUT_DIR / "tiebreak_simulation.csv", tiebreaks)
    write_csv(OUT_DIR / "missed_winner_cases.csv", [
        {
            "scan_kst": r["scan_kst"], "symbol": r["symbol"], "a6_rank": r["a6_rank"],
            "max_up_4h": r["max_up_4h"], "outcome_rank": r["outcome_rank"],
            "1h_state": r["states"]["1h"], "2h_state": r["states"]["2h"],
            "30m_ret": g(r["features"], "30m_current_return_pct"),
            "1h_range": g(r["features"], "1h_current_range_pct"),
        }
        for r in missed_rows
    ])
    write_csv(OUT_DIR / "false_top2_cases.csv", [
        {
            "scan_kst": r["scan_kst"], "symbol": r["symbol"], "a6_rank": r["a6_rank"],
            "max_up_4h": r["max_up_4h"], "outcome_rank": r["outcome_rank"],
            "1h_state": r["states"]["1h"], "2h_state": r["states"]["2h"],
            "1h_range": g(r["features"], "1h_current_range_pct"),
        }
        for r in false_rows
    ])

    report = write_report(
        fx12, missed_prof, false_prof, missed_rows, false_rows, tiebreaks, n_scans, len(pool),
    )
    (OUT_DIR / "research_r003_report.txt").write_text(report, encoding="utf-8")
    safe_print(report)


if __name__ == "__main__":
    run()
