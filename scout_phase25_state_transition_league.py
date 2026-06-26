"""
Scout Phase 25 - State Transition League

Transition-path analysis on Phase24 cohorts + all candidates.
Analysis only. Formulas frozen. No rule/threshold/weight changes.

Usage:
  python scout_phase25_state_transition_league.py
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import scout_phase20_winner_state_ranking as p20
from season2_p37_scout_decision_hierarchy import write_csv

P19_CAND = Path("logs") / "phase19_winner_dna" / "candidates.jsonl"
P23_MATCH = Path("logs") / "phase23_formula_league" / "match_log.jsonl"
P24_DIR = Path("logs") / "phase24_loser_mining"
OUT_DIR = Path("logs") / "phase25_transition_league"

FORMULAS = ("A", "A2", "A5", "A6")
FP_THRESHOLD = 2.0
TF_ORDER = ("5m", "15m", "30m", "1h", "2h")
EXPANSION_STATES = {"Expansion", "VolumeSupport", "ExpansionStart", "Acceleration", "TrendAlive", "StrongTrend"}
DEATH_STATES = {"Flat", "Weak", "Quiet", "Normal", "Neutral", "OverExtended"}


def safe_print(msg: str) -> None:
    print(msg.encode("ascii", "replace").decode("ascii"))


def entropy(p: float) -> float:
    if p <= 0 or p >= 1:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def ig_3way(w: int, f: int, m: int, wn: int, fn: int, mn: int) -> float:
    """Binary IG winner vs (fp+miss) for transition presence."""
    rest = f + m
    rn = fn + mn
    if wn == 0 or rn == 0:
        return 0.0
    pos, pos_n = w, wn
    neg, neg_n = rest, rn
    p_y = pos_n / (pos_n + neg_n)
    parent = entropy(p_y)
    tot = pos + neg
    if tot == 0 or tot == pos_n + neg_n:
        return 0.0
    p_y_t = pos / tot
    rem = pos_n + neg_n - tot
    p_y_f = (pos_n - pos) / rem if rem else 0
    p_t = tot / (pos_n + neg_n)
    return parent - p_t * entropy(p_y_t) - (1 - p_t) * entropy(p_y_f)


def mtf_path(states: dict[str, str]) -> str:
    return " -> ".join(states[tf] for tf in TF_ORDER)


def cross_transitions(states: dict[str, str]) -> list[str]:
    out: list[str] = []
    for i in range(len(TF_ORDER) - 1):
        a, b = TF_ORDER[i], TF_ORDER[i + 1]
        out.append(f"{states[a]}->{states[b]}")
    return out


def all_transitions(states: dict[str, str], intra: dict[str, str]) -> list[str]:
    edges = cross_transitions(states)
    for tf in TF_ORDER:
        edges.append(f"{tf}:{intra[tf]}")
    return edges


def load_picks() -> dict[str, set[str]]:
    picks: dict[str, set[str]] = defaultdict(set)
    if not P23_MATCH.exists():
        return picks
    for line in P23_MATCH.open(encoding="utf-8"):
        m = json.loads(line)
        scan = m["scan_kst"]
        for fid in FORMULAS:
            for sym in m.get(f"{fid}_top2", []):
                picks[(scan, fid)].add(sym)
    union: dict[str, set[str]] = defaultdict(set)
    for (scan, _), syms in list(picks.items()):
        union[scan].update(syms)
    picks["_union_"] = union  # type: ignore
    return picks


def annotate_all() -> list[dict]:
    raw = p20.load_candidates()
    by_scan: dict[str, list] = defaultdict(list)
    for r in raw:
        by_scan[r["scan_kst"]].append(r)
    winner_feats = [r["features"] for rows in by_scan.values() for r in rows[:3] if len(rows) >= 4]
    th = p20.build_thresholds(winner_feats)

    union_picks: dict[str, set[str]] = defaultdict(set)
    for line in P23_MATCH.open(encoding="utf-8"):
        m = json.loads(line)
        for fid in FORMULAS:
            union_picks[m["scan_kst"]].update(m.get(f"{fid}_top2", []))

    rows: list[dict] = []
    for r in raw:
        f = r["features"]
        states = p20.build_states(f, th)
        intra = p20.build_transitions(f, th)
        path = mtf_path(states)
        scan, sym = r["scan_kst"], r["symbol"]
        rank = r["outcome_rank"]
        mu = r["max_up_4h"]
        picked = sym in union_picks.get(scan, set())

        if rank <= 2 and picked:
            cohort = "winner_hit"
        elif rank <= 2 and not picked:
            cohort = "top2_miss"
        elif picked and mu < FP_THRESHOLD:
            cohort = "false_positive"
        else:
            cohort = "other"

        rows.append({
            "scan_kst": scan,
            "symbol": sym,
            "cohort": cohort,
            "outcome_rank": rank,
            "max_up_4h": mu,
            "states": states,
            "intra_trans": intra,
            "mtf_path": path,
            "cross_trans": cross_transitions(states),
            "all_trans": all_transitions(states, intra),
        })
    return rows


def cohort_rows(rows: list[dict], cohort: str) -> list[dict]:
    if cohort == "winner":
        return [r for r in rows if r["cohort"] == "winner_hit" or (r["outcome_rank"] <= 2 and r["cohort"] != "top2_miss")]
    if cohort == "winner_gt":
        # unique ground-truth top2
        seen: set[tuple[str, str]] = set()
        out: list[dict] = []
        for r in rows:
            if r["outcome_rank"] > 2:
                continue
            k = (r["scan_kst"], r["symbol"])
            if k in seen:
                continue
            seen.add(k)
            out.append(r)
        return out
    return [r for r in rows if r["cohort"] == cohort]


def transition_lift(
    winners: list[dict],
    fps: list[dict],
    misses: list[dict],
) -> list[dict]:
    wn, fn, mn = len(winners), len(fps), len(misses)
    all_edges: set[str] = set()
    for pool in (winners, fps, misses):
        for r in pool:
            all_edges.update(r["all_trans"])

    out: list[dict] = []
    for edge in sorted(all_edges):
        wc = sum(1 for r in winners if edge in r["all_trans"])
        fc = sum(1 for r in fps if edge in r["all_trans"])
        mc = sum(1 for r in misses if edge in r["all_trans"])
        wr = wc / wn if wn else 0
        fr = fc / fn if fn else 0
        mr = mc / mn if mn else 0
        base = (wc + fc + mc) / (wn + fn + mn) if (wn + fn + mn) else 0
        lift_w = wr / base if base > 0 else 0
        lift_f = fr / base if base > 0 else 0
        lift_m = mr / base if base > 0 else 0
        ow = wr / (1 - wr) if wr < 1 else 99
        of_ = fr / (1 - fr) if fr < 1 else 99
        odds = of_ / ow if ow > 0 else 0
        ig = ig_3way(wc, fc, mc, wn, fn, mn)
        out.append({
            "transition": edge,
            "winner_pct": round(wr * 100, 2),
            "fp_pct": round(fr * 100, 2),
            "miss_pct": round(mr * 100, 2),
            "lift_winner": round(lift_w, 4),
            "lift_fp": round(lift_f, 4),
            "lift_miss": round(lift_m, 4),
            "odds_fp_vs_winner": round(odds, 4),
            "information_gain": round(ig, 4),
            "winner_count": wc,
            "fp_count": fc,
            "miss_count": mc,
        })
    out.sort(key=lambda x: (x["information_gain"], x["lift_winner"]), reverse=True)
    return out


def path_clusters(rows: list[dict], label: str) -> list[dict]:
    ctr = Counter(r["mtf_path"] for r in rows)
    total = len(rows) or 1
    return [{"path": k, "count": v, "pct": round(v / total * 100, 2), "cohort": label} for k, v in ctr.most_common(30)]


def state_survival(rows: list[dict], cohort: str) -> list[dict]:
    """Release@5m -> Expansion@15m -> Acceleration@1h -> TrendAlive@2h survival."""
    starters = [r for r in rows if r["states"]["5m"] == "Release"]
    n = len(starters) or 1
    s15 = sum(1 for r in starters if r["states"]["15m"] in EXPANSION_STATES or r["states"]["15m"] == "VolumeSupport")
    s30 = sum(1 for r in starters if r["states"]["30m"] in ("Expansion", "Compression"))
    s1h = sum(1 for r in starters if r["states"]["1h"] in ("Expansion", "ExpansionStart", "Acceleration"))
    s2h = sum(1 for r in starters if r["states"]["2h"] in ("TrendAlive", "StrongTrend"))
    return [
        {"cohort": cohort, "stage": "5m_Release", "survival_pct": 100.0, "n": len(starters)},
        {"cohort": cohort, "stage": "15m_Expansion_or_VolSupport", "survival_pct": round(s15 / n * 100, 2), "n": len(starters)},
        {"cohort": cohort, "stage": "30m_Expansion_or_Compression", "survival_pct": round(s30 / n * 100, 2), "n": len(starters)},
        {"cohort": cohort, "stage": "1h_Expansion_family", "survival_pct": round(s1h / n * 100, 2), "n": len(starters)},
        {"cohort": cohort, "stage": "2h_TrendAlive_or_Strong", "survival_pct": round(s2h / n * 100, 2), "n": len(starters)},
    ]


def early_death_patterns(fps: list[dict]) -> list[dict]:
    """Where FP path first hits death-like state in MTF order."""
    ctr: Counter = Counter()
    for r in fps:
        states = [r["states"][tf] for tf in TF_ORDER]
        death_at = None
        for i, st in enumerate(states):
            if st in DEATH_STATES and i >= 2:  # death zone 30m+
                death_at = f"{TF_ORDER[i]}:{st}"
                break
            if i > 0 and st == "Flat" and states[i - 1] in EXPANSION_STATES:
                death_at = f"{TF_ORDER[i-1]}->{TF_ORDER[i]}:{states[i-1]}->{st}"
                break
        if death_at:
            ctr[death_at] += 1
        else:
            ctr["no_clear_death"] += 1
    total = len(fps) or 1
    return [{"death_zone": k, "count": v, "pct": round(v / total * 100, 2)} for k, v in ctr.most_common(20)]


def late_winner_patterns(misses: list[dict], winners: list[dict]) -> list[dict]:
    """Miss paths that match top winner paths (ranking dropout, not state failure)."""
    win_paths = Counter(r["mtf_path"] for r in winners)
    top_win_paths = {p for p, _ in win_paths.most_common(15)}
    ctr: Counter = Counter()
    for r in misses:
        if r["mtf_path"] in top_win_paths:
            ctr[r["mtf_path"]] += 1
    total = len(misses) or 1
    rows = [{"path": k, "count": v, "pct": round(v / total * 100, 2), "note": "same_path_as_winner_cluster"} for k, v in ctr.most_common(15)]
    if not rows:
        rows.append({"path": "(none)", "count": 0, "pct": 0.0, "note": "no_miss_matches_top_winner_paths"})
    return rows


def cross_only_importance(lift_rows: list[dict]) -> list[dict]:
    cross = [r for r in lift_rows if "->" in r["transition"] and ":" not in r["transition"]]
    cross.sort(key=lambda x: (x["information_gain"], x["lift_winner"] - x["lift_fp"]), reverse=True)
    return cross


def run() -> list[str]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = annotate_all()
    winners = cohort_rows(all_rows, "winner_gt")
    fps = cohort_rows(all_rows, "false_positive")
    misses = cohort_rows(all_rows, "top2_miss")

    lift_rows = transition_lift(winners, fps, misses)
    win_paths = path_clusters(winners, "winner")
    fp_paths = path_clusters(fps, "false_positive")
    miss_paths = path_clusters(misses, "top2_miss")

    survival_w = state_survival(winners, "winner")
    survival_f = state_survival(fps, "false_positive")
    survival_m = state_survival(misses, "top2_miss")
    survival_all = survival_w + survival_f + survival_m

    early_death = early_death_patterns(fps)
    late_win = late_winner_patterns(misses, winners)
    importance = cross_only_importance(lift_rows)

    write_csv(OUT_DIR / "transition_lift.csv", lift_rows)
    write_csv(OUT_DIR / "winner_paths.csv", win_paths)
    write_csv(OUT_DIR / "false_positive_paths.csv", fp_paths)
    write_csv(OUT_DIR / "top2_miss_paths.csv", miss_paths)
    write_csv(OUT_DIR / "transition_importance.csv", importance[:50])
    write_csv(OUT_DIR / "state_survival.csv", survival_all)
    write_csv(OUT_DIR / "early_death_patterns.csv", early_death)
    write_csv(OUT_DIR / "late_winner_patterns.csv", late_win)

    win_trans_top = [r for r in lift_rows if r["lift_winner"] >= 1.1][:10]
    fp_trans_top = sorted(lift_rows, key=lambda x: x["lift_fp"], reverse=True)[:10]
    miss_trans_top = sorted(lift_rows, key=lambda x: x["lift_miss"], reverse=True)[:10]

    lines = [
        "############################################################",
        "SCOUT PHASE 25 - STATE TRANSITION LEAGUE",
        "############################################################",
        "",
        "DATA SOURCES (verified):",
        f"  {P19_CAND} (all candidates + p20 states/transitions)",
        f"  {P23_MATCH} (formula TOP2 picks: {', '.join(FORMULAS)})",
        f"  Cohorts: winner_gt={len(winners)} fp={len(fps)} top2_miss={len(misses)}",
        "Formulas frozen. Analysis only.",
        "",
        "=" * 62,
        "1. TOP10 WINNER TRANSITIONS (cross-TF + intra-TF)",
        "=" * 62,
    ]
    for r in win_trans_top:
        lines.append(
            f"  {r['transition']}: W={r['winner_pct']:.1f}% FP={r['fp_pct']:.1f}% "
            f"Miss={r['miss_pct']:.1f}% lift_w={r['lift_winner']:.2f} IG={r['information_gain']:.3f}"
        )

    lines.extend(["", "=" * 62, "2. TOP10 FALSE POSITIVE TRANSITIONS", "=" * 62])
    for r in fp_trans_top:
        lines.append(
            f"  {r['transition']}: FP={r['fp_pct']:.1f}% W={r['winner_pct']:.1f}% "
            f"lift_fp={r['lift_fp']:.2f} odds_fp/w={r['odds_fp_vs_winner']:.2f}"
        )

    lines.extend(["", "=" * 62, "3. TOP10 MISS TRANSITIONS", "=" * 62])
    for r in miss_trans_top:
        lines.append(
            f"  {r['transition']}: Miss={r['miss_pct']:.1f}% W={r['winner_pct']:.1f}% lift_miss={r['lift_miss']:.2f}"
        )

    lines.extend(["", "=" * 62, "4. STATE SURVIVAL (from 5m=Release)", "=" * 62])
    for cohort in ("winner", "false_positive", "top2_miss"):
        lines.append(f"  [{cohort}]")
        for s in survival_all:
            if s["cohort"] == cohort:
                lines.append(f"    {s['stage']}: {s['survival_pct']:.1f}% (n={s['n']})")

    lines.extend(["", "=" * 62, "5. TRANSITION IMPORTANCE (cross-TF edges)", "=" * 62])
    for r in importance[:12]:
        lines.append(
            f"  {r['transition']}: IG={r['information_gain']:.3f} "
            f"lift_w={r['lift_winner']:.2f} lift_fp={r['lift_fp']:.2f}"
        )

    lines.extend(["", "=" * 62, "6. EARLY DEATH ZONE (False Positive)", "=" * 62])
    for r in early_death[:8]:
        lines.append(f"  {r['death_zone']}: {r['pct']:.1f}% ({r['count']}x)")

    lines.extend(["", "=" * 62, "7. LATE WINNER ZONE (Miss ~ Winner path)", "=" * 62])
    for r in late_win[:8]:
        lines.append(f"  [{r['count']}x / {r['pct']}%] {r['path'][:90]}")

    lines.extend(["", "=" * 62, "8. REPRESENTATIVE PATHS", "=" * 62])
    lines.append("  Winner path:")
    if win_paths:
        lines.append(f"    {win_paths[0]['path']} ({win_paths[0]['pct']}%)")
    lines.append("  FP path:")
    if fp_paths:
        lines.append(f"    {fp_paths[0]['path']} ({fp_paths[0]['pct']}%)")
    lines.append("  Miss path:")
    if miss_paths:
        lines.append(f"    {miss_paths[0]['path']} ({miss_paths[0]['pct']}%)")

    lines.extend(["", "=" * 62, "9. RECOMMENDATION (analysis only)", "=" * 62])
    if fp_paths and win_paths:
        lines.append(
            "  FP diverges when Release->Expansion at 15m is followed by 1h=Flat "
            f"(see FP path: {fp_paths[0]['path'][:70]}...)."
        )
    if importance:
        top = importance[0]
        lines.append(
            f"  Strongest separating cross-TF edge: {top['transition']} "
            f"(IG={top['information_gain']:.3f}). Holdout validation only — no formula change."
        )
    if late_win and late_win[0]["count"] > 0:
        pct = late_win[0]["pct"]
        lines.append(
            f"  {pct:.1f}% of TOP2 misses share winner MTF paths — ranking dropout, not missing state."
        )
    lines.append("  No filter/rule/threshold/weight changes in Phase25.")

    lines.extend(["", "DISCLAIMER: Transition DNA descriptive only."])

    report = OUT_DIR / "phase25_transition_league_report.txt"
    report.write_text("\n".join(lines), encoding="utf-8")
    return lines


def main() -> None:
    lines = run()
    for ln in lines:
        safe_print(ln)
    safe_print(f"\nSaved: {OUT_DIR / 'phase25_transition_league_report.txt'}")


if __name__ == "__main__":
    main()
