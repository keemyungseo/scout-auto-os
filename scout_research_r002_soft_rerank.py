"""
Scout Research R002 — Soft Re-Rank (Penalty / Bonus on Frozen A6)

R001 lesson: hard reject collapses winner recall. This experiment applies
continuous score adjustments only — no reject, no formula change, no universe cut.

Final = A6 + w * (transition_bonus + trigger_survival - death_pen - energy_pen - flat_pen)

Usage:
  python scout_research_r002_soft_rerank.py
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
import scout_research_r001_loser_energy_funnel as r1
from season2_scout_mission import mission_summary_lines
from season2_p37_scout_decision_hierarchy import write_csv

OUT_DIR = Path("logs") / "research_r002_soft_rerank"
PENALTY_WEIGHTS = (0, 5, 10, 15, 20, 25, 30)
TOP_N_CANDIDATES = 20
WIN_THRESHOLD = 5.0
LOSS_THRESHOLD = 2.0
HOLDOUT_FRAC = 0.30

LIFECYCLE_STATES = ("Expansion", "Strong", "Neutral", "Weak", "Death")
EXPANSION_1H = {"Expansion", "ExpansionStart", "Acceleration"}
EXPANSION_2H = {"TrendAlive", "StrongTrend"}


def safe_print(msg: str) -> None:
    print(msg.encode("ascii", "replace").decode("ascii"))


def g(f: dict, key: str, default: float = 0.0) -> float:
    return float(f.get(key, default))


def classify_lifecycle(states: dict[str, str]) -> str:
    h1, h2, m15, m5 = states["1h"], states["2h"], states["15m"], states["5m"]
    if h2 in ("Flat", "OverExtended") and h1 == "Flat" and m15 in ("Weak",):
        return "Death"
    if m5 == "Quiet" and h1 == "Flat" and h2 == "Flat":
        return "Death"
    if h1 in EXPANSION_1H and h2 in EXPANSION_2H:
        return "Expansion"
    if h1 == "Acceleration" or h2 == "StrongTrend":
        return "Strong"
    if m15 == "Weak" or m5 in ("Quiet",):
        return "Weak"
    return "Neutral"


def learn_transition_model(train: list[dict]) -> dict:
    """Empirical 5-state transition probabilities + death prob by MTF path."""
    seq_counts: Counter[str] = Counter()
    seq_death: Counter[str] = Counter()
    life_counts: Counter[str] = Counter()
    life_death: Counter[str] = Counter()
    combo_exp: Counter[str] = Counter()
    combo_total: Counter[str] = Counter()

    for r in train:
        life = classify_lifecycle(r["states"])
        seq = r.get("mtf_seq", "")
        combo = r.get("combo", "")
        life_counts[life] += 1
        seq_counts[seq] += 1
        combo_total[combo] += 1
        is_death = life == "Death" or r["max_up_4h"] < LOSS_THRESHOLD
        if is_death:
            life_death[life] += 1
            seq_death[seq] += 1
        if life in ("Expansion", "Strong") or r["outcome_rank"] <= 5:
            combo_exp[combo] += 1

    n = len(train) or 1
    base_death = sum(1 for r in train if classify_lifecycle(r["states"]) == "Death") / n

    def death_prob(key: str, death_c: Counter, total_c: Counter, floor: float = 0.05) -> float:
        t = total_c.get(key, 0)
        if t < 8:
            return floor
        return max(floor, min(0.95, death_c.get(key, 0) / t))

    life_death_prob = {s: death_prob(s, life_death, life_counts, base_death) for s in LIFECYCLE_STATES}
    seq_death_prob = {k: death_prob(k, seq_death, seq_counts, base_death) for k in seq_counts}

    trans_bonus: dict[str, float] = {}
    for combo, t in combo_total.items():
        if t < 8:
            trans_bonus[combo] = 0.0
        else:
            trans_bonus[combo] = combo_exp.get(combo, 0) / t

    return {
        "life_death_prob": life_death_prob,
        "seq_death_prob": seq_death_prob,
        "trans_bonus": trans_bonus,
        "base_death": base_death,
    }


def trigger_survival_bonus(states: dict[str, str]) -> float:
    """Focal Release->Expansion chain survival proxy (phase26 empirical)."""
    bonus = 0.0
    if states["5m"] == "Release":
        bonus += 0.15
    if states["15m"] == "Expansion":
        bonus += 0.25
    if states["30m"] == "Expansion":
        bonus += 0.20
    if states["1h"] in EXPANSION_1H:
        bonus += 0.25
    if states["2h"] in EXPANSION_2H:
        bonus += 0.15
    if states["1h"] == "Flat":
        bonus -= 0.30
    if states["2h"] in ("Flat", "OverExtended"):
        bonus -= 0.20
    return max(-0.5, min(1.0, bonus))


def low_energy_penalty(energy: float, peers: list[float]) -> float:
    if not peers:
        return 0.0
    med = statistics.median(peers)
    q1 = statistics.quantiles(peers, n=4)[0] if len(peers) >= 4 else min(peers)
    if energy >= med:
        return 0.0
    span = med - q1
    if span < 1e-6:
        return 0.5 if energy < med else 0.0
    return max(0.0, min(1.0, (med - energy) / span))


def flat_persistence_penalty(states: dict[str, str]) -> float:
    flat_like = {"Flat", "Quiet", "Neutral", "Compression", "Weak"}
    flat_count = sum(1 for tf in ("5m", "15m", "30m", "1h", "2h") if states[tf] in flat_like)
    return flat_count / 5.0


def soft_adjustment(r: dict, model: dict, peer_energies: list[float]) -> dict:
    states = r["states"]
    life = classify_lifecycle(states)
    seq = r.get("mtf_seq", "")
    combo = r.get("combo", "")

    death_p = model["seq_death_prob"].get(
        seq, model["life_death_prob"].get(life, model["base_death"])
    )
    trans_b = model["trans_bonus"].get(combo, 0.0)
    trig_b = trigger_survival_bonus(states)
    energy_pen = low_energy_penalty(r.get("energy", 0.0), peer_energies)
    flat_pen = flat_persistence_penalty(states)

    net = trans_b + trig_b - death_p - energy_pen - flat_pen
    return {
        "lifecycle": life,
        "death_prob": round(death_p, 4),
        "transition_bonus": round(trans_b, 4),
        "trigger_bonus": round(trig_b, 4),
        "energy_penalty": round(energy_pen, 4),
        "flat_penalty": round(flat_pen, 4),
        "soft_net": round(net, 4),
    }


def score_rows(rows: list[dict], model: dict, weight: float) -> list[dict]:
    energies = [r.get("energy", 0.0) for r in rows]
    a6_vals = [r["a6"] for r in rows]
    mn, mx = min(a6_vals), max(a6_vals)
    span = mx - mn if mx > mn else 1.0

    out: list[dict] = []
    for r in rows:
        adj = soft_adjustment(r, model, energies)
        a6_norm = (r["a6"] - mn) / span
        final = a6_norm * 100.0 + weight * adj["soft_net"]
        out.append({**r, **adj, "a6_norm": round(a6_norm, 4), "final_score": round(final, 4)})
    return out


def prepare_scan(rows: list[dict], profile: dict, th: p20.Thresholds, stats: dict) -> list[dict]:
    scored: list[dict] = []
    for r in rows:
        base = p20.state_match_score(r["states"], r["transitions"], profile)
        a6 = p23.formula_scores_a6(r, rows, base, th, stats)["A6"]
        scored.append({
            **r,
            "a6": a6,
            "energy": r1.energy_score(r["features"]),
            "mtf_seq": r.get("mtf_seq") or " -> ".join(
                f"{tf}:{r['states'][tf]}" for tf in ("5m", "15m", "30m", "1h", "2h")
            ),
        })
    return scored


def pick_top(scored: list[dict], k: int, key: str = "final_score") -> list[dict]:
    return sorted(scored, key=lambda x: x[key], reverse=True)[:k]


def ranking_pool(scored: list[dict], weight: float, key: str = "final_score") -> list[dict]:
    """A6 Top20 then soft re-rank; weight=0 keeps pure A6 on full universe."""
    if weight == 0:
        return sorted(scored, key=lambda x: x["a6"], reverse=True)
    pool = pick_top(scored, TOP_N_CANDIDATES, "a6")
    return sorted(pool, key=lambda x: x[key], reverse=True)


def scan_metrics(rows: list[dict], scored: list[dict], weight: float, key: str = "final_score") -> dict:
    by_rank = sorted(rows, key=lambda x: x["outcome_rank"])
    actual2 = {r["symbol"] for r in by_rank[:2]}
    actual3 = {r["symbol"] for r in by_rank[:3]}
    actual5 = {r["symbol"] for r in by_rank[:5]}
    actual7 = {r["symbol"] for r in by_rank[:7]}

    ranked = ranking_pool(scored, weight, key)
    top2 = ranked[:2]
    top3 = ranked[:3]
    top5 = ranked[:5]
    top7 = ranked[:7]

    a6_only_top2 = pick_top(scored, 2, "a6")
    top2_preserved = {r["symbol"] for r in a6_only_top2} == {r["symbol"] for r in top2}

    def pack(picked: list[dict], actual: set[int] | set, k: int) -> dict:
        syms = {p["symbol"] for p in picked}
        mus = [p["max_up_4h"] for p in picked]
        return {
            "hits": len(syms & actual),
            "recall": len(syms & actual) / k if k else 0,
            "avg_return": statistics.mean(mus) if mus else 0.0,
            "win_rate": sum(1 for p in picked if p["max_up_4h"] >= WIN_THRESHOLD) / max(len(picked), 1),
            "loss_rate": sum(1 for p in picked if p["max_up_4h"] < LOSS_THRESHOLD) / max(len(picked), 1),
            "min_return": min(mus) if mus else 0.0,
        }

    m2 = pack(top2, actual2, 2)
    m3 = pack(top3, actual3, 3)
    m5 = pack(top5, actual5, 5)
    m7 = pack(top7, actual7, 7)

    losers_in_top7 = sum(1 for p in top7 if p["max_up_4h"] < LOSS_THRESHOLD)
    winners_in_top7 = sum(1 for p in top7 if p["symbol"] in actual2)

    dist = Counter(classify_lifecycle(p["states"]) for p in top7)

    quality = (
        m2["recall"] * 25
        + m5["recall"] * 20
        + m5["avg_return"] * 1.5
        + m7["win_rate"] * 15
        + (1 - m7["loss_rate"]) * 10
        - m7["loss_rate"] * 5
    )

    return {
        "weight": weight,
        "top2_hits": m2["hits"],
        "top2_recall": m2["recall"],
        "top3_hits": m3["hits"],
        "top3_recall": m3["recall"],
        "top5_hits": m5["hits"],
        "top5_recall": m5["recall"],
        "top5_avg_return": m5["avg_return"],
        "top5_win_rate": m5["win_rate"],
        "top7_hits": m7["hits"],
        "top7_recall": m7["recall"],
        "top7_avg_return": m7["avg_return"],
        "top7_win_rate": m7["win_rate"],
        "top7_loss_rate": m7["loss_rate"],
        "top7_min_return": m7["min_return"],
        "top7_loser_count": losers_in_top7,
        "top7_winner_recall": winners_in_top7 / 2.0,
        "top2_preserved": top2_preserved,
        "lifecycle_top7": dict(dist),
        "candidate_quality": round(quality, 4),
    }


def aggregate(scan_rows: list[dict]) -> dict:
    if not scan_rows:
        return {}
    n = len(scan_rows)
    return {
        "scans": n,
        "top2_hit_pct": round(sum(r["top2_hits"] for r in scan_rows) / (2 * n) * 100, 2),
        "top3_hit_pct": round(sum(r["top3_hits"] for r in scan_rows) / (3 * n) * 100, 2),
        "top5_recall_pct": round(statistics.mean([r["top5_recall"] for r in scan_rows]) * 100, 2),
        "top5_win_rate_pct": round(statistics.mean([r["top5_win_rate"] for r in scan_rows]) * 100, 2),
        "top5_avg_return": round(statistics.mean([r["top5_avg_return"] for r in scan_rows]), 4),
        "top7_recall_pct": round(statistics.mean([r["top7_recall"] for r in scan_rows]) * 100, 2),
        "top7_win_rate_pct": round(statistics.mean([r["top7_win_rate"] for r in scan_rows]) * 100, 2),
        "top7_avg_return": round(statistics.mean([r["top7_avg_return"] for r in scan_rows]), 4),
        "top7_loss_rate_pct": round(statistics.mean([r["top7_loss_rate"] for r in scan_rows]) * 100, 2),
        "top7_min_return_avg": round(statistics.mean([r["top7_min_return"] for r in scan_rows]), 4),
        "top2_preserve_pct": round(sum(1 for r in scan_rows if r["top2_preserved"]) / n * 100, 2),
        "winner_recall_top7_pct": round(statistics.mean([r["top7_winner_recall"] for r in scan_rows]) * 100, 2),
        "loser_in_top7_avg": round(statistics.mean([r["top7_loser_count"] for r in scan_rows]), 2),
        "candidate_quality": round(statistics.mean([r["candidate_quality"] for r in scan_rows]), 4),
    }


def run_loo_grid(ann_by_scan: dict, annotated: list[dict], th, weights: tuple[int, ...]) -> dict[int, list[dict]]:
    scan_keys = sorted(s for s, rows in ann_by_scan.items() if len(rows) >= 4)
    grid: dict[int, list[dict]] = {w: [] for w in weights}

    for scan in scan_keys:
        rows = ann_by_scan[scan]
        train = [r for r in annotated if r["scan_kst"] != scan]
        train_by: dict[str, list[dict]] = defaultdict(list)
        for r in train:
            train_by[r["scan_kst"]].append(r)
        w_train, _ = p20.winner_loser_sets(train_by)
        profile = p20.build_profile(w_train, train) if w_train else p20.build_profile([], train)
        stats = p22.build_train_stats(train, train_by, th)
        model = learn_transition_model(train)

        base_rows = prepare_scan(rows, profile, th, stats)
        for w in weights:
            scored = score_rows(base_rows, model, w)
            grid[w].append(scan_metrics(rows, scored, w))

    return grid


def temporal_holdout(ann_by_scan: dict, annotated: list[dict], th, weights: tuple[int, ...]) -> dict[int, list[dict]]:
    scans = sorted(s for s, rows in ann_by_scan.items() if len(rows) >= 4)
    cut = int(len(scans) * (1 - HOLDOUT_FRAC))
    train_scans, test_scans = set(scans[:cut]), scans[cut:]

    train = [r for r in annotated if r["scan_kst"] in train_scans]
    train_by: dict[str, list[dict]] = defaultdict(list)
    for r in train:
        train_by[r["scan_kst"]].append(r)
    w_train, _ = p20.winner_loser_sets(train_by)
    profile = p20.build_profile(w_train, train) if w_train else p20.build_profile([], train)
    stats = p22.build_train_stats(train, train_by, th)
    model = learn_transition_model(train)

    grid: dict[int, list[dict]] = {w: [] for w in weights}
    for scan in test_scans:
        rows = ann_by_scan[scan]
        base_rows = prepare_scan(rows, profile, th, stats)
        for w in weights:
            scored = score_rows(base_rows, model, w)
            grid[w].append(scan_metrics(rows, scored, w))
    return grid


def lifecycle_distribution(annotated: list[dict]) -> list[dict]:
    ctr = Counter(classify_lifecycle(r["states"]) for r in annotated)
    total = len(annotated) or 1
    return [
        {"lifecycle": s, "count": ctr.get(s, 0), "pct": round(ctr.get(s, 0) / total * 100, 2)}
        for s in LIFECYCLE_STATES
    ]


def write_report(
    loo_summary: list[dict],
    ho_summary: list[dict],
    best_w: int,
    r001_top2: float,
) -> str:
    base = next(r for r in loo_summary if r["weight"] == 0)
    best = next(r for r in loo_summary if r["weight"] == best_w)

    d_top2 = round(best["top2_hit_pct"] - base["top2_hit_pct"], 2)
    d_top5_ret = round(best["top5_avg_return"] - base["top5_avg_return"], 4)
    d_top7_loss = round(best["top7_loss_rate_pct"] - base["top7_loss_rate_pct"], 2)
    d_quality = round(best["candidate_quality"] - base["candidate_quality"], 4)

    better_than_r001 = best["top2_hit_pct"] > r001_top2
    better_than_hard = best["top2_hit_pct"] > r001_top2  # R001 was 16.2

    adoptable = (
        d_top2 >= -1.0
        and (d_top5_ret > 0 or best["top7_loss_rate_pct"] < base["top7_loss_rate_pct"])
        and best["top2_preserve_pct"] >= 70
    )

    verdict = "REJECT"
    if d_top2 > 0.5 and d_quality > 0:
        verdict = "HOLDOUT_CANDIDATE"
    if adoptable and d_top2 >= 0:
        verdict = "SOFT_RERANK_CANDIDATE"

    lines = [
        "############################################################",
        "SCOUT RESEARCH R002 — SOFT RE-RANK (Frozen A6 + Penalty Layer)",
        "############################################################",
        "",
        "NO formula change | NO reject | NO universe cut",
        "Pool: A6 Top20 -> soft re-rank -> Top2/3/5/7",
        "",
        "=" * 72,
        "1. PENALTY WEIGHT KPI TABLE (LOO Blind, 179 scans)",
        "=" * 72,
        f"  {'W':>3} {'Top2%':>7} {'Top3%':>7} {'T5Rec%':>7} {'T5Win%':>7} {'T5Ret%':>7} "
        f"{'T7Rec%':>7} {'T7Win%':>7} {'T7Loss%':>7} {'T2Keep%':>8} {'Quality':>8}",
    ]
    for r in loo_summary:
        lines.append(
            f"  {r['weight']:>3} {r['top2_hit_pct']:>6.1f}% {r['top3_hit_pct']:>6.1f}% "
            f"{r['top5_recall_pct']:>6.1f}% {r['top5_win_rate_pct']:>6.1f}% {r['top5_avg_return']:>6.2f}% "
            f"{r['top7_recall_pct']:>6.1f}% {r['top7_win_rate_pct']:>6.1f}% {r['top7_loss_rate_pct']:>6.1f}% "
            f"{r['top2_preserve_pct']:>7.1f}% {r['candidate_quality']:>7.2f}"
        )

    lines.extend([
        "",
        "=" * 72,
        "2. TOP2 / TOP5 / TOP7 vs A6 BASELINE (weight=0)",
        "=" * 72,
        f"  {'Metric':<28} {'A6 w=0':>12} {f'Best w={best_w}':>12} {'Delta':>10}",
        f"  {'Top2 Hit %':<28} {base['top2_hit_pct']:>11.1f}% {best['top2_hit_pct']:>11.1f}% {d_top2:>+9.1f}pp",
        f"  {'Top5 Avg Return %':<28} {base['top5_avg_return']:>11.2f}% {best['top5_avg_return']:>11.2f}% {d_top5_ret:>+9.2f}",
        f"  {'Top7 Loss Rate %':<28} {base['top7_loss_rate_pct']:>11.1f}% {best['top7_loss_rate_pct']:>11.1f}% {d_top7_loss:>+9.1f}pp",
        f"  {'Top7 Avg Return %':<28} {base['top7_avg_return']:>11.2f}% {best['top7_avg_return']:>11.2f}% "
        f"{best['top7_avg_return']-base['top7_avg_return']:>+9.2f}",
        f"  {'Top2 Preserved %':<28} {base['top2_preserve_pct']:>11.1f}% {best['top2_preserve_pct']:>11.1f}%",
        f"  {'Candidate Quality':<28} {base['candidate_quality']:>12.2f} {best['candidate_quality']:>12.2f} {d_quality:>+10.2f}",
        "",
        "=" * 72,
        "3. TEMPORAL HOLDOUT (last 30%)",
        "=" * 72,
    ])
    ho_base = next(r for r in ho_summary if r["weight"] == 0)
    ho_best = next(r for r in ho_summary if r["weight"] == best_w)
    lines.append(
        f"  w=0 Top2={ho_base['top2_hit_pct']:.1f}% | w={best_w} Top2={ho_best['top2_hit_pct']:.1f}% "
        f"delta={ho_best['top2_hit_pct']-ho_base['top2_hit_pct']:+.1f}pp"
    )
    lines.append(
        f"  w=0 T7Loss={ho_base['top7_loss_rate_pct']:.1f}% | w={best_w} T7Loss={ho_best['top7_loss_rate_pct']:.1f}%"
    )

    lines.extend([
        "",
        "=" * 72,
        "4. FINAL JUDGMENT",
        "=" * 72,
        f"  1. Soft Re-Rank vs Hard Reject (R001 16.2%): {'YES' if better_than_r001 else 'NO'} "
        f"(best Top2={best['top2_hit_pct']:.1f}%)",
        f"  2. Top5/Top7 quality improved with Top2 preserved: "
        f"{'PARTIAL' if best['top2_preserve_pct']>=70 and d_top5_ret>0 else 'NO'} "
        f"(preserve={best['top2_preserve_pct']:.0f}%, T5ret {d_top5_ret:+.2f})",
        f"  3. Operational adoptable: {'YES' if adoptable else 'NO'} (reject layer forbidden; re-rank only)",
        f"  4. Formula unchanged: YES (A6 frozen, adjustment layer independent)",
        "",
        f"  Best penalty weight: {best_w}",
        f"  VERDICT: {verdict}",
        "",
        "Learning recommendation: NO_ACTION on formula. Soft layer hypothesis only unless holdout confirms.",
        "",
    ])
    lines.extend(mission_summary_lines())
    return "\n".join(lines)


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    annotated, ann_by_scan = r1.load_annotated()
    winner_feats = [
        r["features"]
        for rows in ann_by_scan.values()
        for r in rows[:3]
        if len(rows) >= 4
    ]
    th = p20.build_thresholds(winner_feats)
    for r in annotated:
        r["mtf_seq"] = " -> ".join(
            f"{tf}:{r['states'][tf]}" for tf in ("5m", "15m", "30m", "1h", "2h")
        )

    safe_print("R002 LOO grid search...")
    loo_grid = run_loo_grid(ann_by_scan, annotated, th, PENALTY_WEIGHTS)
    safe_print("R002 holdout...")
    ho_grid = temporal_holdout(ann_by_scan, annotated, th, PENALTY_WEIGHTS)

    loo_summary = [aggregate(loo_grid[w]) | {"weight": w} for w in PENALTY_WEIGHTS]
    ho_summary = [aggregate(ho_grid[w]) | {"weight": w} for w in PENALTY_WEIGHTS]

    best_w = max(PENALTY_WEIGHTS, key=lambda w: (
        aggregate(loo_grid[w])["candidate_quality"],
        aggregate(loo_grid[w])["top2_hit_pct"],
        -aggregate(loo_grid[w])["top7_loss_rate_pct"],
    ))

    write_csv(OUT_DIR / "penalty_weight_kpi_loo.csv", loo_summary)
    write_csv(OUT_DIR / "penalty_weight_kpi_holdout.csv", ho_summary)
    write_csv(OUT_DIR / "lifecycle_distribution.csv", lifecycle_distribution(annotated))

    # per-weight top7 lifecycle shift at best weight
    dist_rows: list[dict] = []
    for sr in loo_grid[best_w]:
        for life, cnt in sr.get("lifecycle_top7", {}).items():
            dist_rows.append({"weight": best_w, "lifecycle": life, "top7_count": cnt})
    write_csv(OUT_DIR / f"top7_lifecycle_w{best_w}.csv", dist_rows)

    report = write_report(loo_summary, ho_summary, best_w, r001_top2=16.2)
    (OUT_DIR / "research_r002_report.txt").write_text(report, encoding="utf-8")
    safe_print(report)


if __name__ == "__main__":
    run()
