"""
Scout Phase 23 - Search Formula League (Parallel Evolution)

Run formulas A..A6 in parallel per scan, league standings, consensus, meta voting.
Base A frozen. LOO + walk-forward for rolling/dynamic (no look-ahead).

Usage:
  python scout_phase23_search_formula_league.py
"""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import scout_phase20_winner_state_ranking as p20
import scout_phase22_search_formula_evolution as p22
from season2_p37_scout_decision_hierarchy import write_csv

OUT_DIR = Path("logs") / "phase23_formula_league"
FORMULAS = ("A", "A1", "A2", "A3", "A4", "A5", "A6")
ROLL_WINDOWS = (20, 50, 100)
MIN_MATCHES_FOR_SWITCH = 100  # rule: keep base until 100+ formula matches

STATE_BUCKETS = (
    "h1_range_high",
    "h2_ma20_high",
    "compression_short",
    "h1_expansion",
    "h1_flat",
)


def safe_print(msg: str) -> None:
    print(msg.encode("ascii", "replace").decode("ascii"))


def g(f: dict, key: str, default: float = 0.0) -> float:
    return float(f.get(key, default))


def formula_scores_a6(row: dict, peers: list[dict], base: float, th: p20.Thresholds, stats: dict) -> dict[str, float]:
    scores = p22.formula_scores(row, peers, base, th, stats)
    # A6 = merge candidates from Phase22 (A2 + A5 bonuses, data-derived IGs)
    b2 = p22.within_scan_pct(row, peers, "1h_current_range_pct")
    b5 = p22.bonus_a5_raw(row, peers, stats)
    scores["A6"] = base + stats["ig_a2"] * b2 + stats["ig_a5"] * b5
    return scores


def scan_state_tags(rows: list[dict], th: p20.Thresholds) -> list[str]:
    if not rows:
        return []
    med_range = statistics.median([g(r["features"], "1h_current_range_pct") for r in rows])
    med_ma20 = statistics.median([g(r["features"], "2h_current_ma20_distance_pct") for r in rows])
    med_comp = statistics.median([g(r["features"], "5m_compression") for r in rows])
    h1_states = Counter(r["states"]["1h"] for r in rows)
    dom_h1 = h1_states.most_common(1)[0][0] if h1_states else "Flat"

    tags: list[str] = []
    if med_range >= th.p75.get("1h_current_range_pct", 5.0):
        tags.append("h1_range_high")
    if med_ma20 >= th.p75.get("2h_current_ma20_distance_pct", 10.0):
        tags.append("h2_ma20_high")
    if med_comp <= th.p25.get("5m_compression", 4.0):
        tags.append("compression_short")
    if dom_h1 in ("Expansion", "ExpansionStart", "Acceleration"):
        tags.append("h1_expansion")
    if dom_h1 == "Flat":
        tags.append("h1_flat")
    if not tags:
        tags.append("neutral")
    return tags


def eval_scan_full(rows: list[dict], profile: dict, th: p20.Thresholds, stats: dict) -> dict:
    for r in rows:
        base = p20.state_match_score(r["states"], r["transitions"], profile)
        r["scores"] = formula_scores_a6(r, rows, base, th, stats)
        r["base_score"] = base

    by_outcome = sorted(rows, key=lambda x: x["outcome_rank"])
    actual_top2 = [x["symbol"] for x in by_outcome[:2]]
    actual_top5 = {x["symbol"] for x in by_outcome[:5]}
    actual_top2_set = set(actual_top2)

    result: dict = {
        "actual_top2": actual_top2,
        "formulas": {},
        "picks_top2": {},
        "picks_top5": {},
    }
    for fid in FORMULAS:
        ranked = sorted(rows, key=lambda x: x["scores"][fid], reverse=True)
        pick2 = [x["symbol"] for x in ranked[:2]]
        pick5 = [x["symbol"] for x in ranked[:5]]
        result["picks_top2"][fid] = pick2
        result["picks_top5"][fid] = pick5
        result["formulas"][fid] = {
            "top2_hit": len(set(pick2) & actual_top2_set),
            "top5_hit": len(set(pick5) & actual_top5),
            "avg_max_up_top2": statistics.mean([x["max_up_4h"] for x in ranked[:2]]) if ranked else 0,
            "rank1_actual": ranked[0]["outcome_rank"] if ranked else None,
        }
    return result


def pairwise_update(rec: dict, a: str, b: str, hit_a: int, hit_b: int, mu_a: float, mu_b: float) -> None:
    key = f"{a}|{b}"
    if hit_a > hit_b or (hit_a == hit_b and mu_a > mu_b):
        rec[key]["wins"] += 1
    elif hit_b > hit_a or (hit_a == hit_b and mu_b > mu_a):
        rec[key]["losses"] += 1
    else:
        rec[key]["draws"] += 1


def consensus_votes(picks_top2: dict[str, list[str]], conf: dict[str, float]) -> dict[str, dict]:
    raw: Counter = Counter()
    weighted: Counter = Counter()
    for fid, syms in picks_top2.items():
        w = conf.get(fid, 1.0)
        for s in syms:
            raw[s] += 1
            weighted[s] += w
    total_w = sum(weighted.values()) or 1.0
    out: dict[str, dict] = {}
    for sym in set(raw):
        out[sym] = {
            "vote_count": raw[sym],
            "weighted_vote": round(weighted[sym], 4),
            "agreement_ratio": round(weighted[sym] / total_w, 4),
        }
    return out


def meta_score(
    row: dict,
    votes: dict[str, dict],
    sym: str,
    state_score_norm: float,
    hist_sim: float,
) -> float:
    v = votes.get(sym, {})
    vote = v.get("vote_count", 0) / len(FORMULAS)
    wvote = v.get("weighted_vote", 0)
    return vote + wvote + hist_sim + state_score_norm


def rolling_hit(history: list[int], window: int) -> float:
    if not history:
        return 0.0
    chunk = history[-window:]
    return sum(chunk) / (2 * len(chunk)) * 100


def run() -> list[str]:
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
    for s in ann_by_scan:
        ann_by_scan[s].sort(key=lambda x: x["outcome_rank"])

    scan_keys = sorted(s for s, rows in ann_by_scan.items() if len(rows) >= 4)

    pairwise: dict[str, dict] = defaultdict(lambda: {"wins": 0, "losses": 0, "draws": 0})
    state_formula_wins: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    state_formula_total: Counter = Counter()
    hit_history: dict[str, list[int]] = {f: [] for f in FORMULAS}
    dynamic_recs: list[dict] = []
    match_log: list[dict] = []
    meta_hits: list[int] = []
    base_hits: list[int] = []
    consensus_hits: list[int] = []

    # walk-forward state-formula performance (prior scans only)
    wf_state_best: dict[str, Counter] = defaultdict(Counter)  # state -> formula top2_hit counts
    wf_state_n: dict[str, Counter] = defaultdict(Counter)

    for i, scan in enumerate(scan_keys):
        rows = ann_by_scan[scan]
        train = [r for r in annotated if r["scan_kst"] != scan]
        train_by: dict[str, list[dict]] = defaultdict(list)
        for r in train:
            train_by[r["scan_kst"]].append(r)
        for s in train_by:
            train_by[s].sort(key=lambda x: x["outcome_rank"])
        w_train, _ = p20.winner_loser_sets(train_by)
        profile = p20.build_profile(w_train, train) if w_train else p20.build_profile([], train)
        stats = p22.build_train_stats(train, train_by, th)
        res = eval_scan_full(rows, profile, th, stats)
        tags = scan_state_tags(rows, th)

        # rolling confidence from PRIOR scans only (walk-forward)
        conf = {f: rolling_hit(hit_history[f], 50) / 100.0 + 0.1 for f in FORMULAS}
        votes = consensus_votes(res["picks_top2"], conf)

        # consensus top2 by weighted vote
        cons_ranked = sorted(votes.items(), key=lambda x: (x[1]["weighted_vote"], x[1]["vote_count"]), reverse=True)
        cons_pick2 = {s for s, _ in cons_ranked[:2]}
        actual2 = set(res["actual_top2"])
        consensus_hits.append(len(cons_pick2 & actual2))

        # dynamic formula recommend from prior state performance
        best_f, best_rate = "A", 0.0
        for tag in tags:
            for fid in FORMULAS:
                w = wf_state_best[tag][fid]
                n = wf_state_n[tag][fid]
                if n >= 5:
                    rate = w / (2 * n)
                    if rate > best_rate:
                        best_rate, best_f = rate, fid
        dynamic_recs.append({
            "scan_kst": scan,
            "state_tags": "|".join(tags),
            "recommended": best_f,
            "confidence_pct": round(best_rate * 100, 1),
            "matches_in_state": sum(wf_state_n[tags[0]].values()) if tags else 0,
        })

        # meta voting top2
        max_base = max(r["base_score"] for r in rows) if rows else 1.0
        meta_scored: list[tuple[str, float]] = []
        for r in rows:
            sym = r["symbol"]
            sn = r["base_score"] / max_base if max_base else 0
            hist_sim = votes.get(sym, {}).get("agreement_ratio", 0)
            ms = meta_score(r, votes, sym, sn, hist_sim)
            meta_scored.append((sym, ms))
        meta_scored.sort(key=lambda x: x[1], reverse=True)
        meta_pick2 = {s for s, _ in meta_scored[:2]}
        meta_hits.append(len(meta_pick2 & actual2))
        base_hits.append(res["formulas"]["A"]["top2_hit"])

        # log match
        entry = {
            "scan_kst": scan,
            "actual_top2": res["actual_top2"],
            "state_tags": tags,
            "consensus_top2": [s for s, _ in cons_ranked[:2]],
            "meta_top2": [s for s, _ in meta_scored[:2]],
            "dynamic_formula": best_f,
            "meta_top2_hit": len(meta_pick2 & actual2),
            "consensus_top2_hit": len(cons_pick2 & actual2),
        }
        for fid in FORMULAS:
            entry[f"{fid}_top2"] = res["picks_top2"][fid]
            entry[f"{fid}_top5_hit"] = res["formulas"][fid]["top5_hit"]
            entry[f"{fid}_hit"] = res["formulas"][fid]["top2_hit"]
            entry[f"{fid}_avg_max_up"] = res["formulas"][fid]["avg_max_up_top2"]
            entry[f"{fid}_rank1"] = res["formulas"][fid]["rank1_actual"]
            hit_history[fid].append(res["formulas"][fid]["top2_hit"])
        match_log.append(entry)

        # pairwise league
        for a in FORMULAS:
            for b in FORMULAS:
                if a >= b:
                    continue
                fa, fb = res["formulas"][a], res["formulas"][b]
                pairwise_update(pairwise, a, b, fa["top2_hit"], fb["top2_hit"], fa["avg_max_up_top2"], fb["avg_max_up_top2"])

        # state formula wins vs A
        for tag in tags:
            state_formula_total[tag] += 1
            hits_a = res["formulas"]["A"]["top2_hit"]
            best_hit = max(res["formulas"][f]["top2_hit"] for f in FORMULAS)
            for fid in FORMULAS:
                if res["formulas"][fid]["top2_hit"] > hits_a:
                    state_formula_wins[tag][fid]["beats_A"] += 1
                if res["formulas"][fid]["top2_hit"] == best_hit:
                    state_formula_wins[tag][fid]["best_in_scan"] += 1
                wf_state_n[tag][fid] += 1
                wf_state_best[tag][fid] += res["formulas"][fid]["top2_hit"]

    n_scans = len(scan_keys)
    total_matches = n_scans * len(FORMULAS)

    # league standings
    standings: list[dict] = []
    for fid in FORMULAS:
        hits = hit_history[fid]
        rank1_vals = [e[f"{fid}_rank1"] for e in match_log if e.get(f"{fid}_rank1")]
        standings.append({
            "formula": fid,
            "top2_hit_pct": round(sum(hits) / (2 * n_scans) * 100, 2),
            "top5_hit_pct": round(statistics.mean([e[f"{fid}_top5_hit"] for e in match_log]) / 5 * 100, 2),
            "avg_max_up_top2": round(statistics.mean([e[f"{fid}_avg_max_up"] for e in match_log]), 4),
            "roll20": round(rolling_hit(hits, 20), 1),
            "roll50": round(rolling_hit(hits, 50), 1),
            "roll100": round(rolling_hit(hits, 100), 1),
            "rank1_median": round(statistics.median(rank1_vals), 1) if rank1_vals else 0,
        })

    standings.sort(key=lambda x: x["top2_hit_pct"], reverse=True)

    # pairwise table
    pw_rows: list[dict] = []
    for a in FORMULAS:
        for b in FORMULAS:
            if a == b:
                continue
            key = f"{a}|{b}" if f"{a}|{b}" in pairwise else f"{b}|{a}"
            rec = pairwise.get(f"{a}|{b}") or pairwise.get(f"{b}|{a}")
            if not rec:
                continue
            # normalize direction a vs b
            if f"{a}|{b}" in pairwise:
                w, l, d = rec["wins"], rec["losses"], rec["draws"]
            else:
                w, l, d = rec["losses"], rec["wins"], rec["draws"]
            tot = w + l + d
            pw_rows.append({
                "formula_a": a, "formula_b": b,
                "wins": w, "losses": l, "draws": d,
                "win_pct": round(w / tot * 100, 1) if tot else 0,
            })

    # state formula win rates
    state_rows: list[dict] = []
    for tag in sorted(state_formula_total.keys()):
        for fid in FORMULAS:
            n = state_formula_total[tag] or 1
            state_rows.append({
                "state": tag,
                "formula": fid,
                "beats_A_pct": round(state_formula_wins[tag][fid].get("beats_A", 0) / n * 100, 1),
                "best_in_scan_pct": round(state_formula_wins[tag][fid].get("best_in_scan", 0) / n * 100, 1),
                "roll_state_top2_pct": round(wf_state_best[tag][fid] / (2 * wf_state_n[tag][fid]) * 100, 1)
                if wf_state_n[tag][fid] else 0,
            })
    state_rows.sort(key=lambda x: x["roll_state_top2_pct"], reverse=True)

    base_t2 = next(s["top2_hit_pct"] for s in standings if s["formula"] == "A")
    meta_t2 = sum(meta_hits) / (2 * n_scans) * 100
    cons_t2 = sum(consensus_hits) / (2 * n_scans) * 100
    dyn_t2_list: list[int] = []
    for e, scan in zip(match_log, scan_keys):
        fid = e.get("dynamic_formula", "A")
        dyn_t2_list.append(e[f"{fid}_hit"])
    dyn_t2 = statistics.mean(dyn_t2_list) / 2 * 100 if dyn_t2_list else 0

    merge_ok = total_matches >= MIN_MATCHES_FOR_SWITCH
    best_st = standings[0]
    delta_best = best_st["top2_hit_pct"] - base_t2

    if meta_t2 >= 66:
        verdict = "KEEP"
    elif meta_t2 >= 50 or cons_t2 >= 50:
        verdict = "MERGE"
    else:
        verdict = "DISCARD" if meta_t2 <= base_t2 else "MERGE"

    lines = [
        "############################################################",
        "SCOUT PHASE 23 - SEARCH FORMULA LEAGUE",
        "############################################################",
        "",
        f"Scans: {n_scans} LOO | Formula matches: {total_matches} | Base A frozen",
        f"Rule: min {MIN_MATCHES_FOR_SWITCH} matches before base switch -> {'MET' if merge_ok else 'NOT YET'}",
        "",
        "=" * 62,
        "1. FORMULA LEAGUE STANDINGS",
        "=" * 62,
        f"{'Rank':<5} {'Formula':<6} {'TOP2%':>7} {'TOP5%':>7} {'Roll20':>7} {'Roll50':>7} {'Roll100':>8}",
    ]
    for i, st in enumerate(standings, 1):
        lines.append(
            f"{i:<5} {st['formula']:<6} {st['top2_hit_pct']:>6.1f}% {st['top5_hit_pct']:>6.1f}% "
            f"{st['roll20']:>6.1f}% {st['roll50']:>6.1f}% {st['roll100']:>7.1f}%"
        )

    lines.extend(["", "=" * 62, "2. PAIRWISE vs A (wins / losses / draws)", "=" * 62])
    for b in FORMULAS:
        if b == "A":
            continue
        row = next((p for p in pw_rows if p["formula_a"] == "A" and p["formula_b"] == b), None)
        if row:
            lines.append(f"  A vs {b}: W{row['wins']} L{row['losses']} D{row['draws']} (A win% {row['win_pct']:.0f}%)")

    lines.extend(["", "=" * 62, "3. STATE x FORMULA (top win rates)", "=" * 62])
    for row in state_rows[:20]:
        lines.append(
            f"  {row['state']} + {row['formula']}: state_TOP2={row['roll_state_top2_pct']:.0f}% "
            f"beats_A={row['beats_A_pct']:.0f}%"
        )

    lines.extend(["", "=" * 62, "4. CONSENSUS EFFECT", "=" * 62])
    lines.append(f"  Consensus TOP2 hit: {cons_t2:.1f}%  |  Base A: {base_t2:.1f}%  |  delta {cons_t2-base_t2:+.1f}pp")

    lines.extend(["", "=" * 62, "5. ROLLING PERFORMANCE (final window)", "=" * 62])
    for st in standings:
        lines.append(f"  {st['formula']}: roll20={st['roll20']:.1f}% roll50={st['roll50']:.1f}% roll100={st['roll100']:.1f}%")

    lines.extend(["", "=" * 62, "6. DYNAMIC FORMULA RECOMMENDATION", "=" * 62])
    for rec in dynamic_recs[-5:]:
        lines.append(
            f"  {rec['scan_kst']}: state={rec['state_tags']} -> {rec['recommended']} "
            f"(conf {rec['confidence_pct']:.0f}%)"
        )
    lines.append(f"  Dynamic-pick TOP2 hit (walk-forward): {dyn_t2:.1f}%")

    lines.extend(["", "=" * 62, "7. META VOTING vs BASE A", "=" * 62])
    lines.append(f"  Meta FinalScore TOP2: {meta_t2:.1f}%")
    lines.append(f"  Base A TOP2:         {base_t2:.1f}%")
    lines.append(f"  Delta:               {meta_t2-base_t2:+.1f}pp")
    lines.append("  Meta = vote_count + weighted_vote + historical_similarity + state_score_norm")

    lines.extend(["", "=" * 62, "8. MERGE READINESS", "=" * 62])
    lines.append(f"  Best single formula: {best_st['formula']} ({best_st['top2_hit_pct']:.1f}%)")
    lines.append(f"  Best vs A: {delta_best:+.1f}pp")
    lines.append(f"  Merge allowed (100+ matches): {merge_ok}")
    if merge_ok and delta_best >= 1.0:
        lines.append(f"  Candidate merge: {best_st['formula']} bonus into league meta layer.")
    else:
        lines.append("  Keep Base A; accumulate more league matches before merge.")

    lines.extend(["", "=" * 62, "SUCCESS CONDITIONS", "=" * 62])
    for label, thr in (("Meaningful", 50), ("Production Candidate", 66), ("Exceptional", 80)):
        hit = max(meta_t2, cons_t2, best_st["top2_hit_pct"])
        lines.append(f"  TOP2 >= {thr}% ({label}): {'YES' if hit >= thr else 'NO'} (best={hit:.1f}%)")

    lines.extend(["", "=" * 62, "VERDICT", "=" * 62])
    lines.append(f"  {verdict}")
    lines.append("  Base A remains frozen. League operates in parallel for state-conditional learning.")

    with (OUT_DIR / "match_log.jsonl").open("w", encoding="utf-8") as f:
        for e in match_log:
            f.write(json.dumps(e, ensure_ascii=True) + "\n")

    write_csv(OUT_DIR / "league_standings.csv", standings)
    write_csv(OUT_DIR / "pairwise_league.csv", pw_rows)
    write_csv(OUT_DIR / "state_formula_winrates.csv", state_rows)
    write_csv(OUT_DIR / "dynamic_recommendations.csv", dynamic_recs)
    write_csv(OUT_DIR / "comparison_meta.csv", [{
        "model": "Base_A", "top2_pct": round(base_t2, 2),
    }, {
        "model": "Consensus", "top2_pct": round(cons_t2, 2),
    }, {
        "model": "Meta_Voting", "top2_pct": round(meta_t2, 2),
    }, {
        "model": "Dynamic_Pick", "top2_pct": round(dyn_t2, 2),
    }, {
        "model": f"Best_{best_st['formula']}", "top2_pct": best_st["top2_hit_pct"],
    }])

    report = OUT_DIR / "phase23_formula_league_report.txt"
    report.write_text("\n".join(lines), encoding="utf-8")
    return lines


def main() -> None:
    lines = run()
    for ln in lines:
        safe_print(ln)
    safe_print(f"\nSaved: {OUT_DIR / 'phase23_formula_league_report.txt'}")


if __name__ == "__main__":
    main()
