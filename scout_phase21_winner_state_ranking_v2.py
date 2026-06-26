"""
Scout Phase 21 - Winner State Ranking Engine V2

State + Transition + Cluster + Lifecycle + Relative Expansion ranking.
NO filter/threshold/rule changes. LOO backtest on Phase19 dataset.

Usage:
  python scout_phase21_winner_state_ranking_v2.py
"""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import scout_phase20_winner_state_ranking as p20
from season2_p37_scout_decision_hierarchy import write_csv

P19_DIR = Path("logs") / "phase19_winner_dna"
CANDIDATES_PATH = P19_DIR / "candidates.jsonl"
OUT_DIR = Path("logs") / "phase21_state_ranking_v2"

WINNER_TOP_N = 3
LOSER_BOTTOM_N = 3
TOP5_N = 5
GOAL_TOP2 = 66.0
GOAL_TOP5 = 80.0

EXPANSION_METRICS = (
    "1h_current_return_pct",
    "1h_current_range_pct",
    "2h_current_return_pct",
    "2h_current_range_pct",
    "2h_current_ma20_distance_pct",
    "30m_current_return_pct",
    "15m_current_volume_ratio",
    "5m_range_energy",
)

TF_ORDER = ("5m", "15m", "30m", "1h", "2h")


def safe_print(msg: str) -> None:
    print(msg.encode("ascii", "replace").decode("ascii"))


def g(f: dict, key: str, default: float = 0.0) -> float:
    return float(f.get(key, default))


def lifecycle_state(f: dict, th: p20.Thresholds) -> str:
    h4 = g(f, "h4_score")
    if h4 >= th.p75.get("h4_score", 0):
        return "Ignition"
    if h4 >= th.p50.get("h4_score", 0):
        return "Growth"
    return "Early"


def mtf_sequence(states: dict[str, str]) -> str:
    return " -> ".join(f"{tf}:{states[tf]}" for tf in TF_ORDER)


def mtf_transition_chain(trans: dict[str, str]) -> str:
    return " -> ".join(f"{tf}:{trans[tf]}" for tf in TF_ORDER)


def annotate_v2(rows: list[dict], th: p20.Thresholds) -> list[dict]:
    base = p20.annotate(rows, th)
    for r in base:
        f = r["features"]
        r["lifecycle"] = lifecycle_state(f, th)
        r["mtf_seq"] = mtf_sequence(r["states"])
        r["mtf_trans_chain"] = mtf_transition_chain(r["transitions"])
    return base


def cohort_slices(by_scan: dict[str, list[dict]]) -> dict[str, list[dict]]:
    w1: list[dict] = []
    w2: list[dict] = []
    w3: list[dict] = []
    top5: list[dict] = []
    losers: list[dict] = []
    winners: list[dict] = []
    for rows in by_scan.values():
        if len(rows) < 4:
            continue
        n = len(rows)
        winners.extend(rows[: min(WINNER_TOP_N, n)])
        if n >= 1:
            w1.append(rows[0])
        if n >= 2:
            w2.append(rows[1])
        if n >= 3:
            w3.append(rows[2])
        top5.extend(rows[: min(TOP5_N, n)])
        losers.extend(rows[-min(LOSER_BOTTOM_N, n):])
    return {"w1": w1, "w2": w2, "w3": w3, "top5": top5, "losers": losers, "winners": winners}


def lift_table(wc: Counter, wt: int, ac: Counter, at: int) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in set(wc) | set(ac):
        wr = wc[k] / wt if wt else 0
        ar = ac[k] / at if at else 0
        out[k] = wr / ar if ar > 0 else (wr * at + 1.0)
    return out


def build_profile_v2(train: list[dict], by_scan_train: dict[str, list[dict]]) -> dict:
    cohorts = cohort_slices(by_scan_train)
    all_n = len(train)
    c = cohorts

    def count_field(rows: list[dict], field: str) -> Counter:
        ctr: Counter = Counter()
        for r in rows:
            ctr[r[field]] += 1
        return ctr

    def count_state(rows: list[dict]) -> Counter:
        ctr: Counter = Counter()
        for r in rows:
            for tf, s in r["states"].items():
                ctr[f"{tf}:{s}"] += 1
        return ctr

    def count_trans(rows: list[dict]) -> Counter:
        ctr: Counter = Counter()
        for r in rows:
            for tf, t in r["transitions"].items():
                ctr[f"{tf}:{t}"] += 1
            ctr[r["mtf_trans_chain"]] += 1
        return ctr

    state_w = count_state(c["winners"])
    state_a = count_state(train)
    trans_w = count_trans(c["winners"])
    trans_a = count_trans(train)
    cluster_w = Counter(r["combo"] for r in c["winners"])
    cluster_a = Counter(r["combo"] for r in train)
    seq_w = Counter(r["mtf_seq"] for r in c["winners"])
    seq_a = Counter(r["mtf_seq"] for r in train)
    seq_w1 = Counter(r["mtf_seq"] for r in c["w1"])
    seq_w2 = Counter(r["mtf_seq"] for r in c["w2"])
    seq_l = Counter(r["mtf_seq"] for r in c["losers"])
    seq_t5 = Counter(r["mtf_seq"] for r in c["top5"])
    lc_w = Counter(r["lifecycle"] for r in c["winners"])
    lc_a = Counter(r["lifecycle"] for r in train)

    # expansion metric IG vs top2 label on training
    top2_set = set()
    for rows in by_scan_train.values():
        if len(rows) >= 2:
            top2_set.add((rows[0]["scan_kst"], rows[0]["symbol"]))
            top2_set.add((rows[1]["scan_kst"], rows[1]["symbol"]))

    metric_ig: dict[str, float] = {}
    for m in EXPANSION_METRICS:
        pos = [g(r["features"], m) for r in train if (r["scan_kst"], r["symbol"]) in top2_set]
        neg = [g(r["features"], m) for r in train if (r["scan_kst"], r["symbol"]) not in top2_set]
        if not pos or not neg:
            metric_ig[m] = 0.0
            continue
        # mean separation normalized
        mu_p, mu_n = statistics.mean(pos), statistics.mean(neg)
        sd = statistics.pstdev([g(r["features"], m) for r in train]) or 1.0
        metric_ig[m] = max(abs(mu_p - mu_n) / sd, 0.0)

    ig_sum = sum(metric_ig.values()) or 1.0
    metric_w = {k: v / ig_sum for k, v in metric_ig.items()}

    profile = {
        "state_lift": lift_table(state_w, len(c["winners"]), state_a, all_n),
        "trans_lift": lift_table(trans_w, len(c["winners"]), trans_a, all_n),
        "cluster_lift": lift_table(cluster_w, len(c["winners"]), cluster_a, all_n),
        "seq_lift": lift_table(seq_w, len(c["winners"]), seq_a, all_n),
        "w1_seq_lift": lift_table(seq_w1, max(len(c["w1"]), 1), seq_a, all_n),
        "w2_seq_lift": lift_table(seq_w2, max(len(c["w2"]), 1), seq_a, all_n),
        "loser_seq_lift": lift_table(seq_l, max(len(c["losers"]), 1), seq_a, all_n),
        "top5_seq_lift": lift_table(seq_t5, max(len(c["top5"]), 1), seq_a, all_n),
        "lifecycle_lift": lift_table(lc_w, len(c["winners"]), lc_a, all_n),
        "expansion_metric_w": metric_w,
    }
    return profile


def relative_expansion(row: dict, peers: list[dict], metric_w: dict[str, float]) -> float:
    f = row["features"]
    score = 0.0
    for m, w in metric_w.items():
        vals = [g(p["features"], m) for p in peers]
        v = g(f, m)
        if not vals:
            continue
        pct = sum(1 for x in vals if x <= v) / len(vals)
        score += w * pct
    return score


def component_scores(row: dict, profile: dict, peers: list[dict]) -> dict[str, float]:
    states, trans = row["states"], row["transitions"]

    state_s = sum(math.log(max(profile["state_lift"].get(f"{tf}:{states[tf]}", 1.0), 0.01)) for tf in TF_ORDER)

    trans_s = sum(math.log(max(profile["trans_lift"].get(f"{tf}:{trans[tf]}", 1.0), 0.01)) for tf in TF_ORDER)
    trans_s += math.log(max(profile["trans_lift"].get(row["mtf_trans_chain"], 1.0), 0.01))

    seq = row["mtf_seq"]
    w1_l = max(profile["w1_seq_lift"].get(seq, 1.0), 0.01)
    w2_l = max(profile["w2_seq_lift"].get(seq, 1.0), 0.01)
    los_l = max(profile["loser_seq_lift"].get(seq, 1.0), 0.01)
    seq_s = math.log(max(profile["seq_lift"].get(seq, 1.0), 0.01))
    seq_s += math.log(w1_l)
    seq_s += math.log(w1_l / w2_l)  # data-driven w1 vs runner-up contrast
    seq_s -= math.log(los_l)  # loser depletion (lift ratio inverse)
    transition_total = trans_s + seq_s

    cluster_s = math.log(max(profile["cluster_lift"].get(row["combo"], 1.0), 0.01))
    lifecycle_s = math.log(max(profile["lifecycle_lift"].get(row["lifecycle"], 1.0), 0.01))
    rel_s = relative_expansion(row, peers, profile["expansion_metric_w"])

    return {
        "state": state_s,
        "transition": transition_total,
        "cluster": cluster_s,
        "lifecycle": lifecycle_s,
        "relative_expansion": rel_s,
    }


def compute_component_ig(train: list[dict], by_scan_train: dict[str, list[dict]], profile: dict) -> dict[str, float]:
    """IG-based component weights from training top2 separation."""
    top2_keys = set()
    for rows in by_scan_train.values():
        if len(rows) >= 2:
            top2_keys.add((rows[0]["scan_kst"], rows[0]["symbol"]))
            top2_keys.add((rows[1]["scan_kst"], rows[1]["symbol"]))

    comps = ["state", "transition", "cluster", "lifecycle", "relative_expansion"]
    ig_vals: dict[str, float] = {}
    for comp in comps:
        pos: list[float] = []
        neg: list[float] = []
        for scan, rows in by_scan_train.items():
            if len(rows) < 4:
                continue
            for r in rows:
                parts = component_scores(r, profile, rows)
                val = parts[comp]
                if (r["scan_kst"], r["symbol"]) in top2_keys:
                    pos.append(val)
                else:
                    neg.append(val)
        if not pos or not neg:
            ig_vals[comp] = 0.0
            continue
        # Cohen-like separation as IG proxy
        mu_p, mu_n = statistics.mean(pos), statistics.mean(neg)
        pooled = statistics.pstdev(pos + neg) or 1.0
        ig_vals[comp] = max((mu_p - mu_n) / pooled, 0.0)

    total = sum(ig_vals.values()) or 1.0
    return {k: v / total for k, v in ig_vals.items()}


def phase20_score(row: dict, profile: dict) -> float:
    return p20.state_match_score(row["states"], row["transitions"], profile)


def phase21_score(row: dict, profile: dict, peers: list[dict], comp_w: dict[str, float]) -> float:
    parts = component_scores(row, profile, peers)
    return sum(comp_w.get(k, 0.2) * parts[k] for k in parts)


def transition_stats(rows: list[dict], field: str = "mtf_seq") -> Counter:
    return Counter(r[field] for r in rows)


def separation_for_sequences(
    w_ctr: Counter, w_total: int, l_ctr: Counter, l_total: int, a_ctr: Counter, a_total: int,
) -> list[dict]:
    rows: list[dict] = []
    for key in a_ctr:
        wc, lc, ac = w_ctr[key], l_ctr[key], a_ctr[key]
        row = p20.separation_row(key, wc, w_total, lc, l_total, ac, a_total)
        row["freq"] = ac
        rows.append(row)
    rows.sort(key=lambda x: (x["information_gain"], x["lift"]), reverse=True)
    return rows


def backtest_row(rows: list[dict], profile_p20: dict, profile_p21: dict, comp_w: dict) -> dict:
    for r in rows:
        r["p20_score"] = phase20_score(r, profile_p20)
        r["p21_score"] = phase21_score(r, profile_p21, rows, comp_w)

    by_outcome = sorted(rows, key=lambda x: x["outcome_rank"])
    actual_top2 = {r["symbol"] for r in by_outcome[:2]}
    actual_top5 = {r["symbol"] for r in by_outcome[:5]}

    h4_s = sorted(rows, key=lambda x: g(x["features"], "h4_score"), reverse=True)
    p20_s = sorted(rows, key=lambda x: x["p20_score"], reverse=True)
    p21_s = sorted(rows, key=lambda x: x["p21_score"], reverse=True)

    def hits(sorted_rows: list[dict], k: int, actual: set[str]) -> int:
        return len({r["symbol"] for r in sorted_rows[:k]} & actual)

    def mus(sorted_rows: list[dict], k: int) -> float:
        return statistics.mean([r["max_up_4h"] for r in sorted_rows[:k]]) if sorted_rows else 0

    def med_rank(sorted_rows: list[dict]) -> int | None:
        if not sorted_rows:
            return None
        return next((r["outcome_rank"] for r in rows if r["symbol"] == sorted_rows[0]["symbol"]), None)

    return {
        "n": len(rows),
        "h4_top2_hit": hits(h4_s, 2, actual_top2),
        "p20_top2_hit": hits(p20_s, 2, actual_top2),
        "p21_top2_hit": hits(p21_s, 2, actual_top2),
        "h4_top5_hit": hits(h4_s, 5, actual_top5),
        "p20_top5_hit": hits(p20_s, 5, actual_top5),
        "p21_top5_hit": hits(p21_s, 5, actual_top5),
        "h4_avg_top2": mus(h4_s, 2),
        "p20_avg_top2": mus(p20_s, 2),
        "p21_avg_top2": mus(p21_s, 2),
        "h4_rank1": med_rank(h4_s),
        "p20_rank1": med_rank(p20_s),
        "p21_rank1": med_rank(p21_s),
    }


def run() -> list[str]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = p20.load_candidates()
    by_scan: dict[str, list[dict]] = defaultdict(list)
    for r in raw:
        by_scan[r["scan_kst"]].append(r)
    for scan in by_scan:
        by_scan[scan].sort(key=lambda x: x["outcome_rank"])

    winner_feats = [
        r["features"] for rows in by_scan.values() for r in rows[:WINNER_TOP_N] if len(rows) >= 4
    ]
    th = p20.build_thresholds(winner_feats)
    annotated = annotate_v2(raw, th)
    ann_by_scan: dict[str, list[dict]] = defaultdict(list)
    for r in annotated:
        ann_by_scan[r["scan_kst"]].append(r)
    for scan in ann_by_scan:
        ann_by_scan[scan].sort(key=lambda x: x["outcome_rank"])

    cohorts_full = cohort_slices(ann_by_scan)

    # global transition analysis (descriptive)
    w_seq = transition_stats(cohorts_full["winners"])
    l_seq = transition_stats(cohorts_full["losers"])
    w2_seq = transition_stats(cohorts_full["w2"])
    t5_seq = transition_stats(cohorts_full["top5"])
    a_seq = transition_stats(annotated)

    w_trans_sep = separation_for_sequences(w_seq, len(cohorts_full["winners"]), l_seq, len(cohorts_full["losers"]), a_seq, len(annotated))
    l_trans_sep = separation_for_sequences(l_seq, len(cohorts_full["losers"]), w_seq, len(cohorts_full["winners"]), a_seq, len(annotated))

    # runner-up vs winner sequence contrast
    w1_only = transition_stats(cohorts_full["w1"])
    ru_contrast: list[dict] = []
    for seq, w1c in w1_only.most_common(50):
        w2c = w2_seq.get(seq, 0)
        if w1c < 3:
            continue
        w1r = w1c / max(len(cohorts_full["w1"]), 1)
        w2r = w2c / max(len(cohorts_full["w2"]), 1)
        ru_contrast.append({
            "sequence": seq,
            "w1_freq": w1c,
            "w2_freq": w2c,
            "w1_rate": round(w1r, 4),
            "w2_rate": round(w2r, 4),
            "contrast": round(w1r - w2r, 4),
            "lift_w1": round(lift_table(w1_only, len(cohorts_full["w1"]), a_seq, len(annotated)).get(seq, 1), 4),
        })
    ru_contrast.sort(key=lambda x: (x["contrast"], x["w1_freq"]), reverse=True)

    # LOO backtest
    bt: list[dict] = []
    comp_w_accum: Counter = Counter()
    n_folds = 0
    scan_keys = [s for s, rows in ann_by_scan.items() if len(rows) >= 4]

    for scan in scan_keys:
        train = [r for r in annotated if r["scan_kst"] != scan]
        train_by: dict[str, list[dict]] = defaultdict(list)
        for r in train:
            train_by[r["scan_kst"]].append(r)
        for s in train_by:
            train_by[s].sort(key=lambda x: x["outcome_rank"])

        prof21 = build_profile_v2(train, train_by)
        w_train, _ = p20.winner_loser_sets(train_by)
        prof20 = p20.build_profile(w_train, train)
        comp_w = compute_component_ig(train, train_by, prof21)
        for k, v in comp_w.items():
            comp_w_accum[k] += v
        n_folds += 1
        bt.append(backtest_row(ann_by_scan[scan], prof20, prof21, comp_w))

    comp_w_avg = {k: comp_w_accum[k] / n_folds for k in comp_w_accum}

    def rate(model: str, k: int) -> float:
        return statistics.mean([b[f"{model}_top{k}_hit"] for b in bt]) / k * 100

    def avg_metric(model: str, field: str) -> float:
        return statistics.mean([b[f"{model}_{field}"] for b in bt])

    def med_metric(model: str) -> float:
        vals = [b[f"{model}_rank1"] for b in bt if b[f"{model}_rank1"]]
        return statistics.median(vals) if vals else 0

    h4_t2, p20_t2, p21_t2 = rate("h4", 2), rate("p20", 2), rate("p21", 2)
    h4_t5, p20_t5, p21_t5 = rate("h4", 5), rate("p20", 5), rate("p21", 5)

    goal_top2 = p21_t2 >= GOAL_TOP2
    goal_top5 = p21_t5 >= GOAL_TOP5

    if goal_top2 and goal_top5:
        verdict = "KEEP"
    elif p21_t2 > p20_t2 or p21_t5 > p20_t5:
        verdict = "MODIFY"
    else:
        verdict = "DISCARD"

    lines = [
        "############################################################",
        "SCOUT PHASE 21 - WINNER STATE RANKING ENGINE V2",
        "############################################################",
        "",
        f"Input: Phase19 | {len(scan_keys)} scans LOO | {len(annotated)} candidates",
        "Pattern B frozen | Ranking layer only",
        "",
        "=" * 62,
        "1. H4 vs Phase20 vs Phase21 (LOO backtest)",
        "=" * 62,
        f"  {'Metric':<22} {'H4':>10} {'Phase20':>10} {'Phase21':>10}",
        f"  {'TOP2 hit %':<22} {h4_t2:>9.1f}% {p20_t2:>9.1f}% {p21_t2:>9.1f}%",
        f"  {'TOP5 hit %':<22} {h4_t5:>9.1f}% {p20_t5:>9.1f}% {p21_t5:>9.1f}%",
        f"  {'Avg max_up TOP2':<22} {avg_metric('h4','avg_top2'):>9.2f}% {avg_metric('p20','avg_top2'):>9.2f}% {avg_metric('p21','avg_top2'):>9.2f}%",
        f"  {'Rank#1 median actual':<22} {med_metric('h4'):>10.1f} {med_metric('p20'):>10.1f} {med_metric('p21'):>10.1f}",
        "",
        f"  Goal TOP2 >= {GOAL_TOP2:.0f}%: {'ACHIEVED' if goal_top2 else 'NOT ACHIEVED'} ({p21_t2:.1f}%)",
        f"  Goal TOP5 >= {GOAL_TOP5:.0f}%: {'ACHIEVED' if goal_top5 else 'NOT ACHIEVED'} ({p21_t5:.1f}%)",
        "",
        "=" * 62,
        "2. WINNER TRANSITION TOP20 (MTF sequence)",
        "=" * 62,
    ]
    for row in w_trans_sep[:20]:
        seq = row["label"].replace(" -> ", "\n    -> ")
        lines.append(
            f"  {row['label']}\n"
            f"    freq={row['freq']} lift={row['lift']:.2f} "
            f"odds={row['odds_ratio']:.2f} win%={row['winner_rate']*100:.1f}%"
        )

    lines.extend(["", "=" * 62, "3. LOSER TRANSITION TOP20", "=" * 62])
    for row in l_trans_sep[:20]:
        lines.append(
            f"  {row['label']}\n"
            f"    freq={row['freq']} lift={row['lift']:.2f} "
            f"lose%={row['loser_rate']*100:.1f}% odds={row['odds_ratio']:.2f}"
        )

    lines.extend(["", "=" * 62, "4. WINNER vs RUNNER-UP SEQUENCE (best contrast)", "=" * 62])
    for row in ru_contrast[:15]:
        lines.append(
            f"  {row['sequence']}\n"
            f"    w1_freq={row['w1_freq']} w2_freq={row['w2_freq']} "
            f"contrast={row['contrast']:+.3f} lift_w1={row['lift_w1']:.2f}"
        )

    lines.extend(["", "=" * 62, "5. COMPONENT IMPORTANCE (IG-based avg weight %)", "=" * 62])
    for k, v in sorted(comp_w_avg.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"  {k}: {v*100:.1f}%")

    lines.extend(["", "=" * 62, "6. SCORE DEFINITION", "=" * 62])
    lines.append("  Phase21 = sum_c [ IG_weight(c) * component_c ]")
    lines.append("  Components: state, transition (per-TF + MTF chain + w1/w2/loser contrast),")
    lines.append("              cluster, lifecycle, relative_expansion (within-scan percentile)")
    lines.append("  IG weights computed per LOO fold on training top2 separation.")
    lines.append("  No hand-tuned coefficients; w1/w2/loser contrast via lift ratios only.")

    lines.extend(["", "=" * 62, "7. VERDICT", "=" * 62])
    lines.append(f"  {verdict}")
    if not goal_top2:
        lines.append(f"  TOP2 gap to goal: {GOAL_TOP2 - p21_t2:.1f}pp remaining.")
    if not goal_top5:
        lines.append(f"  TOP5 gap to goal: {GOAL_TOP5 - p21_t5:.1f}pp remaining.")
    delta20 = p21_t2 - p20_t2
    lines.append(f"  Phase21 vs Phase20 TOP2 delta: {delta20:+.1f}pp")

    lines.extend(["", "DISCLAIMER: Ranking prototype only. No filter/threshold changes."])

    write_csv(OUT_DIR / "winner_transition_top20.csv", w_trans_sep[:20])
    write_csv(OUT_DIR / "loser_transition_top20.csv", l_trans_sep[:20])
    write_csv(OUT_DIR / "runnerup_contrast.csv", ru_contrast[:30])
    write_csv(OUT_DIR / "component_importance.csv", [
        {"component": k, "weight_pct": round(v * 100, 2)} for k, v in comp_w_avg.items()
    ])
    write_csv(OUT_DIR / "loo_backtest.csv", bt)
    write_csv(OUT_DIR / "comparison_summary.csv", [{
        "model": "H4", "top2_hit_pct": round(h4_t2, 2), "top5_hit_pct": round(h4_t5, 2),
        "avg_max_up_top2": round(avg_metric("h4", "avg_top2"), 4), "rank1_median": round(med_metric("h4"), 2),
    }, {
        "model": "Phase20", "top2_hit_pct": round(p20_t2, 2), "top5_hit_pct": round(p20_t5, 2),
        "avg_max_up_top2": round(avg_metric("p20", "avg_top2"), 4), "rank1_median": round(med_metric("p20"), 2),
    }, {
        "model": "Phase21", "top2_hit_pct": round(p21_t2, 2), "top5_hit_pct": round(p21_t5, 2),
        "avg_max_up_top2": round(avg_metric("p21", "avg_top2"), 4), "rank1_median": round(med_metric("p21"), 2),
        "goal_top2_met": goal_top2, "goal_top5_met": goal_top5,
    }])

    report = OUT_DIR / "phase21_state_ranking_v2_report.txt"
    report.write_text("\n".join(lines), encoding="utf-8")
    return lines


def main() -> None:
    lines = run()
    for ln in lines[:25]:
        safe_print(ln)
    safe_print("...")
    for ln in lines[-20:]:
        safe_print(ln)
    safe_print(f"\nSaved: {OUT_DIR / 'phase21_state_ranking_v2_report.txt'}")


if __name__ == "__main__":
    main()
