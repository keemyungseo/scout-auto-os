"""
Scout Phase 26 - Future State Prediction Lab

Predict 1h-forward state survival from current snapshot features (level, delta, rate, duration).
Analysis only. Formulas A/A2/A5/A6 frozen. No ranking/threshold/weight changes.

Input:
  logs/phase19_winner_dna/candidates.jsonl
  logs/phase23_formula_league/match_log.jsonl
  logs/phase24_loser_mining/ (cohort reference)
  logs/phase25_transition_league/ (transition reference)

Usage:
  python scout_phase26_future_state_prediction.py
"""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import scout_phase20_winner_state_ranking as p20
from season2_p37_scout_decision_hierarchy import write_csv

P19_CAND = Path("logs") / "phase19_winner_dna" / "candidates.jsonl"
P23_MATCH = Path("logs") / "phase23_formula_league" / "match_log.jsonl"
OUT_DIR = Path("logs") / "phase26_future_prediction"

FORMULAS = ("A", "A2", "A5", "A6")
FP_THRESHOLD = 2.0
TF_ORDER = ("5m", "15m", "30m", "1h", "2h")
EXPANSION_1H = {"Expansion", "ExpansionStart", "Acceleration"}
EXPANSION_FAMILY = {
    "Expansion", "VolumeSupport", "ExpansionStart", "Acceleration", "TrendAlive", "StrongTrend",
}
CHAIN_STAGES = (
    ("5m", "Release"),
    ("15m", "Expansion"),
    ("30m", "Expansion"),
    ("1h", "Acceleration"),
    ("2h", "TrendAlive"),
)
CHAIN_DEF_KEYS = {
    "1h_current_return_pct", "1h_current_close_position", "1h_current_range_pct",
    "1h_current_body_pct", "1h_return_pct_delta", "1h_return_pct_rate",
    "1h_ma20_distance_pct_delta", "1h_ma20_distance_pct_rate", "1h_range_delta",
}
CHAIN_ALTS: dict[str, set[str]] = {
    "5m": {"Release"},
    "15m": {"Expansion", "VolumeSupport"},
    "30m": {"Expansion", "Compression"},
    "1h": {"Expansion", "ExpansionStart", "Acceleration"},
    "2h": {"TrendAlive", "StrongTrend"},
}


def safe_print(msg: str) -> None:
    print(msg.encode("ascii", "replace").decode("ascii"))


def g(f: dict, key: str, default: float = 0.0) -> float:
    return float(f.get(key, default))


def entropy(p: float) -> float:
    if p <= 0 or p >= 1:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def ig_binary(pos: int, pos_n: int, neg: int, neg_n: int) -> float:
    if pos_n == 0 or neg_n == 0:
        return 0.0
    p_y = pos_n / (pos_n + neg_n)
    parent = entropy(p_y)
    tot = pos + neg
    if tot == 0 or tot == pos_n + neg_n:
        return 0.0
    p_y_t = pos / tot if tot else 0
    rem = pos_n + neg_n - tot
    p_y_f = (pos_n - pos) / rem if rem else 0
    p_t = tot / (pos_n + neg_n)
    return parent - p_t * entropy(p_y_t) - (1 - p_t) * entropy(p_y_f)


def median_split(vals: list[float]) -> float:
    return statistics.median(vals) if vals else 0.0


def load_union_picks() -> dict[str, set[str]]:
    picks: dict[str, set[str]] = defaultdict(set)
    if not P23_MATCH.exists():
        return picks
    for line in P23_MATCH.open(encoding="utf-8"):
        m = json.loads(line)
        scan = m["scan_kst"]
        for fid in FORMULAS:
            picks[scan].update(m.get(f"{fid}_top2", []))
    return picks


def build_feature_variants(f: dict) -> dict[str, float]:
    """Level + delta + rate + duration proxies for predictive comparison."""
    out: dict[str, float] = dict(f)

    for tf in ("15m", "30m", "1h", "2h"):
        for metric in (
            "volume_ratio", "return_pct", "body_pct", "range_pct",
            "compression", "ma20_distance_pct",
        ):
            ck, pk = f"{tf}_current_{metric}", f"{tf}_previous_{metric}"
            if ck not in f or pk not in f:
                continue
            cur, prev = g(f, ck), g(f, pk)
            delta = cur - prev
            denom = max(abs(prev), 0.01)
            out[f"{tf}_{metric}_delta"] = delta
            out[f"{tf}_{metric}_rate"] = delta / denom

    # 5m sequence dynamics
    out["5m_seq_volume_energy_delta"] = g(f, "5m_seq_volume_energy_6") - g(f, "5m_volume_ma_ratio")
    out["5m_compression_delta"] = g(f, "5m_compression") - g(f, "30m_current_compression", g(f, "5m_compression"))
    out["5m_energy_rate"] = g(f, "5m_seq_volume_energy_6") / max(g(f, "5m_volume_ma_ratio"), 0.01)
    out["5m_range_energy_delta"] = g(f, "5m_range_energy") - g(f, "5m_seq_body_energy_6")
    out["1h_range_delta"] = g(f, "1h_current_range_pct") - g(f, "1h_previous_range_pct")
    out["2h_ma20_distance_delta"] = g(f, "2h_current_ma20_distance_pct") - g(f, "2h_previous_ma20_distance_pct")
    out["30m_compression_delta"] = g(f, "30m_current_compression") - g(f, "30m_previous_compression")
    out["15m_volume_delta"] = g(f, "15m_current_volume_ratio") - g(f, "15m_previous_volume_ratio")

    # duration proxies
    out["5m_compression_duration"] = g(f, "5m_compression")
    out["5m_positive_duration"] = g(f, "5m_seq_positive_count_6")
    out["30m_compression_duration"] = g(f, "30m_current_compression")
    out["1h_flat_escape"] = abs(g(f, "1h_current_return_pct")) - abs(g(f, "1h_previous_return_pct"))

    return out


def annotate_rows() -> list[dict]:
    raw = p20.load_candidates()
    by_scan: dict[str, list] = defaultdict(list)
    for r in raw:
        by_scan[r["scan_kst"]].append(r)
    winner_feats = [r["features"] for rows in by_scan.values() for r in rows[:3] if len(rows) >= 4]
    th = p20.build_thresholds(winner_feats)
    picks = load_union_picks()

    rows: list[dict] = []
    for r in raw:
        f = build_feature_variants(r["features"])
        states = p20.build_states(r["features"], th)
        intra = p20.build_transitions(r["features"], th)
        scan, sym = r["scan_kst"], r["symbol"]
        rank, mu = r["outcome_rank"], r["max_up_4h"]
        picked = sym in picks.get(scan, set())

        current_state = states["30m"]
        future_1h = states["1h"]
        trans_30_1h = f"{current_state}->{future_1h}"
        trans_15_1h = f"{states['15m']}->{future_1h}"

        focal = states["5m"] == "Release" and states["15m"] == "Expansion"
        group_a = focal and future_1h in EXPANSION_1H
        group_b = focal and future_1h == "Flat"
        expansion_maintained = states["30m"] == "Expansion" or future_1h in EXPANSION_1H

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
            "formula_picked": picked,
            "states": states,
            "intra_trans": intra,
            "mtf_path": " -> ".join(states[tf] for tf in TF_ORDER),
            "current_state_30m": current_state,
            "future_state_1h": future_1h,
            "transition_30m_1h": trans_30_1h,
            "transition_15m_1h": trans_15_1h,
            "focal_release_expansion": focal,
            "group_a_expansion_maintain": group_a,
            "group_b_flat_transition": group_b,
            "expansion_maintained": expansion_maintained,
            "features": f,
        })
    return rows


def focal_group_a(rows: list[dict]) -> list[dict]:
    """Release -> Expansion -> 1h expansion family (winner-like survival)."""
    return [r for r in rows if r["group_a_expansion_maintain"]]


def focal_group_b(rows: list[dict]) -> list[dict]:
    """Release -> Expansion -> 1h Flat (FP-like death)."""
    return [r for r in rows if r["group_b_flat_transition"]]


def predictive_power(
    retain_rows: list[dict],
    flat_rows: list[dict],
    feature_keys: list[str],
) -> list[dict]:
    """Winner retention vs FP flat transition rate per feature."""
    rn, fn = len(retain_rows), len(flat_rows)
    out: list[dict] = []

    for key in feature_keys:
        all_vals = [g(r["features"], key) for r in retain_rows + flat_rows]
        if len(set(all_vals)) < 2:
            continue
        cut = median_split(all_vals)

        def rate_high(pool: list[dict]) -> float:
            if not pool:
                return 0.0
            return sum(1 for r in pool if g(r["features"], key) >= cut) / len(pool)

        w_r = rate_high(retain_rows)
        f_r = rate_high(flat_rows)
        base = sum(1 for v in all_vals if v >= cut) / len(all_vals) if all_vals else 0

        w_high = sum(1 for r in retain_rows if g(r["features"], key) >= cut)
        f_high = sum(1 for r in flat_rows if g(r["features"], key) >= cut)

        lift_w = w_r / base if base > 0 else 0
        lift_f = f_r / base if base > 0 else 0
        ow = w_r / (1 - w_r) if w_r < 1 else 99
        of_ = f_r / (1 - f_r) if f_r < 1 else 99
        odds = of_ / ow if ow > 0 else 0
        ig = ig_binary(w_high, rn, f_high, fn)

        kind = "level"
        if "_delta" in key:
            kind = "delta"
        elif "_rate" in key:
            kind = "rate"
        elif "duration" in key or key in ("5m_compression", "5m_seq_positive_count_6"):
            kind = "duration"

        out.append({
            "feature": key,
            "feature_kind": kind,
            "definitional_leakage": key in CHAIN_DEF_KEYS,
            "winner_retention_rate": round(w_r, 4),  # Group A: expansion maintain @1h
            "fp_flat_transition_rate": round(f_r, 4),  # Group B: Flat @1h
            "lift_winner": round(lift_w, 4),
            "lift_fp": round(lift_f, 4),
            "odds_fp_vs_winner": round(odds, 4),
            "information_gain": round(ig, 4),
            "split_median": round(cut, 4),
            "retain_n": rn,
            "flat_n": fn,
        })

    out.sort(key=lambda x: (x["information_gain"], abs(x["lift_winner"] - x["lift_fp"])), reverse=True)
    return out


def compare_signal_types(power_rows: list[dict]) -> dict[str, float]:
    """Mean IG by feature_kind — does delta/rate/duration beat level?"""
    by_kind: dict[str, list[float]] = defaultdict(list)
    for r in power_rows:
        by_kind[r["feature_kind"]].append(r["information_gain"])
    return {k: round(statistics.mean(v), 4) if v else 0.0 for k, v in by_kind.items()}


def transition_chain_survival(rows: list[dict], cohort: str) -> list[dict]:
    """Stage survival along ideal winner chain."""
    starters = [r for r in rows if r["states"]["5m"] == "Release"]
    n = len(starters) or 1
    out: list[dict] = []
    prev_pool = starters

    for tf, target in CHAIN_STAGES:
        alts = CHAIN_ALTS.get(tf, {target})
        survived = [r for r in prev_pool if r["states"][tf] in alts]
        out.append({
            "cohort": cohort,
            "stage": f"{tf}_{target}",
            "stage_index": len(out),
            "survival_pct": round(len(survived) / n * 100, 2),
            "survival_n": len(survived),
            "base_n": n,
        })
        prev_pool = survived

    return out


def future_path_table(rows: list[dict], label: str) -> list[dict]:
    ctr = Counter(r["mtf_path"] for r in rows)
    total = len(rows) or 1
    return [
        {"path": k, "count": v, "pct": round(v / total * 100, 2), "cohort": label}
        for k, v in ctr.most_common(25)
    ]


def future_survival_score(
    picked_rows: list[dict],
    top_features: list[dict],
    retain_rows: list[dict],
    flat_rows: list[dict],
) -> list[dict]:
    """
    Descriptive score: P(expansion maintain | feature profile).
    Not applied to ranking.
    """
    if not top_features:
        return []

    # direction: positive IG features where high = more retention
    weights: list[tuple[str, float, int]] = []
    for r in top_features[:12]:
        key = r["feature"]
        wr, fr = r["winner_retention_rate"], r["fp_flat_transition_rate"]
        direction = 1 if wr >= fr else -1
        w = r["information_gain"] * direction
        weights.append((key, w, direction))

    retain_vals = {k: [g(r["features"], k) for r in retain_rows] for k, _, _ in weights}
    flat_vals = {k: [g(r["features"], k) for r in flat_rows] for k, _, _ in weights}
    cuts = {k: median_split(retain_vals[k] + flat_vals[k]) for k, _, _ in weights}

    def raw_score(f: dict) -> float:
        s = 0.0
        for key, w, direction in weights:
            v = g(f, key)
            above = v >= cuts[key]
            s += w * (1.0 if (above and direction > 0) or (not above and direction < 0) else 0.0)
        return s

    retain_scores = [raw_score(r["features"]) for r in retain_rows]
    flat_scores = [raw_score(r["features"]) for r in flat_rows]
    all_scores = retain_scores + flat_scores
    lo, hi = min(all_scores), max(all_scores)
    span = hi - lo if hi > lo else 1.0

    out: list[dict] = []
    for r in picked_rows:
        raw = raw_score(r["features"])
        norm = (raw - lo) / span
        out.append({
            "scan_kst": r["scan_kst"],
            "symbol": r["symbol"],
            "cohort": r["cohort"],
            "outcome_rank": r["outcome_rank"],
            "max_up_4h": r["max_up_4h"],
            "future_state_1h": r["future_state_1h"],
            "transition_30m_1h": r["transition_30m_1h"],
            "focal_release_expansion": r["focal_release_expansion"],
            "group_a": r["group_a_expansion_maintain"],
            "group_b": r["group_b_flat_transition"],
            "future_survival_score": round(norm * 100, 2),
            "raw_score": round(raw, 4),
            "actual_expansion_1h": 1 if r["future_state_1h"] in EXPANSION_1H else 0,
        })

    out.sort(key=lambda x: x["future_survival_score"], reverse=True)
    return out


def run() -> list[str]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = annotate_rows()
    ga = focal_group_a(rows)
    gb = focal_group_b(rows)

    # Group A = expansion maintain (winner path); Group B = 1h Flat (FP death path)
    ga_winners = [r for r in ga if r["outcome_rank"] <= 2]
    gb_fp = [r for r in gb if r["cohort"] == "false_positive"]
    retain_pool = ga
    flat_pool = gb

    sample = rows[0]["features"] if rows else {}
    base_keys = sorted(sample.keys())
    power = predictive_power(retain_pool, flat_pool, base_keys)
    kind_ig = compare_signal_types(power)

    # transition survival by cohort
    surv_w = transition_chain_survival([r for r in rows if r["outcome_rank"] <= 2], "winner_gt")
    surv_fp = transition_chain_survival([r for r in rows if r["cohort"] == "false_positive"], "false_positive")
    surv_focal = transition_chain_survival(
        [r for r in rows if r["focal_release_expansion"]], "focal_release_expansion"
    )
    survival_all = surv_w + surv_fp + surv_focal

    winner_paths = future_path_table(ga_winners if ga_winners else ga, "winner_future")
    fp_paths = future_path_table(gb_fp if gb_fp else gb, "fp_future")

    picked = [r for r in rows if r["formula_picked"]]
    top_feats = [r for r in power if r["information_gain"] > 0 and not r.get("definitional_leakage")][:20]
    if not top_feats:
        top_feats = [r for r in power if r["information_gain"] > 0][:20]
    survival_scores = future_survival_score(picked, top_feats, retain_pool, flat_pool)

    # delta vs level head-to-head on same base metrics
    level_ig = statistics.mean([r["information_gain"] for r in power if r["feature_kind"] == "level"] or [0])
    delta_ig = statistics.mean([r["information_gain"] for r in power if r["feature_kind"] == "delta"] or [0])
    rate_ig = statistics.mean([r["information_gain"] for r in power if r["feature_kind"] == "rate"] or [0])
    dur_ig = statistics.mean([r["information_gain"] for r in power if r["feature_kind"] == "duration"] or [0])

    write_csv(OUT_DIR / "feature_delta_importance.csv", power)
    write_csv(OUT_DIR / "transition_survival_probability.csv", survival_all)
    write_csv(OUT_DIR / "winner_future_path.csv", winner_paths)
    write_csv(OUT_DIR / "fp_future_path.csv", fp_paths)
    write_csv(OUT_DIR / "future_survival_score.csv", survival_scores)

    top_power = [r for r in power if not r.get("definitional_leakage")][:15] or power[:15]
    top_leak = [r for r in power if r.get("definitional_leakage")][:3]
    top_delta = sorted([r for r in power if r["feature_kind"] == "delta"], key=lambda x: x["information_gain"], reverse=True)[:8]
    top_rate = sorted([r for r in power if r["feature_kind"] == "rate"], key=lambda x: x["information_gain"], reverse=True)[:8]

    # score calibration on picked
    if survival_scores:
        picked_focal = [s for s in survival_scores if s["focal_release_expansion"]]
        hi = [s for s in picked_focal if s["future_survival_score"] >= 60]
        hit_rate = sum(s["actual_expansion_1h"] for s in hi) / len(hi) if hi else 0
        lo = [s for s in picked_focal if s["future_survival_score"] < 40]
        miss_rate = 1 - (sum(s["actual_expansion_1h"] for s in lo) / len(lo)) if lo else 0
    else:
        hit_rate, miss_rate = 0.0, 0.0

    lines = [
        "############################################################",
        "SCOUT PHASE 26 - FUTURE STATE PREDICTION LAB",
        "############################################################",
        "",
        "DATA SOURCES (verified):",
        f"  {P19_CAND} (candidates + features)",
        f"  {P23_MATCH} (formula picks: {', '.join(FORMULAS)})",
        f"  Phase24/25 cohort logic reused",
        f"  All candidates: {len(rows)} | focal Release->Expansion: "
        f"{sum(1 for r in rows if r['focal_release_expansion'])}",
        f"  Group A (1h expansion maintain): {len(ga)} | Group B (1h Flat): {len(gb)}",
        f"  Feature compare: Group A n={len(retain_pool)} vs Group B n={len(flat_pool)}",
        f"  (Group A=1h expansion maintain | Group B=1h Flat on Release->Expansion path)",
        "Formulas frozen. Ranking unchanged. Analysis only.",
        "",
        "=" * 62,
        "1. CURRENT -> 1H FUTURE STATE TRANSITION",
        "=" * 62,
        f"  Current anchor: 30m state | Future: 1h state at scan snapshot",
        f"  Group A path: Release -> Expansion -> Expansion/Acceleration @1h (winner survival)",
        f"  Group B path: Release -> Expansion -> Flat @1h (FP death zone)",
        "",
        "=" * 62,
        "2. GROUP A vs GROUP B (focal path)",
        "=" * 62,
        f"  Group A winners (rank<=2): {len(ga_winners)} / {len(ga)}",
        f"  Group B formula FP: {len(gb_fp)} / {len(gb)}",
        f"  Group A max_up<2%: {sum(1 for r in ga if r['max_up_4h'] < FP_THRESHOLD)}",
        f"  Group B max_up<2%: {sum(1 for r in gb if r['max_up_4h'] < FP_THRESHOLD)}",
        "",
        "=" * 62,
        "3. SIGNAL TYPE COMPARISON (mean IG)",
        "=" * 62,
        f"  level:    IG={kind_ig.get('level', 0):.4f}",
        f"  delta:    IG={kind_ig.get('delta', 0):.4f}",
        f"  rate:     IG={kind_ig.get('rate', 0):.4f}",
        f"  duration: IG={kind_ig.get('duration', 0):.4f}",
        f"  Best signal family: {max(kind_ig, key=kind_ig.get) if kind_ig else 'n/a'}",
        "",
        "=" * 62,
        "4. TOP15 PREDICTIVE FEATURES (retention vs flat)",
        "=" * 62,
    ]
    for r in top_power:
        lines.append(
            f"  [{r['feature_kind']}] {r['feature']}: "
            f"retain={r['winner_retention_rate']:.2f} flat={r['fp_flat_transition_rate']:.2f} "
            f"lift_w={r['lift_winner']:.2f} lift_fp={r['lift_fp']:.2f} "
            f"IG={r['information_gain']:.3f} odds={r['odds_fp_vs_winner']:.2f}"
        )

    lines.extend(["", "=" * 62, "5. TOP DELTA / RATE FEATURES", "=" * 62])
    for r in top_delta[:5]:
        lines.append(f"  delta {r['feature']}: IG={r['information_gain']:.3f}")
    for r in top_rate[:5]:
        lines.append(f"  rate  {r['feature']}: IG={r['information_gain']:.3f}")

    lines.extend(["", "=" * 62, "6. TRANSITION CHAIN SURVIVAL", "=" * 62])
    for label in ("winner_gt", "false_positive", "focal_release_expansion"):
        lines.append(f"  [{label}]")
        for s in survival_all:
            if s["cohort"] == label:
                lines.append(f"    {s['stage']}: {s['survival_pct']:.1f}% (n={s['survival_n']}/{s['base_n']})")

    lines.extend(["", "=" * 62, "7. FUTURE SURVIVAL SCORE (formula picks, not ranked)", "=" * 62])
    lines.append(f"  Scored picks: {len(survival_scores)}")
    lines.append(f"  Focal picks score>=60: 1h expansion hit rate {hit_rate*100:.1f}%")
    lines.append(f"  Focal picks score<40:  1h flat/death rate ~{miss_rate*100:.1f}%")
    if survival_scores[:5]:
        lines.append("  Top5 by future_survival_score:")
        for s in survival_scores[:5]:
            lines.append(
                f"    {s['symbol']} score={s['future_survival_score']:.1f} "
                f"1h={s['future_state_1h']} rank={s['outcome_rank']}"
            )

    lines.extend(["", "=" * 62, "8. REPRESENTATIVE FUTURE PATHS", "=" * 62])
    if winner_paths:
        lines.append(f"  Winner future: {winner_paths[0]['path']} ({winner_paths[0]['pct']}%)")
    if fp_paths:
        lines.append(f"  FP future:     {fp_paths[0]['path']} ({fp_paths[0]['pct']}%)")

    lines.extend(["", "=" * 62, "9. RECOMMENDATION (analysis only)", "=" * 62])
    if top_leak:
        lines.append("  (Note: 1h_* features partially define target state — prefer 30m/15m deltas for forward prediction)")
    if top_power:
        best = top_power[0]
        lines.append(
            f"  Strongest forward signal (ex-1h leakage): {best['feature']} ({best['feature_kind']}) "
            f"IG={best['information_gain']:.3f}"
        )
    elif power:
        best = power[0]
        lines.append(
            f"  Strongest signal (definitional): {best['feature']} IG={best['information_gain']:.3f}"
        )
    if delta_ig > level_ig:
        lines.append("  Delta features outperform static levels on average — prefer rate-of-change for 1h survival.")
    else:
        lines.append("  Static levels still competitive — combine level + delta for holdout tests.")
    lines.append(
        "  Group B (Release->Expansion->Flat) confirms Phase25 early death at 1h; "
        "future survival score is descriptive only — no formula/ranking change."
    )
    lines.append("  Holdout validation required before any transition gate enters engine.")
    lines.append("")
    lines.append("DISCLAIMER: Probabilistic state survival — not price prediction.")

    report = OUT_DIR / "phase26_future_prediction_report.txt"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for ln in lines:
        safe_print(ln)
    safe_print(f"\nSaved: {report}")
    return lines


if __name__ == "__main__":
    run()
