"""
Scout Phase 27 - Acceleration Birth Detector (ABD)

Find earliest pre-Acceleration triggers from existing feature dynamics.
Prediction research ONLY. Formulas A/A2/A5/A6 frozen. No ranking/threshold/weight changes.

Input:
  logs/phase19_winner_dna/candidates.jsonl
  logs/phase23_formula_league/match_log.jsonl
  logs/phase25_transition_league/ (track B reference)
  logs/phase26_future_prediction/ (track C reference)

Usage:
  python scout_phase27_acceleration_birth_detector.py
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
P25_LIFT = Path("logs") / "phase25_transition_league" / "transition_importance.csv"
P26_FEAT = Path("logs") / "phase26_future_prediction" / "feature_delta_importance.csv"
OUT_DIR = Path("logs") / "phase27_acceleration_birth"

FORMULAS = ("A", "A2", "A5", "A6")
FP_THRESHOLD = 2.0
EXPANSION_15M = {"Expansion", "VolumeSupport"}
DEATH_2H = {"Flat", "OverExtended"}
TF_EARLY_WEIGHT = {"5m": 3.0, "15m": 2.0, "30m": 1.0, "1h": 0.25, "2h": 0.1}


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


def zscore_map(vals: list[float]) -> list[float]:
    if not vals:
        return []
    mu = statistics.mean(vals)
    sd = statistics.stdev(vals) if len(vals) > 1 else 1.0
    if sd < 1e-9:
        return [0.0] * len(vals)
    return [(v - mu) / sd for v in vals]


def tf_of_feature(key: str) -> str:
    for tf in ("5m", "15m", "30m", "1h", "2h"):
        if key.startswith(tf):
            return tf
    return "other"


def build_abd_features(f: dict) -> dict[str, float]:
    """Extend existing features: level, delta, rate, slope, curvature, velocities."""
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
            rate = delta / denom
            out[f"{tf}_{metric}_delta"] = delta
            out[f"{tf}_{metric}_rate"] = rate
            out[f"{tf}_{metric}_slope"] = delta
            out[f"{tf}_{metric}_curvature"] = delta - rate * 0.5

    # 5m stack
    mom = g(f, "5m_momentum")
    vol_e = g(f, "5m_seq_volume_energy_6")
    comp = g(f, "5m_compression")
    rel = g(f, "5m_release")
    out["5m_momentum_slope"] = mom
    out["5m_momentum_of_momentum"] = mom * vol_e
    out["5m_compression_release_speed"] = rel + max(0.0, g(f, "30m_compression_delta", 0) * -1)
    out["5m_volume_velocity"] = vol_e / max(g(f, "5m_volume_ma_ratio"), 0.01)
    out["5m_body_velocity"] = g(f, "5m_seq_body_energy_6")
    out["5m_range_velocity"] = g(f, "5m_range_energy")
    out["5m_expansion_velocity"] = g(f, "5m_seq_return_sum_6") * out["5m_volume_velocity"]

    out["15m_volume_velocity"] = out.get("15m_volume_ratio_delta", 0)
    out["15m_body_velocity"] = out.get("15m_body_pct_delta", 0)
    out["15m_ma_distance_velocity"] = out.get("15m_ma20_distance_pct_delta", 0)
    out["15m_expansion_velocity"] = out.get("15m_return_pct_rate", 0) * (1 + abs(out.get("15m_volume_ratio_rate", 0)))

    out["30m_return_velocity"] = out.get("30m_return_pct_delta", 0)
    out["30m_ma_distance_velocity"] = out.get("30m_ma20_distance_pct_delta", 0)
    out["30m_body_velocity"] = out.get("30m_body_pct_delta", 0)
    out["30m_compression_release_speed"] = -out.get("30m_compression_delta", 0)
    out["30m_expansion_velocity"] = out.get("30m_return_pct_rate", 0) * out.get("30m_volume_ratio_rate", 0)

    # change-of-change
    for base in ("15m_return_pct", "30m_return_pct", "15m_ma20_distance_pct"):
        d = out.get(f"{base}_delta", 0)
        r = out.get(f"{base}_rate", 0)
        out[f"{base}_change_of_change"] = d * r

    out["volume_rate_stability"] = -abs(out.get("15m_volume_ratio_rate", 0))
    out["5m_positive_duration"] = g(f, "5m_seq_positive_count_6")
    out["5m_compression_duration"] = comp

    return out


def classify_birth_group(states: dict[str, str], max_up: float, intra: dict[str, str]) -> str:
    """
    Mutually exclusive birth groups after 15m expansion.
    A: Expansion->Acceleration within 30-60m (1h=Acceleration)
    B: Expansion->Flat
    C: Expansion->Compression (30m)
    D: Expansion->Death (weak forward outcome)
    """
    if states["15m"] not in EXPANSION_15M:
        return "no_expansion"

    if states["1h"] == "Acceleration":
        return "A_accel_birth"
    if states["1h"] == "Flat":
        return "B_flat"
    if states["30m"] == "Compression":
        return "C_compression"
    if max_up < FP_THRESHOLD:
        return "D_death"
    return "other_expansion"


def load_a6_picks() -> dict[str, list[str]]:
    picks: dict[str, list[str]] = {}
    if not P23_MATCH.exists():
        return picks
    for line in P23_MATCH.open(encoding="utf-8"):
        m = json.loads(line)
        picks[m["scan_kst"]] = m.get("A6_top2", m.get("A_top2", []))
    return picks


def annotate_all() -> tuple[list[dict], p20.Thresholds]:
    raw = p20.load_candidates()
    by_scan: dict[str, list] = defaultdict(list)
    for r in raw:
        by_scan[r["scan_kst"]].append(r)
    winner_feats = [r["features"] for rows in by_scan.values() for r in rows[:3] if len(rows) >= 4]
    th = p20.build_thresholds(winner_feats)
    a6_picks = load_a6_picks()

    rows: list[dict] = []
    for r in raw:
        base_f = r["features"]
        f = build_abd_features(base_f)
        states = p20.build_states(base_f, th)
        intra = p20.build_transitions(base_f, th)
        scan, sym = r["scan_kst"], r["symbol"]
        rank, mu = r["outcome_rank"], r["max_up_4h"]
        birth = classify_birth_group(states, mu, intra)
        picked_a6 = sym in a6_picks.get(scan, [])

        rows.append({
            "scan_kst": scan,
            "symbol": sym,
            "outcome_rank": rank,
            "max_up_4h": mu,
            "states": states,
            "intra_trans": intra,
            "mtf_path": " -> ".join(states[t] for t in ("5m", "15m", "30m", "1h", "2h")),
            "birth_group": birth,
            "formula_a6_picked": picked_a6,
            "is_fp": picked_a6 and mu < FP_THRESHOLD,
            "is_winner": rank <= 2,
            "features": f,
        })
    return rows, th


def feature_importance(
    pos_rows: list[dict],
    neg_rows: list[dict],
    keys: list[str],
) -> list[dict]:
    pn, nn = len(pos_rows), len(neg_rows)
    out: list[dict] = []
    for key in keys:
        all_vals = [g(r["features"], key) for r in pos_rows + neg_rows]
        if len(set(round(v, 6) for v in all_vals)) < 2:
            continue
        cut = median_split(all_vals)

        def hi(pool: list[dict]) -> float:
            return sum(1 for r in pool if g(r["features"], key) >= cut) / len(pool) if pool else 0.0

        pr, nr = hi(pos_rows), hi(neg_rows)
        ph = sum(1 for r in pos_rows if g(r["features"], key) >= cut)
        nh = sum(1 for r in neg_rows if g(r["features"], key) >= cut)
        base = sum(1 for v in all_vals if v >= cut) / len(all_vals)
        ig = ig_binary(ph, pn, nh, nn)
        tf = tf_of_feature(key)
        early_w = TF_EARLY_WEIGHT.get(tf, 0.5)
        out.append({
            "feature": key,
            "timeframe": tf,
            "accel_birth_rate_high": round(pr, 4),
            "non_birth_rate_high": round(nr, 4),
            "lift_birth": round(pr / base if base else 0, 4),
            "odds_birth_vs_other": round((nr / (1 - nr) if nr < 1 else 99) / (pr / (1 - pr) if pr < 1 and pr > 0 else 0.01), 4),
            "information_gain": round(ig, 4),
            "early_priority_score": round(ig * early_w, 4),
            "split_median": round(cut, 4),
        })
    out.sort(key=lambda x: (x["early_priority_score"], x["information_gain"]), reverse=True)
    return out


def loo_stability(rows: list[dict], top_keys: list[str], pos_label: str) -> dict[str, float]:
    """Per-feature IG std across leave-one-scan-out folds."""
    by_scan: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_scan[r["scan_kst"]].append(r)
    scans = sorted(by_scan)
    if len(scans) < 5:
        return {k: 0.0 for k in top_keys}

    igs: dict[str, list[float]] = defaultdict(list)
    for hold in scans:
        train = [r for s, pool in by_scan.items() if s != hold for r in pool if r["birth_group"] != "no_expansion"]
        pos = [r for r in train if r["birth_group"] == pos_label]
        neg = [r for r in train if r["birth_group"] != pos_label and r["birth_group"] != "no_expansion"]
        if len(pos) < 5 or len(neg) < 5:
            continue
        imp = feature_importance(pos, neg, top_keys)
        for row in imp:
            igs[row["feature"]].append(row["information_gain"])

    stability: dict[str, float] = {}
    for k in top_keys:
        vals = igs.get(k, [])
        if len(vals) < 3:
            stability[k] = 0.0
        else:
            stability[k] = round(1.0 - min(statistics.stdev(vals), 1.0), 4)
    return stability


LEAK_1H = {
    "1h_current_return_pct", "1h_body_pct_delta", "1h_body_pct_slope",
    "1h_return_pct_delta", "1h_return_pct_rate", "1h_ma20_distance_pct_delta",
}
def build_triggers(top_feats: list[dict]) -> dict[str, list[tuple[str, float]]]:
    """Score-only trigger composites from top features per timeframe band."""
    clean = [r for r in top_feats if r["feature"] not in LEAK_1H]
    by_tf: dict[str, list[dict]] = defaultdict(list)
    for r in clean:
        by_tf[r["timeframe"]].append(r)

    def topn(tf: str, n: int = 3) -> list[tuple[str, float]]:
        return [(r["feature"], max(r["composite_score"], 0.01)) for r in by_tf.get(tf, [])[:n]]

    t5 = topn("5m", 3)
    t15 = topn("15m", 3)
    t30 = topn("30m", 3)

    triggers: dict[str, list[tuple[str, float]]] = {
        "Trigger_A": t30[:2] + [("volume_rate_stability", 0.8)] if t30 else t15[:2],
        "Trigger_B": [
            ("5m_compression_release_speed", 1.0),
            ("5m_body_velocity", 0.9),
            ("5m_expansion_velocity", 0.8),
        ],
        "Trigger_C": t5[:2] + [("5m_momentum_of_momentum", 0.7)] if t5 else t15[:2],
        "Trigger_D": t15[:2] + t30[:1] if t15 and t30 else t15[:3],
        "Trigger_E": [
            ("15m_expansion_velocity", 1.0),
            ("30m_expansion_velocity", 0.9),
            ("volume_rate_stability", 0.5),
        ],
        "Trigger_F": [
            ("30m_compression_release_speed", 1.0),
            ("5m_positive_duration", 0.8),
            ("5m_momentum_slope", 0.7),
        ],
    }

    # merge learned weights where available
    feat_w = {r["feature"]: max(r["composite_score"], 0.01) for r in clean}
    for tid, comps in triggers.items():
        merged: list[tuple[str, float]] = []
        seen: set[str] = set()
        for name, w in comps:
            if name in seen:
                continue
            seen.add(name)
            merged.append((name, feat_w.get(name, w)))
        if len(merged) < 2 and t15:
            merged = [(t15[0][0], t15[0][1]), (t30[0][0], t30[0][1])] if t30 else t15[:2]
        triggers[tid] = merged[:3]
    return triggers


def trigger_description(tid: str, comps: list[tuple[str, float]]) -> str:
    parts = []
    for name, w in comps:
        short = name.replace("_pct", "").replace("_delta", "_d").replace("_rate", "_r")
        arrow = "^" if w > 0 else "v"
        parts.append(f"{short} {arrow}")
    return f"{tid}: " + " + ".join(parts)


def score_trigger(row: dict, comps: list[tuple[str, float]], norms: dict[str, tuple[float, float]]) -> float:
    s = 0.0
    for name, w in comps:
        v = g(row["features"], name)
        mu, sd = norms.get(name, (0.0, 1.0))
        z = (v - mu) / sd if sd > 1e-9 else 0.0
        s += w * z
    return s


def scan_norms(pool: list[dict], keys: set[str]) -> dict[str, tuple[float, float]]:
    norms: dict[str, tuple[float, float]] = {}
    for k in keys:
        vals = [g(r["features"], k) for r in pool]
        if not vals:
            norms[k] = (0.0, 1.0)
        else:
            sd = statistics.stdev(vals) if len(vals) > 1 else 1.0
            norms[k] = (statistics.mean(vals), sd if sd > 1e-9 else 1.0)
    return norms


def eval_trigger_league(
    by_scan: dict[str, list[dict]],
    triggers: dict[str, list[tuple[str, float]]],
    pool_filter: str = "expansion",
) -> tuple[list[dict], dict[str, dict]]:
    all_keys = {n for comps in triggers.values() for n, _ in comps}
    league_rows: list[dict] = []
    per_scan: dict[str, dict] = {}

    totals: dict[str, dict] = {
        tid: {
            "top2_hits": 0, "top5_hits": 0, "scans": 0,
            "winner_recall_num": 0, "winner_recall_den": 0,
            "fp_top2": 0, "miss_recovery": 0,
            "early_tf_score": 0.0,
        }
        for tid in triggers
    }

    for scan, pool_all in sorted(by_scan.items()):
        if pool_filter == "expansion":
            pool = [r for r in pool_all if r["birth_group"] != "no_expansion"]
        elif pool_filter == "focal":
            pool = [
                r for r in pool_all
                if r["states"]["5m"] == "Release" and r["states"]["15m"] == "Expansion"
            ]
        else:
            pool = pool_all
        if len(pool) < 4:
            continue
        actual2 = {r["symbol"] for r in pool if r["outcome_rank"] <= 2}
        actual5 = {r["symbol"] for r in pool if r["outcome_rank"] <= 5}
        norms = scan_norms(pool, all_keys)
        a6_picks = [r for r in pool if r["formula_a6_picked"]]
        misses = {r["symbol"] for r in pool if r["is_winner"] and not r["formula_a6_picked"]}

        scan_res: dict = {}
        for tid, comps in triggers.items():
            for r in pool:
                r["_trig_score"] = score_trigger(r, comps, norms)
            ranked = sorted(pool, key=lambda x: x["_trig_score"], reverse=True)
            pick2 = {x["symbol"] for x in ranked[:2]}
            pick5 = {x["symbol"] for x in ranked[:5]}
            t2 = len(pick2 & actual2)
            t5 = len(pick5 & actual5)
            wr = len(pick5 & actual2) / len(actual2) if actual2 else 0
            fp2 = sum(1 for s in pick2 if any(r["symbol"] == s and r["is_fp"] for r in pool))
            miss_rec = len(pick5 & misses)
            early = sum(TF_EARLY_WEIGHT.get(tf_of_feature(n), 0.5) for n, _ in comps) / len(comps)

            totals[tid]["top2_hits"] += t2
            totals[tid]["top5_hits"] += t5
            totals[tid]["scans"] += 1
            totals[tid]["winner_recall_num"] += len(pick5 & actual2)
            totals[tid]["winner_recall_den"] += len(actual2)
            totals[tid]["fp_top2"] += fp2
            totals[tid]["miss_recovery"] += miss_rec
            totals[tid]["early_tf_score"] += early

            scan_res[tid] = {"top2": t2, "pick2": list(pick2)}

        per_scan[scan] = scan_res

    for tid, t in totals.items():
        n = t["scans"] or 1
        league_rows.append({
            "trigger": tid,
            "description": trigger_description(tid, triggers[tid]),
            "top2_hit_pct": round(t["top2_hits"] / (2 * n) * 100, 2),
            "top5_hit_pct": round(t["top5_hits"] / (5 * n) * 100, 2),
            "winner_recall_pct": round(t["winner_recall_num"] / t["winner_recall_den"] * 100, 2) if t["winner_recall_den"] else 0,
            "fp_in_top2": t["fp_top2"],
            "miss_recovery_top5": t["miss_recovery"],
            "early_detection_score": round(t["early_tf_score"] / n, 3),
            "scans": n,
        })
    league_rows.sort(key=lambda x: (x["top2_hit_pct"], x["winner_recall_pct"]), reverse=True)
    return league_rows, totals


def false_trigger_clusters(
    rows: list[dict],
    triggers: dict[str, list[tuple[str, float]]],
    best_tid: str,
) -> list[dict]:
    """High trigger score but no acceleration birth."""
    by_scan: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["birth_group"] == "no_expansion":
            continue
        by_scan[r["scan_kst"]].append(r)

    comps = triggers[best_tid]
    keys = {n for n, _ in comps}
    false_rows: list[dict] = []

    for scan, pool in by_scan.items():
        norms = scan_norms(pool, keys)
        for r in pool:
            r["_ts"] = score_trigger(r, comps, norms)
        ranked = sorted(pool, key=lambda x: x["_ts"], reverse=True)
        cutoff = ranked[max(0, len(ranked) // 4)]["_ts"] if ranked else 0
        for r in ranked:
            if r["_ts"] >= cutoff and r["birth_group"] != "A_accel_birth":
                false_rows.append(r)

    if not false_rows:
        return [{"cluster": "(none)", "count": 0, "pct": 0.0, "reject_signal": "n/a"}]

    tag_ctr: Counter = Counter()
    for r in false_rows:
        f = r["features"]
        tags: list[str] = []
        if g(f, "15m_volume_velocity") > 2 and g(f, "30m_return_velocity") < 0.5:
            tags.append("volume_burst_only")
        if g(f, "30m_return_velocity") < 0.2 and g(f, "15m_ma_distance_velocity") < 0.2:
            tags.append("weak_delta")
        if r["states"]["1h"] == "Flat" or r["states"]["30m"] == "Neutral":
            tags.append("flat_tendency")
        if r["states"]["15m"] == "Expansion" and r["states"]["30m"] == "Compression" and r["states"]["1h"] != "Acceleration":
            tags.append("late_expansion")
        if not tags:
            tags.append("unclassified_false_trigger")
        for t in tags:
            tag_ctr[t] += 1

    total = len(false_rows) or 1
    return [
        {"cluster": k, "count": v, "pct": round(v / total * 100, 2), "reject_signal": k}
        for k, v in tag_ctr.most_common(20)
    ]


def early_signal_timeline(top_feats: list[dict], stability: dict[str, float]) -> list[dict]:
    rows: list[dict] = []
    for i, r in enumerate(top_feats[:30]):
        rows.append({
            "rank": i + 1,
            "feature": r["feature"],
            "timeframe": r["timeframe"],
            "information_gain": r["information_gain"],
            "early_priority_score": r["early_priority_score"],
            "loo_stability": stability.get(r["feature"], 0.0),
            "lift_birth": r["lift_birth"],
            "accel_birth_rate_high": r["accel_birth_rate_high"],
            "non_birth_rate_high": r["non_birth_rate_high"],
        })
    return rows


def birth_trigger_ranking(
    triggers: dict[str, list[tuple[str, float]]],
    league: list[dict],
    top_feats: list[dict],
) -> list[dict]:
    feat_rank = {r["feature"]: i + 1 for i, r in enumerate(top_feats)}
    rows: list[dict] = []
    for i, lr in enumerate(league):
        tid = lr["trigger"]
        comps = triggers[tid]
        comp_ranks = [feat_rank.get(n, 99) for n, _ in comps]
        rows.append({
            "rank": i + 1,
            "trigger": tid,
            "description": lr["description"],
            "top2_hit_pct": lr["top2_hit_pct"],
            "top5_hit_pct": lr["top5_hit_pct"],
            "winner_recall_pct": lr["winner_recall_pct"],
            "fp_in_top2": lr["fp_in_top2"],
            "early_detection_score": lr["early_detection_score"],
            "avg_component_feature_rank": round(statistics.mean(comp_ranks), 2),
            "holdout_candidate": "YES" if lr["top2_hit_pct"] >= 35 else "HYPOTHESIS",
        })
    return rows


def meta_simulation(
    by_scan: dict[str, list[dict]],
    triggers: dict[str, list[tuple[str, float]]],
    best_tid: str,
) -> list[dict]:
    """Simulate trigger overlay on A6 picks — no ranking change."""
    comps = triggers[best_tid]
    keys = {n for n, _ in comps}
    rows: list[dict] = []

    base_t2 = base_fp = 0
    sim_t2 = sim_fp = 0
    n = 0

    for scan, pool in sorted(by_scan.items()):
        if len(pool) < 4:
            continue
        actual2 = {r["symbol"] for r in pool if r["outcome_rank"] <= 2}
        a6 = [r for r in pool if r["formula_a6_picked"]]
        if not a6:
            continue
        n += 1
        base_pick2 = {r["symbol"] for r in a6[:2]} if len(a6) >= 2 else {r["symbol"] for r in a6}
        base_t2 += len(base_pick2 & actual2)
        base_fp += sum(1 for s in base_pick2 if any(r["symbol"] == s and r["is_fp"] for r in pool))

        norms = scan_norms(pool, keys)
        for r in a6:
            r["_ts"] = score_trigger(r, comps, norms)
        boosted = sorted(a6, key=lambda x: x["_ts"], reverse=True)
        sim_pick2 = {x["symbol"] for x in boosted[:2]}
        sim_t2 += len(sim_pick2 & actual2)
        sim_fp += sum(1 for s in sim_pick2 if any(r["symbol"] == s and r["is_fp"] for r in pool))

        rows.append({
            "scan_kst": scan,
            "a6_base_top2_hit": len(base_pick2 & actual2),
            "trigger_sim_top2_hit": len(sim_pick2 & actual2),
            "a6_base_fp_top2": sum(1 for s in base_pick2 if any(r["symbol"] == s and r["is_fp"] for r in pool)),
            "trigger_sim_fp_top2": sum(1 for s in sim_pick2 if any(r["symbol"] == s and r["is_fp"] for r in pool)),
            "delta_top2": len(sim_pick2 & actual2) - len(base_pick2 & actual2),
            "delta_fp": sum(1 for s in sim_pick2 if any(r["symbol"] == s and r["is_fp"] for r in pool))
            - sum(1 for s in base_pick2 if any(r["symbol"] == s and r["is_fp"] for r in pool)),
        })

    summary = {
        "scan_kst": "AGGREGATE",
        "a6_base_top2_hit": base_t2,
        "trigger_sim_top2_hit": sim_t2,
        "a6_base_fp_top2": base_fp,
        "trigger_sim_fp_top2": sim_fp,
        "delta_top2": sim_t2 - base_t2,
        "delta_fp": sim_fp - base_fp,
        "base_top2_pct": round(base_t2 / (2 * n) * 100, 2) if n else 0,
        "sim_top2_pct": round(sim_t2 / (2 * n) * 100, 2) if n else 0,
        "scans": n,
    }
    rows.append(summary)  # type: ignore[arg-type]
    return rows


def multi_track_opinions(best_trigger: str, best_t2: float) -> list[str]:
    lines = ["=" * 62, "MULTI-TRACK OPINIONS (independent, no merge)", "=" * 62]

    # Track A
    lines.append("  Track A — Formula League (Phase23):")
    if P23_MATCH.exists():
        hits = []
        for line in P23_MATCH.open(encoding="utf-8"):
            m = json.loads(line)
            a6 = set(m.get("A6_top2", []))
            act = set(m.get("actual_top2", []))
            hits.append(len(a6 & act))
        t2 = sum(hits) / (2 * len(hits)) * 100 if hits else 0
        lines.append(f"    A6 TOP2 hit: {t2:.1f}% — keep frozen base formula")
    else:
        lines.append("    A6 data missing")

    # Track B
    lines.append("  Track B — State Transition (Phase25):")
    if P25_LIFT.exists():
        lines.append("    Strongest edge: Compression->Expansion (IG~0.02) — path context only")
    else:
        lines.append("    Phase25 output missing — run phase25 first")

    # Track C
    lines.append("  Track C — Future Survival (Phase26):")
    if P26_FEAT.exists():
        lines.append("    Forward signal: 30m_return_delta / ma_distance_rate — 1h flat filter candidate")
    else:
        lines.append("    Phase26 output missing")

    # Track D
    lines.append("  Track D — Acceleration Birth Trigger (Phase27):")
    lines.append(f"    Best trigger: {best_trigger} TOP2={best_t2:.1f}% — holdout candidate if >=35%")
    lines.append("  POLICY: Tracks independent. No merge in Phase27.")
    return lines


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, _th = annotate_all()
    exp_rows = [r for r in rows if r["birth_group"] != "no_expansion"]
    pos = [r for r in exp_rows if r["birth_group"] == "A_accel_birth"]
    neg = [r for r in exp_rows if r["birth_group"] != "A_accel_birth"]

    sample_keys = sorted(exp_rows[0]["features"].keys()) if exp_rows else []
    importance = feature_importance(pos, neg, sample_keys)
    top_keys = [r["feature"] for r in importance[:40]]
    stability = loo_stability(exp_rows, top_keys, "A_accel_birth")

    for r in importance:
        r["loo_stability"] = stability.get(r["feature"], 0.0)
        r["composite_score"] = round(
            r["early_priority_score"] * 0.5 + r["information_gain"] * 0.3 + r["loo_stability"] * 0.2,
            4,
        )
    importance.sort(key=lambda x: x["composite_score"], reverse=True)

    triggers = build_triggers(importance[:25])
    by_scan: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_scan[r["scan_kst"]].append(r)

    league, _ = eval_trigger_league(by_scan, triggers, pool_filter="expansion")
    league_focal, _ = eval_trigger_league(by_scan, triggers, pool_filter="focal")
    best = league[0] if league else {"trigger": "Trigger_A", "top2_hit_pct": 0.0}
    best_tid = best["trigger"]

    false_clusters = false_trigger_clusters(exp_rows, triggers, best_tid)
    timeline = early_signal_timeline(importance, stability)
    trigger_rank = birth_trigger_ranking(triggers, league, importance)
    meta = meta_simulation(by_scan, triggers, best_tid)

    # birth group counts
    bg_ctr = Counter(r["birth_group"] for r in rows)

    write_csv(OUT_DIR / "birth_feature_importance.csv", importance[:80])
    write_csv(OUT_DIR / "trigger_league.csv", league)
    write_csv(OUT_DIR / "birth_trigger_ranking.csv", trigger_rank)
    write_csv(OUT_DIR / "false_trigger_clusters.csv", false_clusters)
    write_csv(OUT_DIR / "early_signal_timeline.csv", timeline)
    write_csv(OUT_DIR / "meta_simulation.csv", meta)

    meta_sum = meta[-1] if meta else {}
    lines = [
        "############################################################",
        "SCOUT PHASE 27 - ACCELERATION BIRTH DETECTOR (ABD)",
        "############################################################",
        "",
        "Prediction research ONLY. Formulas frozen. No ranking/threshold/weight changes.",
        f"Candidates: {len(rows)} | post-expansion pool: {len(exp_rows)}",
        f"Birth groups: A={bg_ctr.get('A_accel_birth',0)} B={bg_ctr.get('B_flat',0)} "
        f"C={bg_ctr.get('C_compression',0)} D={bg_ctr.get('D_death',0)}",
        "",
        "=" * 62,
        "STEP 1 — ACCELERATION BIRTH DATASET",
        "=" * 62,
        "  A: 15m Expansion + 1h Acceleration (30-60m birth window)",
        "  B: 15m Expansion + 1h Flat",
        "  C: 15m Expansion + 30m Compression",
        "  D: remaining post-expansion + max_up<2% (often absorbed by B/C)",
        "",
        "=" * 62,
        "STEP 2-3 — EARLIEST SIGNALS (IG x TF priority x LOO stability)",
        "=" * 62,
    ]
    for r in importance:
        if r["feature"] in LEAK_1H:
            continue
        if len([x for x in lines if x.startswith("  [")]) >= 10:
            break
        lines.append(
            f"  [{r['timeframe']}] {r['feature']}: IG={r['information_gain']:.3f} "
            f"early={r['early_priority_score']:.3f} LOO={r['loo_stability']:.2f} "
            f"birth_high={r['accel_birth_rate_high']:.2f}"
        )

    lines.extend(["", "=" * 62, "STEP 4 — TRIGGER CANDIDATES (score only)", "=" * 62])
    for tid, comps in triggers.items():
        lines.append(f"  {trigger_description(tid, comps)}")

    lines.extend(["", "=" * 62, "STEP 5 — TRIGGER LEAGUE (post-expansion pool)", "=" * 62])
    for lr in league:
        lines.append(
            f"  {lr['trigger']}: TOP2={lr['top2_hit_pct']:.1f}% TOP5={lr['top5_hit_pct']:.1f}% "
            f"recall={lr['winner_recall_pct']:.1f}% FP_top2={lr['fp_in_top2']} "
            f"early={lr['early_detection_score']:.2f}"
        )
    lines.extend(["", "  Focal Release->Expansion pool:"])
    for lr in league_focal[:3]:
        lines.append(f"    {lr['trigger']}: TOP2={lr['top2_hit_pct']:.1f}% recall={lr['winner_recall_pct']:.1f}%")

    lines.extend(["", "=" * 62, "STEP 6 — FALSE TRIGGER / REJECT SIGNALS", "=" * 62])
    for fc in false_clusters[:8]:
        lines.append(f"  {fc['reject_signal']}: {fc['pct']:.1f}% ({fc['count']}x)")

    lines.extend(multi_track_opinions(best_tid, best["top2_hit_pct"]))

    lines.extend(["", "=" * 62, "STEP 8 — META SIMULATION (A6 picks + trigger re-rank)", "=" * 62])
    if meta_sum:
        lines.append(f"  Scans: {meta_sum.get('scans', 0)}")
        lines.append(f"  A6 base TOP2: {meta_sum.get('base_top2_pct', 0):.1f}%")
        lines.append(f"  Trigger sim TOP2: {meta_sum.get('sim_top2_pct', 0):.1f}% (delta {meta_sum.get('delta_top2', 0):+d} hits)")
        lines.append(f"  FP in TOP2: {meta_sum.get('a6_base_fp_top2', 0)} -> {meta_sum.get('trigger_sim_fp_top2', 0)} "
                     f"(delta {meta_sum.get('delta_fp', 0):+d})")
    lines.append("  Simulation only — NOT applied to live ranking.")

    lines.extend(["", "=" * 62, "HOLDOUT CANDIDATE", "=" * 62])
    hc = [t for t in trigger_rank if t["holdout_candidate"] == "YES"]
    if hc:
        lines.append(f"  {hc[0]['trigger']}: TOP2 {hc[0]['top2_hit_pct']:.1f}% — validate on holdout scans")
    else:
        lines.append(f"  {best_tid}: TOP2 {best['top2_hit_pct']:.1f}% — hypothesis until holdout confirms")

    lines.append("")
    lines.append("DISCLAIMER: Acceleration birth probability — not price prediction.")

    report = OUT_DIR / "phase27_acceleration_birth_report.txt"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for ln in lines:
        safe_print(ln)
    safe_print(f"\nSaved: {report}")


if __name__ == "__main__":
    run()
