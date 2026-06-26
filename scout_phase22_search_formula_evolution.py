"""
Scout Phase 22 - Search Formula Evolution Lab

Compare Pattern B + Phase20 State Ranking (A) vs ranking bonuses A1-A5.
Same 180 scans, LOO validation. Filter unchanged; ranking-only tweaks.

Usage:
  python scout_phase22_search_formula_evolution.py
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import scout_phase20_winner_state_ranking as p20
from season2_p37_scout_decision_hierarchy import write_csv

P19_DIR = Path("logs") / "phase19_winner_dna"
CANDIDATES_PATH = P19_DIR / "candidates.jsonl"
OUT_DIR = Path("logs") / "phase22_formula_evolution"

WINNER_TOP_N = 3
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

FORMULAS = ("A", "A1", "A2", "A3", "A4", "A5")


def safe_print(msg: str) -> None:
    print(msg.encode("ascii", "replace").decode("ascii"))


def g(f: dict, key: str, default: float = 0.0) -> float:
    return float(f.get(key, default))


def within_scan_pct(row: dict, peers: list[dict], key: str) -> float:
    v = g(row["features"], key)
    vals = [g(p["features"], key) for p in peers]
    if not vals:
        return 0.5
    return sum(1 for x in vals if x <= v) / len(vals)


def train_feature_ig(train: list[dict], by_scan_train: dict[str, list[dict]], fn) -> float:
    top2 = set()
    for rows in by_scan_train.values():
        if len(rows) >= 2:
            top2.add((rows[0]["scan_kst"], rows[0]["symbol"]))
            top2.add((rows[1]["scan_kst"], rows[1]["symbol"]))
    pos, neg = [], []
    for scan, rows in by_scan_train.items():
        if len(rows) < 4:
            continue
        for r in rows:
            val = fn(r, rows)
            if (r["scan_kst"], r["symbol"]) in top2:
                pos.append(val)
            else:
                neg.append(val)
    if not pos or not neg:
        return 0.0
    pooled = statistics.pstdev(pos + neg) or 1.0
    return max((statistics.mean(pos) - statistics.mean(neg)) / pooled, 0.0)


def build_train_stats(train: list[dict], by_scan_train: dict[str, list[dict]], th: p20.Thresholds) -> dict:
    w_train, _ = p20.winner_loser_sets(by_scan_train)
    w_feats = [r["features"] for r in w_train]
    comp_vals = [g(f, "5m_compression") for f in w_feats]
    wm = statistics.mean(comp_vals) if comp_vals else 0.0
    ws = statistics.pstdev(comp_vals) or 1.0

    metric_ig: dict[str, float] = {}
    top2 = set()
    for rows in by_scan_train.values():
        if len(rows) >= 2:
            top2.add((rows[0]["scan_kst"], rows[0]["symbol"]))
            top2.add((rows[1]["scan_kst"], rows[1]["symbol"]))
    for m in EXPANSION_METRICS:
        pos = [g(r["features"], m) for r in train if (r["scan_kst"], r["symbol"]) in top2]
        neg = [g(r["features"], m) for r in train if (r["scan_kst"], r["symbol"]) not in top2]
        if pos and neg:
            sd = statistics.pstdev([g(r["features"], m) for r in train]) or 1.0
            metric_ig[m] = max(abs(statistics.mean(pos) - statistics.mean(neg)) / sd, 0.0)
        else:
            metric_ig[m] = 0.0
    ig_sum = sum(metric_ig.values()) or 1.0
    metric_w = {k: v / ig_sum for k, v in metric_ig.items()}

    return {
        "winner_comp_mean": wm,
        "winner_comp_std": ws,
        "vol_support_p50": p20.percentile([g(f, "15m_current_volume_ratio") for f in w_feats], 0.5) if w_feats else 1.0,
        "expansion_metric_w": metric_w,
        "ig_a1": train_feature_ig(train, by_scan_train, lambda r, p: within_scan_pct(r, p, "2h_current_ma20_distance_pct")),
        "ig_a2": train_feature_ig(train, by_scan_train, lambda r, p: within_scan_pct(r, p, "1h_current_range_pct")),
        "ig_a3": train_feature_ig(train, by_scan_train, lambda r, p: bonus_a3_raw(r, p, th, {"vol_support_p50": p20.percentile([g(x["features"], "15m_current_volume_ratio") for x in w_train], 0.5) if w_train else 1.0})),
        "ig_a4": train_feature_ig(train, by_scan_train, lambda r, p: bonus_a4_raw(r, {"winner_comp_mean": wm, "winner_comp_std": ws})),
        "ig_a5": train_feature_ig(train, by_scan_train, lambda r, p: bonus_a5_raw(r, p, {"expansion_metric_w": metric_w})),
    }


def bonus_a3_raw(row: dict, peers: list[dict], th: p20.Thresholds, stats: dict) -> float:
    f = row["features"]
    release = g(f, "5m_release") > 0 or row.get("states", {}).get("5m") == "Release"
    vol_pct = within_scan_pct(row, peers, "15m_current_volume_ratio")
    vol_state = row.get("states", {}).get("15m") in ("VolumeSupport", "Expansion")
    vol_ok = vol_state or g(f, "15m_current_volume_ratio") >= stats.get("vol_support_p50", 1.0)
    if release and vol_ok:
        return 0.5 + 0.5 * vol_pct
    return 0.25 * vol_pct if release or vol_ok else 0.0


def bonus_a4_raw(row: dict, stats: dict) -> float:
    v = g(row["features"], "5m_compression")
    wm, ws = stats["winner_comp_mean"], stats["winner_comp_std"]
    z = abs(v - wm) / ws
    return math.exp(-0.5 * z * z)


def bonus_a5_raw(row: dict, peers: list[dict], stats: dict) -> float:
    f = row["features"]
    mw = stats["expansion_metric_w"]
    score = 0.0
    for m, w in mw.items():
        score += w * within_scan_pct(row, peers, m)
    return score


def formula_scores(row: dict, peers: list[dict], base: float, th: p20.Thresholds, stats: dict) -> dict[str, float]:
    b1 = within_scan_pct(row, peers, "2h_current_ma20_distance_pct")
    b2 = within_scan_pct(row, peers, "1h_current_range_pct")
    b3 = bonus_a3_raw(row, peers, th, stats)
    b4 = bonus_a4_raw(row, stats)
    b5 = bonus_a5_raw(row, peers, stats)
    return {
        "A": base,
        "A1": base + stats["ig_a1"] * b1,
        "A2": base + stats["ig_a2"] * b2,
        "A3": base + stats["ig_a3"] * b3,
        "A4": base + stats["ig_a4"] * b4,
        "A5": base + stats["ig_a5"] * b5,
    }


def eval_scan(rows: list[dict], profile: dict, th: p20.Thresholds, stats: dict) -> dict:
    for r in rows:
        base = p20.state_match_score(r["states"], r["transitions"], profile)
        r["scores"] = formula_scores(r, rows, base, th, stats)

    by_outcome = sorted(rows, key=lambda x: x["outcome_rank"])
    actual_top2 = {x["symbol"] for x in by_outcome[:2]}
    actual_top5 = {x["symbol"] for x in by_outcome[:5]}
    out: dict = {}
    for fid in FORMULAS:
        ranked = sorted(rows, key=lambda x: x["scores"][fid], reverse=True)
        pick2 = {x["symbol"] for x in ranked[:2]}
        pick5 = {x["symbol"] for x in ranked[:5]}
        out[fid] = {
            "top2_hit": len(pick2 & actual_top2),
            "top5_hit": len(pick5 & actual_top5),
            "avg_max_up_top2": statistics.mean([x["max_up_4h"] for x in ranked[:2]]) if ranked else 0,
            "rank1_actual": ranked[0]["outcome_rank"] if ranked else None,
            "pick1": ranked[0]["symbol"] if ranked else "",
            "pick2": ranked[1]["symbol"] if len(ranked) > 1 else "",
            "actual1": by_outcome[0]["symbol"] if by_outcome else "",
            "actual2": by_outcome[1]["symbol"] if len(by_outcome) > 1 else "",
            "loser": by_outcome[-1]["symbol"] if by_outcome else "",
        }
    return out


def run() -> list[str]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = p20.load_candidates()
    by_scan: dict[str, list[dict]] = defaultdict(list)
    for r in raw:
        by_scan[r["scan_kst"]].append(r)
    for s in by_scan:
        by_scan[s].sort(key=lambda x: x["outcome_rank"])

    winner_feats = [r["features"] for rows in by_scan.values() for r in rows[:WINNER_TOP_N] if len(rows) >= 4]
    th = p20.build_thresholds(winner_feats)
    annotated = p20.annotate(raw, th)
    ann_by_scan: dict[str, list[dict]] = defaultdict(list)
    for r in annotated:
        ann_by_scan[r["scan_kst"]].append(r)
    for s in ann_by_scan:
        ann_by_scan[s].sort(key=lambda x: x["outcome_rank"])

    scan_keys = [s for s, rows in ann_by_scan.items() if len(rows) >= 4]
    bt_rows: list[dict] = []
    example_scan: str | None = None
    example_data: dict | None = None
    best_delta = -1.0

    for scan in scan_keys:
        train = [r for r in annotated if r["scan_kst"] != scan]
        train_by: dict[str, list[dict]] = defaultdict(list)
        for r in train:
            train_by[r["scan_kst"]].append(r)
        for s in train_by:
            train_by[s].sort(key=lambda x: x["outcome_rank"])
        w_train, _ = p20.winner_loser_sets(train_by)
        profile = p20.build_profile(w_train, train) if w_train else p20.build_profile([], train)
        stats = build_train_stats(train, train_by, th)
        res = eval_scan(ann_by_scan[scan], profile, th, stats)
        flat = {"scan_kst": scan}
        for fid in FORMULAS:
            for k, v in res[fid].items():
                flat[f"{fid}_{k}"] = v
        bt_rows.append(flat)
        delta = res["A1"]["top2_hit"] - res["A"]["top2_hit"]
        for fid in ("A1", "A2", "A3", "A4", "A5"):
            delta = max(delta, res[fid]["top2_hit"] - res["A"]["top2_hit"])
        if delta > best_delta:
            best_delta = delta
            example_scan = scan
            example_data = res

    def agg(fid: str, key: str) -> float:
        return statistics.mean([b[f"{fid}_{key}"] for b in bt_rows])

    def med(fid: str, key: str) -> float:
        vals = [b[f"{fid}_{key}"] for b in bt_rows if b.get(f"{fid}_{key}") is not None]
        return statistics.median(vals) if vals else 0

    summary: list[dict] = []
    for fid in FORMULAS:
        summary.append({
            "formula": fid,
            "top2_hit_pct": round(agg(fid, "top2_hit") / 2 * 100, 2),
            "top5_hit_pct": round(agg(fid, "top5_hit") / 5 * 100, 2),
            "avg_max_up_top2": round(agg(fid, "avg_max_up_top2"), 4),
            "rank1_median": round(med(fid, "rank1_actual"), 2),
        })

    base_t2 = summary[0]["top2_hit_pct"]
    deltas: list[tuple[str, float]] = []
    for row in summary[1:]:
        deltas.append((row["formula"], row["top2_hit_pct"] - base_t2))

    best_f, best_d = max(deltas, key=lambda x: x[1])
    worst_f, worst_d = min(deltas, key=lambda x: x[1])

    if best_d >= 5.0:
        verdict = "KEEP"
        verdict_note = f"{best_f} improves TOP2 by {best_d:+.1f}pp vs A; adopt as new ranking base."
    elif best_d >= 1.0:
        verdict = "MERGE"
        verdict_note = f"{best_f} improves TOP2 by {best_d:+.1f}pp vs A; merge bonus into Phase20 ranking."
    else:
        verdict = "DISCARD"
        verdict_note = "No formula bonus beats Base A meaningfully on LOO TOP2; keep Phase20 State Ranking only."

    lines = [
        "############################################################",
        "SCOUT PHASE 22 - SEARCH FORMULA EVOLUTION LAB",
        "############################################################",
        "",
        f"Scans LOO: {len(bt_rows)} | Candidates: {len(annotated)} | Filter: Pattern B frozen",
        "Base ranking: Phase20 State Match Score",
        "",
        "=" * 62,
        "FORMULA COMPARISON",
        "=" * 62,
        f"{'Formula':<8} {'TOP2%':>8} {'TOP5%':>8} {'AvgMaxUp':>10} {'Rank1Med':>10}",
        "-" * 62,
    ]
    for row in summary:
        lines.append(
            f"{row['formula']:<8} {row['top2_hit_pct']:>7.1f}% {row['top5_hit_pct']:>7.1f}% "
            f"{row['avg_max_up_top2']:>9.2f}% {row['rank1_median']:>10.1f}"
        )

    lines.extend(["", "=" * 62, "DELTA vs BASE (A)", "=" * 62])
    for fid, d in sorted(deltas, key=lambda x: x[1], reverse=True):
        tag = "UP" if d > 0 else "DOWN" if d < 0 else "FLAT"
        lines.append(f"  {fid}: {d:+.1f}pp ({tag})")

    lines.extend(["", "=" * 62, "REPRESENTATIVE EXAMPLE (Winner / Runner-up / Loser)", "=" * 62])
    if example_scan and example_data:
        lines.append(f"  Scan: {example_scan}")
        a = example_data["A"]
        lines.append(f"  Actual #1: {a['actual1']} | #2: {a['actual2']} | Bottom: {a['loser']}")
        lines.append(f"  Base A picks: {a['pick1']}, {a['pick2']} (top2_hit={a['top2_hit']})")
        for fid in ("A1", "A2", "A3", "A4", "A5"):
            e = example_data[fid]
            lines.append(f"  {fid} picks: {e['pick1']}, {e['pick2']} (top2_hit={e['top2_hit']})")

    lines.extend(["", "=" * 62, "FORMULA DEFINITIONS (ranking bonus only)", "=" * 62])
    lines.append("  A  = Phase20 State Match Score")
    lines.append("  A1 = A + IG * within_scan_pct(2h_current_ma20_distance_pct)")
    lines.append("  A2 = A + IG * within_scan_pct(1h_current_range_pct)")
    lines.append("  A3 = A + IG * (5m_release AND 15m_volume_support co-occurrence)")
    lines.append("  A4 = A + IG * exp(-0.5*((5m_compression-winner_mean)/winner_std)^2)")
    lines.append("  A5 = A + IG * relative_expansion (IG-weighted within-scan percentiles)")
    lines.append("  IG = training-fold top2 separation (effect size), no hand weights.")

    lines.extend(["", "=" * 62, "VERDICT", "=" * 62])
    lines.append(f"  {verdict}")
    lines.append(f"  {verdict_note}")

    write_csv(OUT_DIR / "formula_comparison.csv", summary)
    write_csv(OUT_DIR / "formula_delta_vs_base.csv", [
        {"formula": f, "top2_delta_pp": round(d, 2)} for f, d in deltas
    ])
    write_csv(OUT_DIR / "loo_by_scan.csv", bt_rows)

    report = OUT_DIR / "phase22_formula_evolution_report.txt"
    report.write_text("\n".join(lines), encoding="utf-8")
    return lines


def main() -> None:
    lines = run()
    for ln in lines:
        safe_print(ln)
    safe_print(f"\nSaved: {OUT_DIR / 'phase22_formula_evolution_report.txt'}")


if __name__ == "__main__":
    main()
