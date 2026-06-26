"""
Scout Phase 24 - Loser Mining & False Positive Lab

Analysis only. Formulas A/A2/A5/A6 frozen. No rule/threshold/weight changes.

Input:
  logs/phase23_formula_league/match_log.jsonl
  logs/phase19_winner_dna/candidates.jsonl

Usage:
  python scout_phase24_loser_mining.py
"""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import scout_phase20_winner_state_ranking as p20
from season2_p37_scout_decision_hierarchy import write_csv

P23_MATCH = Path("logs") / "phase23_formula_league" / "match_log.jsonl"
P19_CAND = Path("logs") / "phase19_winner_dna" / "candidates.jsonl"
OUT_DIR = Path("logs") / "phase24_loser_mining"

FORMULAS = ("A", "A2", "A5", "A6")
FP_THRESHOLD = 2.0  # max_up_4h < 2%


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


def load_index() -> dict[tuple[str, str], dict]:
    idx: dict[tuple[str, str], dict] = {}
    for line in P19_CAND.open(encoding="utf-8"):
        r = json.loads(line)
        idx[(r["scan_kst"], r["symbol"])] = r
    return idx


def combo_key(states: dict[str, str]) -> str:
    return "|".join(f"{k}={states[k]}" for k in ("5m", "15m", "30m", "1h", "2h"))


def build_datasets() -> tuple[list[dict], list[dict], list[dict], dict]:
    if not P23_MATCH.exists():
        raise SystemExit(f"Missing: {P23_MATCH}")
    if not P19_CAND.exists():
        raise SystemExit(f"Missing: {P19_CAND}")

    idx = load_index()
    raw = p20.load_candidates()
    by_scan: dict[str, list[dict]] = defaultdict(list)
    for r in raw:
        by_scan[r["scan_kst"]].append(r)
    winner_feats = [r["features"] for rows in by_scan.values() for r in rows[:3] if len(rows) >= 4]
    th = p20.build_thresholds(winner_feats)
    state_cache: dict[tuple[str, str], dict] = {}

    def enrich(scan: str, sym: str, base: dict) -> dict | None:
        key = (scan, sym)
        cand = idx.get(key)
        if not cand:
            return None
        if key not in state_cache:
            ann = p20.annotate([cand], th)[0]
            state_cache[key] = ann["states"]
        states = state_cache[key]
        return {
            **base,
            "scan_kst": scan,
            "symbol": sym,
            "max_up_4h": cand["max_up_4h"],
            "outcome_rank": cand["outcome_rank"],
            "features": cand["features"],
            "states": states,
            "combo": combo_key(states),
        }

    top2_miss: list[dict] = []
    false_positive: list[dict] = []
    winner_hit: list[dict] = []
    seen_miss: set[tuple[str, str, str]] = set()
    seen_fp: set[tuple[str, str, str]] = set()
    seen_win: set[tuple[str, str, str]] = set()

    match_rows = [json.loads(ln) for ln in P23_MATCH.open(encoding="utf-8")]
    stats = {"scans": len(match_rows), "formulas": FORMULAS}

    for m in match_rows:
        scan = m["scan_kst"]
        actual = m["actual_top2"]
        actual_set = set(actual)
        all_picks: set[str] = set()
        for fid in FORMULAS:
            picks = m.get(f"{fid}_top2", [])
            all_picks.update(picks)
            for sym in actual:
                if sym not in picks:
                    k = (scan, sym, fid)
                    if k not in seen_miss:
                        seen_miss.add(k)
                        row = enrich(scan, sym, {"cohort": "top2_miss", "formula": fid, "miss_type": "not_picked"})
                        if row:
                            top2_miss.append(row)
            for sym in picks:
                k = (scan, sym, fid)
                cand = idx.get((scan, sym))
                if not cand:
                    continue
                mu = cand["max_up_4h"]
                if sym in actual_set:
                    if k not in seen_win:
                        seen_win.add(k)
                        row = enrich(scan, sym, {"cohort": "winner_hit", "formula": fid, "hit": True})
                        if row:
                            winner_hit.append(row)
                if mu < FP_THRESHOLD:
                    if k not in seen_fp:
                        seen_fp.add(k)
                        row = enrich(scan, sym, {
                            "cohort": "false_positive",
                            "formula": fid,
                            "fp_reason": f"max_up_4h<{FP_THRESHOLD}",
                        })
                        if row:
                            false_positive.append(row)

        for sym in actual:
            if sym not in all_picks:
                k = (scan, sym, "ANY")
                if k not in seen_miss:
                    seen_miss.add(k)
                    row = enrich(scan, sym, {"cohort": "top2_miss", "formula": "ANY", "miss_type": "all_formulas_miss"})
                    if row:
                        top2_miss.append(row)

    stats["top2_miss"] = len(top2_miss)
    stats["false_positive"] = len(false_positive)
    stats["winner_hit"] = len(winner_hit)
    return top2_miss, false_positive, winner_hit, stats


def feature_keys(rows: list[dict]) -> list[str]:
    keys: set[str] = set()
    for r in rows:
        keys.update(k for k in r["features"] if k != "price")
    return sorted(keys)


def median_split(vals: list[float]) -> float:
    return statistics.median(vals) if vals else 0.0


def three_way_features(
    winners_gt: list[dict],
    fp: list[dict],
    miss: list[dict],
) -> list[dict]:
    """Compare actual TOP2 (ground truth) vs FP vs miss using feature high/low bins."""
    # Ground-truth winners = unique actual top2 symbols from winner_hit + miss outcome_rank<=2
    gt_winners: dict[tuple[str, str], dict] = {}
    for r in winner_hit_rows(winners_gt, miss):
        gt_winners[(r["scan_kst"], r["symbol"])] = r

    w_rows = list(gt_winners.values())
    if not w_rows:
        w_rows = winners_gt

    keys = feature_keys(w_rows + fp + miss)
    out: list[dict] = []

    for key in keys:
        all_vals = [g(r["features"], key) for r in w_rows + fp + miss]
        cut = median_split(all_vals)

        def rate_high(rows: list[dict]) -> float:
            if not rows:
                return 0.0
            return sum(1 for r in rows if g(r["features"], key) >= cut) / len(rows)

        w_r = rate_high(w_rows)
        f_r = rate_high(fp)
        m_r = rate_high(miss)
        base = sum(1 for v in all_vals if v >= cut) / len(all_vals) if all_vals else 0

        w_high = sum(1 for r in w_rows if g(r["features"], key) >= cut)
        f_high = sum(1 for r in fp if g(r["features"], key) >= cut)
        m_high = sum(1 for r in miss if g(r["features"], key) >= cut)

        lift_w = w_r / base if base > 0 else 0
        lift_f = f_r / base if base > 0 else 0
        lift_m = m_r / base if base > 0 else 0

        ow, of, om = w_r / (1 - w_r) if w_r < 1 else 99, f_r / (1 - f_r) if f_r < 1 else 99, m_r / (1 - m_r) if m_r < 1 else 99
        odds_f = of / ow if ow > 0 else 0
        odds_m = om / ow if ow > 0 else 0

        ig_wf = ig_binary(w_high, len(w_rows), f_high, len(fp))
        ig_wm = ig_binary(w_high, len(w_rows), m_high, len(miss))

        if lift_f > 1.15 and lift_f > lift_w and f_r > w_r:
            tag = "reject"
        elif lift_m > 1.15 and lift_m > lift_w and m_r > w_r:
            tag = "reject"
        elif lift_w > 1.15 and w_r > f_r and w_r > m_r:
            tag = "promote"
        else:
            tag = "neutral"

        out.append({
            "feature": key,
            "winner_freq_high": round(w_r, 4),
            "false_positive_freq_high": round(f_r, 4),
            "loser_miss_freq_high": round(m_r, 4),
            "lift_winner": round(lift_w, 4),
            "lift_fp": round(lift_f, 4),
            "lift_miss": round(lift_m, 4),
            "odds_ratio_fp_vs_winner": round(odds_f, 4),
            "odds_ratio_miss_vs_winner": round(odds_m, 4),
            "ig_winner_vs_fp": round(ig_wf, 4),
            "ig_winner_vs_miss": round(ig_wm, 4),
            "classification": tag,
            "split_median": round(cut, 4),
        })

    return out


def winner_hit_rows(winner_hit: list[dict], miss: list[dict]) -> list[dict]:
    """Ground truth TOP2 symbols (outcome rank 1-2)."""
    gt: dict[tuple[str, str], dict] = {}
    for r in winner_hit:
        gt[(r["scan_kst"], r["symbol"])] = r
    for r in miss:
        if r["outcome_rank"] <= 2:
            gt[(r["scan_kst"], r["symbol"])] = r
    return list(gt.values())


def cluster_table(rows: list[dict], label: str) -> list[dict]:
    ctr = Counter(r["combo"] for r in rows)
    total = len(rows) or 1
    return [
        {"cluster": k, "count": v, "pct": round(v / total * 100, 2), "label": label}
        for k, v in ctr.most_common(20)
    ]


def run() -> list[str]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    lines_intro = [
        "DATA SOURCES (verified):",
        f"  match_log: {P23_MATCH} | keys: scan_kst, actual_top2, {{A,A2,A5,A6}}_top2",
        f"  candidates: {P19_CAND} | keys: scan_kst, symbol, max_up_4h, features, outcome_rank",
        f"  formulas frozen: {', '.join(FORMULAS)}",
    ]

    top2_miss, false_positive, winner_hit, stats = build_datasets()
    gt_winners = winner_hit_rows(winner_hit, top2_miss)

    feat_rows = three_way_features(gt_winners, false_positive, top2_miss)
    promote = sorted([r for r in feat_rows if r["classification"] == "promote"],
                     key=lambda x: (x["lift_winner"], x["ig_winner_vs_fp"]), reverse=True)
    reject = sorted([r for r in feat_rows if r["classification"] == "reject"],
                    key=lambda x: (x["lift_fp"] + x["lift_miss"], x["ig_winner_vs_fp"]), reverse=True)

    win_clusters = cluster_table(gt_winners, "winner")
    fp_clusters = cluster_table(false_positive, "false_positive")
    miss_clusters = cluster_table([r for r in top2_miss if r.get("formula") == "ANY"], "top2_miss_any")

    # save datasets
    def dump_jsonl(path: Path, rows: list[dict]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                slim = {k: v for k, v in r.items() if k != "features"}
                slim["features"] = {k: v for k, v in r["features"].items()}
                f.write(json.dumps(slim, ensure_ascii=True) + "\n")

    dump_jsonl(OUT_DIR / "top2_miss_dataset.jsonl", top2_miss)
    dump_jsonl(OUT_DIR / "false_positive_dataset.jsonl", false_positive)
    dump_jsonl(OUT_DIR / "winner_dataset.jsonl", winner_hit)

    write_csv(OUT_DIR / "winner_vs_falsepositive.csv", feat_rows)
    write_csv(OUT_DIR / "reject_feature_ranking.csv", reject)
    write_csv(OUT_DIR / "failure_clusters.csv", fp_clusters)
    write_csv(OUT_DIR / "false_positive_clusters.csv", fp_clusters)
    write_csv(OUT_DIR / "winner_clusters.csv", win_clusters)
    write_csv(OUT_DIR / "top2_miss_clusters.csv", miss_clusters)
    write_csv(OUT_DIR / "promote_feature_ranking.csv", promote)

    # common miss state
    miss_any = [r for r in top2_miss if r.get("formula") == "ANY"]
    miss_state_ctr: Counter = Counter()
    for r in miss_any:
        for tf, s in r["states"].items():
            miss_state_ctr[f"{tf}:{s}"] += 1
    top_miss_states = miss_state_ctr.most_common(8)

    lines = [
        "############################################################",
        "SCOUT PHASE 24 - LOSER MINING & FALSE POSITIVE LAB",
        "############################################################",
        "",
        *lines_intro,
        "",
        f"Scans: {stats['scans']} | top2_miss rows: {stats['top2_miss']} | "
        f"false_positive rows: {stats['false_positive']} | winner_hit rows: {stats['winner_hit']}",
        f"Ground-truth TOP2 symbols: {len(gt_winners)}",
        "Analysis only. No formula/threshold/weight changes.",
        "",
        "=" * 62,
        "1. TOP10 REJECT FEATURES (failure probability increase)",
        "=" * 62,
    ]
    for r in reject[:10]:
        lines.append(
            f"  {r['feature']}: FP_high={r['false_positive_freq_high']:.2f} "
            f"miss_high={r['loser_miss_freq_high']:.2f} vs winner={r['winner_freq_high']:.2f} "
            f"lift_fp={r['lift_fp']:.2f} IG={r['ig_winner_vs_fp']:.3f}"
        )

    lines.extend(["", "=" * 62, "2. TOP10 PROMOTE FEATURES (success probability increase)", "=" * 62])
    for r in promote[:10]:
        lines.append(
            f"  {r['feature']}: winner_high={r['winner_freq_high']:.2f} "
            f"lift_w={r['lift_winner']:.2f} IG_w_fp={r['ig_winner_vs_fp']:.3f}"
        )

    lines.extend(["", "=" * 62, "3. WINNER CLUSTER (representative)", "=" * 62])
    for c in win_clusters[:5]:
        lines.append(f"  [{c['count']}x / {c['pct']}%] {c['cluster'].replace('|', ' | ')}")

    lines.extend(["", "=" * 62, "4. FALSE POSITIVE CLUSTER (representative)", "=" * 62])
    for c in fp_clusters[:5]:
        lines.append(f"  [{c['count']}x / {c['pct']}%] {c['cluster'].replace('|', ' | ')}")

    lines.extend(["", "=" * 62, "5. TOP2 MISS CLUSTER (all formulas miss)", "=" * 62])
    for c in miss_clusters[:5]:
        lines.append(f"  [{c['count']}x / {c['pct']}%] {c['cluster'].replace('|', ' | ')}")

    lines.extend(["", "=" * 62, "6. COMMON STATE FORMULAS MISS", "=" * 62])
    for st, cnt in top_miss_states:
        lines.append(f"  {st}: {cnt}x in ANY-formula misses")

    lines.extend(["", "=" * 62, "7. NEXT PHASE VALIDATION CANDIDATES (proposal only)", "=" * 62])
    lines.append("  Not rules - hypotheses for holdout validation:")
    if reject:
        lines.append(f"  - Monitor reject signal: {reject[0]['feature']} elevation in picks")
    if promote:
        lines.append(f"  - Monitor promote signal: {promote[0]['feature']} in missed actual TOP2")
    if fp_clusters:
        lines.append(f"  - FP cluster pattern: {fp_clusters[0]['cluster'][:80]}...")
    if miss_clusters:
        lines.append(f"  - Miss cluster pattern: {miss_clusters[0]['cluster'][:80]}...")
    lines.append("  - Test whether demoting FP-cluster picks improves TOP2 without filter change.")

    lines.extend(["", "DISCLAIMER: Descriptive failure mining. No search formula modification in Phase24."])

    report = OUT_DIR / "phase24_loser_mining_report.txt"
    report.write_text("\n".join(lines), encoding="utf-8")
    return lines


def main() -> None:
    lines = run()
    for ln in lines:
        safe_print(ln)
    safe_print(f"\nSaved: {OUT_DIR / 'phase24_loser_mining_report.txt'}")


if __name__ == "__main__":
    main()
