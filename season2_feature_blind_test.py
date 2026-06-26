"""
Scout Season2 - Single Feature Blind Test

Tests ONE price/volume feature at a time against Random baseline.
STRICT NO_LOOKAHEAD | NO_API | NO_TRADING | RESEARCH ONLY.

Usage:
  python season2_feature_blind_test.py --feature return_prev_2h
  python season2_feature_blind_test.py --feature volume_acceleration_ratio
  python season2_feature_blind_test.py --list
  python season2_feature_blind_test.py --update-scoreboard
"""

from __future__ import annotations

import argparse
import random
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path

from season2_blind_loop_001 import (
    END_KST,
    RANDOM_SEED,
    START_KST,
    build_process_record,
    fmt_kst,
    gen_scan_times,
    load_p37_proxy,
    load_physics_index,
    load_top10_index,
    mean_return,
    random_top2,
    resolve_future_extremes,
    resolve_future_return_2h,
    top_k,
)
from season2_p37_scout_decision_hierarchy import load_csv, pf, write_csv

LOGS_DIR = Path("logs")
OUT_DIR = LOGS_DIR / "feature_research"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SCOREBOARD_CSV = LOGS_DIR / "feature_scoreboard.csv"

KST = timezone(timedelta(hours=9))

# Price/volume features only. Process/philosophy features are excluded by policy.
FEATURE_REGISTRY: dict[str, dict] = {
    "return_prev_2h": {
        "name": "return_prev_2h",
        "definition": "2-hour return immediately before scan (close-to-close on 2h candles)",
        "formula": "(close_T0 - close_T0-2h) / close_T0-2h * 100",
        "hypothesis": "Short-term momentum persists for 2h",
        "column": "return_prev_2h_percent",
        "score_from": "column",
        "category": "price",
    },
    "return_prev_4h": {
        "name": "return_prev_4h",
        "definition": "4-hour return before scan",
        "formula": "(close_T0 - close_T0-4h) / close_T0-4h * 100",
        "hypothesis": "Medium momentum predicts next 2h",
        "column": "return_prev_4h_percent",
        "score_from": "column",
        "category": "price",
    },
    "volume_acceleration_ratio": {
        "name": "volume_acceleration_ratio",
        "definition": "Recent volume surge vs prior window",
        "formula": "volume_current / volume_ma6 (normalized in dataset)",
        "hypothesis": "Volume acceleration precedes 2h continuation",
        "column": "volume_acceleration_ratio",
        "score_from": "column",
        "category": "volume",
    },
    "volume_ratio_ma24": {
        "name": "volume_ratio_ma24",
        "definition": "Current volume relative to 24-period average",
        "formula": "volume_current / volume_ma24",
        "hypothesis": "Relative volume flags attention shift",
        "column": "volume_ratio_ma24",
        "score_from": "column",
        "category": "volume",
    },
    "atr_ratio": {
        "name": "atr_ratio",
        "definition": "ATR expansion vs recent baseline",
        "formula": "atr_current / atr_baseline (dataset atr_ratio)",
        "hypothesis": "Volatility expansion precedes continuation",
        "column": "atr_ratio",
        "score_from": "column",
        "category": "price",
    },
    "range_expansion_ratio": {
        "name": "range_expansion_ratio",
        "definition": "Current candle range vs recent average",
        "formula": "range_current / range_avg",
        "hypothesis": "Range expansion signals breakout continuation",
        "column": "range_expansion_ratio",
        "score_from": "column",
        "category": "price",
    },
    "return_24h_percent": {
        "name": "return_24h_percent",
        "definition": "24h gainer rank driver — return over prior 24h",
        "formula": "(close_T0 - close_T0-24h) / close_T0-24h * 100",
        "hypothesis": "Top gainers continue for 2h",
        "column": "return_24h_percent",
        "score_from": "column",
        "category": "price",
    },
    "body_expansion_ratio": {
        "name": "body_expansion_ratio",
        "definition": "Current body size vs 24-candle average body",
        "formula": "body_current / body_avg_24",
        "hypothesis": "Body expansion marks impulse candles",
        "column": "body_expansion_ratio",
        "score_from": "column",
        "category": "price",
    },
}

REJECTED_PROCESS_FEATURES = [
    "ScoutScore_v1",
    "ScoutScore_dynamic",
    "BeliefConsensus",
    "SynchronizationScore",
    "NarrativeScore",
    "AttentionScore",
    "DynamicEnergy",
]


def load_canonical_random_by_scan() -> dict[str, float]:
    """Per-scan Random Top2 returns from Blind Loop 001 (canonical baseline)."""
    path = LOGS_DIR / "blind_loop_001" / "scan_metrics_by_time.csv"
    if not path.exists():
        return {}
    return {
        r["scan_time_kst"]: pf(r["random_top2_mean_return"]) or 0.0
        for r in load_csv(path)
    }


def feature_score(row: dict, spec: dict) -> float:
    val = pf(row.get(spec["column"]))
    if val is None:
        return float("-inf")
    return val


def build_scan_candidates(
    t0: str,
    universe: dict[str, dict],
    physics: dict[tuple[str, str], dict],
) -> list[dict]:
    """Fixed universe per scan — same symbols for every feature test."""
    candidates: list[dict] = []
    for sym, row in universe.items():
        phys = physics.get((t0, sym))
        fut2, fut_source = resolve_future_return_2h(row, phys)
        if fut_source == "missing_or_zero":
            continue
        candidates.append({
            "scan_time_kst": t0,
            "symbol": sym,
            "future_return_2h": fut2,
            "future_return_source": fut_source,
            "row": row,
        })
    return candidates


def run_single_feature_test(feature_key: str) -> dict:
    if feature_key not in FEATURE_REGISTRY:
        raise SystemExit(f"Unknown feature: {feature_key}. Use --list")

    spec = FEATURE_REGISTRY[feature_key]
    top10 = load_top10_index()
    physics = load_physics_index()
    canonical_random = load_canonical_random_by_scan()

    scan_rows: list[dict] = []
    top2_returns: list[float] = []
    random_returns: list[float] = []
    wins_vs_random = 0
    valid_scans = 0

    for t0 in gen_scan_times():
        universe = top10.get(t0)
        if not universe:
            continue

        candidates = build_scan_candidates(t0, universe, physics)
        if len(candidates) < 3:
            continue

        if t0 in canonical_random:
            rand_ret = canonical_random[t0]
        else:
            rng = random.Random(RANDOM_SEED + valid_scans)
            rand_ret = mean_return(random_top2(candidates, rng))

        valid_scans += 1
        random_returns.append(rand_ret)

        scored = []
        for item in candidates:
            score = feature_score(item["row"], spec)
            scored.append({**item, "feature_score": score})

        ranked = sorted(
            scored,
            key=lambda r: (r["feature_score"] == float("-inf"), r["feature_score"]),
            reverse=True,
        )
        selected = ranked[:2]
        feat_ret = mean_return(selected)
        top2_returns.append(feat_ret)
        if feat_ret > rand_ret:
            wins_vs_random += 1

        for rank, item in enumerate(selected, 1):
            scan_rows.append({
                "feature": feature_key,
                "scan_time_kst": t0,
                "rank": rank,
                "symbol": item["symbol"],
                "feature_score": round(item["feature_score"], 4) if item["feature_score"] != float("-inf") else "",
                "future_return_2h_pct": round(item["future_return_2h"], 4),
                "future_return_source": item["future_return_source"],
                "random_top2_mean_return_pct": round(rand_ret, 4),
                "feature_top2_mean_return_pct": round(feat_ret, 4),
                "beat_random_this_scan": "yes" if feat_ret > rand_ret else "no",
                "learning_recommendation": "NO_ACTION",
            })

    avg_feat = statistics.mean(top2_returns) if top2_returns else 0.0
    avg_rand = statistics.mean(random_returns) if random_returns else 0.0
    win_rate = wins_vs_random / valid_scans if valid_scans else 0.0
    passed = avg_feat > avg_rand and win_rate >= 0.50
    # Adoption requires repeated OOS confirmation — never auto-adopt from one blind pass.
    adopted = "no"

    result = {
        "feature": feature_key,
        "category": spec["category"],
        "definition": spec["definition"],
        "formula": spec["formula"],
        "hypothesis": spec["hypothesis"],
        "valid_scans": valid_scans,
        "avg_top2_return_pct": round(avg_feat, 4),
        "random_avg_top2_return_pct": round(avg_rand, 4),
        "vs_random_pct": round(avg_feat - avg_rand, 4),
        "win_rate_vs_random": round(win_rate, 4),
        "passed": "yes" if passed else "no",
        "adopted": adopted,
        "reject_reason": "" if passed else "avg_return_or_win_rate_below_random",
    }

    out_detail = OUT_DIR / f"blind_test_{feature_key}.csv"
    write_csv(out_detail, scan_rows)

    summary_path = OUT_DIR / f"blind_test_{feature_key}_summary.txt"
    summary_path.write_text(
        "\n".join([
            f"Feature: {feature_key}",
            f"Definition: {spec['definition']}",
            f"Formula: {spec['formula']}",
            f"Hypothesis: {spec['hypothesis']}",
            "",
            f"Valid scans: {valid_scans}",
            f"Avg Top2 return: {avg_feat:.4f}%",
            f"Random Avg Top2 return: {avg_rand:.4f}%",
            f"Vs Random: {avg_feat - avg_rand:+.4f}%",
            f"Win rate vs Random: {win_rate:.1%}",
            f"Passed: {'YES' if passed else 'NO'}",
            "",
            "Failure analysis (if failed):",
            _failure_analysis(feature_key, avg_feat, avg_rand, win_rate),
            "",
            "Learning recommendation: NO_ACTION",
        ]),
        encoding="utf-8",
    )
    return result


def _failure_analysis(key: str, avg: float, rand: float, win_rate: float) -> str:
    if avg > rand and win_rate >= 0.5:
        return "None — candidate passed blind test."
    lines = [f"- Random beat feature by {rand - avg:.4f}% on average"]
    if win_rate < 0.5:
        lines.append(f"- Won only {win_rate:.1%} of scans vs Random")
    if key in ("return_prev_2h", "return_prev_4h", "return_24h_percent"):
        lines.append("- Likely cause: momentum exhaustion in top-gainer universe")
    if "volume" in key:
        lines.append("- Likely cause: volume spike without price follow-through (fakeout)")
    if key in ("atr_ratio", "range_expansion_ratio"):
        lines.append("- Likely cause: volatility expansion = reversal not continuation")
    lines.append("- Noise: small universe (~10 symbols) amplifies variance")
    return "\n".join(lines)


def update_scoreboard(run_all: bool = False) -> list[dict]:
    rows: list[dict] = []

    # Falsified process features from Blind Loop 001 (not re-tested)
    rows.append({
        "feature": "ScoutScore_v1",
        "category": "process_REJECTED",
        "avg_top2_return_pct": 0.5986,
        "random_avg_top2_return_pct": 0.9214,
        "vs_random_pct": -0.3228,
        "win_rate_vs_random": "",
        "valid_scans": 57,
        "passed": "no",
        "adopted": "no",
        "reject_reason": "Blind Loop 001 falsified — process/philosophy",
        "test_date": "2026-06-09",
        "learning_recommendation": "NO_ACTION",
    })
    for name in ("BeliefConsensus", "SynchronizationScore", "NarrativeScore", "DynamicEnergy"):
        rows.append({
            "feature": name,
            "category": "process_REJECTED",
            "avg_top2_return_pct": "",
            "random_avg_top2_return_pct": 0.9214,
            "vs_random_pct": "",
            "win_rate_vs_random": "",
            "valid_scans": 57,
            "passed": "no",
            "adopted": "no",
            "reject_reason": "Policy — not price/volume; do not test",
            "test_date": "",
            "learning_recommendation": "NO_ACTION",
        })

    keys = list(FEATURE_REGISTRY.keys()) if run_all else list(FEATURE_REGISTRY.keys())
    for key in keys:
        result = run_single_feature_test(key)
        rows.append({
            "feature": result["feature"],
            "category": result["category"],
            "avg_top2_return_pct": result["avg_top2_return_pct"],
            "random_avg_top2_return_pct": result["random_avg_top2_return_pct"],
            "vs_random_pct": result["vs_random_pct"],
            "win_rate_vs_random": result["win_rate_vs_random"],
            "valid_scans": result["valid_scans"],
            "passed": result["passed"],
            "adopted": result["adopted"],
            "reject_reason": result["reject_reason"],
            "test_date": datetime.now(KST).strftime("%Y-%m-%d"),
            "learning_recommendation": "NO_ACTION",
        })

    write_csv(SCOREBOARD_CSV, rows)
    return rows


def print_scoreboard(rows: list[dict]) -> None:
    print("")
    print("Feature Scoreboard")
    print("")
    print(f"{'Feature':<28} {'AvgRet%':>8} {'Random%':>8} {'vsRand':>8} {'WinRate':>8} {'Pass':>5} {'Adopt':>5}")
    print("-" * 80)
    for r in rows:
        avg = r.get("avg_top2_return_pct", "")
        rnd = r.get("random_avg_top2_return_pct", "")
        vs = r.get("vs_random_pct", "")
        wr = r.get("win_rate_vs_random", "")
        print(
            f"{r['feature']:<28} "
            f"{avg!s:>8} {rnd!s:>8} {vs!s:>8} "
            f"{wr!s:>8} {r.get('passed',''):>5} {r.get('adopted',''):>5}"
        )
    print("")
    adopted = [r for r in rows if r.get("adopted") == "yes"]
    passed = [r for r in rows if r.get("passed") == "yes" and r.get("category") != "process_REJECTED"]
    print(f"Price/volume passed: {len(passed)} | Adopted: {len(adopted)}")
    print(f"Random baseline: {rows[0].get('random_avg_top2_return_pct', '0.9214')}% (57 scans)")
    print("")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scout Season2 single-feature blind test")
    parser.add_argument("--feature", help="Feature key to test")
    parser.add_argument("--list", action="store_true", help="List testable features")
    parser.add_argument("--update-scoreboard", action="store_true", help="Run all features and update scoreboard")
    args = parser.parse_args()

    if args.list:
        print("Testable price/volume features:")
        for key, spec in FEATURE_REGISTRY.items():
            print(f"  {key}: {spec['definition']}")
        print("\nRejected (do not test):", ", ".join(REJECTED_PROCESS_FEATURES))
        return

    if args.update_scoreboard:
        rows = update_scoreboard(run_all=True)
        print_scoreboard(rows)
        print(f"Scoreboard saved: {SCOREBOARD_CSV}")
        return

    if not args.feature:
        parser.print_help()
        return

    result = run_single_feature_test(args.feature)
    print(f"Feature: {result['feature']}")
    print(f"Avg Top2: {result['avg_top2_return_pct']}% | Random: {result['random_avg_top2_return_pct']}%")
    print(f"Vs Random: {result['vs_random_pct']:+.4f}% | Win rate: {result['win_rate_vs_random']:.1%}")
    print(f"Passed: {result['passed']}")
    print(f"Detail: {OUT_DIR / f'blind_test_{args.feature}.csv'}")


if __name__ == "__main__":
    main()
