"""Candidate scoring, champion gates, and statistical comparison vs Random/A6."""

from __future__ import annotations

import statistics
from collections import defaultdict

from scout_auto_os.engine.research.zero_base.candidates import CANDIDATE_ENGINES

MIN_CHAMPION_SAMPLES = 300
MIN_CONSISTENT_DAYS = 3


def candidate_score(row: dict) -> float:
    return (
        float(row.get("avg_return_2h") or 0)
        + float(row.get("big_winner_capture_rate") or 0)
        + float(row.get("win_rate") or 0)
        - float(row.get("trap_rate") or 0)
        - abs(float(row.get("max_drawdown_avg") or 0))
    )


def _profit_factor(returns: list[float]) -> float:
    wins = sum(r for r in returns if r > 0)
    losses = abs(sum(r for r in returns if r < 0))
    if losses <= 0:
        return round(wins, 2) if wins else 0.0
    return round(wins / losses, 2)


def aggregate_candidate_metrics(samples: list[dict]) -> dict:
    if not samples:
        return {"sample_count": 0, "score": 0}
    r2h = [float(s.get("return_2h", 0)) for s in samples]
    traps = sum(1 for s in samples if s.get("label_trap"))
    big = sum(1 for s in samples if s.get("label_big_winner"))
    mdd = [float(s.get("max_drawdown_2h", 0)) for s in samples]
    wins = sum(1 for x in r2h if x >= 3.0)
    ttp = [s.get("time_to_peak") for s in samples if s.get("time_to_peak") is not None]
    tt3 = [s.get("time_to_3pct") for s in samples if s.get("time_to_3pct") is not None]
    tt5 = [s.get("time_to_5pct") for s in samples if s.get("time_to_5pct") is not None]
    row = {
        "sample_count": len(samples),
        "return_30m_avg": round(statistics.mean([float(s.get("return_30m", 0)) for s in samples]), 4),
        "return_1h_avg": round(statistics.mean([float(s.get("return_1h", 0)) for s in samples]), 4),
        "avg_return_2h": round(statistics.mean(r2h), 4),
        "return_4h_avg": round(statistics.mean([float(s.get("return_4h", 0)) for s in samples]), 4),
        "return_6h_avg": round(statistics.mean([float(s.get("return_6h", 0)) for s in samples]), 4),
        "median_return_2h": round(statistics.median(r2h), 4),
        "max_return_2h_avg": round(statistics.mean([float(s.get("max_return_2h", 0)) for s in samples]), 4),
        "min_return_2h_avg": round(statistics.mean([float(s.get("min_return_2h", 0)) for s in samples]), 4),
        "max_drawdown_avg": round(statistics.mean(mdd), 4),
        "time_to_peak_avg": round(statistics.mean(ttp), 1) if ttp else None,
        "time_to_3pct_avg": round(statistics.mean(tt3), 1) if tt3 else None,
        "time_to_5pct_avg": round(statistics.mean(tt5), 1) if tt5 else None,
        "trap_rate": round(traps / len(samples) * 100, 2),
        "big_winner_capture_rate": round(big / len(samples) * 100, 2),
        "win_rate": round(wins / len(samples) * 100, 2),
        "profit_factor": _profit_factor(r2h),
        "downside_capture_avg": round(statistics.mean([float(s.get("downside_capture", 0)) for s in samples]), 4),
    }
    row["score"] = round(candidate_score(row), 2)
    return row


def compare_vs_baseline(candidate: dict, random: dict, a6: dict) -> dict:
    return {
        "avg_return_2h_delta_vs_random": round(
            float(candidate.get("avg_return_2h", 0)) - float(random.get("avg_return_2h", 0)), 4,
        ),
        "win_rate_delta_vs_random": round(
            float(candidate.get("win_rate", 0)) - float(random.get("win_rate", 0)), 2,
        ),
        "trap_rate_delta_vs_random": round(
            float(candidate.get("trap_rate", 0)) - float(random.get("trap_rate", 0)), 2,
        ),
        "big_winner_delta_vs_random": round(
            float(candidate.get("big_winner_capture_rate", 0)) - float(random.get("big_winner_capture_rate", 0)), 2,
        ),
        "avg_return_2h_delta_vs_a6": round(
            float(candidate.get("avg_return_2h", 0)) - float(a6.get("avg_return_2h", 0)), 4,
        ),
        "beats_random_return": float(candidate.get("avg_return_2h", 0)) > float(random.get("avg_return_2h", 0)),
        "beats_random_trap": float(candidate.get("trap_rate", 99)) < float(random.get("trap_rate", 0)),
        "beats_a6_return": float(candidate.get("avg_return_2h", 0)) > float(a6.get("avg_return_2h", 0)),
    }


def day_consistency(samples: list[dict]) -> int:
    days: dict[str, list[float]] = defaultdict(list)
    for s in samples:
        scan = s.get("scan_time_kst", "")[:10]
        if scan:
            days[scan].append(float(s.get("return_2h", 0)))
    positive_days = sum(1 for rets in days.values() if statistics.mean(rets) > 0)
    return positive_days


def champion_eligible(candidate: dict, comparison: dict, samples: list[dict]) -> dict:
    n = int(candidate.get("sample_count") or 0)
    days = day_consistency(samples)
    checks = {
        "sample_count_ok": n >= MIN_CHAMPION_SAMPLES,
        "beats_random_return": comparison.get("beats_random_return", False),
        "beats_random_trap": comparison.get("beats_random_trap", False),
        "beats_a6_return": comparison.get("beats_a6_return", False),
        "consistent_days_ok": days >= MIN_CONSISTENT_DAYS,
        "median_positive": float(candidate.get("median_return_2h", 0)) > 0,
        "drawdown_ok": abs(float(candidate.get("max_drawdown_avg", 0))) <= 12.0,
    }
    eligible = all(checks.values())
    return {
        "champion_eligible": eligible,
        "champion_checks": checks,
        "positive_days": days,
        "tier": "champion_candidate" if eligible else ("verification_needed" if n >= 100 else "hypothesis"),
    }


def build_champion_board(
    candidate_results: list[dict],
    random_stats: dict,
    sample_index: dict[str, list[dict]],
) -> list[dict]:
    a6 = next((c for c in candidate_results if c.get("engine") == "A6_CURRENT"), {})
    board: list[dict] = []
    for c in candidate_results:
        if c.get("engine") == "RANDOM_BASELINE":
            continue
        cmp = compare_vs_baseline(c, random_stats, a6)
        champ = champion_eligible(c, cmp, sample_index.get(c.get("engine", ""), []))
        board.append({**c, **cmp, **champ})
    board.sort(key=lambda x: x.get("score", 0), reverse=True)
    for i, row in enumerate(board, 1):
        row["board_rank"] = i
    return board


def feature_diagnostics(samples_by_engine: dict[str, list[dict]]) -> list[dict]:
    """Patterns that outperform / underperform on validation set."""
    out: list[dict] = []
    for eng, samples in samples_by_engine.items():
        if eng == "RANDOM_BASELINE" or len(samples) < 20:
            continue
        r2h = [float(s.get("return_2h", 0)) for s in samples]
        traps = sum(1 for s in samples if s.get("label_trap"))
        out.append({
            "engine": eng,
            "sample_count": len(samples),
            "avg_return_2h": round(statistics.mean(r2h), 4),
            "trap_rate": round(traps / len(samples) * 100, 2),
            "verdict": "promising" if statistics.mean(r2h) > 1.0 and traps / len(samples) < 0.25 else "discard",
        })
    out.sort(key=lambda x: x["avg_return_2h"], reverse=True)
    return out
