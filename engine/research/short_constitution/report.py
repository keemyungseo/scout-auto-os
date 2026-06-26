"""Short Constitution V1 report."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

KST = timezone(timedelta(hours=9))


def completeness_score(short_blind: dict, long_blind: dict, shift_summary: dict) -> int:
    s_avg = float(short_blind.get("avg_return_2h", 0))
    l_avg = float(long_blind.get("avg_return_2h", 5.26))
    s_sharpe = float(short_blind.get("sharpe", 0))
    ndcg = float(short_blind.get("rank_ndcg5", 0))
    overlap = float(shift_summary.get("top15_overlap_pct", 0))

    score = 0.0
    if s_avg > 0:
        score += 20
    if s_avg >= l_avg * 0.5:
        score += 15
    if s_avg >= l_avg * 0.8:
        score += 10
    if s_sharpe > 3:
        score += 15
    if ndcg > 0.7:
        score += 15
    if ndcg > 0.85:
        score += 5
    if overlap < 60:
        score += 10
    if float(short_blind.get("win_rate", 0)) > 45:
        score += 10
    return min(100, int(round(score)))


def build_decision(
    best_label: dict,
    label_ranking: list[dict],
    shift_summary: dict,
    long_blind: dict,
    score: int,
    leak: dict,
) -> dict:
    s_avg = float(best_label.get("avg_return_2h", 0))
    l_avg = float(long_blind.get("avg_return_2h", 0))
    independent = shift_summary.get("independent_structure", False)

    return {
        "q1_independent_features": (
            f"**{'YES hypothesis' if independent else 'PARTIAL'}** - "
            f"top-15 overlap {shift_summary.get('top15_overlap_pct')}% "
            f"(long-only: {len(shift_summary.get('long_only_top15', []))}, "
            f"short-only: {len(shift_summary.get('short_only_top15', []))})."
        ),
        "q2_best_short_label": (
            f"**{best_label.get('label_name')}** (`{best_label.get('label_id')}`) "
            f"blind avg 2h **{s_avg}%**"
        ),
        "q3_feature_reuse": (
            "**Short-specific label + same 170 feature scaffold** - "
            "importance structure differs; not a sign flip of long features."
            if independent
            else "**Reuse with relabeling** - high overlap suggests shared cross-sectional signals."
        ),
        "q4_data_signal": (
            f"**{'YES hypothesis' if s_avg > 2 and s_avg > 0 else 'WEAK / unknown'}** - "
            f"short blind {s_avg}% vs long {l_avg}% on 15d window."
        ),
        "q5_completeness_score": f"**{score}/100** - research completeness (not LIVE readiness).",
        "q6_next_steps": _next_steps(score, s_avg, l_avg, leak),
    }


def _next_steps(score: int, s_avg: float, l_avg: float, leak: dict) -> str:
    steps = []
    if not leak.get("passed"):
        steps.append("Fix leak check failures before any shadow run.")
    if s_avg < l_avg * 0.5:
        steps.append("Short label discovery round 2 - bear-only scan filter + dedicated short features (research).")
    if score < 50:
        steps.append("Accumulate 90d+ history via Research Infrastructure before Short Constitution V2.")
    else:
        steps.append("Shadow short ranker parallel to long; regime-gated activation only.")
    steps.append("Re-run blind after infrastructure reaches 90d calendar.")
    return " | ".join(steps)


def build_report(
    meta: dict,
    label_ranking: list[dict],
    best: dict,
    long_cmp: dict,
    decision: dict,
    regime_rows: list[dict],
) -> str:
    lines = [
        "# Short Constitution Research V1",
        "",
        "Independent short search constitution - not a sign flip of Long.",
        "",
        f"- Direction: **short** | Model: **catboost_ranker** | Samples: **{meta.get('sample_count')}**",
        f"- Best label: **{best.get('label_id')}** avg 2h **{best.get('avg_return_2h')}%**",
        f"- Long constitution (frozen): avg 2h **{long_cmp.get('long_avg_return_2h')}%**",
        f"- Leak check: **{'PASS' if meta.get('leak_passed') else 'FAIL'}**",
        "",
        "## 1. Independent feature structure?",
        "",
        decision.get("q1_independent_features", ""),
        "",
        "## 2. Best short label",
        "",
        decision.get("q2_best_short_label", ""),
        "",
        "## 3. Reuse long features?",
        "",
        decision.get("q3_feature_reuse", ""),
        "",
        "## 4. Sufficient data signal?",
        "",
        decision.get("q4_data_signal", ""),
        "",
        "## 5. Completeness score",
        "",
        decision.get("q5_completeness_score", ""),
        "",
        "## 6. Next steps",
        "",
        decision.get("q6_next_steps", ""),
        "",
        "## Label ranking (blind)",
        "",
        "| Rank | Label | Avg 2h | Sharpe | NDCG5 | vs baseline |",
        "|------|-------|--------|--------|-------|-------------|",
    ]
    for r in label_ranking[:15]:
        lines.append(
            f"| {r.get('label_rank')} | {r.get('label_id')} | {r.get('avg_return_2h')} | "
            f"{r.get('sharpe')} | {r.get('rank_ndcg5')} | {r.get('vs_baseline_pct')}% |",
        )

    lines.extend([
        "",
        "## Short regime performance (best label)",
        "",
        "| Regime | Scans | Avg 2h | Win% |",
        "|--------|-------|--------|------|",
    ])
    for r in regime_rows[:12]:
        lines.append(
            f"| {r.get('regime')} | {r.get('scan_count')} | {r.get('avg_return_2h')} | {r.get('win_rate')} |",
        )

    lines.extend([
        "",
        "Probabilistic - 15d calendar; no price targets.",
        "",
        f"_Generated {datetime.now(KST).isoformat()}_",
    ])
    return "\n".join(lines)
