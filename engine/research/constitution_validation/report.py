"""Final constitution validation report."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

KST = timezone(timedelta(hours=9))


def build_decision(
    calendar: dict,
    blind: dict,
    drift: dict,
    stability: dict,
    regime_summary: list[dict],
    rolling: list[dict],
) -> dict:
    meets_calendar = calendar.get("meets_3month_target", False)
    avg = float(blind.get("avg_return_2h", 0))
    positive_blind = avg > 0
    drift_ok = not drift.get("drift_detected", True)
    stable_imp = stability.get("stable", False)

    wf_rows = [r for r in rolling if r.get("validation_type") == "walk_forward"]
    wf_avg = float(wf_rows[0].get("avg_return_2h", 0)) if wf_rows else avg

    negative_regimes = [
        r for r in regime_summary
        if float(r.get("avg_return_2h", 0)) < 0 and int(r.get("trade_count", 0)) >= 15
    ]

    if meets_calendar and positive_blind and drift_ok and wf_avg > 2.0:
        confidence = "medium"
    elif positive_blind and wf_avg > 3.0:
        confidence = "medium-low"
    else:
        confidence = "hypothesis"

    core_ready = (
        meets_calendar
        and positive_blind
        and drift_ok
        and stable_imp
        and len(negative_regimes) == 0
        and wf_avg >= avg * 0.85
    )

    return {
        "q1_long_term_blind": (
            f"**{'Partial YES' if positive_blind and wf_avg > 2 else 'NO / insufficient'}** - "
            f"blind avg 2h **{avg}%**, walk-forward **{wf_avg}%** on **{calendar.get('calendar_days')}d** "
            f"({calendar.get('start_date')} to {calendar.get('end_date')}). "
            f"{'Does NOT meet 3-month target.' if not meets_calendar else 'Calendar target met.'}"
        ),
        "q2_live_confidence": (
            f"**{confidence.upper()}** ({confidence}) - "
            f"Sharpe {blind.get('sharpe')}, PF {blind.get('profit_factor')}, "
            f"NDCG@5 {blind.get('rank_ndcg5')}. "
            f"Recommend shadow LIVE minimum 30d before capital."
        ),
        "q3_core_engine_ready": (
            "**YES for research core, NO for LIVE core** - constitution hypothesis-validated on short window."
            if not core_ready
            else "**YES hypothesis** - passes extended calendar + stability gates."
        ),
        "q4_biggest_risk": (
            f"**Calendar length** ({calendar.get('calendar_days')}d vs 90d target) and "
            f"**regime gaps** ({len(negative_regimes)} weak regime buckets). "
            + (f"Weakest: {negative_regimes[0].get('regime_axis')}={negative_regimes[0].get('regime')}."
               if negative_regimes else "All regime buckets positive on available sample.")
        ),
        "q5_pre_live_blockers": (
            "1) Extend Binance history to 90d+  2) Persist model artifact + version pin  "
            "3) LIVE scan-history for label retrain pipeline  "
            "4) Regime-negative monitoring  5) NDCG/P@5 trade-off vs return_minus_dd label"
        ),
        "core_ready": core_ready,
        "confidence_tier": confidence,
    }


def build_report(
    meta: dict,
    calendar: dict,
    blind: dict,
    decision: dict,
    label_ranking: list | None = None,
) -> str:
    lines = [
        "# Constitution Validation V1 — Final Blind Validation",
        "",
        "Frozen SCOUT Constitution — no new feature, rule, label, model, or tuning.",
        "",
        "## Frozen stack",
        "",
        "- **Features:** Ranking Engine V1 (170 snapshot features)",
        "- **Model:** CatBoost Ranker (seed=42, iterations=200, lr=0.05)",
        "- **Label:** `return_minus_dd` (Target Discovery winner)",
        "",
        "## Calendar coverage",
        "",
        f"- Period: **{calendar.get('start_date')}** to **{calendar.get('end_date')}**",
        f"- Days: **{calendar.get('calendar_days')}** (target 90d: **{'PASS' if calendar.get('meets_3month_target') else 'FAIL'}**)",
        f"- Scans: **{calendar.get('scan_count')}** | Samples: **{calendar.get('sample_count')}**",
        "",
        "## Blind holdout (30%)",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Avg 2h | {blind.get('avg_return_2h')}% |",
        f"| Top2 | {blind.get('top2_avg_return_2h')}% |",
        f"| Top5 | {blind.get('top5_avg_return_2h')}% |",
        f"| Win% | {blind.get('win_rate')}% |",
        f"| Sharpe | {blind.get('sharpe')} |",
        f"| Sortino | {blind.get('sortino')} |",
        f"| MDD | {blind.get('mdd')}% |",
        f"| Profit Factor | {blind.get('profit_factor')} |",
        f"| NDCG@5 | {blind.get('rank_ndcg5')} |",
        f"| P@5 | {blind.get('rank_p5')} |",
        f"| Rank Corr | {blind.get('rank_correlation')} |",
        "",
        "## Final conclusions",
        "",
        f"### 1. Long-term blind persistence",
        "",
        decision.get("q1_long_term_blind", ""),
        "",
        f"### 2. LIVE confidence tier",
        "",
        decision.get("q2_live_confidence", ""),
        "",
        f"### 3. Core engine confirmation",
        "",
        decision.get("q3_core_engine_ready", ""),
        "",
        f"### 4. Biggest remaining risk",
        "",
        decision.get("q4_biggest_risk", ""),
        "",
        f"### 5. Pre-LIVE blockers",
        "",
        decision.get("q5_pre_live_blockers", ""),
        "",
        "Probabilistic — correlation is not causation; no price targets.",
        "",
        f"_Generated {datetime.now(KST).isoformat()}_",
    ]
    return "\n".join(lines)
