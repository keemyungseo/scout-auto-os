"""Target Discovery Engine V1 report."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

KST = timezone(timedelta(hours=9))


def build_decision(
    baseline: dict,
    best: dict,
    label_ranking: list[dict],
    sig: dict,
    learnability: list[dict],
) -> dict:
    improved = float(best.get("avg_return_2h", 0)) > float(baseline.get("avg_return_2h", 0))
    better_than_return = any(
        r.get("beats_baseline")
        and r.get("category") in ("return", "mfe", "risk_adjusted", "efficiency")
        for r in label_ranking
    )

    easiest = sorted(learnability, key=lambda x: -float(x.get("rank_correlation", 0)))[:3]
    easiest_names = ", ".join(f"`{e['label_id']}`" for e in easiest)

    return {
        "q1_best_label": (
            f"Best blind label: **{best.get('label_name')}** (`{best.get('label_id')}`) "
            f"avg 2h **{best.get('avg_return_2h')}%**"
        ),
        "q2_better_than_return": (
            f"**{'YES hypothesis' if better_than_return else 'NO'}** - "
            f"non-return labels {'beat' if better_than_return else 'did not beat'} baseline on blind avg 2h."
        ),
        "q3_generalization": (
            "**Hypothesis pass** - best label beats baseline on blind holdout."
            if improved and sig.get("significant_hypothesis")
            else "**Fail / insufficient** - no label significantly beats baseline on ~15d window."
        ),
        "q4_live": (
            "**Shadow only** - label swap requires retraining pipeline; not wired to LIVE."
            if improved
            else "**REJECT for LIVE** - baseline label remains best for realized 2h picks."
        ),
        "q5_vs_ranking_engine": (
            f"**{'YES hypothesis' if improved else 'NO'}** - lift {sig.get('lift_pct')}% "
            f"(z~{sig.get('approx_z')}) vs Ranking V1 baseline label."
        ),
        "q6_learnability": f"Highest baseline rank correlation: {easiest_names}.",
        "improved": improved,
    }


def build_report(
    meta: dict,
    label_ranking: list[dict],
    baseline: dict,
    best: dict,
    decision: dict,
) -> str:
    lines = [
        "# Target Discovery Engine V1",
        "",
        "Label research - same CatBoost ranker and snapshot features; training target only changes.",
        "",
        f"- Baseline label (max_up_4h): avg 2h **{baseline.get('avg_return_2h')}**",
        f"- Best label: **{best.get('label_name')}** avg 2h **{best.get('avg_return_2h')}**",
        f"- Candidates tested: **{meta.get('candidate_count', 0)}**",
        "",
        "## 1. Best blind label",
        "",
        decision.get("q1_best_label", ""),
        "",
        "## 2. Better than return-only labels?",
        "",
        decision.get("q2_better_than_return", ""),
        "",
        "## 3. Generalization",
        "",
        decision.get("q3_generalization", ""),
        "",
        "## 4. LIVE applicability",
        "",
        decision.get("q4_live", ""),
        "",
        "## 5. Meaningful vs Ranking Engine?",
        "",
        decision.get("q5_vs_ranking_engine", ""),
        "",
        "## 6. Learnability",
        "",
        decision.get("q6_learnability", ""),
        "",
        "## Label ranking (blind avg 2h)",
        "",
        "| Rank | Label | Category | Avg 2h | Top2 | NDCG5 | P@5 | Sharpe | vs baseline |",
        "|------|-------|----------|--------|------|-------|-----|--------|-------------|",
    ]
    for r in label_ranking[:25]:
        lines.append(
            f"| {r.get('label_rank')} | {r.get('label_id')} | {r.get('category')} | "
            f"{r.get('avg_return_2h')} | {r.get('top2_avg_return_2h')} | {r.get('rank_ndcg5')} | "
            f"{r.get('rank_p5')} | {r.get('sharpe')} | {r.get('vs_baseline_pct')}% |",
        )

    lines.extend([
        "",
        "Probabilistic - ~15 day calendar; labels from forward klines for training only.",
        "Blind trading metrics always use realized 2h return.",
        "",
        f"_Generated {datetime.now(KST).isoformat()}_",
    ])
    return "\n".join(lines)
