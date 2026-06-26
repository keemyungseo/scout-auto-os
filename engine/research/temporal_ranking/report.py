"""Temporal Ranking Engine V1 report."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

KST = timezone(timedelta(hours=9))


def build_temporal_report(
    meta: dict,
    seq_comparison: list[dict],
    baseline: dict,
    best: dict,
    importance: dict,
    decision: dict,
    leak: dict,
) -> str:
    lines = [
        "# Temporal Ranking Engine V1",
        "",
        "Time-series search AI — snapshot + leak-safe scan history temporal features.",
        "",
        f"- Baseline (Ranking V1 snapshot CatBoost): avg 2h **{baseline.get('avg_return_2h')}**",
        f"- Best temporal model: **{best.get('model')}** seq={best.get('seq_len')} avg 2h **{best.get('avg_return_2h')}** "
        f"(snapshot baseline retained unless temporal beats it)",
        f"- Leak check: **{'PASS' if leak.get('passed') else 'FAIL'}**",
        "",
        "## 1. Did temporal features improve blind performance?",
        "",
        decision.get("q1_temporal_improves", ""),
        "",
        "## 2. Best time window (sequence length)",
        "",
        decision.get("q2_best_window", ""),
        "",
        "## Sequence length comparison",
        "",
        "| Seq len | Model | Avg 2h | Top2 | NDCG5 | P@5 | Sharpe |",
        "|---------|-------|--------|------|-------|-----|--------|",
    ]
    for r in seq_comparison[:20]:
        lines.append(
            f"| {r.get('seq_len')} | {r.get('model')} | {r.get('avg_return_2h')} | "
            f"{r.get('top2_avg_return_2h')} | {r.get('rank_ndcg5')} | {r.get('rank_p5')} | {r.get('rank_sharpe', r.get('sharpe'))} |",
        )

    lines.extend([
        "",
        "## 3. Delta vs absolute importance",
        "",
        decision.get("q3_delta_vs_absolute", ""),
        "",
        "### Top delta-dominant features",
        "",
    ])
    for r in importance.get("delta_vs_absolute", [])[:10]:
        if r.get("delta_wins"):
            lines.append(
                f"- `{r['base_feature']}` — delta {r['delta_share_pct']}% vs absolute",
            )

    lines.extend([
        "",
        "## 4. Statistically meaningful vs Ranking V1?",
        "",
        decision.get("q4_significance", ""),
        "",
        "## 5. LIVE applicability",
        "",
        decision.get("q5_live", ""),
        "",
        "## 6. Generalization",
        "",
        decision.get("q6_generalization", ""),
        "",
        "Probabilistic — ~15 day calendar; no price targets.",
        "",
        f"_Generated {datetime.now(KST).isoformat()}_",
    ])
    return "\n".join(lines)


def build_decision(
    baseline: dict,
    best: dict,
    seq_comparison: list[dict],
    importance: dict,
    sig: dict,
    leak: dict,
) -> dict:
    improved = float(best.get("avg_return_2h", 0)) > float(baseline.get("avg_return_2h", 0))
    ndcg_improved = float(best.get("rank_ndcg5", 0)) >= float(baseline.get("rank_ndcg5", 0))

    by_seq: dict[int, list[dict]] = {}
    for r in seq_comparison:
        by_seq.setdefault(r["seq_len"], []).append(r)
    best_seq = max(
        by_seq.keys(),
        key=lambda s: max(float(x.get("avg_return_2h", 0)) for x in by_seq[s]),
    ) if by_seq else 0

    level = importance.get("level_pct", {})
    delta_pct = level.get("delta", 0) + level.get("accel", 0)

    return {
        "q1_temporal_improves": (
            f"**Hypothesis YES** — temporal best avg 2h {best.get('avg_return_2h')}% vs snapshot {baseline.get('avg_return_2h')}%"
            if improved and ndcg_improved
            else f"**NO proven lift** - snapshot baseline {baseline.get('avg_return_2h')}% vs temporal {best.get('avg_return_2h')}%"
        ),
        "q2_best_window": f"Best sequence length: **{best_seq}** scans (~{best_seq * 2}h history at 2h cadence).",
        "q3_delta_vs_absolute": (
            f"Temporal delta+accel features account for **{delta_pct}%** of combined importance. "
            f"Delta-dominant bases listed in report."
        ),
        "q4_significance": (
            f"Approx lift {sig.get('lift_pct')}% (z~{sig.get('approx_z')}). "
            f"{'Hypothesis significant' if sig.get('significant_hypothesis') else 'Not significant on short window'}."
        ),
        "q5_live": (
            "**Shadow only** — requires LIVE scan history cache per symbol; not wired to scanner."
            if improved else "**REJECT for LIVE** - no blind improvement vs snapshot ranker."
        ),
        "q6_generalization": (
            "**Hypothesis pass** - temporal beats baseline on blind holdout."
            if improved and leak.get("passed")
            else (
                "**Fail / insufficient** - temporal does not beat snapshot on blind holdout."
                if not improved
                else "**Fail / insufficient** - leak check failed."
            )
        ),
        "improved": improved,
    }
