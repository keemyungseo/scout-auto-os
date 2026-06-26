"""Formula League V2 report."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

KST = timezone(timedelta(hours=9))


def build_formula_report(
    meta: dict,
    top_formulas: list[dict],
    survivors: list[dict],
    baseline: dict,
    best_survivor: dict | None,
) -> str:
    lift = meta.get("blind_lift_pct", 0)
    beats = meta.get("best_beats_baseline", False)

    lines = [
        "# Formula League V2 — Search Formula Evolution",
        "",
        "**Frozen:** Direction Champion, Entry V2, Execution stack, Portfolio, Router.",
        "**Research:** Search Formula only.",
        "",
        f"- Formulas generated: **{meta.get('formulas_generated')}**",
        f"- Blind scans: **{meta.get('blind_scan_count')}** / {meta.get('total_scans')}",
        f"- Survivors: **{meta.get('survivor_count')}**",
        "",
        "## Final question: Blind lift vs A6 baseline (Search only)",
        "",
    ]
    if beats and best_survivor:
        lines.append(
            f"**Hypothesis YES** — best survivor `{best_survivor.get('formula_id', '')[:50]}` "
            f"avg 2h **{best_survivor.get('avg_return_2h')}%** vs A6 **{baseline.get('avg_return_2h')}%** "
            f"(lift **{lift}%**). Probabilistic — small calendar window.",
        )
    else:
        lines.append(
            f"**No proven lift** — A6 baseline avg 2h **{baseline.get('avg_return_2h')}%** "
            f"remains competitive. Specialized formulas lack cross-round stability.",
        )

    lines.extend([
        "",
        "## A6 frozen baseline (blind)",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Avg 2h | {baseline.get('avg_return_2h')} |",
        f"| Win rate 2h | {baseline.get('win_rate_2h')} |",
        f"| Hit top3 rate | {baseline.get('hit_rate_top3_pct')} |",
        f"| Sharpe-like | {baseline.get('sharpe_like')} |",
        f"| Stability | {baseline.get('stability')} |",
        "",
        "## Top formulas (blind generalization score)",
        "",
        "| Formula | Avg 2h | Hit top3% | Win 2h | Stability | Gen score |",
        "|---------|--------|-----------|--------|-----------|-----------|",
    ])
    for r in top_formulas[:15]:
        lines.append(
            f"| `{str(r.get('formula_id', ''))[:40]}` | {r.get('avg_return_2h')} | "
            f"{r.get('hit_rate_top3_pct')} | {r.get('win_rate_2h')} | {r.get('stability')} | "
            f"{r.get('generalization_score')} |",
        )

    lines.extend([
        "",
        "## LIVE-ready candidates",
        "",
    ])
    live_candidates = [s for s in survivors if s.get("survived") and s["formula_id"] != "A6_frozen"][:5]
    if live_candidates:
        for s in live_candidates:
            lines.append(
                f"- `{s['formula_id']}` — round win rate {s.get('round_win_rate')}, "
                f"gen score {s.get('avg_generalization_score')}",
            )
    else:
        lines.append("- **None promoted** — keep `A6_frozen` for LIVE Search.")

    lines.extend([
        "",
        "## Survivor system",
        "",
        "Rounds: temporal blind, weekly, monthly, regime, volatility.",
        "Survival = beat A6 in >=55% of blind rounds with positive generalization score.",
        "",
        "## DNA summary",
        "",
        f"Top survivor features: {meta.get('top_dna_features', 'n/a')}",
        "",
        "Generalization > raw average return.",
        "",
        f"_Generated {datetime.now(KST).isoformat()}_",
    ])
    return "\n".join(lines)
