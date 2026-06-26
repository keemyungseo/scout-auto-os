"""Ranking Engine V1 decision report."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

KST = timezone(timedelta(hours=9))


def build_ranking_report(
    meta: dict,
    model_comparison: list[dict],
    baselines: list[dict],
    top_features: list[dict],
    decision: dict,
) -> str:
    best_model = decision.get("best_model", "n/a")
    best_bl = next((b for b in baselines if b["strategy"] == "current_search_a6"), {})
    best_rank = next((m for m in model_comparison if m["model"] == best_model), {})

    lines = [
        "# Ranking Engine V1 — Search AI",
        "",
        "Rule-free ranking from full feature matrix. Random seed fixed.",
        "",
        f"- Samples: **{meta.get('sample_count')}** | Features: **{meta.get('feature_count')}**",
        f"- Blind scans: **{meta.get('blind_scans')}** | Models trained: **{meta.get('models_trained')}**",
        f"- Best model: **{best_model}**",
        "",
        "## Decision",
        "",
        f"### 1. Ranking Engine vs Formula League V2 (blind)",
        decision.get("vs_formula_league_v2", "unknown"),
        "",
        f"### 2. Ranking Engine vs Execution Engine proxy (blind)",
        decision.get("vs_execution", "unknown"),
        "",
        f"### 3. Can Search Formula be fully replaced?",
        decision.get("replace_search_formula", "unknown"),
        "",
        f"### 4. Top ranking features",
        "",
    ]
    for f in top_features[:10]:
        lines.append(f"- `{f.get('feature')}` — combined {f.get('combined_score')}")

    lines.extend([
        "",
        f"### 5. LIVE applicability",
        decision.get("live_applicability", "unknown"),
        "",
        f"### 6. Blockers if not LIVE",
        decision.get("blockers", "unknown"),
        "",
        "## Blind comparison (avg 2h / NDCG@5 / P@5)",
        "",
        "| Strategy | Avg 2h | Win% | NDCG5 | P@5 | Sharpe |",
        "|----------|--------|------|-------|-----|--------|",
    ])
    for row in baselines + [r for r in model_comparison if r.get("split") == "blind"]:
        lines.append(
            f"| {row.get('strategy') or row.get('model')} | {row.get('avg_return_2h')} | "
            f"{row.get('win_rate')} | {row.get('rank_ndcg5')} | {row.get('rank_p5')} | {row.get('sharpe')} |",
        )

    lines.extend([
        "",
        "## Model comparison (blind)",
        "",
        f"Best ranking model avg 2h: **{best_rank.get('avg_return_2h')}** vs A6 **{best_bl.get('avg_return_2h')}**",
        "",
        "Probabilistic — short calendar window; no price targets.",
        "",
        f"_Generated {datetime.now(KST).isoformat()}_",
    ])
    return "\n".join(lines)


def build_decision(
    model_comparison: list[dict],
    baselines: list[dict],
    top_features: list[dict],
) -> dict:
    a6 = next((b for b in baselines if b["strategy"] == "current_search_a6"), {})
    formula = next((b for b in baselines if b["strategy"] == "formula_league_v2"), {})
    execution = next((b for b in baselines if b["strategy"] == "execution_score_proxy"), {})
    blind_models = [m for m in model_comparison if m.get("split") == "blind" and m.get("model")]
    by_ndcg = sorted(blind_models, key=lambda x: -float(x.get("rank_ndcg5", 0)))
    by_ret = sorted(blind_models, key=lambda x: -float(x.get("avg_return_2h", 0)))
    best = by_ndcg[0] if by_ndcg else {}

    def _cmp(rank_row, base_row, name: str) -> str:
        ra = float(rank_row.get("avg_return_2h", 0))
        ba = float(base_row.get("avg_return_2h", 0))
        ndcg_r = float(rank_row.get("rank_ndcg5", 0))
        ndcg_b = float(base_row.get("rank_ndcg5", 0))
        if ra > ba and ndcg_r >= ndcg_b:
            return f"**Hypothesis YES** — {name} avg 2h {ra}% vs {ba}% (NDCG {ndcg_r} vs {ndcg_b})."
        if ra > ba:
            return f"**Mixed** — higher avg 2h ({ra}% vs {ba}%) but weaker ranking quality (NDCG {ndcg_r})."
        return f"**NO** — {name} remains stronger on blind ({ba}% vs {ra}%)."

    vs_formula = _cmp(best, formula, "Ranking vs Formula V2")
    vs_exec = _cmp(best, execution, "Ranking vs Execution proxy")

    replace = (
        "**Not yet** — keep A6_frozen; ranking shows promise but sample depth insufficient."
        if float(best.get("avg_return_2h", 0)) <= float(a6.get("avg_return_2h", 0))
        else "**Hypothesis partial** — ranking beats A6 on blind avg 2h; requires extended walk-forward before config change."
    )

    live = (
        "**REJECT for LIVE** — insufficient blind calendar and regime coverage."
        if float(best.get("trade_count", 0)) < 100
        else "**HYPOTHESIS** — candidate for shadow mode only."
    )
    blockers = (
        "Short blind window (~15 days); observation features need causal LIVE pipeline; "
        "no multi-regime validation; model not registered in live scanner."
    )

    top_feat = ", ".join(f["feature"] for f in top_features[:5]) if top_features else "n/a"

    return {
        "best_model": best.get("model"),
        "best_model_by_return": by_ret[0].get("model") if by_ret else None,
        "vs_formula_league_v2": vs_formula,
        "vs_execution": vs_exec,
        "replace_search_formula": replace,
        "top_features": top_feat,
        "live_applicability": live,
        "blockers": blockers,
    }
