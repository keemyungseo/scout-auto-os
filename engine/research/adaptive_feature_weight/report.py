"""Adaptive feature weight report."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

KST = timezone(timedelta(hours=9))


def build_adaptive_report(
    meta: dict,
    comparison: dict,
    top_conditional: list[dict],
    patterns: list[dict],
    decision: dict,
) -> str:
    lines = [
        "# Adaptive Feature Weight Engine V1",
        "",
        "Conditional importance analysis on **frozen** CatBoost Ranker — no model retrain.",
        "",
        f"- Features: **{meta.get('feature_count')}** | Conditions: **{meta.get('condition_count')}**",
        f"- Train scans: {meta.get('train_scans')} | Blind scans: {meta.get('blind_scans')}",
        "",
        "## Final question: Does conditional weight improve blind performance?",
        "",
        decision.get("summary", ""),
        "",
        "## Uniform vs Adaptive (blind)",
        "",
        "| Metric | Uniform | Adaptive | Delta |",
        "|--------|---------|----------|-------|",
    ]
    for key, label in (
        ("avg_return_2h", "Avg 2h"),
        ("top2_avg_return_2h", "Top2 avg"),
        ("top5_avg_return_2h", "Top5 avg"),
        ("rank_ndcg5", "NDCG@5"),
        ("rank_p5", "P@5"),
        ("precision_top3", "Precision top3"),
        ("sharpe", "Sharpe"),
    ):
        u = comparison.get("uniform", {}).get(key, 0)
        a = comparison.get("adaptive", {}).get(key, 0)
        d = round(float(a) - float(u), 4)
        lines.append(f"| {label} | {u} | {a} | {d} |")

    lines.extend(["", "## Conditional patterns (data-driven)", ""])
    for p in patterns[:8]:
        lines.append(f"- **{p.get('condition_id')}**: top feature `{p.get('top_feature')}` — {p.get('interpretation')}")

    lines.extend(["", "## Top conditional features", ""])
    for r in top_conditional[:15]:
        lines.append(
            f"- [{r.get('condition_id')}] `{r.get('feature')}` "
            f"combined={r.get('combined_score')} (n={r.get('sample_count')})",
        )

    lines.extend([
        "",
        "## When / why features matter",
        "",
        decision.get("when_why", ""),
        "",
        "Probabilistic — short calendar window.",
        "",
        f"_Generated {datetime.now(KST).isoformat()}_",
    ])
    return "\n".join(lines)


def infer_patterns(conditional_rows: list[dict]) -> list[dict]:
    by_cond: dict[str, list[dict]] = {}
    for r in conditional_rows:
        by_cond.setdefault(r["condition_id"], []).append(r)

    hints = {
        "high_volatility": "range/ATR proxies dominate",
        "breakout": "volume and release rank rise",
        "compression": "compression and body features dominate",
        "momentum": "momentum and return rank dominate",
        "bull_leader": "direction confidence and trend features",
        "sideway": "mean-reversion / margin features",
        "volume_surge": "volume ratio and OBV proxies",
        "reversal": "previous return vs current return spread",
    }

    patterns: list[dict] = []
    for cid, rows in sorted(by_cond.items()):
        rows.sort(key=lambda x: -float(x.get("combined_score", 0)))
        top = rows[0] if rows else {}
        feat = top.get("feature", "")
        # Secondary feature when release rank dominates all conditions
        if feat == "ctx_rank_5m_release" and len(rows) > 1:
            feat = rows[1].get("feature", feat)
            top = rows[1]
        interp = hints.get(cid, "condition-specific feature shift")
        if "range" in feat or "atr" in feat.lower():
            interp = "range/ATR family important in this state"
        elif "volume" in feat:
            interp = "volume family important in this state"
        elif "momentum" in feat or "return" in feat:
            interp = "momentum/return family important in this state"
        patterns.append({
            "condition_id": cid,
            "top_feature": feat,
            "combined_score": top.get("combined_score"),
            "interpretation": interp,
        })
    return patterns


def build_decision(comparison: dict, improved: bool, patterns: list[dict]) -> dict:
    u = comparison.get("uniform", {})
    a = comparison.get("adaptive", {})
    if improved:
        summary = (
            f"**Hypothesis YES** — adaptive weights improve blind avg 2h "
            f"({a.get('avg_return_2h')}% vs {u.get('avg_return_2h')}%) with NDCG "
            f"{a.get('rank_ndcg5')} vs {u.get('rank_ndcg5')}. Probabilistic."
        )
    else:
        summary = (
            f"**NO proven lift** — uniform frozen ranker remains competitive "
            f"(avg 2h {u.get('avg_return_2h')}% vs adaptive {a.get('avg_return_2h')}%). "
            "Conditional importance is descriptive, not yet operational."
        )

    when_why = "; ".join(
        f"In `{p['condition_id']}`, `{p['top_feature']}` leads ({p['interpretation']})"
        for p in patterns[:6]
    )
    return {"summary": summary, "when_why": when_why, "improved": improved}
