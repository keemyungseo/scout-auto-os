"""Rule Portfolio Engine V1 report."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

KST = timezone(timedelta(hours=9))


def _library_row(profile: dict) -> dict:
    row = {k: v for k, v in profile.items() if not k.endswith("_json")}
    return row


def build_portfolio_report(
    meta: dict,
    library: list[dict],
    cluster_summary: list[dict],
    router: dict,
) -> str:
    baseline = next((r for r in library if r["rule_id"] == "execution_score_v1"), {})
    specialized = [r for r in library if r["rule_id"] != "execution_score_v1" and r["direction"] == "long"]
    top3 = sorted(specialized, key=lambda r: -float(r.get("avg_return_2h", 0)))[:3]

    beats_universal = router.get("router_beats_baseline", False)
    answer = (
        "**Yes (hypothesis)** — regime-routed specialized rules beat universal Execution Score on this window"
        if beats_universal
        else "**No (current evidence)** — universal Execution Score remains safer; specialized rules are regime-fragile"
    )

    lines = [
        "# Rule Portfolio Engine V1",
        "",
        "Frozen: Direction Champion, Entry Rule V2, Entry Score, Execution Score baseline.",
        "",
        f"- Rules in library: **{meta.get('rule_count')}** (incl. universal baseline)",
        f"- Long mined rules: **{meta.get('long_rule_count')}**",
        f"- Scans: {meta.get('scan_count')} | Execution groups: {meta.get('group_count')}",
        "",
        "## Can specialized rules outperform a universal execution rule?",
        "",
        answer,
        "",
        f"| Strategy | Avg 2h | Trades |",
        f"|----------|--------|--------|",
        f"| Regime router (specialized) | {router.get('router_avg_return_2h')} | {router.get('router_trade_count')} |",
        f"| Universal Execution Score | {router.get('baseline_avg_return_2h')} | {router.get('router_trade_count')} |",
        f"| Lift | {router.get('lift_pct')}% | routes={router.get('routes_used')} |",
        "",
        "## Universal baseline",
        "",
        f"- Execution Score long avg: **{baseline.get('avg_return_2h', 'n/a')}%** "
        f"(trades={baseline.get('trade_count')}, precision={baseline.get('precision_pct')}%)",
        "",
        "## Top specialized rules (long)",
        "",
    ]
    for r in top3:
        lines.append(
            f"- `{r['rule_id'][:60]}` — avg={r['avg_return_2h']}%, "
            f"best={r['best_regime']}, worst={r['worst_regime']}, tags={r['status_tags']}",
        )

    lines.extend(
        [
            "",
            "## Clusters",
            "",
            "| Cluster | Rules | Mean avg 2h | Top rule | Regime affinity |",
            "|---------|-------|-------------|----------|-----------------|",
        ],
    )
    for c in cluster_summary:
        lines.append(
            f"| {c['cluster_id']} | {c['rule_count']} | {c['avg_return_mean']} | "
            f"{c['top_rule_id'][:40]} | {c['regime_affinity']} |",
        )

    lines.extend(
        [
            "",
            "## Portfolio design",
            "",
            "At runtime the engine should:",
            "1. Classify scan regime + volatility band (scan-time only)",
            "2. Select rule from `rule_metadata.csv` matching preferred regime",
            "3. Fall back to `execution_score_v1` when sample_size < threshold or avoid_regime matches",
            "",
            "## Caveats",
            "",
            "- Small calendar window — cluster and regime labels are **hypothesis-level**",
            "- Generalization test **REJECT**ed the top discovered rule on bear/sideway negatives",
            "- Regime router uses in-sample best-rule-per-regime (upper-bound estimate, not blind)",
            "",
            f"_Generated {datetime.now(KST).isoformat()}_",
        ],
    )
    return "\n".join(lines)


def flatten_library_rows(profiles: list[dict]) -> list[dict]:
    return [_library_row(p) for p in profiles]
