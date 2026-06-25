"""Directional DNA discovery report."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

KST = timezone(timedelta(hours=9))


def build_dna_report(
    dna_summaries: list[dict],
    global_importance: list[dict],
    cluster_stats: list[dict],
    validation_rows: list[dict],
    live_candidates: list[dict],
    meta: dict,
) -> str:
    lines = [
        "# SCOUT Directional DNA Discovery V1 Report",
        "",
        f"Generated: {datetime.now(KST).strftime('%Y-%m-%d %H:%M:%S')} KST",
        f"Validation scans: {meta.get('validation_scans', 0)} | Train: {meta.get('train_scans', 0)} | Blind: {meta.get('blind_scans', 0)}",
        "",
        "## Pattern DNA Summary",
    ]
    for d in dna_summaries:
        lines.append(
            f"- **{d['engine']}**: n={d['sample_count']} success={d['success_rate']}% "
            f"sig_features={d['significant_features']} clusters={d.get('cluster_count', 0)}"
        )
        lines.append(f"  top DNA: {', '.join(d.get('top_features', [])[:5])}")

    lines.extend(["", "## Global Feature Importance TOP20"])
    for r in global_importance[:20]:
        lines.append(
            f"- #{r.get('importance_rank')} `{r['feature']}` effect={r['effect_size']} "
            f"delta={r['delta']} p={r['p_approx']} sig={r.get('significant')}"
        )

    lines.extend(["", "## Cluster Performance"])
    for c in cluster_stats:
        lines.append(
            f"- {c['formula_name']}: n={c['sample_count']} avg2h={c['avg_return_2h']}% "
            f"win={c['win_rate']}% PF={c['profit_factor']} trap={c['trap_rate']}%"
        )

    lines.extend(["", "## Blind Validation vs Random / Pattern Champion"])
    for v in validation_rows:
        if v.get("split") != "blind":
            continue
        lines.append(
            f"- {v['engine']} / {v['formula_name']}: avg2h={v['avg_return_2h']}% "
            f"Δrandom={v.get('delta_vs_random')} Δchampion={v.get('delta_vs_champion', 0)}"
        )

    lines.extend(["", "## LIVE Cluster Candidates (research only — not auto-applied)"])
    if live_candidates:
        for lc in live_candidates:
            lines.append(
                f"- `{lc['formula_name']}` engine={lc['engine']} blind_avg2h={lc['blind_avg2h']}% "
                f"tier={lc['tier']} reason={lc['reason']}"
            )
    else:
        lines.append("- None passed blind improvement gates yet")

    lines.extend([
        "",
        "## Principles",
        "- No future data in features or clustering train",
        "- No manual feature selection — all ranked by statistics",
        "- Long/Short researched independently",
        "- Zero-base: cluster formulas compete on blind holdout only",
        "",
        "*Research/Lab only. LIVE engines unchanged.*",
    ])
    return "\n".join(lines)
