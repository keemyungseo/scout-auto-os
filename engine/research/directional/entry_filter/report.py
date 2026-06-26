"""Markdown report for Direction Champion Entry DNA."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from season2_scout_mission import evaluate_convergence, mission_summary_lines

from scout_auto_os.engine.research.directional.entry_filter.constants import (
    LONG_DIRECTION_CHAMPION,
    SHORT_DIRECTION_CHAMPION,
)

KST = timezone(timedelta(hours=9))


def _feature_lines(importance: list[dict], limit: int = 15) -> list[str]:
    lines: list[str] = []
    for r in importance[:limit]:
        arrow = "↑ winner higher" if r.get("winner_favors_higher") else "↓ winner lower"
        sig = "*" if r.get("significant") else ""
        lines.append(
            f"- #{r['importance_rank']}{sig} `{r['feature']}` "
            f"effect={r['effect_size']} Δ={r['delta']} "
            f"(W={r['winner_mean']} L={r['loser_mean']}) {arrow}"
        )
    return lines


def build_entry_dna_report(
    meta: dict,
    long_split: dict,
    short_split: dict,
    long_profile: dict,
    short_profile: dict,
    long_importance: list[dict],
    short_importance: list[dict],
    common_dna: list[dict],
) -> str:
    lines = [
        "# Direction Champion Entry DNA Report (V1)",
        "",
        "## Purpose",
        "- **Not prediction** — distinguish good vs bad **entries** after Direction Champion selection.",
        "- Basis for **Entry Filter Engine V1** (research only; LIVE not modified).",
        "",
        "## Data window",
        f"- Lookback: **{meta.get('lookback_months')} months** (from latest scan in bundle)",
        f"- Date range: {meta.get('date_min')} → {meta.get('date_max')}",
        f"- Scans used: {meta.get('scan_count')}",
        f"- Champion engines: `{LONG_DIRECTION_CHAMPION}` / `{SHORT_DIRECTION_CHAMPION}`",
        f"- Picks per scan: top **{meta.get('top_k')}** (direction champion standard)",
        "",
        "## Signal counts",
        f"- Long signals: **{meta.get('long_signal_count')}** | Short signals: **{meta.get('short_signal_count')}**",
        "",
        "### Long winner/loser split (by return_2h)",
        f"- Winners (top 20%): n={long_split.get('winner_count')} threshold≥{long_split.get('winner_threshold')}%",
        f"- Losers (bottom 20%): n={long_split.get('loser_count')} threshold≤{long_split.get('loser_threshold')}%",
        f"- Median return_2h: {long_split.get('median_return_2h')}%",
        "",
        "### Short winner/loser split (by short return_2h)",
        f"- Winners: n={short_split.get('winner_count')} threshold≥{short_split.get('winner_threshold')}%",
        f"- Losers: n={short_split.get('loser_count')} threshold≤{short_split.get('loser_threshold')}%",
        f"- Median return_2h: {short_split.get('median_return_2h')}%",
        "",
        "## Long Winner DNA (feature importance)",
    ]
    lines.extend(_feature_lines(long_importance))
    lines.extend([
        "",
        f"Winner favors **higher**: {', '.join(f'`{f}`' for f in long_profile.get('winner_higher_features', [])[:8])}",
        f"Winner favors **lower**: {', '.join(f'`{f}`' for f in long_profile.get('winner_lower_features', [])[:8])}",
        "",
        "## Short Winner DNA",
    ])
    lines.extend(_feature_lines(short_importance))
    lines.extend([
        "",
        f"Winner favors **higher**: {', '.join(f'`{f}`' for f in short_profile.get('winner_higher_features', [])[:8])}",
        f"Winner favors **lower**: {', '.join(f'`{f}`' for f in short_profile.get('winner_lower_features', [])[:8])}",
        "",
        "## Common DNA (both directions)",
    ])
    if common_dna:
        for r in common_dna[:12]:
            same = "same sign" if r.get("same_winner_direction") else "opposite sign"
            lines.append(
                f"- #{r['common_rank']} `{r['feature']}` combined_effect={r['combined_effect']} "
                f"long={r['long_effect_size']} short={r['short_effect_size']} ({same})"
            )
    else:
        lines.append("- No overlapping top features (sample may be small)")

    lines.extend([
        "",
        "## Entry Filter V1 hypotheses (probabilistic)",
        "- Filter candidates where winner-DNA features align (direction-specific).",
        "- Reject entries matching loser-DNA profile (high trap correlation — verify empirically).",
        "- Long and Short filters remain **independent**.",
        "",
        f"_Generated {datetime.now(KST).isoformat()}_",
    ])

    conv = evaluate_convergence(
        "situation_archetype",
        improves=["real_vs_fake_trend_discrimination", "relative_ranking_between_candidates"],
        sample_size=int(meta.get("long_signal_count", 0)) + int(meta.get("short_signal_count", 0)),
        confidence="medium" if meta.get("scan_count", 0) >= 30 else "hypothesis",
    )
    lines.append(f"**Convergence tier:** {conv['tier']} | {', '.join(conv['convergence_criteria_met']) or 'background'}")
    lines.extend(mission_summary_lines())
    return "\n".join(lines)
