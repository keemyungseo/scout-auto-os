"""Execution Engine V1 report."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

KST = timezone(timedelta(hours=9))


def build_execution_report(
    meta: dict,
    blind_rows: list[dict],
    combined_blind: dict,
) -> str:
    lines = [
        "# Execution Engine V1",
        "",
        "Top5 observe → **Execution Score** → Top2 execute.",
        "Search rules unchanged — execution layer only.",
        "",
        "## Method",
        "",
        "- Pipeline: Direction Champion → Entry Rule V2 → Entry Score → Top5 PASS",
        f"- Observation: first **{meta.get('observation_minutes')}m** bar after scan (bundle resolution)",
        "- Execution features: obs return, volume, VWAP dev, ATR expansion, breakout, false-breakout penalty",
        f"- Weights tuned on train scans ({meta.get('train_scan_count')}), blind on {meta.get('blind_scan_count')}",
        "- Forward 2h return: **evaluation only**",
        "",
        "## Blind comparison",
        "",
        "| Strategy | Direction | Trades | Avg 2h | Win% | Lift vs Top5 | Lift vs Entry Top2 |",
        "|----------|-----------|--------|--------|------|--------------|-------------------|",
    ]

    for r in blind_rows:
        lift5 = r.get("lift_vs_top5_pct", "")
        lift2 = r.get("lift_vs_entry_top2_pct", "")
        if r["strategy"] != "top2_execution":
            lift5 = lift2 = ""
        lines.append(
            f"| {r['strategy']} | {r['direction']} | {r['trade_count']} | {r['avg_return_2h']} | "
            f"{r['win_rate_pct']} | {lift5} | {lift2} |",
        )

    lines.extend(
        [
            "",
            "## Combined blind (long + short)",
            "",
            f"- Top5 all avg 2h: **{combined_blind.get('top5_avg')}**",
            f"- Top2 entry-score avg 2h: **{combined_blind.get('entry_top2_avg')}**",
            f"- Top2 execution avg 2h: **{combined_blind.get('exec_top2_avg')}**",
            f"- Execution lift vs Top5: **{combined_blind.get('lift_vs_top5_pct')}%**",
            f"- Execution lift vs entry Top2: **{combined_blind.get('lift_vs_entry_top2_pct')}%**",
            "",
            "## Interpretation",
            "",
            "Positive lift on blind suggests execution timing adds value beyond search ranking.",
            "Unknown / negative lift is valid — keep Entry Top2 until more observation data exists.",
            "",
            f"_Generated {datetime.now(KST).isoformat()}_",
        ],
    )

    try:
        from season2_scout_mission import evaluate_convergence, mission_summary_lines

        conv = evaluate_convergence(
            "functional_role",
            improves=["relative_ranking_between_candidates", "trend_persistence_estimation"],
            sample_size=int(meta.get("total_picks", 0)),
            confidence="medium" if meta.get("blind_scan_count", 0) >= 15 else "hypothesis",
        )
        lines.append(
            f"**Convergence tier:** {conv['tier']} | {', '.join(conv['convergence_criteria_met']) or 'background'}",
        )
        lines.extend(mission_summary_lines())
    except ImportError:
        pass

    return "\n".join(lines)
