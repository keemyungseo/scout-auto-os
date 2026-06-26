"""Online State Machine V1 report."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from scout_auto_os.engine.research.online_state_machine.constants import BAR_MINUTES
from scout_auto_os.engine.research.online_state_machine.diagram import build_mermaid_diagram, top_transition_paths

KST = timezone(timedelta(hours=9))


def build_state_report(
    meta: dict,
    matrix_rows: list[dict],
    state_stats: list[dict],
) -> str:
    lines = [
        "# Online State Machine V1",
        "",
        "Post-entry **online state estimation** — no entry prediction, no price forecast.",
        "State updates causally as each bar closes (simulated on forward bundle for research labels).",
        "",
        "## Method",
        "",
        f"- Update cadence: **{BAR_MINUTES}m bars** (bundle resolution; design target was 5m)",
        "- Features at each step: body, range, volume, ATR, momentum, slope, MFE, MAE, drawdown, acceleration",
        "- State rules are empirical thresholds on **observable-to-date** dynamics only",
        "",
        "## Sample",
        "",
        f"| Field | Value |",
        f"|-------|-------|",
        f"| Date range | {meta.get('date_min')} .. {meta.get('date_max')} |",
        f"| Scans | {meta.get('scan_count')} |",
        f"| Long signals | {meta.get('long_signal_count')} |",
        f"| Short signals | {meta.get('short_signal_count')} |",
        f"| Timeline rows | {meta.get('timeline_row_count')} |",
        f"| Transitions | {meta.get('transition_count')} |",
        "",
    ]

    for direction in ("long", "short"):
        lines.extend(
            [
                f"## Top transitions — {direction.upper()}",
                "",
            ],
        )
        for path in top_transition_paths(matrix_rows, direction):
            lines.append(f"- {path}")
        lines.append("")

        dir_stats = [s for s in state_stats if s["direction"] == direction]
        if dir_stats:
            lines.extend(
                [
                    f"## State statistics — {direction.upper()}",
                    "",
                    "| State | Obs | Avg return | Avg duration (min) | Top next |",
                    "|-------|-----|------------|------------------|----------|",
                ],
            )
            for s in sorted(dir_stats, key=lambda x: -int(x["observation_count"] or 0)):
                lines.append(
                    f"| {s['state']} | {s['observation_count']} | {s['avg_return_pct']} | "
                    f"{s['avg_duration_min']} | {s['next_state_top']} |",
                )
            lines.append("")

    lines.extend(
        [
            "## State transition diagram — LONG",
            "",
            build_mermaid_diagram(matrix_rows, "long"),
            "",
            "## State transition diagram — SHORT",
            "",
            build_mermaid_diagram(matrix_rows, "short"),
            "",
            "## Interpretation",
            "",
            "- Transition probabilities describe **historical cohort behaviour**, not forecasts.",
            "- Suitable for **holding-policy hypotheses** (extend in HEALTHY/ACCELERATION, tighten in EXHAUSTION/FAKE).",
            "- Operational use requires live bar feed at intended cadence.",
            "",
            f"_Generated {datetime.now(KST).isoformat()}_",
        ],
    )

    try:
        from season2_scout_mission import evaluate_convergence, mission_summary_lines

        conv = evaluate_convergence(
            "situation_evolution",
            improves=["trend_persistence_estimation", "real_vs_fake_trend_discrimination"],
            sample_size=int(meta.get("long_signal_count", 0)) + int(meta.get("short_signal_count", 0)),
            confidence="medium" if meta.get("scan_count", 0) >= 30 else "hypothesis",
        )
        lines.append(
            f"**Convergence tier:** {conv['tier']} | {', '.join(conv['convergence_criteria_met']) or 'background'}",
        )
        lines.extend(mission_summary_lines())
    except ImportError:
        pass

    return "\n".join(lines)
