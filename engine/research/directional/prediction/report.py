"""Markdown reports for Cluster Prediction Engine V1."""

from __future__ import annotations

from season2_scout_mission import evaluate_convergence, mission_summary_lines

from scout_auto_os.engine.research.directional.prediction.picker import (
    LONG_DIRECTION_CHAMPION,
    SHORT_DIRECTION_CHAMPION,
)


def _convergence_footer(sample_count: int) -> list[str]:
    conv = evaluate_convergence(
        "interaction_mining",
        improves=["relative_ranking_between_candidates", "trend_persistence_estimation"],
        sample_size=sample_count,
        confidence="medium" if sample_count >= 50 else "hypothesis",
    )
    return [
        "",
        f"**Convergence tier:** {conv['tier']} | criteria: {', '.join(conv['convergence_criteria_met']) or 'none yet'}",
        *mission_summary_lines(),
    ]


def build_cluster_prediction_report(
    sample_predictions: list[dict],
    meta: dict,
) -> str:
    lines = [
        "# Cluster Prediction Report (V1)",
        "",
        "## Mission",
        "- Scan-time P(cluster) from DNA cluster formulas (features only, no future data).",
        "- Expected return = Σ P(cluster) × train cluster avg2h.",
        "- Research / Lab only — does not modify LIVE logic.",
        "",
        "## Configuration",
        f"- Validation scans: {meta.get('validation_scans')}",
        f"- Blind scans: {meta.get('blind_scans')}",
        f"- Long formulas: {meta.get('long_formula_count')}",
        f"- Short formulas: {meta.get('short_formula_count')}",
        f"- Best long cluster (blind): {meta.get('best_long_cluster')}",
        f"- Best short cluster (blind): {meta.get('best_short_cluster')}",
        "",
        "## Example predictions (latest scan sample)",
    ]
    for row in sample_predictions[:5]:
        lines.extend([
            "",
            f"### {row['symbol']} @ {row['scan_time_kst']}",
            f"- Long score **{row['long_score']}** | expected **{row['long_expected_return_2h']:+.2f}%** | cluster `{row['top_long_cluster']}` **{row['long_probability_pct']}%**",
            f"- Short score **{row['short_score']}** | expected **{row['short_expected_return_2h']:+.2f}%** | cluster `{row['top_short_cluster']}` **{row['short_probability_pct']}%**",
            f"- Recommend **{row['recommended_direction']}** | confidence **{row['confidence_pct']}%** | holding ~{row['expected_holding_min']}m",
        ])
    lines.extend(_convergence_footer(meta.get("blind_scans", 0)))
    return "\n".join(lines)


def build_prediction_engine_report(
    comparison: list[dict],
    slot_summary: dict,
    meta: dict,
) -> str:
    lines = [
        "# Prediction Engine Report (V1)",
        "",
        "## Blind comparison (same top-k, same scans)",
        "",
        "| Method | Direction | Samples | avg2h | win% | trap% | score |",
        "|--------|-----------|---------|-------|------|-------|-------|",
    ]
    for row in comparison:
        lines.append(
            f"| {row['method']} | {row['direction']} | {row['sample_count']} | "
            f"{row['avg_return_2h']} | {row['win_rate']} | {row['trap_rate']} | {row['score']} |"
        )

    lines.extend([
        "",
        "## 3+3 slot simulation (prediction engine)",
        f"- Long slots: {slot_summary.get('long_slots')} | picks={slot_summary.get('long_picks')} | avg2h={slot_summary.get('long_avg_return_2h')}%",
        f"- Short slots: {slot_summary.get('short_slots')} | picks={slot_summary.get('short_picks')} | avg2h={slot_summary.get('short_avg_return_2h')}%",
        f"- Combined avg2h: **{slot_summary.get('combined_avg_return_2h')}%** (n={slot_summary.get('combined_sample_count')})",
        "",
        "## Baselines compared",
        f"- Zero-Base Champion: A6 (long) / MOMENTUM proxy (short)",
        f"- Direction Champion: {LONG_DIRECTION_CHAMPION} / {SHORT_DIRECTION_CHAMPION}",
        "- Cluster Champion: best blind cluster formula per direction",
        "- Prediction Engine: rank by long_score / short_score",
        "",
        f"_Generated {meta.get('generated_at')}_",
    ])
    lines.extend(_convergence_footer(slot_summary.get("combined_sample_count", 0)))
    return "\n".join(lines)
