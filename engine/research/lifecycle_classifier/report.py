"""Lifecycle Classifier V1 report."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from scout_auto_os.engine.research.lifecycle_classifier.constants import LABEL_SHORT, PROB_DISPLAY_ORDER

KST = timezone(timedelta(hours=9))


def _readiness_tier(macro_f1: float, fake_recall: float, continuous_precision: float) -> str:
    if macro_f1 >= 0.45 and fake_recall >= 0.35 and continuous_precision >= 0.40:
        return "medium — differentiated holding hypotheses plausible"
    if macro_f1 >= 0.30 and (fake_recall >= 0.25 or continuous_precision >= 0.30):
        return "hypothesis — weak separation; holding split not yet reliable"
    return "insufficient — entry features do not yet separate lifecycle types reliably"


def build_classifier_report(
    meta: dict,
    long_metrics: dict,
    short_metrics: dict,
    long_binary: dict,
    short_binary: dict,
) -> str:
    lines = [
        "# Lifecycle Classifier V1 Report",
        "",
        "Entry-time **lifecycle type classification** only.",
        "No price prediction, return regression, or expected return.",
        "",
        "## Method",
        "",
        "- **Input:** scan-time features (DNA, pattern, cluster score, rule margin, entry score, etc.)",
        "- **Target:** post-hoc lifecycle label from Signal Lifecycle Engine (used only as training label)",
        "- **Model:** multinomial logistic regression with class balancing (numpy)",
        "- **Split:** temporal train/validation by scan time",
        "",
        "## Sample",
        "",
        f"| Field | Value |",
        f"|-------|-------|",
        f"| Date range | {meta.get('date_min')} .. {meta.get('date_max')} |",
        f"| Total signals | {meta.get('total_signals')} |",
        f"| Train scans | {meta.get('train_scan_count')} |",
        f"| Validation scans | {meta.get('val_scan_count')} |",
        f"| Feature dimensions | {meta.get('feature_dim')} |",
        f"| Long classes (train) | {meta.get('long_class_count')} |",
        f"| Short classes (train) | {meta.get('short_class_count')} |",
        "",
        "## Validation metrics",
        "",
    ]

    for direction, m, binary in (("LONG", long_metrics, long_binary), ("SHORT", short_metrics, short_binary)):
        agg = m["aggregate"]
        lines.extend(
            [
                f"### {direction}",
                "",
                f"- Accuracy: **{m['accuracy']}** | Top-2 accuracy: **{m['top2_accuracy']}**",
                f"- Macro F1: **{agg['macro_f1']}** | Weighted F1: **{agg['weighted_f1']}**",
                f"- Macro precision: {agg['macro_precision']} | Macro recall: {agg['macro_recall']}",
                "",
                "| Label | Support | Precision | Recall | F1 |",
                "|-------|---------|-----------|--------|-----|",
            ],
        )
        for row in m["per_class"]:
            if row["support"] == 0:
                continue
            lines.append(
                f"| {row['label']} | {row['support']} | {row['precision']} | "
                f"{row['recall']} | {row['f1']} |",
            )
        lines.extend(
            [
                "",
                f"**Fake vs Continuous binary view** (pred={binary['positive']}): "
                f"precision={binary['precision']} recall={binary['recall']} f1={binary['f1']}",
                "",
            ],
        )

    long_tier = _readiness_tier(
        long_metrics["aggregate"]["macro_f1"],
        long_binary["recall"] if long_binary["positive"] == "Fake Breakout" else 0,
        next((r["precision"] for r in long_metrics["per_class"] if r["label"] == "Continuous Trend"), 0),
    )
    short_tier = _readiness_tier(
        short_metrics["aggregate"]["macro_f1"],
        short_binary["recall"],
        next((r["precision"] for r in short_metrics["per_class"] if r["label"] == "Continuous Trend"), 0),
    )

    lines.extend(
        [
            "## Holding-strategy readiness (probabilistic)",
            "",
            "Can entry-time features support **different holding strategies** by lifecycle type?",
            "",
            f"- **Long:** {long_tier}",
            f"- **Short:** {short_tier}",
            "",
            "Interpretation stays non-operational until out-of-sample validation on new regimes.",
            "Unknown is valid when macro F1 remains near majority-class baseline.",
            "",
            "## Probability output columns",
            "",
            "Per signal: `prob_" + "`, `prob_".join(LABEL_SHORT[l] for l in PROB_DISPLAY_ORDER) + "`",
            "",
            f"_Generated {datetime.now(KST).isoformat()}_",
        ],
    )

    try:
        from season2_scout_mission import evaluate_convergence, mission_summary_lines

        conv = evaluate_convergence(
            "behaviour_grammar",
            improves=["real_vs_fake_trend_discrimination", "trend_persistence_estimation"],
            sample_size=int(meta.get("total_signals", 0)),
            confidence="medium" if meta.get("val_scan_count", 0) >= 20 else "hypothesis",
        )
        lines.append(f"**Convergence tier:** {conv['tier']} | {', '.join(conv['convergence_criteria_met']) or 'background'}")
        lines.extend(mission_summary_lines())
    except ImportError:
        pass

    return "\n".join(lines)
