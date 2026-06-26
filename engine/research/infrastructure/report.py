"""Research dashboard and report builders."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

KST = timezone(timedelta(hours=9))


def build_decision(
    status: dict,
    readiness: dict,
    quality: dict,
    gaps: list[dict],
) -> dict:
    days = int(status.get("calendar_days", 0))
    days_needed = int(readiness.get("days_remaining_90d", 90))

    weak = ", ".join(
        f"{g['regime_axis']}/{g['regime']} (need +{g['gap_scans_needed']})"
        for g in gaps[:5]
    ) or "none on current axes"

    return {
        "q1_90d_validation": (
            f"**{'YES' if readiness.get('can_validate_90d') else 'NO'}** - "
            f"{days}d available, need {90 - days}d more for 90d blind validation."
        ),
        "q2_regime_gaps": f"**{len(gaps)} insufficient buckets** - weakest: {weak}.",
        "q3_days_to_live": (
            f"**{days_needed} calendar days** (+ regime coverage) before LIVE validation at 90d target. "
            f"Recommend {max(days_needed, 30)}d shadow accumulation minimum."
        ),
        "q4_auto_accumulation": (
            "**YES** - SQLite history DB + append_scan API + forward_labeler "
            "can grow dataset without new research code. Run runner after each scan batch."
        ),
        "auto_ready": quality.get("passed", False) and days >= 15,
    }


def build_research_dashboard(
    status: dict,
    calendar: list[dict],
    coverage: list[dict],
    quality: dict,
    decision: dict,
) -> str:
    lines = [
        "# SCOUT Research Infrastructure V1 — Dashboard",
        "",
        "Automatic blind dataset builder — frozen constitution, no new research.",
        "",
        "## Dataset status",
        "",
        f"- Version: **{status.get('dataset_version')}**",
        f"- Calendar: **{status.get('start_date')}** to **{status.get('end_date')}** ({status.get('calendar_days')}d)",
        f"- Scans: **{status.get('scan_count')}** | Samples: **{status.get('sample_count')}** | Labeled: **{status.get('labeled_count')}**",
        f"- Label coverage: **{status.get('label_coverage_pct')}%**",
        "",
        "## Validation windows",
        "",
        "| Window | Coverage | Scans | Ready |",
        "|--------|----------|-------|-------|",
    ]
    for c in calendar:
        lines.append(
            f"| {c['window_days']}d | {c['coverage_pct']}% | {c['scan_count']} | "
            f"{'YES' if c['validation_ready'] else 'NO'} |",
        )

    lines.extend([
        "",
        "## Regime coverage",
        "",
        "| Axis | Regime | Scans | Sufficient |",
        "|------|--------|-------|------------|",
    ])
    for r in coverage:
        suff = "YES" if r["sufficient"] else f"NO (+{r['gap_scans_needed']})"
        lines.append(
            f"| {r['regime_axis']} | {r['regime']} | {r['scan_count']} | {suff} |",
        )

    lines.extend([
        "",
        "## Integrity",
        "",
        f"- Quality: **{quality.get('summary')}**",
        f"- Duplicates: {quality.get('duplicate_count')} | Leak features: {quality.get('leak_feature_count')} | Label errors: {quality.get('label_error_count')}",
        "",
        "## Final questions",
        "",
        f"1. 90d validation possible? {decision.get('q1_90d_validation')}",
        f"2. Regime gaps? {decision.get('q2_regime_gaps')}",
        f"3. Days to LIVE validation? {decision.get('q3_days_to_live')}",
        f"4. Auto accumulation? {decision.get('q4_auto_accumulation')}",
        "",
        f"_Updated {datetime.now(KST).isoformat()}_",
    ])
    return "\n".join(lines)


def build_quality_report(quality: dict) -> str:
    lines = [
        "# Research Infrastructure — Quality Report",
        "",
        f"**{quality.get('summary')}**",
        "",
        f"- Duplicates: {quality.get('duplicate_count')}",
        f"- Missing label scans: {quality.get('missing_label_scans')}",
        f"- Leak features (sampled): {quality.get('leak_feature_count')}",
        f"- Label consistency errors: {quality.get('label_error_count')}",
        f"- Orphan labels: {quality.get('orphan_label_count')}",
        "",
    ]
    if quality.get("issues"):
        lines.append("## Sample issues")
        lines.append("")
        for issue in quality["issues"][:20]:
            lines.append(f"- `{issue.get('check')}` scan={issue.get('scan_kst')} sym={issue.get('symbol', '')}")
    return "\n".join(lines)
