"""Markdown report for Signal Lifecycle Engine V1."""

from __future__ import annotations

import statistics
from collections import defaultdict

from scout_auto_os.engine.research.signal_lifecycle.constants import (
    BAR_MINUTES,
    MIN_TRACK_HOURS,
    PREFERRED_TRACK_HOURS,
)


def _mean(vals: list[float]) -> float | None:
    return round(statistics.mean(vals), 4) if vals else None


def _pct(n: int, d: int) -> float:
    return round(n / d * 100, 2) if d else 0.0


def aggregate_clusters(shapes: list[dict]) -> list[dict]:
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in shapes:
        buckets[(row["direction"], row["lifecycle_label"])].append(row)

    clusters: list[dict] = []
    for (direction, label), rows in sorted(buckets.items()):
        n = len(rows)
        clusters.append(
            {
                "direction": direction,
                "lifecycle_label": label,
                "signal_count": n,
                "avg_mfe_pct": _mean([float(r["mfe_full"]) for r in rows]),
                "avg_mae_pct": _mean([float(r["mae_full"]) for r in rows]),
                "avg_peak_time_min": _mean([float(r["peak_time_min"]) for r in rows]),
                "avg_end_time_min": _mean([float(r["track_hours"]) * 60 for r in rows]),
                "avg_return_2h": _mean([float(r["return_2h"]) for r in rows]),
                "avg_return_6h": _mean([float(r["return_6h"]) for r in rows]),
                "avg_return_shift_2h_to_6h": _mean([float(r["return_shift_2h_to_6h"]) for r in rows]),
                "success_2h_rate_pct": _pct(sum(1 for r in rows if r["success_2h"]), n),
                "success_6h_rate_pct": _pct(sum(1 for r in rows if r["success_6h"]), n),
                "sustained_winner_pct": _pct(
                    sum(1 for r in rows if r["evaluation_shift"] == "sustained_winner"), n,
                ),
                "fade_after_2h_pct": _pct(
                    sum(1 for r in rows if r["evaluation_shift"] == "winner_2h_fade_6h"), n,
                ),
                "late_runner_6h_pct": _pct(
                    sum(1 for r in rows if r["evaluation_shift"] == "loser_2h_runner_6h"), n,
                ),
            },
        )
    return clusters


def build_lifecycle_report(
    meta: dict,
    lifecycle_rows: list[dict],
    shapes: list[dict],
    clusters: list[dict],
) -> str:
    lines = [
        "# Signal Lifecycle Engine V1",
        "",
        "Research-only lifecycle analysis of Direction Champion signals.",
        "No rules, thresholds, or predictions were modified.",
        "",
        "## Method",
        "",
        f"- Resolution: **{BAR_MINUTES}m bars** (forward bundle; target was 5m — finer data not in bundle)",
        f"- Minimum track: **{MIN_TRACK_HOURS}h**; preferred: **{PREFERRED_TRACK_HOURS}h** when bars available",
        "- Labels emerge from measured trajectory shape (MFE/MAE, peak timing, 2h vs 6h return shift)",
        "- Confidence: descriptive cohort statistics only",
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
        "",
        "## 2h vs 6h evaluation shift (all signals)",
        "",
    ]

    for direction in ("long", "short"):
        sub = [s for s in shapes if s["direction"] == direction]
        if not sub:
            continue
        n = len(sub)
        lines.extend(
            [
                f"### {direction.upper()}",
                "",
                f"- Sustained winner (>=3% at 2h and 6h): {_pct(sum(1 for s in sub if s['evaluation_shift']=='sustained_winner'), n)}%",
                f"- Winner at 2h, fade by 6h: {_pct(sum(1 for s in sub if s['evaluation_shift']=='winner_2h_fade_6h'), n)}%",
                f"- Under 3% at 2h, runner by 6h: {_pct(sum(1 for s in sub if s['evaluation_shift']=='loser_2h_runner_6h'), n)}%",
                f"- Avg return shift (6h - 2h): {_mean([float(s['return_shift_2h_to_6h']) for s in sub])}%",
                "",
            ],
        )

    lines.extend(["## Lifecycle type summary", ""])

    for direction in ("long", "short"):
        dir_clusters = [c for c in clusters if c["direction"] == direction]
        if not dir_clusters:
            continue
        lines.append(f"### {direction.upper()}")
        lines.append("")
        lines.append(
            "| Label | N | Avg MFE | Avg peak (min) | Avg 2h | Avg 6h | 2h win% | 6h win% | Fade 2h->6h% |",
        )
        lines.append("|-------|---|---------|----------------|--------|--------|---------|---------|--------------|")
        for c in sorted(dir_clusters, key=lambda x: -x["signal_count"]):
            lines.append(
                f"| {c['lifecycle_label']} | {c['signal_count']} | "
                f"{c['avg_mfe_pct']} | {c['avg_peak_time_min']} | "
                f"{c['avg_return_2h']} | {c['avg_return_6h']} | "
                f"{c['success_2h_rate_pct']} | {c['success_6h_rate_pct']} | "
                f"{c['fade_after_2h_pct']} |",
            )
        lines.append("")

    lines.extend(_scout_findings(shapes, clusters))
    lines.extend(_mission_lines(meta))
    return "\n".join(lines)


def _scout_findings(shapes: list[dict], clusters: list[dict]) -> list[str]:
    lines = [
        "## What SCOUT finds vs misses (empirical)",
        "",
        "Interpretation stays probabilistic — correlation with champion ranking, not causation.",
        "",
    ]
    for direction in ("long", "short"):
        sub = [c for c in clusters if c["direction"] == direction]
        if not sub:
            continue
        strong = sorted(sub, key=lambda x: -(x["success_6h_rate_pct"] or 0))[:3]
        weak = sorted(sub, key=lambda x: (x["success_6h_rate_pct"] or 0))[:3]
        lines.append(f"### {direction.upper()} — relatively strong lifecycle shapes")
        for c in strong:
            if c["signal_count"] < 3:
                continue
            lines.append(
                f"- **{c['lifecycle_label']}** (n={c['signal_count']}): "
                f"6h success {_pct_rate(c['success_6h_rate_pct'])}%, avg MFE {c['avg_mfe_pct']}%",
            )
        lines.append("")
        lines.append(f"### {direction.upper()} — weak / trap-prone shapes")
        for c in weak:
            if c["lifecycle_label"] in ("Dead Signal", "Fake Breakout") or (c["fade_after_2h_pct"] or 0) > 20:
                lines.append(
                    f"- **{c['lifecycle_label']}** (n={c['signal_count']}): "
                    f"fade after 2h {_pct_rate(c['fade_after_2h_pct'])}%, avg 6h {c['avg_return_6h']}%",
                )
        lines.append("")
    return lines


def _pct_rate(v: float | None) -> str:
    return f"{v:.1f}" if v is not None else "?"


def _mission_lines(meta: dict) -> list[str]:
    try:
        from season2_scout_mission import evaluate_convergence, mission_summary_lines

        conv = evaluate_convergence(
            "signal_lifecycle",
            improves=["trend_persistence_estimation", "real_vs_fake_trend_discrimination"],
            sample_size=int(meta.get("long_signal_count", 0)) + int(meta.get("short_signal_count", 0)),
            confidence="medium" if meta.get("scan_count", 0) >= 30 else "hypothesis",
        )
        lines = [
            "## Mission convergence",
            "",
            f"**Convergence tier:** {conv['tier']} | "
            f"{', '.join(conv['convergence_criteria_met']) or 'background'}",
            "",
        ]
        lines.extend(mission_summary_lines())
        lines.append("")
        return lines
    except ImportError:
        return []
