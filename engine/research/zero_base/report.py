"""Generate zero_base_report.md."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

KST = timezone(timedelta(hours=9))


def build_zero_base_report(
    candidate_results: list[dict],
    champion_board: list[dict],
    random_stats: dict,
    feature_diag: list[dict],
    meta: dict,
) -> str:
    a6 = next((c for c in candidate_results if c.get("engine") == "A6_CURRENT"), {})
    better = [c for c in champion_board if c.get("beats_a6_return")]
    worse = [c for c in champion_board if not c.get("beats_a6_return") and c.get("engine") != "A6_CURRENT"]
    promising = [f for f in feature_diag if f.get("verdict") == "promising"]
    discard = [f for f in feature_diag if f.get("verdict") == "discard"]

    lines = [
        "# SCOUT Zero-Base Discovery Report",
        "",
        f"Generated: {datetime.now(KST).strftime('%Y-%m-%d %H:%M:%S')} KST",
        f"Split: train<{meta.get('train_cutoff', '2026-06-01')} | validation>={meta.get('train_cutoff')}",
        f"Validation scans: {meta.get('validation_scans', 0)} | Train scans: {meta.get('train_scans', 0)}",
        f"Random draws per scan: {meta.get('random_draws', 100)}",
        "",
        "## A6 Baseline (validation)",
        f"- avg_return_2h: {a6.get('avg_return_2h', 0)}%",
        f"- win_rate: {a6.get('win_rate', 0)}%",
        f"- trap_rate: {a6.get('trap_rate', 0)}%",
        f"- sample_count: {a6.get('sample_count', 0)}",
        "",
        "## Random Baseline (validation)",
        f"- avg_return_2h: {random_stats.get('avg_return_2h', 0)}% (std={random_stats.get('return_2h_std', 0)})",
        f"- win_rate: {random_stats.get('win_rate', 0)}%",
        f"- trap_rate: {random_stats.get('trap_rate', 0)}%",
        f"- big_winner_capture: {random_stats.get('big_winner_capture_rate', 0)}%",
        "",
        "## Candidate Engine Rankings (Champion Board)",
    ]
    for row in champion_board[:15]:
        lines.append(
            f"- #{row.get('board_rank')} **{row.get('engine')}** "
            f"score={row.get('score')} avg2h={row.get('avg_return_2h')}% "
            f"win={row.get('win_rate')}% trap={row.get('trap_rate')}% "
            f"[{row.get('tier')}] champion={row.get('champion_eligible')}"
        )

    lines.extend(["", "## Better than A6 (validation avg_return_2h)"])
    if better:
        for c in better[:10]:
            lines.append(
                f"- {c.get('engine')}: +{c.get('avg_return_2h_delta_vs_a6', 0)}% vs A6 "
                f"(n={c.get('sample_count')})"
            )
    else:
        lines.append("- None met A6 avg_return_2h on validation set")

    lines.extend(["", "## Worse than A6"])
    for c in worse[:8]:
        lines.append(f"- {c.get('engine')}: delta={c.get('avg_return_2h_delta_vs_a6', 0)}%")

    lines.extend(["", "## Promising Zero-Base Patterns"])
    for p in promising[:8]:
        lines.append(f"- {p.get('engine')}: avg2h={p.get('avg_return_2h')}% trap={p.get('trap_rate')}%")

    lines.extend(["", "## Patterns to Discard"])
    for d in discard[:8]:
        lines.append(f"- {d.get('engine')}: avg2h={d.get('avg_return_2h')}% trap={d.get('trap_rate')}%")

    lines.extend([
        "",
        "## Next Experiments",
        "- Expand validation window beyond June when more history available",
        "- Cross-check champion candidates on blind holdout dates (min 3 positive days)",
        "- Lab Stream only — Main Stream unchanged until user approval",
        "- Never replace A6 from single-loss reaction; require n>=300 + champion gates",
        "",
        "*Research/Lab Stream only. No LIVE order logic modified.*",
    ])
    return "\n".join(lines)
