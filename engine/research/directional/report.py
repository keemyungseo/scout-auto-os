"""Directional zero-base report."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

KST = timezone(timedelta(hours=9))


def build_directional_report(
    long_board: list[dict],
    short_board: list[dict],
    long_random: dict,
    short_random: dict,
    pattern_stats: list[dict],
    slot_sim: dict,
    a6_long: dict,
    meta: dict,
) -> str:
    lines = [
        "# SCOUT Directional Zero-Base V1 Report",
        "",
        f"Generated: {datetime.now(KST).strftime('%Y-%m-%d %H:%M:%S')} KST",
        f"Validation scans: {meta.get('validation_scans', 0)}",
        "",
        "## Long TOP Engines",
    ]
    for r in long_board[:5]:
        lines.append(
            f"- #{r.get('board_rank')} **{r.get('engine')}** avg2h={r.get('avg_return_2h')}% "
            f"win={r.get('win_rate')}% PF={r.get('profit_factor')} slot={r.get('slot_eligible')} "
            f"[{r.get('tier')}]"
        )
    lines.extend([
        "",
        "## Short TOP Engines",
    ])
    for r in short_board[:5]:
        lines.append(
            f"- #{r.get('board_rank')} **{r.get('engine')}** short_avg2h={r.get('avg_return_2h')}% "
            f"win={r.get('win_rate')}% PF={r.get('profit_factor')} slot={r.get('slot_eligible')} "
            f"[{r.get('tier')}]"
        )
    lines.extend([
        "",
        "## vs Random Baseline",
        f"- Long Random: avg2h={long_random.get('avg_return_2h')}% win={long_random.get('win_rate')}%",
        f"- Short Random: short_avg2h={short_random.get('avg_return_2h')}% win={short_random.get('win_rate')}%",
        f"- A6 Long (baseline only): avg2h={a6_long.get('avg_return_2h')}% win={a6_long.get('win_rate')}%",
    ])
    for r in long_board[:3]:
        if r.get("engine") != "RANDOM_LONG":
            lines.append(f"- Long {r['engine']} vs Random: delta={float(r.get('avg_return_2h',0))-float(long_random.get('avg_return_2h',0)):.4f}%")
    for r in short_board[:3]:
        if r.get("engine") != "RANDOM_SHORT":
            lines.append(f"- Short {r['engine']} vs Random: delta={float(r.get('avg_return_2h',0))-float(short_random.get('avg_return_2h',0)):.4f}%")

    lines.extend(["", "## Pattern Performance"])
    for p in pattern_stats:
        lines.append(
            f"- {p.get('pattern')} ({p.get('side')}): n={p.get('sample_count')} "
            f"long_avg2h={p.get('long_avg_return_2h')} short_avg2h={p.get('short_avg_return_2h')}"
        )

    lines.extend([
        "",
        "## 3 Long + 3 Short Slot Simulation",
        f"- Long slots filled: {slot_sim.get('long_slots_filled')}/{meta.get('max_long_slots', 3)} "
        f"engines={slot_sim.get('long_slot_engines')}",
        f"- Short slots filled: {slot_sim.get('short_slots_filled')}/{meta.get('max_short_slots', 3)} "
        f"engines={slot_sim.get('short_slot_engines')}",
        f"- Empty long slots: {slot_sim.get('long_slots_empty')} | empty short slots: {slot_sim.get('short_slots_empty')}",
        f"- Combined avg2h: {slot_sim.get('combined_avg_return_2h')}% (n={slot_sim.get('combined_sample_count')})",
        f"- Long leg: {slot_sim.get('long_avg_return_2h')}% | Short leg: {slot_sim.get('short_avg_return_2h')}%",
        "",
        "## Quality Gate (empty slot if fail)",
        f"- min_samples={meta.get('quality_min_samples')} min_win={meta.get('quality_min_win')}% "
        f"min_avg2h={meta.get('quality_min_avg2h')}% must beat random",
        "",
        "*Research/Lab only. No LIVE order changes. Long/Short evaluated independently.*",
    ])
    return "\n".join(lines)
