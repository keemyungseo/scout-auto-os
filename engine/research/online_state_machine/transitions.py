"""State transition extraction and aggregation."""

from __future__ import annotations

from collections import defaultdict


def extract_transitions(
    signal_id: str,
    direction: str,
    scan_kst: str,
    symbol: str,
    timeline: list[dict],
) -> tuple[list[dict], str]:
    """Return transition rows and arrow sequence string."""
    if not timeline:
        return [], ""

    transitions: list[dict] = []
    states_seen: list[str] = []
    prev_state: str | None = None
    enter_min = 0
    enter_return = 0.0

    for row in timeline:
        st = row["state"]
        if not states_seen or states_seen[-1] != st:
            states_seen.append(st)
        if prev_state is None:
            prev_state = st
            enter_min = int(row["minutes_from_entry"])
            enter_return = float(row["return_pct"])
            continue
        if st != prev_state:
            dur = int(row["minutes_from_entry"]) - enter_min
            transitions.append(
                {
                    "signal_id": signal_id,
                    "direction": direction,
                    "scan_time_kst": scan_kst,
                    "symbol": symbol,
                    "from_state": prev_state,
                    "to_state": st,
                    "transition_minutes": int(row["minutes_from_entry"]),
                    "duration_in_from_state_min": dur,
                    "return_at_transition_pct": float(row["return_pct"]),
                    "return_delta_in_state_pct": round(float(row["return_pct"]) - enter_return, 4),
                    "mfe_at_transition_pct": float(row["mfe_pct"]),
                    "mae_at_transition_pct": float(row["mae_pct"]),
                },
            )
            prev_state = st
            enter_min = int(row["minutes_from_entry"])
            enter_return = float(row["return_pct"])

    last = timeline[-1]
    if prev_state and prev_state != "EXIT":
        dur = int(last["minutes_from_entry"]) - enter_min
        transitions.append(
            {
                "signal_id": signal_id,
                "direction": direction,
                "scan_time_kst": scan_kst,
                "symbol": symbol,
                "from_state": prev_state,
                "to_state": "EXIT",
                "transition_minutes": int(last["minutes_from_entry"]),
                "duration_in_from_state_min": dur,
                "return_at_transition_pct": float(last["return_pct"]),
                "return_delta_in_state_pct": round(float(last["return_pct"]) - enter_return, 4),
                "mfe_at_transition_pct": float(last["mfe_pct"]),
                "mae_at_transition_pct": float(last["mae_pct"]),
            },
        )
        states_seen.append("EXIT")

    seq = " -> ".join(states_seen)
    return transitions, seq


def build_transition_matrix(transitions: list[dict], direction: str) -> list[dict]:
    sub = [t for t in transitions if t["direction"] == direction]
    counts: dict[tuple[str, str], list[dict]] = defaultdict(list)
    from_totals: dict[str, int] = defaultdict(int)
    for t in sub:
        key = (t["from_state"], t["to_state"])
        counts[key].append(t)
        from_totals[t["from_state"]] += 1

    rows: list[dict] = []
    for (frm, to), items in sorted(counts.items()):
        n = len(items)
        total_from = from_totals[frm] or 1
        rows.append(
            {
                "direction": direction,
                "from_state": frm,
                "to_state": to,
                "transition_count": n,
                "transition_probability_pct": round(n / total_from * 100, 2),
                "avg_return_at_transition_pct": round(
                    sum(x["return_at_transition_pct"] for x in items) / n, 4,
                ),
                "avg_duration_in_from_state_min": round(
                    sum(x["duration_in_from_state_min"] for x in items) / n, 2,
                ),
            },
        )
    return rows


def build_state_statistics(timeline_rows: list[dict], transitions: list[dict]) -> list[dict]:
    """Per-state dwell stats from timeline + transition aggregates."""
    by_dir_state: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in timeline_rows:
        key = (row["direction"], row["state"])
        by_dir_state[key].append(row)

    trans_dur: dict[tuple[str, str], list[float]] = defaultdict(list)
    for t in transitions:
        trans_dur[(t["direction"], t["from_state"])].append(float(t["duration_in_from_state_min"]))

    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for (direction, state), items in sorted(by_dir_state.items()):
        seen.add((direction, state))
        rets = [float(x["return_pct"]) for x in items]
        minutes = [int(x["minutes_from_entry"]) for x in items]
        durs = trans_dur.get((direction, state), [])
        rows.append(
            {
                "direction": direction,
                "state": state,
                "observation_count": len(items),
                "avg_return_pct": round(sum(rets) / len(rets), 4),
                "avg_minutes_from_entry": round(sum(minutes) / len(minutes), 2),
                "avg_duration_min": round(sum(durs) / len(durs), 2) if durs else None,
                "next_state_top": _top_next_state(transitions, direction, state),
            },
        )

    for (direction, state), durs in trans_dur.items():
        if (direction, state) in seen:
            continue
        rows.append(
            {
                "direction": direction,
                "state": state,
                "observation_count": 0,
                "avg_return_pct": None,
                "avg_minutes_from_entry": None,
                "avg_duration_min": round(sum(durs) / len(durs), 2),
                "next_state_top": _top_next_state(transitions, direction, state),
            },
        )
    return rows


def _top_next_state(transitions: list[dict], direction: str, from_state: str) -> str | None:
    counts: dict[str, int] = defaultdict(int)
    for t in transitions:
        if t["direction"] == direction and t["from_state"] == from_state:
            counts[t["to_state"]] += 1
    if not counts:
        return None
    return max(counts, key=counts.get)
