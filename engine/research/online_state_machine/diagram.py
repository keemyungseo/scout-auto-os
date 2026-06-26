"""Mermaid state transition diagrams."""

from __future__ import annotations

from collections import defaultdict


def build_mermaid_diagram(matrix_rows: list[dict], direction: str, min_prob_pct: float = 5.0) -> str:
    sub = [r for r in matrix_rows if r["direction"] == direction]
    if not sub:
        return f"```mermaid\nflowchart LR\n  empty[\"No transitions — {direction}\"]\n```"

    nodes: set[str] = set()
    edges: list[tuple[str, str, float, float]] = []
    for r in sub:
        frm, to = r["from_state"], r["to_state"]
        prob = float(r["transition_probability_pct"])
        if prob < min_prob_pct:
            continue
        nodes.add(frm)
        nodes.add(to)
        edges.append((frm, to, prob, float(r["avg_return_at_transition_pct"])))

    lines = ["```mermaid", "flowchart TD", f"  subgraph {direction.upper()}[\"{direction.upper()} state machine\"]"]
    id_map: dict[str, str] = {}
    for i, node in enumerate(sorted(nodes)):
        nid = f"s{i}"
        id_map[node] = nid
        label = node.replace("_", " ")
        lines.append(f"    {nid}[\"{label}\"]")

    for frm, to, prob, avg_ret in sorted(edges, key=lambda x: -x[2]):
        a, b = id_map[frm], id_map[to]
        lines.append(f"    {a} -->|\"{prob:.0f}% avgRet {avg_ret:+.1f}%\"| {b}")

    lines.append("  end")
    lines.append("```")
    return "\n".join(lines)


def top_transition_paths(matrix_rows: list[dict], direction: str, top_n: int = 5) -> list[str]:
    sub = sorted(
        [r for r in matrix_rows if r["direction"] == direction],
        key=lambda x: -int(x["transition_count"]),
    )
    out: list[str] = []
    for r in sub[:top_n]:
        out.append(
            f"{r['from_state']} -> {r['to_state']}: "
            f"P={r['transition_probability_pct']}% n={r['transition_count']} "
            f"avgDur={r['avg_duration_in_from_state_min']}min "
            f"avgRet={r['avg_return_at_transition_pct']}%",
        )
    return out
