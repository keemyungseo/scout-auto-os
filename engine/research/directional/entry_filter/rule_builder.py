"""Build Entry Filter Rule V1 from optimized thresholds."""

from __future__ import annotations

from scout_auto_os.engine.research.directional.entry_filter.threshold_optimizer import Operator


def _op_symbol(operator: Operator) -> str:
    return ">=" if operator == "gte" else "<="


def _feature_family(name: str) -> str:
    for token in (
        "body_pct", "range_pct", "return_pct", "ma20_distance_pct",
        "close_position", "momentum", "compression", "volume_ratio",
        "volume_energy", "range_energy",
    ):
        if token in name:
            return token
    return "other"


def select_rule_features(
    long_best: list[dict],
    short_best: list[dict],
    dna_sets: dict,
    max_rules: int = 4,
) -> tuple[list[dict], list[dict]]:
    """Pick diverse AND rules — DNA top features first, then best lift/F1."""

    def _pick(best_rows: list[dict], direction: str) -> list[dict]:
        dna_ranked: list[str] = list(dna_sets.get(direction, []))
        dna_set = set(dna_ranked) | set(dna_sets.get("common", set()))
        best_map = {r["feature"]: r for r in best_rows}
        chosen: list[dict] = []
        families: set[str] = set()

        def _try_add(row: dict) -> bool:
            if row["feature"] in {c["feature"] for c in chosen}:
                return False
            fam = _feature_family(row["feature"])
            if fam in families and fam in ("compression", "other"):
                return False
            if row.get("pass_count", 0) < 15:
                return False
            chosen.append(row)
            families.add(fam)
            return True

        for feat in dna_ranked[:12]:
            if len(chosen) >= max_rules:
                break
            row = best_map.get(feat)
            if row and row.get("use_in_filter"):
                _try_add(row)

        ranked = sorted(
            best_rows,
            key=lambda r: (
                r.get("use_in_filter", False),
                r.get("lift", 0) * max(r.get("f1", 0), 0.01),
                r.get("expected_return_lift_2h", 0),
            ),
            reverse=True,
        )
        for row in ranked:
            if len(chosen) >= max_rules:
                break
            if row["feature"] not in dna_set and len(chosen) >= max_rules - 1:
                continue
            if row.get("use_in_filter"):
                _try_add(row)

        return chosen[:max_rules]

    return _pick(long_best, "long"), _pick(short_best, "short")


def merge_both_directions(
    long_best: list[dict],
    short_best: list[dict],
) -> list[dict]:
    """Tag features usable in long only, short only, or both."""
    long_map = {r["feature"]: r for r in long_best}
    short_map = {r["feature"]: r for r in short_best}
    all_feats = set(long_map) | set(short_map)
    rows: list[dict] = []
    for feat in all_feats:
        lr = long_map.get(feat)
        sr = short_map.get(feat)
        if lr and sr:
            same_op = lr.get("operator") == sr.get("operator")
            scope = "both" if same_op else "split"
        elif lr:
            scope = "long_only"
        else:
            scope = "short_only"
        rows.append({
            "feature": feat,
            "scope": scope,
            "long_threshold": lr.get("threshold") if lr else None,
            "long_operator": lr.get("operator") if lr else None,
            "long_use": lr.get("use_in_filter") if lr else None,
            "long_lift": lr.get("lift") if lr else None,
            "short_threshold": sr.get("threshold") if sr else None,
            "short_operator": sr.get("operator") if sr else None,
            "short_use": sr.get("use_in_filter") if sr else None,
            "short_lift": sr.get("lift") if sr else None,
        })
    rows.sort(key=lambda x: -(float(x.get("long_lift") or 0) + float(x.get("short_lift") or 0)))
    return rows


def format_rule_block(direction: str, rules: list[dict], combined_stats: dict) -> list[str]:
    lines = [f"## {direction.upper()} Entry Filter Rule V1", "", "```"]
    if not rules:
        lines.append("# No rules passed quality gate — manual review required")
    else:
        lines.append("IF")
        for i, r in enumerate(rules):
            joiner = "AND" if i < len(rules) - 1 else ""
            sym = _op_symbol(r["operator"])
            line = f"  {r['feature']} {sym} {r['threshold']}"
            if joiner:
                line += f"  {joiner}"
            lines.append(line)
        lines.extend(["THEN", "  PASS", "ELSE", "  REJECT"])
    lines.append("```")
    lines.extend([
        "",
        f"- Combined pass n={combined_stats.get('pass_count')} | "
        f"precision={combined_stats.get('precision')} | recall={combined_stats.get('recall')} | "
        f"f1={combined_stats.get('f1')}",
        f"- Avg return 2h (pass)={combined_stats.get('avg_return_2h_pass')}% | "
        f"4h={combined_stats.get('avg_return_4h_pass')}% | "
        f"win_rate={combined_stats.get('win_rate_pass_pct')}%",
        "",
    ])
    return lines


def build_entry_filter_rule_markdown(
    long_rules: list[dict],
    short_rules: list[dict],
    long_stats: dict,
    short_stats: dict,
    meta: dict,
) -> str:
    lines = [
        "# Entry Filter Rule V1",
        "",
        "Research-derived threshold rules for Direction Champion entries.",
        "**Not prediction. No ML. Scan-time features only.**",
        "",
        f"Signals analyzed: long={meta.get('long_signals')} short={meta.get('short_signals')}",
        f"Winner/Loser split: top/bottom {meta.get('winner_quantile', 20)}%",
        "",
    ]
    lines.extend(format_rule_block("long", long_rules, long_stats))
    lines.extend(format_rule_block("short", short_rules, short_stats))
    lines.extend([
        "## Per-feature conditions",
        "",
        "### Long",
    ])
    for r in long_rules:
        sym = _op_symbol(r["operator"])
        lines.append(
            f"- `{r['feature']}` {sym} **{r['threshold']}** | f1={r.get('f1')} lift={r.get('lift')} "
            f"use={r.get('use_in_filter')} | avg2h_pass={r.get('avg_return_2h_pass')}%"
        )
    lines.extend(["", "### Short"])
    for r in short_rules:
        sym = _op_symbol(r["operator"])
        lines.append(
            f"- `{r['feature']}` {sym} **{r['threshold']}** | f1={r.get('f1')} lift={r.get('lift')} "
            f"use={r.get('use_in_filter')} | avg2h_pass={r.get('avg_return_2h_pass')}%"
        )
    return "\n".join(lines)


def build_rules_json(long_rules: list[dict], short_rules: list[dict]) -> dict:
    """Machine-readable rules for future LIVE wiring (not auto-applied)."""

    def _serialize(rules: list[dict]) -> list[dict]:
        return [
            {
                "feature": r["feature"],
                "operator": r["operator"],
                "threshold": r["threshold"],
                "use_in_filter": r.get("use_in_filter", False),
                "f1": r.get("f1"),
                "lift": r.get("lift"),
            }
            for r in rules
        ]

    return {
        "version": "v1",
        "long": {"logic": "AND", "conditions": _serialize(long_rules)},
        "short": {"logic": "AND", "conditions": _serialize(short_rules)},
    }
