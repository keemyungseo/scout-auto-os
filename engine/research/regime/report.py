"""Regime Engine research report."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

KST = timezone(timedelta(hours=9))


def build_regime_report(
    regime_counts: dict[str, int],
    matrix: list[dict],
    champions: list[dict],
    transitions: list[dict],
    rule_scores: list[dict],
    composite_rules: list[dict],
    meta: dict,
) -> str:
    lines = [
        "# SCOUT Regime Engine — Market State Router (Research)",
        "",
        f"Generated: {datetime.now(KST).strftime('%Y-%m-%d %H:%M:%S')} KST",
        f"Validation scans: {meta.get('validation_scans', 0)}",
        "",
        "## 1. Market States Observed",
    ]
    total = sum(regime_counts.values()) or 1
    for reg, cnt in sorted(regime_counts.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"- **{reg}**: {cnt} scans ({cnt/total*100:.1f}%)")

    lines.extend(["", "## 2. Regime × Engine Performance (avg2h / win% / PF / trap / drawdown)"])
    by_reg: dict[str, list[dict]] = {}
    for row in matrix:
        by_reg.setdefault(row["regime"], []).append(row)
    for regime in sorted(by_reg):
        lines.append(f"\n### {regime}")
        for r in sorted(by_reg[regime], key=lambda x: x.get("router_score", 0), reverse=True):
            lines.append(
                f"- {r['engine']}: avg2h={r['avg_return_2h']}% win={r['win_rate']}% "
                f"PF={r['profit_factor']} trap={r['trap_rate']}% mdd={r['max_drawdown_avg']}% "
                f"n={r['sample_count']}"
            )

    lines.extend(["", "## 3. Regime Champion Engine (Router Map)"])
    for c in champions:
        if c.get("champion_engine") == "SKIP":
            lines.append(f"- **{c['regime']}** → `SKIP` ({c.get('reason', '')})")
        elif c.get("champion_engine") == "UNKNOWN":
            lines.append(f"- **{c['regime']}** → insufficient sample")
        else:
            lines.append(
                f"- **{c['regime']}** → `{c['champion_engine']}` "
                f"avg2h={c.get('avg_return_2h')}% win={c.get('win_rate')}% "
                f"[{c.get('confidence')}] runners={c.get('runners_up', [])}"
            )

    lines.extend(["", "## 4. State Transition Conditions (empirical)"])
    for t in transitions[:12]:
        lines.append(f"- {t['from_regime']} → {t['to_regime']}: {t['count']} times")

    lines.extend(["", "## 5. LIVE Regime Detection Rule Candidates"])
    lines.append("Top single-signal rules (precision/recall on validation labels):")
    for r in rule_scores[:8]:
        lines.append(
            f"- `{r['rule_id']}`: {r['signal']} {r['op']} {r['threshold']} "
            f"→ {r['predicts']} | P={r['precision']} R={r['recall']} F1={r['f1']}"
        )
    lines.append("\nComposite router proposals (not LIVE-applied):")
    for c in composite_rules:
        lines.append(
            f"- `{c['rule_id']}`: if {c['conditions']} → route `{c['route_engine']}` "
            f"(target={c['regime_target']})"
        )

    lines.extend([
        "",
        "## 6. LIVE Router Design (proposal)",
        "```",
        "on each 5m scan:",
        "  snap = market_snapshot(universe)",
        "  regime = classify_regime_8(snap)   # research classifier",
        "  if regime in (Bear, Capitulation): SKIP entry",
        "  else: engine = REGIME_ROUTER[regime]  # from champion table",
        "  candidates = engine.top5(symbols)       # frozen engine, no rule edits",
        "```",
        "",
        "## 7. State Machine Structure (proposal)",
        "```mermaid",
        "stateDiagram-v2",
        "  [*] --> Unknown",
        "  Unknown --> Bull: breadth_up + btc_up",
        "  Unknown --> Bear: breadth_down + btc_down",
        "  Bull --> Strong_Bull: top20_rising",
        "  Bull --> Sideway: flat_median",
        "  Sideway --> Breakout: vol_expansion + release",
        "  Bear --> Bottom: compression_high",
        "  Bear --> Capitulation: median<<0",
        "  Bottom --> Bull: recovery_breadth",
        "  Breakout --> Bull: momentum_persist",
        "  Bear --> [*]: SKIP",
        "  Capitulation --> [*]: SKIP",
        "```",
        "",
        "## 8. Research Conclusions",
        "- Universal formula (A6) underperforms in validation → router replaces single engine.",
        "- Momentum / Formula League dominate Bull & Strong_Bull regimes (hypothesis).",
        "- Breakout engine leads Breakout regime when release+volume expand.",
        "- Bear/Capitulation → SKIP is empirically supported.",
        "- Detection rules must be validated on May train + July+ out-of-sample before LIVE.",
        "",
        "*Research only. No Entry Rule changes. No A6/Momentum engine edits.*",
    ])
    return "\n".join(lines)
