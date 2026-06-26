"""Value Gate policy test V2 report writer."""

from __future__ import annotations

import csv
import json
from pathlib import Path


class PolicyTestReport:
    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def write_all(
        self,
        comparisons: list[dict],
        long_short: list[dict],
        false_skips: list[dict],
        false_accepts: list[dict],
        bands: list[dict],
        recommended: dict,
        verdict: str,
        answers: dict,
    ) -> Path:
        self._write_csv("policy_comparison.csv", comparisons)
        self._write_csv("policy_long_short.csv", long_short)
        self._write_csv("policy_false_skip.csv", false_skips)
        self._write_csv("policy_false_accept.csv", false_accepts)
        self._write_csv("policy_band_analysis.csv", bands)
        (self.out_dir / "recommended_policy.json").write_text(
            json.dumps(recommended, indent=2), encoding="utf-8",
        )
        return self._write_md(comparisons, recommended, verdict, answers)

    def _write_csv(self, name: str, rows: list[dict]) -> None:
        path = self.out_dir / name
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        fields = list(rows[0].keys())
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    def _write_md(
        self,
        comparisons: list[dict],
        recommended: dict,
        verdict: str,
        answers: dict,
    ) -> Path:
        path = self.out_dir / "value_gate_policy_report.md"
        v1 = next((c for c in comparisons if c["policy"] == "A"), comparisons[0])
        lines = [
            "# Predator Value Gate Policy Test V2",
            "",
            "## Policy comparison (157 replay trades)",
            "",
            "| Policy | Total ROI | Sharpe | MDD | WinRate | Trades | FalseSkip | FalseAccept | R/Risk |",
            "|--------|-----------|--------|-----|---------|--------|-----------|-------------|--------|",
        ]
        for c in comparisons:
            lines.append(
                f"| {c['policy']} {c['policy_name']} | {c['total_roi']} | {c['sharpe']} | {c['mdd']} | "
                f"{c['win_rate']}% | {c['trade_count']} | {c['false_skip_count']} | "
                f"{c['false_accept_count']} | {c['return_per_risk']} |"
            )
        lines.extend([
            "",
            f"## Recommended: Policy **{recommended.get('policy', '?')}** — {recommended.get('policy_name', '')}",
            "",
            f"Selection score: {recommended.get('selection_score', 0)} (vs V1 false_skip={v1['false_skip_count']})",
            "",
            "## Final Questions",
            "",
        ])
        for i, (q, a) in enumerate(answers.items(), 1):
            lines.append(f"{i}. **{q}** — {a}")
        lines.extend(["", f"## Verdict: `{verdict}`", ""])
        path.write_text("\n".join(lines), encoding="utf-8")
        return path
