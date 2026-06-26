"""Predator Value Gate report writer."""

from __future__ import annotations

import csv
import json
from pathlib import Path


class ValueGateReport:
    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def write_all(
        self,
        shadow_trades: list[dict],
        baseline_metrics: dict,
        gated_metrics: dict,
        false_skips: list[dict],
        false_accepts: list[dict],
        best_missed: dict | None,
        worst_accepted: dict | None,
        verdict: str,
        answers: dict,
    ) -> Path:
        self._write_decisions(shadow_trades)
        self._write_shadow_trades(shadow_trades)
        self._write_comparison(baseline_metrics, gated_metrics, false_skips, false_accepts, best_missed, worst_accepted)
        self._write_false_skip(false_skips)
        self._write_false_accept(false_accepts)
        return self._write_report(
            baseline_metrics, gated_metrics, false_skips, false_accepts,
            best_missed, worst_accepted, verdict, answers,
        )

    def _write_decisions(self, trades: list[dict]) -> None:
        path = self.out_dir / "value_gate_decision.csv"
        fields = [
            "trade_key", "scan_kst", "symbol", "direction", "value_score",
            "gate_action", "gate_reason", "recommended_size",
            "baseline_action", "gated_action", "actual_roi",
            "predicted_dna_type", "actual_dna_type",
        ]
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for t in trades:
                w.writerow({
                    "trade_key": t["trade_key"],
                    "scan_kst": t["scan_kst"],
                    "symbol": t["symbol"],
                    "direction": t["direction"],
                    "value_score": t["value_score"],
                    "gate_action": t["gate_action"],
                    "gate_reason": t["gate_reason"],
                    "recommended_size": t["recommended_size"],
                    "baseline_action": t["baseline"]["action"],
                    "gated_action": t["gated"]["action"],
                    "actual_roi": t["actual_roi"],
                    "predicted_dna_type": t["predicted_dna_type"],
                    "actual_dna_type": t["actual_dna_type"],
                })

    def _write_shadow_trades(self, trades: list[dict]) -> None:
        path = self.out_dir / "value_gate_shadow_trades.csv"
        fields = [
            "trade_key", "symbol", "direction", "actual_roi",
            "baseline_size", "baseline_weighted_roi",
            "gated_action", "gated_size", "gated_weighted_roi", "gated_taken",
            "value_score", "trade_contract_json",
        ]
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for t in trades:
                w.writerow({
                    "trade_key": t["trade_key"],
                    "symbol": t["symbol"],
                    "direction": t["direction"],
                    "actual_roi": t["actual_roi"],
                    "baseline_size": t["baseline"]["size"],
                    "baseline_weighted_roi": t["baseline"]["weighted_roi"],
                    "gated_action": t["gated"]["action"],
                    "gated_size": t["gated"]["size"],
                    "gated_weighted_roi": t["gated"]["weighted_roi"],
                    "gated_taken": int(bool(t["gated"].get("taken"))),
                    "value_score": t["value_score"],
                    "trade_contract_json": json.dumps(t["trade_contract"], ensure_ascii=False),
                })

    def _write_comparison(
        self,
        baseline: dict,
        gated: dict,
        false_skips: list[dict],
        false_accepts: list[dict],
        best_missed: dict | None,
        worst_accepted: dict | None,
    ) -> None:
        path = self.out_dir / "value_gate_comparison.csv"
        fields = [
            "metric", "baseline_predator", "value_gate", "delta",
        ]
        rows = [
            ("total_roi", baseline["total_roi"], gated["total_roi"]),
            ("avg_roi", baseline["avg_roi"], gated["avg_roi"]),
            ("win_rate", baseline["win_rate"], gated["win_rate"]),
            ("sharpe", baseline["sharpe"], gated["sharpe"]),
            ("mdd", baseline["mdd"], gated["mdd"]),
            ("trade_count", baseline["trade_count"], gated["trade_count"]),
            ("skipped_trades_avg_roi", baseline.get("skipped_avg_roi", 0), gated["skipped_avg_roi"]),
            ("accepted_trades_avg_roi", baseline["accepted_avg_roi"], gated["accepted_avg_roi"]),
            ("false_skip_count", 0, len(false_skips)),
            ("false_accept_count", 0, len(false_accepts)),
            (
                "best_missed_trade_roi",
                0,
                round(float(best_missed["actual_roi"]), 4) if best_missed else 0,
            ),
            (
                "worst_accepted_trade_roi",
                round(float(worst_accepted["actual_roi"]), 4) if worst_accepted else 0,
                0,
            ),
        ]
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for name, b, g in rows:
                try:
                    delta = round(float(g) - float(b), 4)
                except (TypeError, ValueError):
                    delta = ""
                w.writerow({
                    "metric": name,
                    "baseline_predator": b,
                    "value_gate": g,
                    "delta": delta,
                })

    def _write_false_skip(self, rows: list[dict]) -> None:
        path = self.out_dir / "false_skip_cases.csv"
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        fields = list(rows[0].keys())
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    def _write_false_accept(self, rows: list[dict]) -> None:
        path = self.out_dir / "false_accept_cases.csv"
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        fields = list(rows[0].keys())
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    def _write_report(
        self,
        baseline: dict,
        gated: dict,
        false_skips: list[dict],
        false_accepts: list[dict],
        best_missed: dict | None,
        worst_accepted: dict | None,
        verdict: str,
        answers: dict,
    ) -> Path:
        path = self.out_dir / "value_gate_report.md"
        lines = [
            "# Predator Value Gate V1",
            "",
            "## Replay comparison (157 trades)",
            "",
            "| Metric | Baseline | Value Gate | Delta |",
            "|--------|----------|------------|-------|",
            f"| Total ROI | {baseline['total_roi']} | {gated['total_roi']} | {round(gated['total_roi']-baseline['total_roi'],4)} |",
            f"| Avg ROI | {baseline['avg_roi']} | {gated['avg_roi']} | {round(gated['avg_roi']-baseline['avg_roi'],4)} |",
            f"| Win Rate | {baseline['win_rate']}% | {gated['win_rate']}% | {round(gated['win_rate']-baseline['win_rate'],2)} |",
            f"| Sharpe | {baseline['sharpe']} | {gated['sharpe']} | {round(gated['sharpe']-baseline['sharpe'],4)} |",
            f"| MDD | {baseline['mdd']} | {gated['mdd']} | {round(gated['mdd']-baseline['mdd'],4)} |",
            f"| Trade Count | {baseline['trade_count']} | {gated['trade_count']} | {gated['trade_count']-baseline['trade_count']} |",
            f"| Skipped avg ROI | — | {gated['skipped_avg_roi']} | — |",
            f"| Accepted avg ROI | {baseline['accepted_avg_roi']} | {gated['accepted_avg_roi']} | — |",
            "",
            f"**False Skip:** {len(false_skips)} | **False Accept:** {len(false_accepts)}",
            "",
        ]
        if best_missed:
            lines.append(
                f"**Best Missed:** {best_missed['symbol']} {best_missed['direction']} "
                f"ROI={best_missed['actual_roi']}% score={best_missed['value_score']}"
            )
        if worst_accepted:
            lines.append(
                f"**Worst Accepted:** {worst_accepted['symbol']} {worst_accepted['direction']} "
                f"ROI={worst_accepted['actual_roi']}% score={worst_accepted['value_score']}"
            )
        lines.extend(["", "## Final Questions", ""])
        for i, (q, a) in enumerate(answers.items(), 1):
            lines.append(f"{i}. **{q}** — {a}")
        lines.extend(["", f"## Verdict: `{verdict}`", ""])
        path.write_text("\n".join(lines), encoding="utf-8")
        return path
