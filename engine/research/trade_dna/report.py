"""Trade DNA statistics and markdown reports."""

from __future__ import annotations

import statistics
from datetime import datetime, timezone, timedelta
from pathlib import Path

from scout_auto_os.engine.research.trade_dna.clustering import infer_archetype_label
from scout_auto_os.engine.research.trade_dna.curve_builder import TradeDNARecord

KST = timezone(timedelta(hours=9))


def _now_kst() -> str:
    return datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    import csv
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def type_statistics(
    cluster_rows: list[dict],
    records: list[TradeDNARecord],
) -> list[dict]:
    by_type: dict[str, list[dict]] = {}
    rec_map = {r.trade_key: r for r in records}
    for row in cluster_rows:
        by_type.setdefault(row["trade_type_id"], []).append(row)

    stats: list[dict] = []
    for tid, rows in sorted(by_type.items()):
        archetype = infer_archetype_label(rows)
        vol_30 = statistics.mean(float(r.get("vol_30m", 1)) for r in rows)
        stats.append({
            "trade_type_id": tid,
            "data_derived_label": archetype,
            "trade_count": len(rows),
            "winner_count": sum(int(r.get("is_winner", 0)) for r in rows),
            "winner_rate_pct": round(sum(int(r.get("is_winner", 0)) for r in rows) / len(rows) * 100, 2),
            "avg_hold_proxy_min": 120,
            "avg_roi_2h": round(statistics.mean(float(r["final_roi_2h"]) for r in rows), 4),
            "avg_peak_roi": round(statistics.mean(float(r["peak_roi"]) for r in rows), 4),
            "avg_drawdown": round(statistics.mean(float(r["max_drawdown"]) for r in rows), 4),
            "avg_peak_timing_min": round(statistics.mean(int(r["peak_timing_min"]) for r in rows), 1),
            "avg_alive_delta": round(statistics.mean(float(r["alive_delta_proxy"]) for r in rows), 4),
            "avg_exit_pressure": round(statistics.mean(float(r["exit_pressure_proxy"]) for r in rows), 4),
            "avg_volume_30m": round(vol_30, 4),
            "long_count": sum(1 for r in rows if r.get("direction") == "long"),
            "short_count": sum(1 for r in rows if r.get("direction") == "short"),
        })
    return stats


def winner_loser_analysis(cluster_rows: list[dict]) -> dict:
    winners = [r for r in cluster_rows if int(r.get("is_winner", 0))]
    losers = [r for r in cluster_rows if not int(r.get("is_winner", 0))]
    w_types = len({r["trade_type_id"] for r in winners})
    l_types = len({r["trade_type_id"] for r in losers})
    return {
        "winner_type_count": w_types,
        "loser_type_count": l_types,
        "winners_multi_type": w_types > 1,
        "losers_multi_pattern": l_types > 1,
        "winner_dominant_type": max(
            {r["trade_type_id"] for r in winners}, key=lambda t: sum(1 for r in winners if r["trade_type_id"] == t),
        ) if winners else "n/a",
        "loser_dominant_type": max(
            {r["trade_type_id"] for r in losers}, key=lambda t: sum(1 for r in losers if r["trade_type_id"] == t),
        ) if losers else "n/a",
    }


class TradeDNAReport:
    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def write_all(
        self,
        cluster_rows: list[dict],
        records: list[TradeDNARecord],
        cluster_meta: dict,
        type_stats: list[dict],
        exit_table: list[dict],
        lift: dict,
        entry_pred: dict,
        wl: dict,
    ) -> Path:
        _write_csv(self.out_dir / "trade_cluster.csv", cluster_rows)
        _write_csv(self.out_dir / "trade_type_statistics.csv", type_stats)
        _write_csv(self.out_dir / "trade_type_exit_table.csv", exit_table)

        self._write_examples(cluster_rows, type_stats)
        self._write_entry_predictor(entry_pred)
        self._write_main_report(cluster_meta, type_stats, exit_table, lift, entry_pred, wl)

        return self.out_dir / "trade_dna.md"

    def _write_examples(self, cluster_rows: list[dict], type_stats: list[dict]) -> None:
        lines = ["# Trade Type Examples", "", f"_Generated: {_now_kst()}_", ""]
        label_map = {s["trade_type_id"]: s["data_derived_label"] for s in type_stats}
        for tid in sorted({r["trade_type_id"] for r in cluster_rows}):
            sub = [r for r in cluster_rows if r["trade_type_id"] == tid]
            archetype = label_map.get(tid, "unknown")
            lines.append(f"## {tid} — {archetype}")
            lines.append(f"")
            lines.append(f"Sample count: {len(sub)}")
            best = sorted(sub, key=lambda r: -float(r["final_roi_2h"]))[:3]
            worst = sorted(sub, key=lambda r: float(r["final_roi_2h"]))[:3]
            lines.append("")
            lines.append("**Top 3 by 2h ROI:**")
            for r in best:
                lines.append(
                    f"- {r['symbol']} {r['direction']} scan={r['scan_kst']} "
                    f"roi_2h={r['final_roi_2h']}% peak={r['peak_roi']}% @ {r['peak_timing_min']}m"
                )
            lines.append("")
            lines.append("**Bottom 3:**")
            for r in worst:
                lines.append(
                    f"- {r['symbol']} {r['direction']} roi_2h={r['final_roi_2h']}% peak={r['peak_roi']}%"
                )
            lines.append("")
        (self.out_dir / "trade_type_examples.md").write_text("\n".join(lines), encoding="utf-8")

    def _write_entry_predictor(self, entry_pred: dict) -> None:
        lines = [
            "# Entry → Trade Type Predictability",
            "",
            f"_Generated: {_now_kst()}_",
            "",
            f"- CV accuracy: **{entry_pred.get('cv_accuracy', 0)}**",
            f"- Random baseline: {entry_pred.get('random_baseline', 0)}",
            f"- Lift vs random: {entry_pred.get('lift_vs_random', 0)}",
            f"- Predictable at entry: **{entry_pred.get('predictable', False)}**",
            "",
            "## Pattern → Type concentration",
            "",
        ]
        for p in entry_pred.get("pattern_dominance", []):
            lines.append(
                f"- `{p['live_pattern']}` → {p['dominant_type']} "
                f"({p['concentration_pct']}% of {p['sample_count']} samples)"
            )
        (self.out_dir / "entry_to_trade_type.md").write_text("\n".join(lines), encoding="utf-8")

    def _write_main_report(
        self,
        meta: dict,
        type_stats: list[dict],
        exit_table: list[dict],
        lift: dict,
        entry_pred: dict,
        wl: dict,
    ) -> None:
        n_types = meta.get("n_clusters", 0)
        lines = [
            "# Trade DNA Engine V1",
            "",
            f"_Generated: {_now_kst()}_",
            "",
            "## Summary",
            "",
            f"- Trades analyzed: **{meta.get('n_trades', 0)}**",
            f"- Trade Types discovered (data-driven): **{n_types}**",
            f"- Cluster selection: silhouette-max KMeans (k=2..10)",
            "",
            "## Five Questions",
            "",
            "### 1. Replay에 실제로 몇 개의 Trade Type이 존재하는가?",
            f"**{n_types}개** (TYPE_0 .. TYPE_{max(n_types - 1, 0)}). "
            f"Silhouette scores: {meta.get('silhouette_scores', [])}",
            "",
            "### 2. Winner는 모두 같은 DNA인가?",
            (
                f"**아니오 — {wl['winner_type_count']}개 Type에 분포.** "
                f"Dominant winner type: {wl['winner_dominant_type']}"
                if wl["winners_multi_type"] else
                f"**Winner가 단일 Type에 집중:** {wl['winner_dominant_type']}"
            ),
            "",
            "### 3. Loser도 하나의 패턴인가?",
            (
                f"**아니오 — {wl['loser_type_count']}개 실패 Type.** "
                f"Dominant loser type: {wl['loser_dominant_type']}"
                if wl["losers_multi_pattern"] else
                f"**Loser 단일 패턴:** {wl['loser_dominant_type']}"
            ),
            "",
            "### 4. Trade Type을 Entry 시점에서 예측 가능한가?",
            (
                f"**{'부분적으로 가능' if entry_pred.get('predictable') else '현재 불충분'}** — "
                f"CV accuracy {entry_pred.get('cv_accuracy')} vs random {entry_pred.get('random_baseline')} "
                f"(+{entry_pred.get('lift_vs_random')} lift)"
            ),
            "",
            "### 5. Type별 전용 Exit 적용 시 ROI 개선 예상?",
            f"**+{lift.get('expected_lift_pp', 0)}%p avg per trade** "
            f"(baseline hold_2h avg {lift.get('baseline_avg_roi')}% → type-specific exit avg {lift.get('type_exit_avg_roi')}%). "
            f"TYPE_0: hold_240m | TYPE_1: full_dynamic early exit",
            "",
            "## Type Statistics",
            "",
            "| Type | Data Label | Count | Win% | Avg ROI 2h | Peak ROI | Peak Time | Best Exit |",
            "|------|------------|-------|------|------------|----------|-----------|-----------|",
        ]
        exit_map = {r["trade_type_id"]: r for r in exit_table}
        for s in type_stats:
            ex = exit_map.get(s["trade_type_id"], {})
            lines.append(
                f"| {s['trade_type_id']} | {s['data_derived_label']} | {s['trade_count']} | "
                f"{s['winner_rate_pct']}% | {s['avg_roi_2h']} | {s['avg_peak_roi']} | "
                f"{s['avg_peak_timing_min']}m | {ex.get('best_exit_mode', 'n/a')} |"
            )
        lines.extend([
            "",
            "## Season3 Implication",
            "",
            "1. Entry → classify Trade Type (DNA)",
            "2. Type → dedicated Exit policy",
            "3. Type → Position management (hold horizon, trail vs fixed)",
            "",
            "No new indicators. No threshold tuning. DNA from replay curves only.",
        ])
        (self.out_dir / "trade_dna.md").write_text("\n".join(lines), encoding="utf-8")
