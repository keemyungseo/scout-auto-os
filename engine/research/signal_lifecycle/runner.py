"""Signal Lifecycle Engine V1 orchestrator."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scout_auto_os.engine.research.directional.entry_filter.collector import (
    collect_direction_champion_signals,
    filter_scans_last_months,
)
from scout_auto_os.engine.research.directional.entry_filter.constants import (
    CHAMPION_TOP_K,
    LOOKBACK_MONTHS,
    LONG_DIRECTION_CHAMPION,
    SHORT_DIRECTION_CHAMPION,
)
from scout_auto_os.engine.research.safe import research_safe
from scout_auto_os.engine.research.signal_lifecycle.report import aggregate_clusters, build_lifecycle_report
from scout_auto_os.engine.research.signal_lifecycle.shape_classifier import (
    classify_lifecycle_shape,
    evaluation_flags,
    shape_feature_row,
)
from scout_auto_os.engine.research.signal_lifecycle.timeline import build_signal_timeline
from scout_auto_os.engine.research.zero_base.runner import load_candidates_jsonl, load_forward_klines

KST = timezone(timedelta(hours=9))


def _signal_id(direction: str, scan_kst: str, symbol: str) -> str:
    safe_scan = scan_kst.replace(" ", "T").replace(":", "")
    return f"{direction}_{safe_scan}_{symbol}"


class SignalLifecycleRunner:
    def __init__(
        self,
        data_dir: Path,
        pkg_root: Path,
        candidates_path: Path,
        forward_path: Path,
        lookback_months: int = LOOKBACK_MONTHS,
    ) -> None:
        self.data_dir = data_dir
        self.pkg_root = pkg_root
        self.out_dir = data_dir / "signal_lifecycle"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.candidates_path = candidates_path
        self.forward_path = forward_path
        self.lookback_months = lookback_months

    @research_safe("signal_lifecycle")
    def run(self, top_k: int = CHAMPION_TOP_K) -> dict:
        print("[SIGNAL LIFECYCLE] started")
        by_scan = load_candidates_jsonl(self.candidates_path)
        fwd = load_forward_klines(self.forward_path)
        all_scans = sorted(by_scan.keys())
        scans = filter_scans_last_months(all_scans, self.lookback_months)

        long_signals, short_signals = collect_direction_champion_signals(
            by_scan, fwd, scans, top_k=top_k,
        )

        timeline_rows: list[dict] = []
        lifecycle_rows: list[dict] = []
        shape_rows: list[dict] = []

        for direction, engine, signals in (
            ("long", LONG_DIRECTION_CHAMPION, long_signals),
            ("short", SHORT_DIRECTION_CHAMPION, short_signals),
        ):
            for sig in signals:
                scan_kst = sig["scan_time_kst"]
                symbol = sig["symbol"]
                sid = _signal_id(direction, scan_kst, symbol)
                klines = fwd.get((scan_kst, symbol))
                if not klines:
                    continue

                timeline, summary = build_signal_timeline(
                    klines, direction, scan_kst, symbol, sid,
                )
                if not timeline:
                    continue

                label = classify_lifecycle_shape(summary)
                flags = evaluation_flags(summary)
                timeline_rows.extend(timeline)
                shape_rows.append(shape_feature_row(summary, label, engine))

                lifecycle_rows.append(
                    {
                        "signal_id": sid,
                        "direction": direction,
                        "engine": engine,
                        "scan_time_kst": scan_kst,
                        "symbol": symbol,
                        "lifecycle_label": label,
                        "track_hours": summary["track_hours"],
                        "peak_time_min": summary["peak_time_min"],
                        "end_time_min": summary["end_time_min"],
                        "peak_return_pct": summary["peak_return_pct"],
                        "mfe_full": summary["mfe_full"],
                        "mae_full": summary["mae_full"],
                        "return_2h": summary["return_2h"],
                        "return_6h": summary["return_6h"],
                        "return_12h": summary.get("return_12h"),
                        "return_at_end": summary["return_at_end"],
                        **flags,
                        "dominant_phase": _dominant_phase(timeline),
                    },
                )

        clusters = aggregate_clusters(shape_rows)

        dates = [s[:10] for s in scans]
        meta = {
            "lookback_months": self.lookback_months,
            "date_min": min(dates) if dates else None,
            "date_max": max(dates) if dates else None,
            "scan_count": len(scans),
            "top_k": top_k,
            "long_signal_count": len([r for r in lifecycle_rows if r["direction"] == "long"]),
            "short_signal_count": len([r for r in lifecycle_rows if r["direction"] == "short"]),
            "timeline_row_count": len(timeline_rows),
            "generated_at": datetime.now(KST).isoformat(),
        }

        from season2_p37_scout_decision_hierarchy import write_csv

        write_csv(self.out_dir / "signal_lifecycle.csv", lifecycle_rows)
        write_csv(self.out_dir / "signal_timeline.csv", timeline_rows)
        write_csv(self.out_dir / "signal_shape.csv", shape_rows)
        write_csv(self.out_dir / "lifecycle_cluster.csv", clusters)

        report = build_lifecycle_report(meta, lifecycle_rows, shape_rows, clusters)
        report_path = self.out_dir / "lifecycle_report.md"
        report_path.write_text(report, encoding="utf-8")
        (self.out_dir / "lifecycle_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        reports_dir = self.pkg_root / "research_bundle" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        bundle_map = {
            "lifecycle_report.md": "signal_lifecycle_v1_report.md",
            "signal_lifecycle.csv": "signal_lifecycle_v1.csv",
            "signal_timeline.csv": "signal_timeline_v1.csv",
            "signal_shape.csv": "signal_shape_v1.csv",
            "lifecycle_cluster.csv": "lifecycle_cluster_v1.csv",
            "lifecycle_meta.json": "signal_lifecycle_v1_meta.json",
        }
        for src_name, dst_name in bundle_map.items():
            src = self.out_dir / src_name
            if src.exists():
                (reports_dir / dst_name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

        print("[SIGNAL LIFECYCLE] report generated")
        return {
            "meta": meta,
            "cluster_count": len(clusters),
            "report_path": str(report_path),
        }


def _dominant_phase(timeline: list[dict]) -> str:
    counts: dict[str, int] = {}
    for row in timeline:
        ph = row.get("lifecycle_phase", "")
        counts[ph] = counts.get(ph, 0) + 1
    if not counts:
        return "unknown"
    return max(counts, key=counts.get)
