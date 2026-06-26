"""Online State Machine V1 orchestrator."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scout_auto_os.engine.research.directional.engines import rank_long, rank_short
from scout_auto_os.engine.research.directional.entry_filter.collector import filter_scans_last_months
from scout_auto_os.engine.research.directional.entry_filter.constants import (
    CHAMPION_TOP_K,
    LOOKBACK_MONTHS,
    LONG_DIRECTION_CHAMPION,
    SHORT_DIRECTION_CHAMPION,
)
from scout_auto_os.engine.research.online_state_machine.online_features import enrich_timeline_online
from scout_auto_os.engine.research.online_state_machine.report import build_state_report
from scout_auto_os.engine.research.online_state_machine.state_estimator import annotate_states
from scout_auto_os.engine.research.online_state_machine.transitions import (
    build_state_statistics,
    build_transition_matrix,
    extract_transitions,
)
from scout_auto_os.engine.research.safe import research_safe
from scout_auto_os.engine.research.signal_lifecycle.timeline import build_signal_timeline
from scout_auto_os.engine.research.zero_base.runner import load_candidates_jsonl, load_forward_klines

KST = timezone(timedelta(hours=9))


def _signal_id(direction: str, scan_kst: str, symbol: str) -> str:
    safe_scan = scan_kst.replace(" ", "T").replace(":", "")
    return f"{direction}_{safe_scan}_{symbol}"


class OnlineStateMachineRunner:
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
        self.out_dir = data_dir / "online_state_machine"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.candidates_path = candidates_path
        self.forward_path = forward_path
        self.lookback_months = lookback_months

    @research_safe("online_state_machine")
    def run(self, top_k: int = CHAMPION_TOP_K) -> dict:
        print("[ONLINE STATE MACHINE] started")
        by_scan = load_candidates_jsonl(self.candidates_path)
        fwd = load_forward_klines(self.forward_path)
        all_scans = sorted(by_scan.keys())
        scans = filter_scans_last_months(all_scans, self.lookback_months)

        timeline_rows: list[dict] = []
        transition_rows: list[dict] = []
        sequence_rows: list[dict] = []

        for scan_kst in scans:
            rows = by_scan.get(scan_kst, [])
            if not rows:
                continue
            for direction, engine in (
                ("long", LONG_DIRECTION_CHAMPION),
                ("short", SHORT_DIRECTION_CHAMPION),
            ):
                rank_fn = rank_long if direction == "long" else rank_short
                for sym in rank_fn(rows, engine, top_k):
                    klines = fwd.get((scan_kst, sym))
                    if not klines:
                        continue
                    sid = _signal_id(direction, scan_kst, sym)
                    raw_tl, _ = build_signal_timeline(klines, direction, scan_kst, sym, sid)
                    if not raw_tl:
                        continue
                    enriched = enrich_timeline_online(raw_tl)
                    stated = annotate_states(enriched)
                    trans, seq = extract_transitions(sid, direction, scan_kst, sym, stated)
                    transition_rows.extend(trans)
                    sequence_rows.append(
                        {
                            "signal_id": sid,
                            "direction": direction,
                            "scan_time_kst": scan_kst,
                            "symbol": sym,
                            "state_sequence": seq,
                            "transition_count": len(trans),
                            "final_state": stated[-1]["state"] if stated else None,
                            "final_return_pct": stated[-1]["return_pct"] if stated else None,
                        },
                    )
                    for row in stated:
                        timeline_rows.append(
                            {
                                "signal_id": sid,
                                "direction": direction,
                                "scan_time_kst": scan_kst,
                                "symbol": sym,
                                "minutes_from_entry": row["minutes_from_entry"],
                                "state": row["state"],
                                "return_pct": row["return_pct"],
                                "mfe_pct": row["mfe_pct"],
                                "mae_pct": row["mae_pct"],
                                "drawdown_from_peak_pct": row["drawdown_from_peak_pct"],
                                "velocity_pct_per_hour": row["velocity_pct_per_hour"],
                                "acceleration_pct_per_hour2": row["acceleration_pct_per_hour2"],
                                "body_pct": row["body_pct"],
                                "range_pct": row["range_pct"],
                                "volume_ratio": row["volume_ratio"],
                                "atr_ratio": row["atr_ratio"],
                                "momentum_pct": row["momentum_pct"],
                                "slope_return": row["slope_return"],
                            },
                        )

        matrix_rows = build_transition_matrix(transition_rows, "long") + build_transition_matrix(
            transition_rows, "short",
        )
        state_stats = build_state_statistics(timeline_rows, transition_rows)

        dates = [s[:10] for s in scans]
        meta = {
            "lookback_months": self.lookback_months,
            "date_min": min(dates) if dates else None,
            "date_max": max(dates) if dates else None,
            "scan_count": len(scans),
            "long_signal_count": len([s for s in sequence_rows if s["direction"] == "long"]),
            "short_signal_count": len([s for s in sequence_rows if s["direction"] == "short"]),
            "timeline_row_count": len(timeline_rows),
            "transition_count": len(transition_rows),
            "generated_at": datetime.now(KST).isoformat(),
        }

        from season2_p37_scout_decision_hierarchy import write_csv

        write_csv(self.out_dir / "state_timeline.csv", timeline_rows)
        write_csv(self.out_dir / "state_sequence.csv", sequence_rows)
        write_csv(self.out_dir / "transition_matrix.csv", matrix_rows)
        write_csv(self.out_dir / "state_statistics.csv", state_stats)

        report = build_state_report(meta, matrix_rows, state_stats)
        report_path = self.out_dir / "state_report.md"
        report_path.write_text(report, encoding="utf-8")
        (self.out_dir / "state_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        reports_dir = self.pkg_root / "research_bundle" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        bundle_map = {
            "state_report.md": "online_state_machine_v1_report.md",
            "state_timeline.csv": "state_timeline_v1.csv",
            "state_sequence.csv": "state_sequence_v1.csv",
            "transition_matrix.csv": "transition_matrix_v1.csv",
            "state_statistics.csv": "state_statistics_v1.csv",
            "state_meta.json": "online_state_machine_v1_meta.json",
        }
        for src_name, dst_name in bundle_map.items():
            src = self.out_dir / src_name
            if src.exists():
                (reports_dir / dst_name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

        print("[ONLINE STATE MACHINE] report generated")
        return {
            "meta": meta,
            "report_path": str(report_path),
            "transition_count": len(transition_rows),
        }
