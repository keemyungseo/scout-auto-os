"""Direction Champion Entry DNA orchestrator."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scout_auto_os.engine.research.directional.dna.collector import numeric_feature_keys
from scout_auto_os.engine.research.directional.entry_filter.analyzer import (
    compare_winner_loser_features,
    find_common_dna,
    split_winner_loser,
    summarize_dna_profile,
)
from scout_auto_os.engine.research.directional.entry_filter.collector import (
    collect_direction_champion_signals,
    filter_scans_last_months,
)
from scout_auto_os.engine.research.directional.entry_filter.constants import (
    CHAMPION_TOP_K,
    LOOKBACK_MONTHS,
)
from scout_auto_os.engine.research.directional.entry_filter.report import build_entry_dna_report
from scout_auto_os.engine.research.safe import research_safe
from scout_auto_os.engine.research.zero_base.runner import load_candidates_jsonl, load_forward_klines

KST = timezone(timedelta(hours=9))


def _flatten_signal(s: dict) -> dict:
    row = {k: v for k, v in s.items() if k != "features"}
    for fk, fv in s.get("features", {}).items():
        row[f"feat_{fk}"] = fv
    return row


class DirectionChampionEntryDnaRunner:
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
        self.out_dir = data_dir / "zero_base"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.candidates_path = candidates_path
        self.forward_path = forward_path
        self.lookback_months = lookback_months

    @research_safe("direction_champion_entry_dna")
    def run(self, top_k: int = CHAMPION_TOP_K) -> dict:
        print("[ENTRY DNA] started")
        by_scan = load_candidates_jsonl(self.candidates_path)
        fwd = load_forward_klines(self.forward_path)
        all_scans = sorted(by_scan.keys())
        scans = filter_scans_last_months(all_scans, self.lookback_months)

        long_signals, short_signals = collect_direction_champion_signals(
            by_scan, fwd, scans, top_k=top_k,
        )

        sample_feats = long_signals[0]["features"] if long_signals else (
            short_signals[0]["features"] if short_signals else {}
        )
        feature_keys = numeric_feature_keys(sample_feats)

        long_w, long_l, long_split = split_winner_loser(long_signals)
        short_w, short_l, short_split = split_winner_loser(short_signals)

        long_importance = compare_winner_loser_features(long_w, long_l, feature_keys, "long")
        short_importance = compare_winner_loser_features(short_w, short_l, feature_keys, "short")
        long_profile = summarize_dna_profile(long_importance)
        short_profile = summarize_dna_profile(short_importance)
        common_dna = find_common_dna(long_importance, short_importance)

        dates = [s[:10] for s in scans]
        meta = {
            "lookback_months": self.lookback_months,
            "date_min": min(dates) if dates else None,
            "date_max": max(dates) if dates else None,
            "scan_count": len(scans),
            "top_k": top_k,
            "long_signal_count": len(long_signals),
            "short_signal_count": len(short_signals),
            "feature_count": len(feature_keys),
            "generated_at": datetime.now(KST).isoformat(),
        }

        from season2_p37_scout_decision_hierarchy import write_csv

        signal_rows = [_flatten_signal(s) for s in long_signals + short_signals]
        write_csv(self.out_dir / "direction_champion_signals.csv", signal_rows)
        write_csv(self.out_dir / "entry_dna_feature_importance_long.csv", long_importance)
        write_csv(self.out_dir / "entry_dna_feature_importance_short.csv", short_importance)
        write_csv(self.out_dir / "entry_dna_common.csv", common_dna)

        winner_loser_stats = [
            {"direction": "long", "cohort": "winner", **long_split},
            {"direction": "long", "cohort": "loser", **{k: long_split.get(k) for k in long_split}},
            {"direction": "short", "cohort": "winner", **short_split},
        ]
        write_csv(self.out_dir / "entry_dna_winner_loser_stats.csv", winner_loser_stats)

        report = build_entry_dna_report(
            meta, long_split, short_split,
            long_profile, short_profile,
            long_importance, short_importance, common_dna,
        )
        report_path = self.out_dir / "direction_champion_entry_dna_report.md"
        report_path.write_text(report, encoding="utf-8")
        (self.out_dir / "entry_dna_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        reports_dir = self.pkg_root / "research_bundle" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        bundle_map = {
            "direction_champion_entry_dna_report.md": "direction_champion_entry_dna_v1_report.md",
            "direction_champion_signals.csv": "direction_champion_signals_v1.csv",
            "entry_dna_feature_importance_long.csv": "entry_dna_feature_importance_long_v1.csv",
            "entry_dna_feature_importance_short.csv": "entry_dna_feature_importance_short_v1.csv",
            "entry_dna_common.csv": "entry_dna_common_v1.csv",
            "entry_dna_meta.json": "entry_dna_meta_v1.json",
        }
        for src_name, dst_name in bundle_map.items():
            src = self.out_dir / src_name
            if src.exists():
                (reports_dir / dst_name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

        print("[ENTRY DNA] report generated")
        return {
            "meta": meta,
            "long_profile": long_profile,
            "short_profile": short_profile,
            "common_dna_count": len(common_dna),
            "report_path": str(report_path),
        }
