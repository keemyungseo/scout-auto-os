"""CSV / JSON persistence for Research Engine."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from season2_p37_scout_decision_hierarchy import write_csv

SCAN_FIELDS = [
    "scan_time_kst", "total_symbols", "market_regime", "btc_1h_return",
    "btc_4h_return", "alt_market_strength", "top20_symbols",
]

CANDIDATE_FIELDS = [
    "scan_time_kst", "rank", "symbol", "current_price", "a6_score", "expected_ev",
    "reason_1h", "reason_2h", "range_pct", "volume_ratio", "atr_pct",
    "momentum_15m", "momentum_1h", "compression_score", "breakout_score",
    "btc_context", "selected_by_live_engine",
]

FORWARD_FIELDS = [
    "scan_time_kst", "symbol", "rank", "price_at_scan",
    "price_30m", "return_30m", "price_1h", "return_1h",
    "price_2h", "return_2h", "price_4h", "return_4h",
    "price_6h", "return_6h", "price_12h", "return_12h",
    "max_return_2h", "min_return_2h", "max_drawdown_2h",
    "label_success_2h", "label_big_winner", "label_trap",
]

FORMULA_LEAGUE_FIELDS = [
    "formula_name", "sample_count", "win_rate_30m", "win_rate_1h", "win_rate_2h",
    "avg_return_2h", "median_return_2h", "max_drawdown_avg",
    "big_winner_capture_rate", "trap_rate", "score",
]

FEATURE_LEAGUE_FIELDS = [
    "feature_name", "condition", "sample_count", "win_rate_2h",
    "avg_return_2h", "median_return_2h", "big_winner_rate", "trap_rate", "comment",
]


class ResearchStore:
    def __init__(self, data_dir: Path) -> None:
        self.root = data_dir / "research"
        self.root.mkdir(parents=True, exist_ok=True)
        self.scans_path = self.root / "research_scans.csv"
        self.candidates_path = self.root / "research_candidates.csv"
        self.forward_path = self.root / "research_forward_results.csv"
        self.formula_path = self.root / "formula_league.csv"
        self.feature_path = self.root / "feature_league.csv"
        self.report_path = self.root / "research_report.json"
        self.picks_path = self.root / "formula_picks.jsonl"
        self._ensure_headers()

    def _ensure_headers(self) -> None:
        mapping = [
            (self.scans_path, SCAN_FIELDS),
            (self.candidates_path, CANDIDATE_FIELDS),
            (self.forward_path, FORWARD_FIELDS),
            (self.formula_path, FORMULA_LEAGUE_FIELDS),
            (self.feature_path, FEATURE_LEAGUE_FIELDS),
        ]
        for path, fields in mapping:
            if not path.exists():
                with path.open("w", newline="", encoding="utf-8") as f:
                    csv.DictWriter(f, fieldnames=fields).writeheader()

    def append_rows(self, path: Path, fields: list[str], rows: list[dict]) -> None:
        if not rows:
            return
        with path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            for row in rows:
                w.writerow({k: row.get(k, "") for k in fields})

    def append_scan(self, row: dict) -> None:
        self.append_rows(self.scans_path, SCAN_FIELDS, [row])

    def append_candidates(self, rows: list[dict]) -> None:
        self.append_rows(self.candidates_path, CANDIDATE_FIELDS, rows)

    def append_forward(self, rows: list[dict]) -> None:
        self.append_rows(self.forward_path, FORWARD_FIELDS, rows)

    def write_formula_league(self, rows: list[dict]) -> None:
        write_csv(self.formula_path, rows)

    def write_feature_league(self, rows: list[dict]) -> None:
        write_csv(self.feature_path, rows)

    def write_report(self, payload: dict) -> None:
        self.report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def append_formula_picks(self, scan_time_kst: str, picks: dict[str, list[str]]) -> None:
        with self.picks_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"scan_time_kst": scan_time_kst, "picks": picks}, ensure_ascii=False) + "\n")

    def read_forward_all(self) -> list[dict]:
        if not self.forward_path.exists():
            return []
        with self.forward_path.open(encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def read_formula_league(self) -> list[dict]:
        if not self.formula_path.exists():
            return []
        with self.formula_path.open(encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def read_feature_league(self) -> list[dict]:
        if not self.feature_path.exists():
            return []
        with self.feature_path.open(encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def snapshot(self) -> dict:
        report = {}
        if self.report_path.exists():
            try:
                report = json.loads(self.report_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        return {
            "report": report,
            "scans_csv": str(self.scans_path),
            "candidates_csv": str(self.candidates_path),
            "forward_csv": str(self.forward_path),
            "formula_league": self.read_formula_league()[:10],
            "feature_league": self.read_feature_league()[:10],
        }
