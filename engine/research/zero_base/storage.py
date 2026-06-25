"""Zero-Base output paths and CSV writers."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from season2_p37_scout_decision_hierarchy import write_csv

CANDIDATE_RESULT_FIELDS = [
    "engine", "eval_interval", "split", "top_k", "sample_count",
    "avg_return_2h", "median_return_2h", "win_rate", "profit_factor",
    "trap_rate", "big_winner_capture_rate", "max_drawdown_avg",
    "return_30m_avg", "return_1h_avg", "return_4h_avg", "return_6h_avg",
    "time_to_peak_avg", "time_to_3pct_avg", "time_to_5pct_avg",
    "downside_capture_avg", "score",
    "avg_return_2h_delta_vs_random", "avg_return_2h_delta_vs_a6",
    "beats_random_return", "beats_a6_return", "champion_eligible", "tier",
]

RANDOM_BASELINE_FIELDS = [
    "scan_time_kst", "eval_interval", "split", "draw_id", "symbols",
    "avg_return_2h", "sample_count",
]

CHAMPION_BOARD_FIELDS = [
    "board_rank", "engine", "sample_count", "score", "avg_return_2h",
    "win_rate", "trap_rate", "big_winner_capture_rate", "max_drawdown_avg",
    "avg_return_2h_delta_vs_random", "avg_return_2h_delta_vs_a6",
    "champion_eligible", "tier", "positive_days",
]

FEATURE_DIAG_FIELDS = ["engine", "sample_count", "avg_return_2h", "trap_rate", "verdict"]


class ZeroBaseStore:
    def __init__(self, data_dir: Path) -> None:
        self.root = data_dir / "zero_base"
        self.root.mkdir(parents=True, exist_ok=True)
        self.candidate_path = self.root / "candidate_results.csv"
        self.random_path = self.root / "random_baseline.csv"
        self.champion_path = self.root / "champion_board.csv"
        self.feature_path = self.root / "feature_diagnostics.csv"
        self.report_path = self.root / "zero_base_report.md"
        self.picks_path = self.root / "zero_base_picks.jsonl"
        self.meta_path = self.root / "zero_base_meta.json"

    def write_candidate_results(self, rows: list[dict]) -> None:
        write_csv(self.candidate_path, rows)

    def write_random_baseline(self, rows: list[dict]) -> None:
        write_csv(self.random_path, rows)

    def write_champion_board(self, rows: list[dict]) -> None:
        write_csv(self.champion_path, rows)

    def write_feature_diagnostics(self, rows: list[dict]) -> None:
        write_csv(self.feature_path, rows)

    def write_report(self, text: str) -> None:
        self.report_path.write_text(text, encoding="utf-8")

    def write_meta(self, payload: dict) -> None:
        self.meta_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def append_picks(self, row: dict) -> None:
        with self.picks_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def read_champion_board(self) -> list[dict]:
        if not self.champion_path.exists():
            return []
        with self.champion_path.open(encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def snapshot(self) -> dict:
        board = self.read_champion_board()[:10]
        meta = {}
        if self.meta_path.exists():
            try:
                meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        return {"champion_board_top": board, "meta": meta, "report_path": str(self.report_path)}
