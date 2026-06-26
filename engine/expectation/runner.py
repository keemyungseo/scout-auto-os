"""Expectation Engine V1 orchestrator."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from scout_auto_os.engine.expectation.curve_builder import ExpectedPath, ExpectedPathStore, build_expected_path
from scout_auto_os.engine.expectation.dynamic_thesis import (
    ThesisTransition,
    maybe_extension_thesis,
)
from scout_auto_os.engine.expectation.expectation_score import ExpectationScoreResult, compute_expectation_score
from scout_auto_os.engine.expectation.progress_tracker import ProgressSnapshot, compute_progress
from scout_auto_os.engine.expectation.thesis_state_machine import ThesisStateResult, compute_thesis_state
from scout_auto_os.engine.position_evaluation.thesis import ThesisStore, TradeThesis
from scout_auto_os.storage.db import now_kst

PKG_FALLBACK_DATA = Path(__file__).resolve().parents[2] / "data"


@dataclass
class ExpectationReview:
    progress: ProgressSnapshot
    score: ExpectationScoreResult
    state: ThesisStateResult
    path: ExpectedPath
    extension: ThesisTransition | None = None
    new_thesis: TradeThesis | None = None
    new_path: ExpectedPath | None = None


class ExpectationRunner:
    def __init__(self, config: dict, data_dir: Path, thesis_store: ThesisStore | None = None) -> None:
        ex = config.get("expectation", {})
        self.enabled = bool(ex.get("enabled", True))
        self.data_dir = data_dir
        self.path_store = ExpectedPathStore(data_dir)
        self.thesis_store = thesis_store
        self._prior_state: dict[str, str] = {}
        self.dir = data_dir / "expectation"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.review_path = self.dir / "expectation_review.csv"
        self.transition_path = self.dir / "thesis_transition.csv"
        self._ensure_csv(
            self.review_path,
            [
                "timestamp", "thesis_id", "position_id", "symbol", "side",
                "elapsed_minutes", "expected_roi", "current_roi", "expected_progress",
                "progress_ratio", "progress_delta", "expectation_score", "expectation_label",
                "thesis_state", "thesis_version", "curve_version", "thesis_transition_reason",
            ],
        )
        self._ensure_csv(
            self.transition_path,
            [
                "timestamp", "position_id", "symbol", "transition_type",
                "prior_thesis_id", "new_thesis_id", "reason", "thesis_version",
            ],
        )

    @staticmethod
    def _ensure_csv(path: Path, fields: list[str]) -> None:
        if not path.exists():
            with path.open("w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=fields).writeheader()

    def create_path_for_thesis(self, thesis: TradeThesis) -> ExpectedPath:
        try:
            ep = build_expected_path(
                thesis.thesis_id,
                thesis.position_id,
                thesis.symbol,
                thesis.side,
                thesis.expected_return_pct,
                thesis.expected_horizon_min,
                thesis.success_probability,
                self.data_dir,
            )
        except FileNotFoundError:
            ep = build_expected_path(
                thesis.thesis_id,
                thesis.position_id,
                thesis.symbol,
                thesis.side,
                thesis.expected_return_pct,
                thesis.expected_horizon_min,
                thesis.success_probability,
                PKG_FALLBACK_DATA,
            )
        self.path_store.append(ep)
        return ep

    def evaluate(
        self,
        thesis: TradeThesis,
        *,
        elapsed_min: int,
        current_roi: float,
        peak_roi: float,
        momentum_alive: bool,
        trend_alive: bool,
        volume_alive: bool,
        exit_pressure: float,
    ) -> ExpectationReview | None:
        if not self.enabled:
            return None
        path = self.path_store.get_by_thesis(thesis.thesis_id)
        if not path:
            path = self.create_path_for_thesis(thesis)

        progress = compute_progress(path, elapsed_min, current_roi)
        score = compute_expectation_score(
            progress,
            momentum_alive=momentum_alive,
            trend_alive=trend_alive,
            volume_alive=volume_alive,
            peak_roi=peak_roi,
            expected_peak_window=path.expected_peak_window_min,
            elapsed_min=elapsed_min,
            exit_pressure=exit_pressure,
            entry_success_prob=thesis.success_probability,
        )
        prior = self._prior_state.get(thesis.position_id, "EARLY")
        state = compute_thesis_state(
            progress,
            score,
            prior_state=prior,
            elapsed_min=elapsed_min,
            expected_horizon=thesis.expected_horizon_min,
            expected_return=thesis.expected_return_pct,
            current_roi=current_roi,
            max_hold_min=thesis.max_hold_minutes,
        )
        self._prior_state[thesis.position_id] = state.state

        ext_trans = None
        new_thesis = None
        new_path = None
        if state.state in ("THESIS_COMPLETE", "OUTPERFORM"):
            new_thesis, new_path, ext_trans = maybe_extension_thesis(
                thesis, path,
                current_roi=current_roi,
                elapsed_min=elapsed_min,
                thesis_state=state.state,
                data_dir=self.data_dir,
            )
            if ext_trans and self.thesis_store and new_thesis and new_path:
                self.thesis_store.append(new_thesis)
                self.path_store.append(new_path)
                self._log_transition(thesis.position_id, thesis.symbol, ext_trans)
                thesis = new_thesis
                path = new_path
                state = ThesisStateResult("OUTPERFORM", "extension_thesis_active_trailing", state.state)

        self._log_review(thesis, path, progress, score, state)
        return ExpectationReview(
            progress=progress,
            score=score,
            state=state,
            path=path,
            extension=ext_trans,
            new_thesis=new_thesis,
            new_path=new_path,
        )

    def _log_review(self, thesis, path, progress, score, state) -> None:
        row = {
            "timestamp": now_kst(),
            "thesis_id": thesis.thesis_id,
            "position_id": thesis.position_id,
            "symbol": thesis.symbol,
            "side": thesis.side,
            "elapsed_minutes": progress.current_elapsed,
            "expected_roi": progress.expected_roi_now,
            "current_roi": progress.current_roi,
            "expected_progress": progress.expected_progress,
            "progress_ratio": progress.progress_ratio,
            "progress_delta": progress.progress_delta,
            "expectation_score": score.score,
            "expectation_label": score.label,
            "thesis_state": state.state,
            "thesis_version": path.thesis_version,
            "curve_version": path.curve_version,
            "thesis_transition_reason": state.transition_reason,
        }
        fields = list(row.keys())
        with self.review_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writerow(row)

    def _log_transition(self, position_id: str, symbol: str, trans: ThesisTransition) -> None:
        row = {
            "timestamp": now_kst(),
            "position_id": position_id,
            "symbol": symbol,
            "transition_type": trans.transition_type,
            "prior_thesis_id": trans.prior_thesis_id,
            "new_thesis_id": trans.new_thesis_id,
            "reason": trans.reason,
            "thesis_version": trans.thesis_version,
        }
        with self.transition_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            w.writerow(row)

    def on_close(self, position_id: str) -> None:
        self._prior_state.pop(position_id, None)

    def write_report(self) -> Path:
        out = self.dir / "expectation_report.md"
        reviews = 0
        if self.review_path.exists():
            reviews = max(0, sum(1 for _ in self.review_path.open(encoding="utf-8")) - 1)
        paths = 0
        if self.path_store.path.exists():
            paths = sum(1 for ln in self.path_store.path.read_text(encoding="utf-8").splitlines() if ln.strip())
        body = "\n".join([
            "# Expectation Engine V1 Report",
            "",
            f"- Expected paths: **{paths}**",
            f"- Expectation reviews: **{reviews}**",
            "",
            "## Lifecycle stack",
            "",
            "Search Engine -> Trade Thesis -> **Expectation Engine** -> Position Evaluation -> Exit",
            "",
            "## Final questions",
            "",
            "1. Entry expectation vs current compared each review via progress_ratio.",
            "2. Expectation Score 0-100 from research curve + alive signals - exit pressure.",
            "3. HEI +30%: THESIS_COMPLETE -> Extension Thesis with trailing mode.",
            "4. MET 2d: elapsed >> horizon -> THESIS_FAILED / EXIT_READY + score drop.",
            "5. Long/Short share framework; curves from separate research sources.",
            "6. Pre-LIVE: validate expectation_review.csv on real positions; tune extension thresholds.",
            "",
            f"_Generated {now_kst()}_",
        ])
        out.write_text(body, encoding="utf-8")
        return out
