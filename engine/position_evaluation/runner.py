"""Position Evaluation Engine V1 runner."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

from scout_research_r006_pilot_execution_engine import Bar

from scout_auto_os.engine.position_evaluation.decision import PositionDecision, decide
from scout_auto_os.engine.position_evaluation.evaluator import PositionEvaluator
from scout_auto_os.engine.position_evaluation.logger import PositionEvaluationLogger
from scout_auto_os.engine.position_evaluation.manual_guard import (
    can_enter,
    guard_row,
    is_protected,
)
from scout_auto_os.engine.position_evaluation.side_rules import protective_stop_hit
from scout_auto_os.engine.position_evaluation.thesis import (
    ThesisStore,
    TradeThesis,
    build_thesis_for_entry,
)
from scout_auto_os.engine.expectation.runner import ExpectationRunner
from scout_auto_os.engine.state_exit_engine import StateExitDecision
from scout_auto_os.storage.db import now_kst

KST_FMT = "%Y-%m-%d %H:%M:%S"
FORCED_REVIEW_MINUTES = (30, 60, 120)


class PositionEvaluationRunner:
    def __init__(self, config: dict, data_dir: Path) -> None:
        pe = config.get("position_evaluation", {})
        self.enabled = bool(pe.get("enabled", True))
        self.min_review_sec = int(pe.get("min_review_interval_sec", 60))
        self.default_review_min = int(pe.get("review_interval_minutes", 5))
        self.override_state_exit = bool(pe.get("override_state_exit", True))
        self.store = ThesisStore(data_dir)
        self.logger = PositionEvaluationLogger(data_dir)
        self.evaluator = PositionEvaluator()
        self.expectation = ExpectationRunner(config, data_dir, thesis_store=self.store)
        self._last_eval: dict[str, float] = {}
        self.data_dir = data_dir

    def create_thesis_for_position(
        self,
        position_id: str,
        symbol: str,
        side: str,
        entry_time: str,
        entry_price: float,
        *,
        source: str = "BOT",
        auto_manage: bool = True,
        engine: str = "",
        entry_score: float = 0.0,
        rank: int = 1,
        expected_ev: float = 0.0,
        primary_reason: str = "",
        manual_lock: bool = False,
    ) -> TradeThesis:
        if manual_lock or source.upper() == "MANUAL" or not auto_manage:
            source = "MANUAL"
            auto_manage = False
        thesis = build_thesis_for_entry(
            position_id, symbol, side, entry_time, entry_price,
            source=source,
            auto_manage=auto_manage,
            engine=engine,
            entry_score=entry_score,
            rank=rank,
            expected_ev=expected_ev,
            primary_reason=primary_reason,
        )
        self.store.append(thesis)
        if self.expectation.enabled:
            self.expectation.create_path_for_thesis(thesis)
        if is_protected({"source": source, "auto_manage": int(auto_manage), "manual_lock": int(manual_lock)}):
            self.logger.log_guard(guard_row(
                {"symbol": symbol, "position_id": position_id, "source": source,
                 "auto_manage": int(auto_manage), "manual_lock": int(manual_lock)},
                "thesis_created_manual",
                "manual thesis — bot will not enter/exit/modify",
            ))
        return thesis

    @staticmethod
    def hold_minutes(entry_time: str, now: str | None = None) -> int:
        now = now or now_kst()
        try:
            t0 = datetime.strptime(entry_time, KST_FMT)
            t1 = datetime.strptime(now, KST_FMT)
            return int((t1 - t0).total_seconds() / 60)
        except ValueError:
            return 0

    def _due(self, position_id: str, thesis: TradeThesis, elapsed: int) -> bool:
        now = time.time()
        interval = max(self.min_review_sec, thesis.review_interval_minutes * 60)
        if now - self._last_eval.get(position_id, 0) >= interval:
            return True
        if elapsed in FORCED_REVIEW_MINUTES:
            return True
        if elapsed >= thesis.max_hold_minutes:
            return True
        return False

    def evaluate_position(
        self,
        position: dict,
        bars: list[Bar],
        pnl_pct: float,
        state_decision: StateExitDecision | None = None,
    ) -> tuple[PositionDecision | None, StateExitDecision | None]:
        if not self.enabled:
            return None, state_decision

        pid = position["position_id"]
        protected = is_protected(position)
        side = position.get("side", "LONG")
        thesis = self.store.get_by_position(pid)
        if not thesis:
            thesis = self.create_thesis_for_position(
                pid, position["symbol"], side,
                position["entry_time"], float(position["entry_price"]),
                source=position.get("source", "BOT"),
                auto_manage=bool(int(position.get("auto_manage", 1))),
                engine=position.get("engine", ""),
                entry_score=float(position.get("a6_score") or 0),
                manual_lock=bool(int(position.get("manual_lock", 0))),
            )

        elapsed = self.hold_minutes(position["entry_time"])
        if protected:
            dec = decide(
                self.evaluator.evaluate(
                    thesis, pid, side, float(position["entry_price"]),
                    float(position.get("current_price") or bars[-1].c if bars else position["entry_price"]),
                    elapsed, bars or [],
                ),
                thesis,
                is_manual=True,
            )
            self.logger.log_review(self._review_row(thesis, position, dec, pnl_pct, elapsed, bars))
            self.logger.log_guard(guard_row(position, "observe_only", dec.action))
            return dec, StateExitDecision(False, review_reason="manual_observe_only")

        if not self._due(pid, thesis, elapsed) and not (state_decision and state_decision.should_exit):
            return None, state_decision

        current_px = float(position.get("current_price") or (bars[-1].c if bars else position["entry_price"]))
        metrics = self.evaluator.evaluate(
            thesis, pid, side, float(position["entry_price"]),
            current_px, elapsed, bars or [],
        )
        exp_review = self.expectation.evaluate(
            thesis,
            elapsed_min=elapsed,
            current_roi=metrics.current_roi,
            peak_roi=metrics.peak_roi,
            momentum_alive=metrics.momentum_alive,
            trend_alive=metrics.trend_alive,
            volume_alive=metrics.volume_alive,
            exit_pressure=metrics.exit_pressure_score,
        )
        if exp_review and exp_review.new_thesis:
            thesis = exp_review.new_thesis
        dec = decide(metrics, thesis, expectation=exp_review)

        if protective_stop_hit(side, bars or [], float(position["entry_price"]), thesis.initial_stop_pct):
            dec = PositionDecision("EXIT", True, dec.reason_lines + ["side-aware protective stop hit"])

        merged = self._merge_with_state(dec, state_decision, metrics, thesis, elapsed)
        self.logger.log_review(self._review_row(thesis, position, merged, pnl_pct, elapsed, bars, metrics, exp_review))
        self.logger.log_decision({
            "thesis_id": thesis.thesis_id,
            "position_id": pid,
            "symbol": position["symbol"],
            "action": merged.action,
            "should_exit": merged.should_exit,
            "action_reason": merged.action_reason,
        })
        self._last_eval[pid] = time.time()

        exit_dec = StateExitDecision(
            merged.should_exit,
            reason=f"pe_{merged.action.lower()}",
            review_reason=merged.action,
        ) if merged.should_exit else StateExitDecision(False, review_reason=merged.action)

        if self.override_state_exit and merged.should_exit:
            return merged, exit_dec
        if state_decision and state_decision.should_exit:
            return merged, state_decision
        return merged, exit_dec if merged.should_exit else StateExitDecision(False, review_reason=merged.action)

    def _merge_with_state(self, dec, state_decision, metrics, thesis, elapsed):
        if dec.should_exit:
            return dec
        if state_decision and state_decision.should_exit and state_decision.reason == "protective_sl":
            return PositionDecision("EXIT", True, dec.reason_lines + ["legacy state protective_sl overridden by side rules"])
        if metrics.current_roi >= thesis.expected_return_pct * 2 and elapsed >= thesis.expected_horizon_min:
            if state_decision and state_decision.review_reason == "alive_score_strong_hold":
                return PositionDecision(
                    "TRAIL", False,
                    dec.reason_lines + ["override alive_score_strong_hold — ROI well above target, trail not infinite hold"],
                )
        return dec

    def _review_row(self, thesis, position, dec, pnl_pct, elapsed, bars, metrics=None, exp_review=None):
        side = position.get("side", thesis.side)
        if metrics is None:
            metrics = self.evaluator.evaluate(
                thesis, position["position_id"], side,
                float(position["entry_price"]),
                float(position.get("current_price") or position["entry_price"]),
                elapsed, bars or [],
            )
        row = {
            "thesis_id": thesis.thesis_id,
            "position_id": position["position_id"],
            "symbol": position["symbol"],
            "side": side,
            "source": position.get("source", thesis.source),
            "auto_manage": position.get("auto_manage", thesis.auto_manage),
            "entry_time": position["entry_time"],
            "entry_price": position["entry_price"],
            "current_price": metrics.current_price,
            "roi": metrics.current_roi,
            "elapsed_minutes": elapsed,
            "expected_horizon": thesis.expected_horizon_min,
            "expected_return": thesis.expected_return_pct,
            "success_probability": thesis.success_probability,
            "mfe": metrics.mfe,
            "mae": metrics.mae,
            "peak_roi": metrics.peak_roi,
            "drawdown_from_peak": metrics.drawdown_from_peak,
            "thesis_validity_score": metrics.thesis_validity_score,
            "exit_pressure_score": metrics.exit_pressure_score,
            "hold_confidence": metrics.hold_confidence,
            "action": dec.action,
            "action_reason": dec.action_reason,
            "thesis_update_reason": dec.thesis_update_reason or (exp_review.extension.reason if exp_review and exp_review.extension else ""),
            "should_exit": int(dec.should_exit),
        }
        if exp_review:
            row.update({
                "expected_roi": exp_review.progress.expected_roi_now,
                "expected_progress": exp_review.progress.expected_progress,
                "progress_ratio": exp_review.progress.progress_ratio,
                "progress_delta": exp_review.progress.progress_delta,
                "expectation_score": exp_review.score.score,
                "thesis_state": exp_review.state.state,
                "thesis_version": exp_review.path.thesis_version,
                "curve_version": exp_review.path.curve_version,
                "thesis_transition_reason": exp_review.state.transition_reason,
            })
        return row

    def on_close(self, position_id: str) -> None:
        self._last_eval.pop(position_id, None)
        self.evaluator.reset_peak(position_id)
        self.expectation.on_close(position_id)

    def write_report(self) -> Path:
        out = self.data_dir / "position_evaluation" / "position_evaluation_report.md"
        theses = self.store.path.read_text(encoding="utf-8").count("\n") if self.store.path.exists() else 0
        reviews = 0
        if self.logger.review_path.exists():
            reviews = max(0, sum(1 for _ in self.logger.review_path.open(encoding="utf-8")) - 1)
        body = "\n".join([
            "# Position Evaluation Engine V1 Report",
            "",
            f"- Trade theses recorded: **{theses}**",
            f"- Review rows logged: **{reviews}**",
            f"- Enabled: **{self.enabled}**",
            "",
            "## Final questions",
            "",
            "1. Entry thesis recorded per position in `trade_thesis.jsonl` with thesis_id.",
            "2. BOT positions evaluated against expected horizon/return each review cycle.",
            "3. HEI +30% hold override: TRAIL action replaces infinite alive_score hold when ROI >> target.",
            "4. MET 2d+ hold: max_hold_minutes + forced reviews at 30/60/120m trigger EXIT candidates.",
            "5. MANUAL/WLD: source=MANUAL or manual_lock blocks all bot exit/entry on symbol.",
            "6. Pre-LIVE risk: validate on live position_review.csv; tune review_interval on production load.",
            "",
            f"_Generated {now_kst()}_",
        ])
        out.write_text(body, encoding="utf-8")
        return out

    @staticmethod
    def entry_allowed(symbol: str, occupied: set[str], locked: set[str], open_positions: list[dict]) -> bool:
        return can_enter(symbol, occupied, locked, open_positions)
