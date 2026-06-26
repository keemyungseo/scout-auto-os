"""Policy B runtime shadow — observe only, no live order impact."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

from scout_auto_os.engine.predator.policies import POLICIES, policy_b_soft_50s
from scout_auto_os.engine.predator.predator_output import enrich_predator_candidate
from scout_auto_os.engine.predator.short_watch import ShortFalseAcceptWatch
from scout_auto_os.engine.predator.trade_contract import build_trade_contract
from scout_auto_os.engine.predator.value_gate import GateAction, is_manual_protected
from scout_auto_os.engine.predator.timestamp_fix import resolve_replay_timestamp
from scout_auto_os.engine.predator.value_gate_shadow_logger import ValueGateShadowLogger
from scout_auto_os.storage.db import now_kst

SHADOW_MODES = ("live", "replay")

DEFAULT_POLICY = {"policy": "B", "policy_name": "Soft 50s"}


def load_recommended_policy(data_dir: Path) -> dict:
    path = data_dir / "value_gate_policy" / "recommended_policy.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return dict(DEFAULT_POLICY)


from scout_auto_os.engine.predator.prediction_key import (
    make_prediction_key,
    make_scan_id,
    prediction_key_from_row,
)


class ShadowPredictionCache:
    """Read-only cache keyed by trade_key / scan_id — never symbol|side alone."""

    def __init__(self, trade_dna_dir: Path) -> None:
        value_rows = self._load_csv(trade_dna_dir / "value_prediction.csv")
        dna_rows = self._load_csv(trade_dna_dir / "dna_prediction_model.csv")
        self._by_value: dict[str, dict] = {}
        self._by_dna: dict[str, dict] = {}
        for row in value_rows:
            pk = prediction_key_from_row(row)
            if pk:
                self._by_value[pk] = row
        for row in dna_rows:
            pk = prediction_key_from_row(row)
            if pk:
                self._by_dna[pk] = row

    @staticmethod
    def _load_csv(path: Path) -> list[dict]:
        if not path.exists():
            return []
        with path.open(encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def lookup(self, prediction_key: str) -> dict:
        if not prediction_key:
            return self._fallback("missing_prediction_key")
        val = self._by_value.get(prediction_key, {})
        dna = self._by_dna.get(prediction_key, {})
        if not val and not dna:
            return self._fallback("key_not_found", prediction_key=prediction_key)
        runner = float(dna.get("runner_probability", 0.5))
        pred_type = dna.get("predicted_type", val.get("trade_type_id", "TYPE_0"))
        return {
            "value_score": float(val.get("value_score", 50)),
            "runner_probability": runner,
            "predicted_dna_type": pred_type,
            "predicted_roi": float(val.get("pred_expected_roi", 0)),
            "predicted_peak_roi": float(val.get("pred_expected_peak_roi", 0)),
            "predicted_drawdown": float(val.get("pred_expected_drawdown", 0)),
            "predicted_win_prob": float(val.get("pred_expected_win_prob", runner)),
            "entry_score": float(dna.get("entry_score", val.get("entry_score", 0))),
            "prediction_source": "trade_key_cache",
            "prediction_key": prediction_key,
        }

    @staticmethod
    def _fallback(reason: str, *, prediction_key: str = "") -> dict:
        return {
            "value_score": 50.0,
            "runner_probability": 0.5,
            "predicted_dna_type": "TYPE_0",
            "predicted_roi": 0.0,
            "predicted_peak_roi": 0.0,
            "predicted_drawdown": 0.0,
            "predicted_win_prob": 0.5,
            "entry_score": 0.0,
            "prediction_source": reason,
            "prediction_key": prediction_key,
        }


def evaluate_policy_b_shadow(row: dict, *, manual_context: dict | None = None) -> dict:
    ctx = manual_context or {}
    if is_manual_protected(ctx):
        return {
            "action": GateAction.NO_ACTION.value,
            "recommended_size": 0.0,
            "reason": "manual_protected",
        }
    return policy_b_soft_50s(row)


def baseline_decision(
    *,
    symbol: str,
    occupied: set[str],
    locked: set[str],
    can_enter: bool,
    manual_context: dict | None = None,
) -> tuple[str, float, str]:
    if is_manual_protected(manual_context or {}):
        return GateAction.NO_ACTION.value, 0.0, "manual_protected"
    if symbol in locked:
        return GateAction.SKIP.value, 0.0, "manual_lock"
    if symbol in occupied:
        return GateAction.SKIP.value, 0.0, "already_occupied"
    if not can_enter:
        return GateAction.SKIP.value, 0.0, "risk_or_slots_blocked"
    return GateAction.ENTER.value, 1.0, "baseline_predator"


class ValueGateRuntimeShadow:
    """
    Shadow-only Policy B layer.
    Never calls ExecutionEngine or modifies entry sizing.
    """

    def __init__(self, data_dir: Path, *, enabled: bool = True, mode: str = "live") -> None:
        if mode not in SHADOW_MODES:
            raise ValueError(f"mode must be one of {SHADOW_MODES}")
        self.data_dir = data_dir
        self.enabled = enabled
        self.mode = mode
        self.out_dir = data_dir / "runtime_shadow"
        self.logger = ValueGateShadowLogger(self.out_dir)
        self.short_watch = ShortFalseAcceptWatch(self.out_dir)
        rec = load_recommended_policy(data_dir)
        self.policy_key = rec.get("policy", "B")
        self.policy_name = rec.get("policy_name", POLICIES.get(self.policy_key, {}).get("name", "Soft 50s"))
        self.pred_cache = ShadowPredictionCache(data_dir / "trade_dna")
        self._contracts_today = 0

    def record_candidate(
        self,
        scan_time: str,
        candidate: dict,
        *,
        side: str = "long",
        occupied: set[str] | None = None,
        locked: set[str] | None = None,
        can_enter: bool = True,
        manual_context: dict | None = None,
    ) -> dict | None:
        """Log one predator candidate — shadow only. Returns None if replay row lacks timestamp."""
        occupied = occupied or set()
        locked = locked or set()
        sym = candidate.get("symbol", "").upper()
        side_u = side.upper()

        if self.mode == "replay":
            ts = scan_time or resolve_replay_timestamp(candidate)
            if not ts:
                return None
        else:
            ts = scan_time or now_kst()

        trade_key = candidate.get("trade_key") or make_scan_id(ts, sym, side_u)
        scan_id = make_scan_id(ts, sym, side_u)
        prediction_key = make_prediction_key(
            trade_key=trade_key, scan_id=scan_id, scan_time=ts, symbol=sym, side=side_u,
        )
        preds = self.pred_cache.lookup(prediction_key)
        preds["symbol"] = sym
        preds["direction"] = side_u.lower()

        policy_row = {
            "value_score": preds["value_score"],
            "runner_probability": preds["runner_probability"],
            "predicted_dna_type": preds["predicted_dna_type"],
            "predicted_drawdown": preds["predicted_drawdown"],
            "predicted_win_prob": preds["predicted_win_prob"],
        }
        pb = evaluate_policy_b_shadow(policy_row, manual_context=manual_context)
        b_dec, b_size = pb["action"], float(pb["recommended_size"])
        b_reason = pb.get("reason", "")

        base_dec, base_size, base_reason = baseline_decision(
            symbol=sym,
            occupied=occupied,
            locked=locked,
            can_enter=can_enter,
            manual_context=manual_context,
        )

        manual_lock = sym in locked or bool((manual_context or {}).get("manual_lock"))
        source = (manual_context or {}).get("source", candidate.get("source", "BOT"))
        auto_manage = (manual_context or {}).get("auto_manage", candidate.get("auto_manage", True))

        enriched = enrich_predator_candidate(
            {"symbol": sym, "side": side_u.lower(), "entry_score": preds.get("entry_score", 0)},
            preds,
            position=manual_context,
        )
        contract = build_trade_contract({
            **enriched,
            "recommended_size": b_size,
            "gate_action": b_dec,
            "gate_reason": b_reason,
        })
        self._contracts_today += 1

        row = {
            "timestamp": ts,
            "scan_id": scan_id,
            "trade_key": trade_key,
            "prediction_key": prediction_key,
            "symbol": sym,
            "side": side_u,
            "baseline_decision": base_dec,
            "baseline_size": base_size,
            "policy_b_decision": b_dec,
            "policy_b_size": b_size,
            "value_score": round(float(preds["value_score"]), 2),
            "runner_prob": round(float(preds["runner_probability"]), 4),
            "predicted_dna_type": preds["predicted_dna_type"],
            "predicted_roi": round(float(preds["predicted_roi"]), 4),
            "predicted_peak_roi": round(float(preds["predicted_peak_roi"]), 4),
            "predicted_drawdown": round(float(preds["predicted_drawdown"]), 4),
            "predicted_win_prob": round(float(preds["predicted_win_prob"]), 4),
            "reason": f"baseline={base_reason}; policy_b={b_reason}; src={preds.get('prediction_source')}",
            "manual_lock": int(bool(manual_lock)),
            "source": str(source).upper() if str(source).upper() in ("MANUAL", "BOT", "AUTO") else "UNKNOWN",
            "auto_manage": int(bool(auto_manage)),
            "trade_contract": contract,
            "shadow_mode": self.mode,
        }
        self.logger.append(row)
        self.short_watch.maybe_record(row)
        return row

    def on_scan(
        self,
        scan_time: str,
        candidates: list[dict],
        *,
        occupied: set[str],
        locked: set[str],
        can_enter: bool,
    ) -> list[dict]:
        if not self.enabled:
            return []
        logged = []
        for c in candidates:
            sym = c.get("symbol", "").upper()
            side = c.get("side", "long")
            manual_ctx = None
            if sym in locked:
                manual_ctx = {"symbol": sym, "manual_lock": 1, "auto_manage": 0, "source": "MANUAL"}
            cand = {
                **c,
                "trade_key": c.get("trade_key") or make_scan_id(scan_time, sym, side),
            }
            row = self.record_candidate(
                scan_time, cand, side=side,
                occupied=occupied, locked=locked, can_enter=can_enter,
                manual_context=manual_ctx,
            )
            if row is not None:
                logged.append(row)
        self.refresh_summary()
        return logged

    def replay_backfill(self, replay_rows: list[dict]) -> tuple[list[dict], list[dict]]:
        """Rebuild shadow CSV from replay bundle — original timestamps only."""
        if self.mode != "replay":
            raise ValueError("replay_backfill requires mode='replay'")
        self.logger.reset()
        watch_path = self.short_watch.path
        if watch_path.exists():
            watch_path.unlink()
        self.short_watch.__init__(self.out_dir)
        logged: list[dict] = []
        skipped: list[dict] = []
        for r in replay_rows:
            ts = resolve_replay_timestamp(r)
            if not ts:
                skipped.append({
                    "trade_key": r.get("trade_key", ""),
                    "symbol": r.get("symbol", ""),
                    "diagnosis": "MISSING_ORIGINAL_TIMESTAMP",
                })
                continue
            candidate = {
                "symbol": r["symbol"],
                "side": r.get("direction", r.get("side", "long")),
                "entry_score": r.get("entry_score", 0),
                "trade_key": r.get("trade_key", ""),
                "scan_kst": ts,
            }
            row = self.record_candidate(
                ts,
                candidate,
                side=r.get("direction", r.get("side", "long")),
                occupied=set(),
                locked=set(),
                can_enter=True,
            )
            if row is not None:
                logged.append(row)
        self.refresh_summary()
        return logged, skipped

    def refresh_summary(self) -> dict:
        today = now_kst()[:10]
        rows = self.logger.read_all() if self.mode == "replay" else self.logger.rows_today()
        scores = [float(r["value_score"]) for r in rows if r.get("value_score")]
        enter_n = sum(1 for r in rows if r.get("policy_b_decision") == GateAction.ENTER.value)
        skip_n = sum(1 for r in rows if r.get("policy_b_decision") == GateAction.SKIP.value)
        no_action_n = sum(1 for r in rows if r.get("policy_b_decision") == GateAction.NO_ACTION.value)
        shadow_only_n = sum(1 for r in rows if r.get("policy_b_decision") == GateAction.SHADOW_ONLY.value)
        enter_by_side: dict[str, int] = {}
        skip_by_side: dict[str, int] = {}
        for r in rows:
            s = r.get("side", "LONG")
            if r.get("policy_b_decision") == GateAction.ENTER.value:
                enter_by_side[s] = enter_by_side.get(s, 0) + 1
            if r.get("policy_b_decision") == GateAction.SKIP.value:
                skip_by_side[s] = skip_by_side.get(s, 0) + 1
        summary = {
            "last_update": now_kst(),
            "policy_name": self.policy_name,
            "policy_key": self.policy_key,
            "mode": "SHADOW_ONLY",
            "shadow_logger_mode": self.mode,
            "total_candidates_today": len(rows),
            "policy_enter_count": enter_n,
            "policy_skip_count": skip_n,
            "policy_no_action_count": no_action_n,
            "shadow_only_count": shadow_only_n,
            "short_watch_count": self.short_watch.count_today(today),
            "avg_value_score": round(statistics.mean(scores), 2) if scores else 0,
            "enter_by_side": enter_by_side,
            "skip_by_side": skip_by_side,
        }
        self.logger.write_summary(summary)
        return summary

    def write_report(self) -> Path:
        path = self.out_dir / "value_gate_runtime_shadow_report.md"
        s = self.logger.read_summary() or self.refresh_summary()
        lines = [
            "# Value Gate Runtime Shadow V1",
            "",
            f"**Policy:** {s.get('policy_name')} ({s.get('policy_key')}) — **SHADOW ONLY**",
            "",
            "## Status",
            "",
            f"- Last update: {s.get('last_update')}",
            f"- Candidates today: {s.get('total_candidates_today')}",
            f"- Policy ENTER: {s.get('policy_enter_count')}",
            f"- Policy SKIP: {s.get('policy_skip_count')}",
            f"- Short watch: {s.get('short_watch_count')}",
            "",
            "## Safety",
            "",
            "- No live order impact",
            "- manual_lock / MANUAL → NO_ACTION",
            "",
            "## Join keys",
            "",
            "`timestamp`, `symbol`, `side`, `scan_id` → future label columns",
        ]
        path.write_text("\n".join(lines), encoding="utf-8")
        return path
