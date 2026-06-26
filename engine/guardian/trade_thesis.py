"""Guardian Trade Thesis — why we entered, for Guardian evaluation context."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass

from scout_auto_os.engine.guardian.decision_rules import expected_horizon_minutes
from scout_auto_os.engine.predator.prediction_key import make_prediction_key, prediction_key_from_row
from scout_auto_os.engine.predator.trade_contract import build_trade_contract

PREDATOR_VERSION = "season3_predator_v1"
DEFAULT_FORMULA = "policy_b_soft_50s"


@dataclass
class GuardianTradeThesis:
    thesis_id: str
    contract_id: str
    prediction_key: str
    symbol: str
    side: str
    predator_version: str
    formula_name: str
    expected_roi: float
    expected_peak_roi: float
    expected_horizon: int
    expected_drawdown: float
    expected_win_prob: float
    value_score: float
    predicted_dna: str
    entry_reason: str
    confidence: float

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "GuardianTradeThesis":
        return cls(
            thesis_id=d["thesis_id"],
            contract_id=d["contract_id"],
            prediction_key=d.get("prediction_key", d.get("contract_id", "")),
            symbol=d["symbol"],
            side=str(d.get("side", "long")).lower(),
            predator_version=d.get("predator_version", PREDATOR_VERSION),
            formula_name=d.get("formula_name", DEFAULT_FORMULA),
            expected_roi=float(d.get("expected_roi", 0)),
            expected_peak_roi=float(d.get("expected_peak_roi", 0)),
            expected_horizon=int(d.get("expected_horizon", 120)),
            expected_drawdown=float(d.get("expected_drawdown", 0)),
            expected_win_prob=float(d.get("expected_win_prob", 0)),
            value_score=float(d.get("value_score", 0)),
            predicted_dna=d.get("predicted_dna", d.get("dna_type", "")),
            entry_reason=d.get("entry_reason", ""),
            confidence=float(d.get("confidence", 0)),
        )

    def contract_summary(self) -> dict:
        return {
            "contract_id": self.contract_id,
            "prediction_key": self.prediction_key,
            "expected_roi": self.expected_roi,
            "expected_peak_roi": self.expected_peak_roi,
            "expected_horizon": self.expected_horizon,
            "expected_drawdown": self.expected_drawdown,
            "expected_win_prob": self.expected_win_prob,
            "value_score": self.value_score,
            "predicted_dna": self.predicted_dna,
            "formula_name": self.formula_name,
        }


def _thesis_id_for(contract_id: str) -> str:
    digest = hashlib.sha256(contract_id.encode()).hexdigest()[:12]
    return f"th_{digest}"


def _score_band(score: float) -> str:
    if score < 50:
        return "below_50"
    if score < 60:
        return "50_59"
    if score < 70:
        return "60_69"
    if score < 80:
        return "70_79"
    return "80_plus"


def compute_confidence(
    *,
    value_score: float,
    expected_win_prob: float,
    runner_probability: float = 0.5,
) -> float:
    """0–100 explainable confidence from entry signals — not ML."""
    vs = max(0.0, min(100.0, value_score))
    wp = max(0.0, min(1.0, expected_win_prob)) * 100.0
    rp = max(0.0, min(1.0, runner_probability)) * 100.0
    return round(0.50 * vs + 0.30 * wp + 0.20 * rp, 2)


def build_entry_reason(
    *,
    gate_reason: str = "",
    predicted_dna: str = "",
    value_score: float = 0.0,
    side: str = "long",
    runner_probability: float = 0.5,
) -> str:
    """Human-readable why-we-entered string."""
    parts: list[str] = []
    if gate_reason:
        parts.append(f"gate={gate_reason}")
    else:
        parts.append("gate=policy_b_enter")
    parts.append(f"dna={predicted_dna or 'UNKNOWN'}")
    if predicted_dna == "TYPE_0":
        parts.append(f"runner={runner_probability:.2f}")
    parts.append(f"side={side.lower()}")
    parts.append(f"score_band={_score_band(value_score)}")
    parts.append(f"value_score={value_score:.1f}")
    return "; ".join(parts)


def build_thesis_from_predator_entry(
    candidate: dict,
    contract: dict,
    *,
    gate_reason: str = "",
    formula_name: str = DEFAULT_FORMULA,
    predator_version: str = PREDATOR_VERSION,
) -> GuardianTradeThesis:
    """Create thesis at Predator entry — attach to Guardian downstream."""
    trade_key = candidate.get("trade_key", "")
    scan_time = candidate.get("scan_kst") or candidate.get("timestamp", "")
    sym = contract.get("symbol", candidate.get("symbol", "")).upper()
    side = str(contract.get("side", candidate.get("side", "long"))).lower()
    prediction_key = make_prediction_key(
        trade_key=trade_key,
        scan_time=scan_time,
        symbol=sym,
        side=side,
    )
    contract_id = trade_key or prediction_key
    dna = contract.get("dna_type", candidate.get("predicted_dna_type", ""))
    value_score = float(contract.get("value_score", 0))
    win_prob = float(contract.get("expected_win_prob", 0))
    runner = float(candidate.get("runner_probability", contract.get("runner_probability", 0.5)))
    horizon = int(contract.get("expected_horizon") or expected_horizon_minutes(contract))

    return GuardianTradeThesis(
        thesis_id=_thesis_id_for(contract_id),
        contract_id=contract_id,
        prediction_key=prediction_key,
        symbol=sym,
        side=side,
        predator_version=predator_version,
        formula_name=formula_name,
        expected_roi=float(contract.get("expected_roi", 0)),
        expected_peak_roi=float(contract.get("expected_peak_roi", 0)),
        expected_horizon=horizon,
        expected_drawdown=float(contract.get("expected_drawdown", 0)),
        expected_win_prob=win_prob,
        value_score=value_score,
        predicted_dna=dna,
        entry_reason=build_entry_reason(
            gate_reason=gate_reason or contract.get("gate_reason", ""),
            predicted_dna=dna,
            value_score=value_score,
            side=side,
            runner_probability=runner,
        ),
        confidence=compute_confidence(
            value_score=value_score,
            expected_win_prob=win_prob,
            runner_probability=runner,
        ),
    )


def build_thesis_from_replay_row(
    row: dict,
    *,
    formula_name: str = DEFAULT_FORMULA,
) -> GuardianTradeThesis:
    """Rebuild thesis from 157-trade replay bundle row."""
    contract = build_trade_contract({
        "symbol": row.get("symbol", ""),
        "side": row.get("side", row.get("direction", "long")),
        "predicted_roi": row.get("predicted_roi", 0),
        "predicted_peak_roi": row.get("predicted_peak_roi", 0),
        "predicted_drawdown": row.get("predicted_drawdown", 0),
        "predicted_win_prob": row.get("predicted_win_prob", 0),
        "value_score": row.get("value_score", 0),
        "predicted_dna_type": row.get("predicted_dna_type", ""),
        "gate_action": "ENTER",
        "gate_reason": row.get("gate_reason", ""),
    })
    contract_id = row.get("trade_key", "")
    return build_thesis_from_predator_entry(
        {
            **row,
            "trade_key": contract_id,
            "scan_kst": row.get("scan_kst", contract_id.split("|")[0] if "|" in contract_id else ""),
            "runner_probability": row.get("runner_probability", 0.5),
        },
        contract,
        gate_reason=row.get("gate_reason", ""),
        formula_name=formula_name,
    )


def thesis_to_json_line(thesis: GuardianTradeThesis) -> str:
    return json.dumps(thesis.to_dict(), ensure_ascii=False)
