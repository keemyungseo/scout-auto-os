"""Trade Thesis — entry purpose document per position."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

SEARCH_ENGINE_VERSION = "ranking_engine_v1_frozen"
LONG_MODEL_VERSION = "catboost_ranker_return_minus_dd"
SHORT_MODEL_VERSION = "catboost_ranker_risk_adjusted_short"


@dataclass
class TradeThesis:
    thesis_id: str
    position_id: str
    symbol: str
    side: str
    entry_time: str
    entry_price: float
    entry_score: float
    search_engine_version: str
    model_version: str
    label_version: str
    rank: int
    expected_horizon_min: int
    expected_return_pct: float
    success_probability: float
    primary_reason: str
    secondary_reasons: list[str] = field(default_factory=list)
    initial_stop_pct: float = 8.0
    initial_take_profit_pct: float = 10.0
    invalid_condition: str = "thesis_invalid_if_roi_below_stop_after_min_hold"
    max_hold_minutes: int = 240
    review_interval_minutes: int = 5
    source: str = "BOT"
    auto_manage: bool = True
    engine: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["secondary_reasons"] = list(self.secondary_reasons)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TradeThesis":
        sec = d.get("secondary_reasons") or []
        if isinstance(sec, str):
            try:
                sec = json.loads(sec)
            except json.JSONDecodeError:
                sec = [sec] if sec else []
        return cls(
            thesis_id=d["thesis_id"],
            position_id=d["position_id"],
            symbol=d["symbol"],
            side=d["side"],
            entry_time=d["entry_time"],
            entry_price=float(d["entry_price"]),
            entry_score=float(d.get("entry_score") or 0),
            search_engine_version=d.get("search_engine_version", SEARCH_ENGINE_VERSION),
            model_version=d.get("model_version", LONG_MODEL_VERSION),
            label_version=d.get("label_version", "return_minus_dd"),
            rank=int(d.get("rank") or 1),
            expected_horizon_min=int(d.get("expected_horizon_min") or 120),
            expected_return_pct=float(d.get("expected_return_pct") or 3.0),
            success_probability=float(d.get("success_probability") or 0.5),
            primary_reason=d.get("primary_reason") or "",
            secondary_reasons=list(sec),
            initial_stop_pct=float(d.get("initial_stop_pct") or 8.0),
            initial_take_profit_pct=float(d.get("initial_take_profit_pct") or 10.0),
            invalid_condition=d.get("invalid_condition") or "thesis_invalid_if_roi_below_stop_after_min_hold",
            max_hold_minutes=int(d.get("max_hold_minutes") or 240),
            review_interval_minutes=int(d.get("review_interval_minutes") or 5),
            source=d.get("source", "BOT"),
            auto_manage=bool(d.get("auto_manage", True)),
            engine=d.get("engine", ""),
        )


class ThesisStore:
    def __init__(self, data_dir: Path) -> None:
        self.dir = data_dir / "position_evaluation"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "trade_thesis.jsonl"
        self._cache: dict[str, TradeThesis] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        if not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                t = TradeThesis.from_dict(json.loads(line))
                self._cache[t.thesis_id] = t
                self._cache[t.position_id] = t
            except (json.JSONDecodeError, KeyError, TypeError):
                continue

    def append(self, thesis: TradeThesis) -> None:
        self._cache[thesis.thesis_id] = thesis
        self._cache[thesis.position_id] = thesis
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(thesis.to_dict(), ensure_ascii=False) + "\n")

    def get_by_position(self, position_id: str) -> TradeThesis | None:
        return self._cache.get(position_id)

    def get(self, thesis_id: str) -> TradeThesis | None:
        return self._cache.get(thesis_id)


def build_thesis_for_entry(
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
) -> TradeThesis:
    side = side.upper()
    is_short = side == "SHORT"
    is_manual = source.upper() == "MANUAL" or not auto_manage

    if is_manual:
        return TradeThesis(
            thesis_id=f"th_{uuid.uuid4().hex[:12]}",
            position_id=position_id,
            symbol=symbol,
            side=side,
            entry_time=entry_time,
            entry_price=entry_price,
            entry_score=entry_score,
            search_engine_version="manual",
            model_version="manual",
            label_version="manual",
            rank=0,
            expected_horizon_min=0,
            expected_return_pct=0.0,
            success_probability=0.0,
            primary_reason="manual_position",
            secondary_reasons=[],
            initial_stop_pct=0.0,
            initial_take_profit_pct=0.0,
            invalid_condition="manual_never_auto_exit",
            max_hold_minutes=99999,
            review_interval_minutes=60,
            source="MANUAL",
            auto_manage=False,
            engine=engine or "USER",
        )

    exp_return = max(3.0, float(expected_ev) if expected_ev else 3.0)
    horizon = 90 if is_short else 120
    label = "risk_adjusted_short" if is_short else "return_minus_dd"
    model = SHORT_MODEL_VERSION if is_short else LONG_MODEL_VERSION

    return TradeThesis(
        thesis_id=f"th_{uuid.uuid4().hex[:12]}",
        position_id=position_id,
        symbol=symbol,
        side=side,
        entry_time=entry_time,
        entry_price=entry_price,
        entry_score=entry_score,
        search_engine_version=SEARCH_ENGINE_VERSION,
        model_version=model,
        label_version=label,
        rank=rank,
        expected_horizon_min=horizon,
        expected_return_pct=exp_return,
        success_probability=min(0.95, max(0.35, entry_score / 100.0)) if entry_score else 0.5,
        primary_reason=primary_reason or f"{engine}_entry_rank_{rank}",
        secondary_reasons=[f"expected_horizon_{horizon}m", f"label_{label}"],
        initial_stop_pct=8.0,
        initial_take_profit_pct=10.0,
        invalid_condition="roi_below_initial_stop_after_30m",
        max_hold_minutes=240,
        review_interval_minutes=5,
        source="BOT",
        auto_manage=True,
        engine=engine,
    )
