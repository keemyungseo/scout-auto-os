"""Directional forward evaluation — long vs short success criteria."""

from __future__ import annotations

import statistics

from scout_auto_os.engine.research.zero_base.forward_eval import compute_forward_metrics
from scout_auto_os.engine.research.zero_base.ranking import _profit_factor

RETURN_KEYS = ("return_30m", "return_1h", "return_2h", "return_4h", "return_6h")

QUALITY_MIN_SAMPLES = 50
QUALITY_MIN_WIN_RATE = 35.0
QUALITY_MIN_AVG_2H = 0.3
MAX_LONG_SLOTS = 3
MAX_SHORT_SLOTS = 3


def to_short_metrics(long_m: dict) -> dict:
    """short_return = -forward_return; success when price falls."""
    if not long_m:
        return {}
    sm = dict(long_m)
    for k in RETURN_KEYS:
        if k in long_m:
            sm[k] = round(-float(long_m[k]), 4)
    sm["direction"] = "short"
    r2h = float(long_m.get("return_2h", 0))
    max_up = float(long_m.get("max_return_2h", 0))
    min_down = float(long_m.get("min_return_2h", 0))
    sm["short_return_2h"] = round(-r2h, 4)
    sm["max_drawdown_2h"] = round(max_up, 4)  # adverse move for short
    sm["label_success_2h"] = r2h <= -3.0
    sm["label_big_winner"] = r2h <= -4.0 or min_down <= -4.0
    sm["label_trap"] = max_up >= 3.0 and r2h >= -1.0
    rets = []
    entry = float(long_m.get("price_at_scan", 1))
    # approximate time_to for short from stored long metrics
    if long_m.get("time_to_3pct") is not None:
        sm["time_to_3pct"] = long_m.get("time_to_3pct")
    if long_m.get("time_to_5pct") is not None:
        sm["time_to_5pct"] = long_m.get("time_to_5pct")
    sm["downside_capture"] = round(max_up, 4)
    return sm


def to_long_metrics(long_m: dict) -> dict:
    if not long_m:
        return {}
    m = dict(long_m)
    m["direction"] = "long"
    m["short_return_2h"] = round(-float(long_m.get("return_2h", 0)), 4)
    return m


def aggregate_directional(samples: list[dict], direction: str) -> dict:
    if not samples:
        return {"sample_count": 0, "direction": direction, "score": 0}
    if direction == "short":
        rets = [float(s.get("short_return_2h", -float(s.get("return_2h", 0)))) for s in samples]
    else:
        rets = [float(s.get("return_2h", 0)) for s in samples]
    traps = sum(1 for s in samples if s.get("label_trap"))
    big = sum(1 for s in samples if s.get("label_big_winner"))
    mdd = [float(s.get("max_drawdown_2h", 0)) for s in samples]
    wins = sum(1 for x in rets if x >= 3.0)
    row = {
        "direction": direction,
        "sample_count": len(samples),
        "return_30m_avg": round(statistics.mean([float(s.get("return_30m", 0)) for s in samples]), 4),
        "return_1h_avg": round(statistics.mean([float(s.get("return_1h", 0)) for s in samples]), 4),
        "avg_return_2h": round(statistics.mean(rets), 4),
        "median_return_2h": round(statistics.median(rets), 4),
        "return_4h_avg": round(statistics.mean([float(s.get("return_4h", 0)) for s in samples]), 4),
        "return_6h_avg": round(statistics.mean([float(s.get("return_6h", 0)) for s in samples]), 4),
        "win_rate": round(wins / len(rets) * 100, 2),
        "profit_factor": _profit_factor(rets),
        "max_drawdown_avg": round(statistics.mean(mdd), 4),
        "trap_rate": round(traps / len(samples) * 100, 2),
        "big_winner_capture_rate": round(big / len(samples) * 100, 2),
        "time_to_3pct_avg": _avg_optional(samples, "time_to_3pct"),
        "time_to_5pct_avg": _avg_optional(samples, "time_to_5pct"),
    }
    row["score"] = round(
        row["avg_return_2h"] + row["win_rate"] * 0.1 + row["big_winner_capture_rate"] * 0.2 - row["trap_rate"] * 0.3,
        2,
    )
    return row


def _avg_optional(samples: list[dict], key: str) -> float | None:
    vals = [s[key] for s in samples if s.get(key) is not None]
    return round(statistics.mean(vals), 1) if vals else None


def passes_quality(agg: dict, random_agg: dict) -> bool:
    n = int(agg.get("sample_count") or 0)
    if n < QUALITY_MIN_SAMPLES:
        return False
    if float(agg.get("win_rate") or 0) < QUALITY_MIN_WIN_RATE:
        return False
    if float(agg.get("avg_return_2h") or 0) < QUALITY_MIN_AVG_2H:
        return False
    if float(agg.get("avg_return_2h") or 0) <= float(random_agg.get("avg_return_2h") or 0):
        return False
    return True


def build_champion_board(
    engine_aggs: list[dict],
    random_agg: dict,
    direction: str,
    exclude_baseline: tuple[str, ...] = (),
) -> list[dict]:
    rows = [a for a in engine_aggs if a.get("direction") == direction and a.get("engine") not in exclude_baseline]
    rows.sort(key=lambda x: x.get("score", 0), reverse=True)
    board: list[dict] = []
    for i, row in enumerate(rows, 1):
        qualified = passes_quality(row, random_agg)
        board.append({
            **row,
            "board_rank": i,
            "slot_eligible": qualified,
            "beats_random": float(row.get("avg_return_2h", 0)) > float(random_agg.get("avg_return_2h", 0)),
            "tier": "champion_candidate" if qualified else "hypothesis",
        })
    return board


def simulate_slots(
    long_board: list[dict],
    short_board: list[dict],
    long_samples_by_engine: dict,
    short_samples_by_engine: dict,
) -> dict:
    """3 long + 3 short slots; empty if quality gate fails."""
    long_filled = [b for b in long_board if b.get("slot_eligible")][:MAX_LONG_SLOTS]
    short_filled = [b for b in short_board if b.get("slot_eligible")][:MAX_SHORT_SLOTS]
    long_rets: list[float] = []
    short_rets: list[float] = []
    for b in long_filled:
        for s in long_samples_by_engine.get(b["engine"], []):
            long_rets.append(float(s.get("return_2h", 0)))
    for b in short_filled:
        for s in short_samples_by_engine.get(b["engine"], []):
            short_rets.append(float(s.get("short_return_2h", -float(s.get("return_2h", 0)))))
    combined = long_rets + short_rets
    return {
        "long_slots_filled": len(long_filled),
        "short_slots_filled": len(short_filled),
        "long_slot_engines": [b["engine"] for b in long_filled],
        "short_slot_engines": [b["engine"] for b in short_filled],
        "long_slots_empty": MAX_LONG_SLOTS - len(long_filled),
        "short_slots_empty": MAX_SHORT_SLOTS - len(short_filled),
        "combined_avg_return_2h": round(statistics.mean(combined), 4) if combined else 0,
        "long_avg_return_2h": round(statistics.mean(long_rets), 4) if long_rets else 0,
        "short_avg_return_2h": round(statistics.mean(short_rets), 4) if short_rets else 0,
        "combined_sample_count": len(combined),
    }
