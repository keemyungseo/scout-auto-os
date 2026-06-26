"""Guardian position status — read-only merge from runtime + review logs."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

from scout_auto_os.engine.control.manual_lock import ManualLockStore
from scout_auto_os.storage.db import Database, now_kst

KST_FMT = "%Y-%m-%d %H:%M:%S"


def _safe_float(v, default: float = 0.0) -> float:
    try:
        if v is None or v == "":
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _safe_int(v, default: int = 0) -> int:
    try:
        if v is None or v == "":
            return default
        return int(float(v))
    except (TypeError, ValueError):
        return default


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _latest_by_symbol(rows: list[dict], symbol_key: str = "symbol") -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in rows:
        sym = (r.get(symbol_key) or "").upper()
        if sym:
            out[sym] = r
    return out


def _load_thesis_by_symbol(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        sym = (obj.get("symbol") or "").upper()
        if sym:
            out[sym] = obj
    return out


def _load_runtime_open(data_dir: Path) -> list[dict]:
    db_path = data_dir / "trades.db"
    if not db_path.exists():
        return []
    try:
        db = Database(db_path)
        try:
            rows = db.fetchall("SELECT * FROM positions WHERE status='OPEN' ORDER BY entry_time DESC")
            return [dict(r) for r in rows]
        finally:
            db.conn.close()
    except Exception:
        return []


def _normalize_source(source: str, auto_manage: bool, manual_lock: bool) -> str:
    if manual_lock or str(source).upper() == "MANUAL" or not auto_manage:
        return "MANUAL"
    if str(source).upper() in ("AUTO", "BOT"):
        return "BOT"
    return "UNKNOWN"


def _elapsed_minutes(entry_time: str) -> int:
    if not entry_time:
        return 0
    try:
        t0 = datetime.strptime(entry_time[:19], KST_FMT)
        t1 = datetime.strptime(now_kst(), KST_FMT)
        return max(0, int((t1 - t0).total_seconds() / 60))
    except ValueError:
        return 0


def _guardian_from_reviews(
    action: str,
    thesis_state: str,
    manual_lock: bool,
    source: str,
    auto_manage: bool,
) -> tuple[str, str]:
    if manual_lock or source == "MANUAL" or not auto_manage:
        return "NO_ACTION_MANUAL_POSITION", "manual position — bot does not touch"
    if action:
        return action, ""
    state_map = {
        "EXIT_READY": ("EXIT_READY", "expectation EXIT_READY"),
        "THESIS_FAILED": ("THESIS_FAILED", "thesis failed"),
        "THESIS_COMPLETE": ("THESIS_COMPLETE", "target exceeded — trail/watch"),
        "OUTPERFORM": ("OUTPERFORM", "outperforming expected curve"),
    }
    if thesis_state in state_map:
        return state_map[thesis_state]
    return "HOLD", "observe"


def _empty_position(symbol: str = "") -> dict:
    sym = symbol.upper() or "—"
    return {
        "symbol": sym,
        "side": "—",
        "source": "UNKNOWN",
        "auto_manage": False,
        "manual_lock": False,
        "entry_time": "",
        "entry_price": 0.0,
        "current_price": 0.0,
        "roi": 0.0,
        "elapsed_minutes": 0,
        "mfe": 0.0,
        "mae": 0.0,
        "peak_roi": 0.0,
        "drawdown_from_peak": 0.0,
        "thesis_id": "",
        "expected_roi": 0.0,
        "expected_horizon": 0,
        "expected_progress": 0.0,
        "progress_ratio": 0.0,
        "expectation_score": 0.0,
        "thesis_state": "",
        "guardian_action": "NO_DATA",
        "guardian_reason": "no position data available",
        "exit_pressure_score": 0.0,
        "hold_confidence": 0.0,
        "dry_run": True,
        "last_updated": now_kst(),
    }


def _merge_position(
    base: dict,
    review: dict | None,
    expectation: dict | None,
    thesis: dict | None,
    locked: set[str],
) -> dict:
    sym = base["symbol"].upper()
    manual_lock = sym in locked or bool(_safe_int(base.get("manual_lock")))
    auto_manage = bool(_safe_int(base.get("auto_manage", 1)))
    source_raw = base.get("source", "AUTO")
    source = _normalize_source(source_raw, auto_manage, manual_lock)

    if manual_lock or sym in locked:
        auto_manage = False
        manual_lock = True
        source = "MANUAL"

    entry_time = base.get("entry_time", "")
    entry_price = _safe_float(base.get("entry_price"))
    current_price = _safe_float(base.get("current_price"), entry_price)
    roi = _safe_float(base.get("unrealized_pnl_pct"))
    if review:
        roi = _safe_float(review.get("roi"), roi)
        current_price = _safe_float(review.get("current_price"), current_price)

    elapsed = _safe_int(review.get("elapsed_minutes") if review else 0)
    if not elapsed and entry_time:
        elapsed = _elapsed_minutes(entry_time)

    expected_horizon = _safe_int(review.get("expected_horizon") if review else 0)
    if thesis and not expected_horizon:
        expected_horizon = _safe_int(thesis.get("expected_horizon_min"))

    expected_roi = _safe_float(expectation.get("expected_roi") if expectation else 0)
    if not expected_roi and review:
        expected_roi = _safe_float(review.get("expected_return"))

    expected_progress = _safe_float(expectation.get("expected_progress") if expectation else 0)
    progress_ratio = _safe_float(expectation.get("progress_ratio") if expectation else 0)
    expectation_score = _safe_float(expectation.get("expectation_score") if expectation else 0)
    thesis_state = (expectation.get("thesis_state") if expectation else "") or ""

    action = (review.get("action") if review else "") or ""
    action_reason = (review.get("action_reason") if review else "") or ""
    guardian_action, default_reason = _guardian_from_reviews(
        action, thesis_state, manual_lock, source, auto_manage,
    )
    guardian_reason = action_reason or default_reason
    if manual_lock:
        guardian_action = "NO_ACTION_MANUAL_POSITION"
        guardian_reason = "manual position — bot does not touch"

    last_updated = now_kst()
    for src in (review, expectation):
        if src and src.get("timestamp"):
            last_updated = src["timestamp"]

    return {
        "symbol": sym,
        "side": (base.get("side") or "LONG").upper(),
        "source": source,
        "auto_manage": auto_manage,
        "manual_lock": manual_lock,
        "entry_time": entry_time,
        "entry_price": round(entry_price, 6),
        "current_price": round(current_price, 6),
        "roi": round(roi, 4),
        "elapsed_minutes": elapsed,
        "mfe": round(_safe_float(review.get("mfe") if review else 0), 4),
        "mae": round(_safe_float(review.get("mae") if review else 0), 4),
        "peak_roi": round(_safe_float(review.get("peak_roi") if review else 0), 4),
        "drawdown_from_peak": round(_safe_float(review.get("drawdown_from_peak") if review else 0), 4),
        "thesis_id": base.get("thesis_id") or (review.get("thesis_id") if review else "") or (thesis.get("thesis_id") if thesis else ""),
        "expected_roi": round(expected_roi, 4),
        "expected_horizon": expected_horizon,
        "expected_progress": round(expected_progress, 4),
        "progress_ratio": round(progress_ratio, 4),
        "expectation_score": round(expectation_score, 2),
        "thesis_state": thesis_state,
        "guardian_action": guardian_action,
        "guardian_reason": guardian_reason,
        "exit_pressure_score": round(_safe_float(review.get("exit_pressure_score") if review else 0), 2),
        "hold_confidence": round(_safe_float(review.get("hold_confidence") if review else 0), 2),
        "dry_run": True,
        "last_updated": last_updated,
    }


def build_guardian_positions(data_dir: Path, control_dir: Path) -> dict:
    """Merge runtime OPEN positions with evaluation / expectation logs."""
    pe_dir = data_dir / "position_evaluation"
    exp_dir = data_dir / "expectation"
    locked = ManualLockStore(control_dir).locked_symbols()

    reviews = _latest_by_symbol(_read_csv(pe_dir / "position_review.csv"))
    expectations = _latest_by_symbol(_read_csv(exp_dir / "expectation_review.csv"))
    theses = _load_thesis_by_symbol(pe_dir / "trade_thesis.jsonl")

    runtime = _load_runtime_open(data_dir)
    positions: list[dict] = []

    seen: set[str] = set()
    for row in runtime:
        sym = row["symbol"].upper()
        seen.add(sym)
        positions.append(_merge_position(
            row, reviews.get(sym), expectations.get(sym), theses.get(sym), locked,
        ))

    for sym, review in reviews.items():
        if sym in seen:
            continue
        base = {
            "symbol": sym,
            "side": review.get("side", "LONG"),
            "source": review.get("source", "AUTO"),
            "auto_manage": review.get("auto_manage", 1),
            "manual_lock": review.get("manual_lock", 0),
            "entry_time": review.get("entry_time", ""),
            "entry_price": review.get("entry_price", 0),
            "current_price": review.get("current_price", 0),
            "unrealized_pnl_pct": review.get("roi", 0),
            "thesis_id": review.get("thesis_id", ""),
        }
        positions.append(_merge_position(
            base, review, expectations.get(sym), theses.get(sym), locked,
        ))
        seen.add(sym)

    for sym in locked:
        if sym not in seen:
            stub = _empty_position(sym)
            stub.update({
                "side": "LONG",
                "source": "MANUAL",
                "auto_manage": False,
                "manual_lock": True,
                "guardian_action": "NO_ACTION_MANUAL_POSITION",
                "guardian_reason": "manual lock active — bot does not touch",
            })
            positions.append(stub)

    positions.sort(key=lambda p: p["symbol"])

    return {
        "ok": True,
        "dry_run": True,
        "count": len(positions),
        "positions": positions,
        "data_sources": {
            "runtime_db": bool(runtime),
            "position_review": (pe_dir / "position_review.csv").exists(),
            "expectation_review": (exp_dir / "expectation_review.csv").exists(),
            "trade_thesis": (pe_dir / "trade_thesis.jsonl").exists(),
            "manual_locks": sorted(locked),
        },
    }
