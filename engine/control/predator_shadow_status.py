"""Predator Value Gate shadow status for Command Center (read-only)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

CANDIDATE_FIELDS = (
    "timestamp", "symbol", "side",
    "baseline_decision", "baseline_size",
    "policy_b_decision", "policy_b_size",
    "value_score", "runner_prob", "predicted_dna_type",
    "predicted_roi", "predicted_drawdown", "predicted_win_prob",
    "reason", "manual_lock", "source", "auto_manage",
)

WATCH_FIELDS = (
    "timestamp", "symbol", "value_score", "runner_prob",
    "predicted_dna_type", "predicted_drawdown", "predicted_win_prob", "reason",
)


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        with path.open(encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except (OSError, csv.Error):
        return []


def _tail(rows: list[dict], n: int) -> list[dict]:
    return list(reversed(rows[-n:])) if rows else []


def _pick(row: dict, fields: tuple[str, ...]) -> dict:
    return {k: row.get(k, "") for k in fields}


def build_predator_shadow_status(
    data_dir: Path,
    *,
    bot_paused: bool = False,
    bot_emergency: bool = False,
    recent_limit: int = 30,
    watch_limit: int = 20,
) -> dict:
    shadow_dir = data_dir / "runtime_shadow"
    summary_path = shadow_dir / "value_gate_shadow_summary.json"
    shadow_path = shadow_dir / "value_gate_runtime_shadow.csv"
    watch_path = shadow_dir / "short_false_accept_watch.csv"

    summary: dict = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            summary = {}

    shadow_rows = _read_csv(shadow_path)
    watch_rows = _read_csv(watch_path)
    watch_symbols = {r.get("symbol", "").upper() for r in watch_rows}

    recent = []
    for row in _tail(shadow_rows, recent_limit):
        item = _pick(row, CANDIDATE_FIELDS)
        item["on_short_watch"] = int(item.get("symbol", "").upper() in watch_symbols)
        recent.append(item)

    watch_recent = [_pick(r, WATCH_FIELDS) for r in _tail(watch_rows, watch_limit)]

    return {
        "ok": True,
        "dry_run": True,
        "mode": "SHADOW_ONLY",
        "bot_paused": bot_paused,
        "bot_emergency": bot_emergency,
        "policy_name": summary.get("policy_name", "Soft 50s"),
        "policy_key": summary.get("policy_key", "B"),
        "last_update": summary.get("last_update", ""),
        "total_candidates_today": summary.get("total_candidates_today", 0),
        "policy_enter_count": summary.get("policy_enter_count", 0),
        "policy_skip_count": summary.get("policy_skip_count", 0),
        "shadow_only_count": summary.get("shadow_only_count", 0),
        "short_watch_count": summary.get("short_watch_count", len(watch_rows)),
        "avg_value_score": summary.get("avg_value_score", 0),
        "enter_by_side": summary.get("enter_by_side", {}),
        "skip_by_side": summary.get("skip_by_side", {}),
        "recent_candidates": recent,
        "short_watch": watch_recent,
        "data_sources": {
            "summary": summary_path.exists(),
            "shadow_csv": shadow_path.exists(),
            "watch_csv": watch_path.exists(),
        },
    }
