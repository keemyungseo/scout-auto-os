"""Load Guardian outcome analysis inputs."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from scout_auto_os.engine.guardian.trade_thesis import GuardianTradeThesis

TIMELINE_CSV = "guardian_timeline.csv"
TRANSITION_STATS_CSV = "guardian_transition_statistics.csv"
DECISION_LOG_CSV = "guardian_decision_log.csv"
THESIS_JSONL = "trade_thesis.jsonl"


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        with path.open(encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except (OSError, csv.Error):
        return []


def load_theses(path: Path) -> dict[str, GuardianTradeThesis]:
    out: dict[str, GuardianTradeThesis] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            t = GuardianTradeThesis.from_dict(json.loads(line))
            out[t.contract_id] = t
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    return out


def load_outcome_inputs(data_dir: Path) -> dict:
    guardian_dir = data_dir / "guardian"
    timeline = _read_csv(guardian_dir / TIMELINE_CSV)
    by_trade: dict[str, list[dict]] = defaultdict(list)
    for row in timeline:
        by_trade[row.get("trade_id", "")].append(row)
    for tid in by_trade:
        by_trade[tid].sort(key=lambda r: int(float(r.get("elapsed_minutes", 0))))

    decision_by_contract = {}
    for row in _read_csv(guardian_dir / DECISION_LOG_CSV):
        cid = row.get("contract_id", "")
        if cid:
            decision_by_contract[cid] = row

    return {
        "timeline_by_trade": dict(by_trade),
        "transition_stats": _read_csv(guardian_dir / TRANSITION_STATS_CSV),
        "decision_log": decision_by_contract,
        "theses": load_theses(guardian_dir / THESIS_JSONL),
        "paths": {
            "timeline": guardian_dir / TIMELINE_CSV,
            "transitions": guardian_dir / TRANSITION_STATS_CSV,
            "decision_log": guardian_dir / DECISION_LOG_CSV,
            "thesis": guardian_dir / THESIS_JSONL,
        },
    }
