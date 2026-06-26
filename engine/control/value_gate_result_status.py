"""Value Gate labeled shadow results for Command Center (read-only)."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

FALSE_SKIP_FIELDS = (
    "symbol", "side", "value_score", "runner_prob", "predicted_dna_type",
    "actual_roi_2h", "actual_roi_4h", "actual_peak_roi", "actual_drawdown", "reason",
)

FALSE_ACCEPT_FIELDS = (
    "symbol", "side", "value_score", "runner_prob", "predicted_dna_type",
    "predicted_drawdown", "actual_roi_2h", "actual_roi_4h", "actual_drawdown", "reason",
)

BAND_FIELDS = (
    "band", "count", "avg_roi_2h", "avg_roi_4h", "avg_peak_roi",
    "win_rate", "mdd", "false_skip_rate", "false_accept_rate",
)

REEVAL_CSV = "value_gate_cache_fix_reevaluation.csv"
BAND_CSV = "value_gate_cache_fix_band_calibration.csv"
FALSE_SKIP_CSV = "value_gate_cache_fix_false_skip.csv"
FALSE_ACCEPT_CSV = "value_gate_cache_fix_false_accept.csv"
REPORT_MD = "value_gate_cache_fix_report.md"
LABEL_SUMMARY_JSON = "value_gate_shadow_label_summary.json"
SHADOW_CSV = "value_gate_runtime_shadow.csv"
LABELED_CSV = "value_gate_runtime_shadow_labeled.csv"


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        with path.open(encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except (OSError, csv.Error):
        return []


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _f(val: str | float | int | None, default: float = 0.0) -> float:
    try:
        return float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _pick(row: dict, fields: tuple[str, ...]) -> dict:
    return {k: row.get(k, "") for k in fields}


def _parse_verdict(report_text: str) -> str:
    m = re.search(r"\*\*Verdict:\*\*\s*`([^`]+)`", report_text)
    return m.group(1).strip() if m else ""


def _sort_false_skips(rows: list[dict]) -> list[dict]:
    def sort_key(r: dict) -> float:
        peak = _f(r.get("actual_peak_roi"))
        roi2 = _f(r.get("actual_roi_2h"))
        return max(peak, roi2)

    return sorted(rows, key=sort_key, reverse=True)


def _map_false_skip(row: dict) -> dict:
    out = _pick(row, FALSE_SKIP_FIELDS)
    if not out.get("reason"):
        out["reason"] = row.get("false_skip_reason", "")
    return out


def _map_false_accept(row: dict) -> dict:
    out = _pick(row, FALSE_ACCEPT_FIELDS)
    if not out.get("reason"):
        out["reason"] = row.get("false_accept_reason", "")
    out["is_short"] = int(
        row.get("is_short_false_accept") == "1"
        or str(row.get("side", "")).upper() == "SHORT"
    )
    return out


def _map_band(row: dict) -> dict:
    return {
        "band": row.get("score_band", row.get("band", "")),
        "count": row.get("count", ""),
        "avg_roi_2h": row.get("avg_roi_2h", ""),
        "avg_roi_4h": row.get("avg_roi_4h", ""),
        "avg_peak_roi": row.get("avg_peak_roi", ""),
        "win_rate": row.get("win_rate", ""),
        "mdd": row.get("mdd", ""),
        "false_skip_rate": row.get("false_skip_rate", ""),
        "false_accept_rate": row.get("false_accept_rate", ""),
    }


def _build_data_health(data_dir: Path, shadow_dir: Path, label_summary: dict) -> dict:
    labeled_path = shadow_dir / LABELED_CSV
    shadow_path = shadow_dir / SHADOW_CSV

    rule_mm = 0
    tk_mm = 0
    if labeled_path.exists():
        try:
            from scout_auto_os.engine.predator.labeled_reevaluation import (
                check_policy_b_decision_consistency,
                check_trade_key_policy_mismatch,
                load_labeled_rows,
            )

            labeled_rows = load_labeled_rows(data_dir)
            rule_mm = len(check_policy_b_decision_consistency(labeled_rows))
            tk_mm = len(check_trade_key_policy_mismatch(labeled_rows, data_dir))
        except Exception:
            pass

    future_count = 0
    if shadow_path.exists():
        try:
            from scout_auto_os.engine.predator.timestamp_fix import validate_shadow_timestamps

            shadow_rows = _read_csv(shadow_path)
            future_count = validate_shadow_timestamps(shadow_rows).get("future_timestamp_count", 0)
        except Exception:
            pass

    return {
        "policy_rule_mismatch_count": rule_mm,
        "trade_key_mismatch_count": tk_mm,
        "future_timestamp_count": future_count,
        "labeled_4h_rows": label_summary.get("labeled_4h_rows", 0),
        "waiting_rows": label_summary.get("waiting_rows", 0),
        "healthy": rule_mm == 0 and tk_mm == 0 and future_count == 0,
    }


def _empty_summary() -> dict:
    return {
        "policy_name": "Soft 50s",
        "policy_key": "B",
        "status": "SHADOW",
        "labeled_rows": 0,
        "enter_count": 0,
        "skip_count": 0,
        "weighted_roi": 0.0,
        "sharpe": 0.0,
        "mdd": 0.0,
        "win_rate": 0.0,
        "false_skip_count": 0,
        "false_accept_count": 0,
        "decision_mismatch_count": 0,
        "trade_key_mismatch_count": 0,
        "verdict": "",
        "long_enter_count": 0,
        "short_enter_count": 0,
        "long_skip_count": 0,
        "short_skip_count": 0,
        "long_false_skip": 0,
        "short_false_skip": 0,
        "long_false_accept": 0,
        "short_false_accept": 0,
        "accepted_avg_roi": 0.0,
        "skipped_avg_roi": 0.0,
    }


def build_value_gate_result_status(
    data_dir: Path,
    *,
    false_skip_limit: int = 20,
    false_accept_limit: int = 20,
) -> dict:
    shadow_dir = data_dir / "runtime_shadow"
    paths = {
        "reevaluation": shadow_dir / REEVAL_CSV,
        "band_calibration": shadow_dir / BAND_CSV,
        "false_skip": shadow_dir / FALSE_SKIP_CSV,
        "false_accept": shadow_dir / FALSE_ACCEPT_CSV,
        "report": shadow_dir / REPORT_MD,
        "label_summary": shadow_dir / LABEL_SUMMARY_JSON,
    }

    label_summary = _read_json(paths["label_summary"])
    data_health = _build_data_health(data_dir, shadow_dir, label_summary)

    reeval_rows = _read_csv(paths["reevaluation"])
    reeval = reeval_rows[0] if reeval_rows else {}

    report_text = ""
    if paths["report"].exists():
        try:
            report_text = paths["report"].read_text(encoding="utf-8")
        except OSError:
            report_text = ""

    verdict = _parse_verdict(report_text) or reeval.get("verdict", "")

    has_data = any(p.exists() for p in paths.values())
    summary = _empty_summary()
    if reeval:
        summary.update({
            "policy_name": reeval.get("policy_name", "Soft 50s"),
            "policy_key": reeval.get("policy", "B"),
            "status": "SHADOW",
            "labeled_rows": int(_f(reeval.get("trade_count", label_summary.get("total_rows", 0)))),
            "enter_count": int(_f(reeval.get("enter_count"))),
            "skip_count": int(_f(reeval.get("skip_count"))),
            "weighted_roi": _f(reeval.get("weighted_roi")),
            "sharpe": _f(reeval.get("sharpe")),
            "mdd": _f(reeval.get("mdd")),
            "win_rate": _f(reeval.get("win_rate")),
            "false_skip_count": int(_f(reeval.get("false_skip_count"))),
            "false_accept_count": int(_f(reeval.get("false_accept_count"))),
            "decision_mismatch_count": data_health["policy_rule_mismatch_count"],
            "trade_key_mismatch_count": data_health["trade_key_mismatch_count"],
            "verdict": verdict,
            "long_enter_count": int(_f(reeval.get("long_enter_count"))),
            "short_enter_count": int(_f(reeval.get("short_enter_count"))),
            "long_skip_count": int(_f(reeval.get("long_skip_count"))),
            "short_skip_count": int(_f(reeval.get("short_skip_count"))),
            "long_false_skip": int(_f(reeval.get("long_false_skip"))),
            "short_false_skip": int(_f(reeval.get("short_false_skip"))),
            "long_false_accept": int(_f(reeval.get("long_false_accept"))),
            "short_false_accept": int(_f(reeval.get("short_false_accept"))),
            "accepted_avg_roi": _f(reeval.get("accepted_avg_roi")),
            "skipped_avg_roi": _f(reeval.get("skipped_avg_roi")),
        })
    elif label_summary:
        summary.update({
            "labeled_rows": label_summary.get("total_rows", 0),
            "false_skip_count": label_summary.get("false_skip_count", 0),
            "false_accept_count": label_summary.get("false_accept_count", 0),
            "decision_mismatch_count": data_health["policy_rule_mismatch_count"],
            "trade_key_mismatch_count": data_health["trade_key_mismatch_count"],
        })

    band_rows = [_map_band(r) for r in _read_csv(paths["band_calibration"])]
    false_skips = [
        _map_false_skip(r)
        for r in _sort_false_skips(_read_csv(paths["false_skip"]))[:false_skip_limit]
    ]
    false_accepts = [
        _map_false_accept(r)
        for r in _read_csv(paths["false_accept"])[:false_accept_limit]
    ]

    return {
        "ok": True,
        "dry_run": True,
        "mode": "SHADOW_ONLY",
        "summary": summary,
        "band_calibration": band_rows,
        "false_skips": false_skips,
        "false_accepts": false_accepts,
        "data_health": data_health,
        "data_sources": {k: v.exists() for k, v in paths.items()},
        "has_data": has_data,
    }
