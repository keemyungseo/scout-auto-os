"""Runtime shadow timestamp resolution and validation."""

from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scout_auto_os.storage.db import now_kst

KST = timezone(timedelta(hours=9))

TIMESTAMP_FIELDS = (
    "scan_kst",
    "scan_time",
    "scan_timestamp",
    "entry_time",
    "entry_timestamp",
    "trade_timestamp",
    "timestamp",
)

DIAG_FIELDS = (
    "scan_id", "symbol", "side", "timestamp", "scan_id_time",
    "diagnosis", "label_source", "unique_timestamp_ok", "future_ok",
    "error_reason",
)

BACKUP_NAME = "value_gate_runtime_shadow.bad_timestamp_backup.csv"
FIX_REPORT = "timestamp_fix_report.md"
TIMESTAMP_DIAG_CSV = "timestamp_diagnostics.csv"


def _parse_kst(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=KST)


def resolve_replay_timestamp(row: dict) -> str | None:
    """Replay backfill: original scan/entry time only — never synthetic."""
    for key in TIMESTAMP_FIELDS:
        val = row.get(key)
        if val and str(val).strip():
            return str(val).strip()
    trade_key = row.get("trade_key", "")
    if "|" in trade_key:
        return trade_key.split("|")[0]
    return None


def validate_shadow_timestamps(
    rows: list[dict],
    *,
    now_s: str | None = None,
) -> dict:
    """Fail if all timestamps identical, any future, or all missing."""
    now_str = now_s or now_kst()
    now_dt = _parse_kst(now_str)
    timestamps = [r.get("timestamp", "") for r in rows if r.get("timestamp")]
    scan_ids = [r.get("scan_id", "") for r in rows if r.get("scan_id")]
    missing = sum(1 for r in rows if not r.get("timestamp"))
    unique_ts = len(set(timestamps))
    unique_scan = len(set(scan_ids))
    future_count = 0
    for t in timestamps:
        try:
            if _parse_kst(t) > now_dt:
                future_count += 1
        except ValueError:
            pass
    all_same = len(rows) > 1 and unique_ts <= 1
    ok = (
        len(rows) > 0
        and not all_same
        and future_count == 0
        and missing == 0
        and unique_ts > 1
    )
    return {
        "ok": ok,
        "total_rows": len(rows),
        "unique_timestamp_count": unique_ts,
        "unique_scan_id_count": unique_scan,
        "future_timestamp_count": future_count,
        "missing_timestamp_count": missing,
        "all_timestamps_identical": all_same,
        "validated_at": now_str,
    }


def backup_shadow_csv(shadow_dir: Path) -> Path | None:
    src = shadow_dir / "value_gate_runtime_shadow.csv"
    if not src.exists():
        return None
    dst = shadow_dir / BACKUP_NAME
    shutil.copy2(src, dst)
    return dst


def diagnose_rows(rows: list[dict], *, now_s: str | None = None) -> list[dict]:
    now_str = now_s or now_kst()
    now_dt = _parse_kst(now_str)
    out: list[dict] = []
    for row in rows:
        ts = row.get("timestamp", "")
        scan_id = row.get("scan_id", "")
        scan_id_time = scan_id.split("|")[0] if "|" in scan_id else ""
        diagnosis = "OK"
        error = ""
        if not ts:
            diagnosis = "MISSING_ORIGINAL_TIMESTAMP"
            error = "empty timestamp"
        elif ts != scan_id_time and scan_id_time:
            diagnosis = "TIMESTAMP_SCAN_ID_MISMATCH"
            error = f"timestamp={ts} scan_id_time={scan_id_time}"
        else:
            try:
                dt = _parse_kst(ts)
                if dt > now_dt:
                    diagnosis = "FUTURE_TIMESTAMP"
                    error = f"{ts} > now {now_str}"
            except ValueError:
                diagnosis = "PARSE_ERROR"
                error = f"bad format: {ts!r}"
        out.append({
            "scan_id": scan_id,
            "symbol": row.get("symbol", ""),
            "side": row.get("side", ""),
            "timestamp": ts,
            "scan_id_time": scan_id_time,
            "diagnosis": diagnosis,
            "label_source": row.get("source", ""),
            "unique_timestamp_ok": "",
            "future_ok": int(diagnosis != "FUTURE_TIMESTAMP"),
            "error_reason": error,
        })
    return out


def write_timestamp_fix_report(
    shadow_dir: Path,
    *,
    validation: dict,
    skipped: list[dict],
    mode: str,
    backup_path: Path | None,
) -> Path:
    path = shadow_dir / FIX_REPORT
    lines = [
        "# Runtime Shadow Timestamp Fix V1 — Report",
        "",
        f"**Mode:** {mode}",
        f"**Validated at:** {validation.get('validated_at', '')}",
        "",
        "## Root cause",
        "",
        "Replay backfill previously set `timestamp = now_kst()` (wall clock) instead of",
        "original `scan_kst` from replay bundle. All 157 rows shared one future timestamp.",
        "",
        "## Validation",
        "",
        f"- Total rows: {validation.get('total_rows', 0)}",
        f"- Unique timestamps: {validation.get('unique_timestamp_count', 0)}",
        f"- Unique scan_ids: {validation.get('unique_scan_id_count', 0)}",
        f"- Future timestamps: {validation.get('future_timestamp_count', 0)}",
        f"- Missing timestamps: {validation.get('missing_timestamp_count', 0)}",
        f"- All identical: {validation.get('all_timestamps_identical', False)}",
        f"- **Validation OK:** {validation.get('ok', False)}",
        "",
        f"- Backup: `{backup_path}`" if backup_path else "- Backup: none (no prior file)",
        f"- Skipped (missing original timestamp): {len(skipped)}",
        "",
        "## Final answers",
        "",
        "1. Cause: `record_candidate` used `now_kst()` for `timestamp` column during replay backfill.",
        "2. Replay mode: uses `scan_kst` / `trade_key` — never `datetime.now()`.",
        "3. scan_id: deterministic `scan_kst|SYMBOL|side`.",
        f"4. Unique timestamp count: {validation.get('unique_timestamp_count', 0)}.",
        f"5. Future timestamp count: {validation.get('future_timestamp_count', 0)}.",
        "6. Labeler: re-run with `--mode replay` after fix.",
        "7. Command Center summary: valid after labeled CSV refresh.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_timestamp_diagnostics(shadow_dir: Path, diag_rows: list[dict]) -> Path:
    path = shadow_dir / TIMESTAMP_DIAG_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=DIAG_FIELDS, extrasaction="ignore")
        w.writeheader()
        for row in diag_rows:
            w.writerow({k: row.get(k, "") for k in DIAG_FIELDS})
    return path


def run_timestamp_validation(
    shadow_dir: Path,
    *,
    now_s: str | None = None,
    mode: str = "replay",
    skipped: list[dict] | None = None,
    backup_path: Path | None = None,
) -> dict:
    csv_path = shadow_dir / "value_gate_runtime_shadow.csv"
    rows = []
    if csv_path.exists():
        with csv_path.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    validation = validate_shadow_timestamps(rows, now_s=now_s)
    diag_rows = diagnose_rows(rows, now_s=now_s)
    write_timestamp_diagnostics(shadow_dir, diag_rows)
    write_timestamp_fix_report(
        shadow_dir,
        validation=validation,
        skipped=skipped or [],
        mode=mode,
        backup_path=backup_path,
    )
    return {"validation": validation, "diagnostics": diag_rows}
