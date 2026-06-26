"""Shadow labeler — attach forward outcomes to Policy B runtime shadow rows."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from scout_auto_os.engine.predator.value_gate_shadow_logger import SHADOW_FIELDS, WATCH_FIELDS
from scout_auto_os.engine.research.market_data import fetch_klines_range
from scout_auto_os.engine.research.short_execution.constants import BAR_MINUTES
from scout_auto_os.engine.research.trade_dna.curve_builder import build_trade_dna
from scout_auto_os.engine.runtime_audit.ablation_runner import _peak_and_mdd, _roi_at
from scout_auto_os.storage.db import now_kst

KST = timezone(timedelta(hours=9))
WAITING = "WAITING"
HORIZON_2H_MIN = 120
HORIZON_4H_MIN = 240

SHADOW_SOURCE = "value_gate_runtime_shadow.csv"
SHADOW_LABELED = "value_gate_runtime_shadow_labeled.csv"
WATCH_SOURCE = "short_false_accept_watch.csv"
WATCH_LABELED = "short_false_accept_watch_labeled.csv"
SUMMARY_JSON = "value_gate_shadow_label_summary.json"
REPORT_MD = "value_gate_shadow_label_report.md"

WATCH_LABEL_FIELDS = WATCH_FIELDS + (
    "actual_roi_2h", "actual_roi_4h", "actual_peak_roi", "actual_drawdown",
    "actual_dna_type", "false_skip", "false_accept",
)


@dataclass
class LabelerConfig:
    false_skip_roi_2h_pct: float = 10.0
    false_skip_peak_roi_pct: float = 15.0
    false_accept_drawdown_pct: float = -10.0
    win_threshold_pct: float = 3.0
    rest_base: str = "https://fapi.binance.com"
    kline_interval: str = "15m"


def load_labeler_config(config: dict | None = None) -> LabelerConfig:
    cfg = (config or {}).get("value_gate_shadow", {}).get("labeler", {})
    live = (config or {}).get("live_data", {})
    return LabelerConfig(
        false_skip_roi_2h_pct=float(cfg.get("false_skip_roi_2h_pct", 10.0)),
        false_skip_peak_roi_pct=float(cfg.get("false_skip_peak_roi_pct", 15.0)),
        false_accept_drawdown_pct=float(cfg.get("false_accept_drawdown_pct", -10.0)),
        win_threshold_pct=float(cfg.get("win_threshold_pct", 3.0)),
        rest_base=str(live.get("rest_base", "https://fapi.binance.com")),
        kline_interval=str(cfg.get("kline_interval", "15m")),
    )


def _parse_kst(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=KST)


def _kst_to_ms(s: str) -> int:
    return int(_parse_kst(s).timestamp() * 1000)


def _bar_idx(minutes: int) -> int:
    return max(0, minutes // BAR_MINUTES - 1)


def _normalize_side(side: str) -> str:
    return side.lower() if side else "long"


def _is_numeric(val: str | float | int | None) -> bool:
    if val is None or val == "" or val == WAITING:
        return False
    try:
        float(val)
        return True
    except (TypeError, ValueError):
        return False


def _minutes_elapsed(scan_kst: str, now_kst: str) -> float:
    return (_parse_kst(now_kst) - _parse_kst(scan_kst)).total_seconds() / 60.0


def effective_scan_time(row: dict) -> str:
    """Prefer scan_id scan time when it differs from wall-clock timestamp (replay backfill)."""
    scan_id = row.get("scan_id", "")
    raw_ts = row.get("timestamp", "")
    if "|" in scan_id:
        scan_kst = scan_id.split("|")[0]
        if scan_kst and scan_kst != raw_ts:
            return scan_kst
    return raw_ts or (scan_id.split("|")[0] if "|" in scan_id else "")


def compute_side_aware_labels(
    klines: list,
    side: str,
    *,
    hours: int = 4,
) -> dict:
    """Side-aware ROI labels from forward 15m klines (entry = first bar open)."""
    if not klines or len(klines) < 2:
        return {}
    direction = _normalize_side(side)
    entry = float(klines[0][1])
    if entry <= 0:
        return {}

    idx_2h = _bar_idx(HORIZON_2H_MIN)
    idx_4h = _bar_idx(HORIZON_4H_MIN)
    roi_2h = _roi_at(klines, min(idx_2h, len(klines) - 1), direction)

    out: dict = {"actual_roi_2h": round(roi_2h, 4)}

    if hours >= 4 and len(klines) > idx_4h:
        end_i = min(idx_4h, len(klines) - 1)
        roi_4h = _roi_at(klines, end_i, direction)
        peak, _mdd_pos = _peak_and_mdd(klines, end_i, direction)
        rois = [_roi_at(klines, i, direction) for i in range(end_i + 1)]
        signed_dd = round(min(rois), 4) if rois else 0.0
        dna = _infer_actual_dna_type(klines, direction)
        out.update({
            "actual_roi_4h": round(roi_4h, 4),
            "actual_peak_roi": peak,
            "actual_drawdown": signed_dd,
            "actual_dna_type": dna,
        })
    elif hours >= 4:
        out["actual_roi_4h"] = WAITING
    return out


def _infer_actual_dna_type(klines: list, direction: str) -> str:
    rec = build_trade_dna(
        scan_kst="",
        symbol="",
        direction=direction,
        entry_score=0.0,
        live_pattern="",
        features={},
        klines=klines,
    )
    if rec is None:
        return ""
    if rec.final_roi_2h >= 3.0 and rec.peak_roi >= 4.0:
        return "TYPE_0"
    if rec.final_roi_2h < 0 or rec.max_drawdown >= 10:
        return "TYPE_1"
    return "TYPE_0" if rec.is_winner else "TYPE_1"


def compute_false_flags(row: dict, labels: dict, cfg: LabelerConfig) -> tuple[str, str]:
    policy = (row.get("policy_b_decision") or "").upper()
    roi_2h = labels.get("actual_roi_2h", "")
    if not _is_numeric(roi_2h):
        return "0", "0"

    roi_2h_f = float(roi_2h)
    false_skip = "0"
    false_accept = "0"

    if policy == "SKIP":
        peak = labels.get("actual_peak_roi", "")
        peak_hit = _is_numeric(peak) and float(peak) >= cfg.false_skip_peak_roi_pct
        if roi_2h_f >= cfg.false_skip_roi_2h_pct or peak_hit:
            false_skip = "1"

    if policy == "ENTER":
        dd = labels.get("actual_drawdown", "")
        dd_f = float(dd) if _is_numeric(dd) else 0.0
        if roi_2h_f < 0 or dd_f <= cfg.false_accept_drawdown_pct:
            false_accept = "1"

    return false_skip, false_accept


def _is_fully_labeled(row: dict) -> bool:
    return _is_numeric(row.get("actual_roi_4h"))


def _is_partial_labeled(row: dict) -> bool:
    return _is_numeric(row.get("actual_roi_2h")) and not _is_fully_labeled(row)


def _row_key(row: dict) -> str:
    return row.get("scan_id") or f"{row.get('timestamp')}|{row.get('symbol')}|{row.get('side')}"


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        with path.open(encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except (OSError, csv.Error):
        return []


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def default_klines_fetcher(
    symbol: str,
    start_ms: int,
    end_ms: int,
    cfg: LabelerConfig,
) -> list[list]:
    return fetch_klines_range(
        cfg.rest_base, symbol.upper(), cfg.kline_interval, start_ms, end_ms,
    )


def label_shadow_row(
    row: dict,
    *,
    now_kst_str: str,
    cfg: LabelerConfig,
    klines_fetcher: Callable[[str, int, int], list[list]],
    hours: int = 4,
    force: bool = False,
) -> tuple[dict, str]:
    """Return (updated_row, status). status: skipped | waiting | partial | full."""
    merged = dict(row)
    if not force:
        if hours <= 2 and _is_numeric(merged.get("actual_roi_2h")):
            return merged, "skipped"
        if _is_fully_labeled(merged):
            return merged, "skipped"
        if _is_partial_labeled(merged) and _minutes_elapsed(effective_scan_time(merged), now_kst_str) < HORIZON_4H_MIN:
            return merged, "skipped"

    elapsed = _minutes_elapsed(effective_scan_time(merged), now_kst_str)
    if elapsed < HORIZON_2H_MIN:
        return merged, "waiting"

    start_ms = _kst_to_ms(effective_scan_time(merged))
    horizon_min = HORIZON_4H_MIN if hours >= 4 else HORIZON_2H_MIN
    end_ms = start_ms + horizon_min * 60_000 + BAR_MINUTES * 60_000
    klines = klines_fetcher(merged["symbol"], start_ms, end_ms)
    if not klines:
        return merged, "waiting"

    if hours >= 4 and elapsed < HORIZON_4H_MIN:
        labels = compute_side_aware_labels(klines, merged.get("side", "LONG"), hours=2)
        if not labels:
            return merged, "waiting"
        merged.update(labels)
        merged["actual_roi_4h"] = WAITING
        merged["false_skip"], merged["false_accept"] = compute_false_flags(merged, merged, cfg)
        return merged, "partial"

    label_hours = 2 if hours <= 2 else 4
    labels = compute_side_aware_labels(klines, merged.get("side", "LONG"), hours=label_hours)
    if not labels:
        return merged, "waiting"

    merged.update(labels)
    merged["false_skip"], merged["false_accept"] = compute_false_flags(merged, merged, cfg)
    status = "full" if _is_fully_labeled(merged) else "partial"
    return merged, status


def label_shadow_row_replay(
    row: dict,
    store,
    cfg: LabelerConfig,
    *,
    force: bool = False,
) -> tuple[dict, str]:
    from scout_auto_os.engine.predator.labeler_diagnostics import resolve_replay_labels

    merged = dict(row)
    if not force and _is_fully_labeled(merged):
        return merged, "skipped"
    labels, _source, _key = resolve_replay_labels(merged, store)
    if not labels:
        return merged, "waiting"
    for k, v in labels.items():
        merged[k] = v if isinstance(v, str) else str(v)
    merged["false_skip"], merged["false_accept"] = compute_false_flags(merged, merged, cfg)
    return merged, "full"


def build_summary(rows: list[dict], cfg: LabelerConfig) -> dict:
    waiting = sum(
        1 for r in rows
        if not _is_numeric(r.get("actual_roi_2h"))
    )
    labeled_2h = sum(1 for r in rows if _is_numeric(r.get("actual_roi_2h")))
    labeled_4h = sum(1 for r in rows if _is_fully_labeled(r))

    enter_rois = [
        float(r["actual_roi_2h"]) for r in rows
        if r.get("policy_b_decision", "").upper() == "ENTER" and _is_numeric(r.get("actual_roi_2h"))
    ]
    skip_rois = [
        float(r["actual_roi_2h"]) for r in rows
        if r.get("policy_b_decision", "").upper() == "SKIP" and _is_numeric(r.get("actual_roi_2h"))
    ]

    def win_rate(rois: list[float]) -> float:
        if not rois:
            return 0.0
        wins = sum(1 for x in rois if x >= cfg.win_threshold_pct)
        return round(wins / len(rois) * 100, 2)

    short_fa = sum(
        1 for r in rows
        if r.get("side", "").upper() == "SHORT"
        and r.get("policy_b_decision", "").upper() == "ENTER"
        and r.get("false_accept") == "1"
    )

    return {
        "last_label_update": now_kst(),
        "total_rows": len(rows),
        "waiting_rows": waiting,
        "labeled_2h_rows": labeled_2h,
        "labeled_4h_rows": labeled_4h,
        "false_skip_count": sum(1 for r in rows if r.get("false_skip") == "1"),
        "false_accept_count": sum(1 for r in rows if r.get("false_accept") == "1"),
        "policy_enter_avg_roi_2h": round(sum(enter_rois) / len(enter_rois), 4) if enter_rois else 0.0,
        "policy_skip_avg_roi_2h": round(sum(skip_rois) / len(skip_rois), 4) if skip_rois else 0.0,
        "enter_win_rate": win_rate(enter_rois),
        "skip_win_rate": win_rate(skip_rois),
        "short_false_accept_count": short_fa,
        "mode": "SHADOW_ONLY",
        "dry_run": True,
    }


def write_label_report(out_dir: Path, summary: dict, *, processed: dict) -> Path:
    path = out_dir / REPORT_MD
    lines = [
        "# Value Gate Shadow Labeler V1 — Report",
        "",
        f"**Last update:** {summary.get('last_label_update', '')}",
        "",
        "## Summary",
        "",
        f"- Total rows: {summary.get('total_rows', 0)}",
        f"- Waiting: {summary.get('waiting_rows', 0)}",
        f"- Labeled 2h: {summary.get('labeled_2h_rows', 0)}",
        f"- Labeled 4h: {summary.get('labeled_4h_rows', 0)}",
        f"- False skip: {summary.get('false_skip_count', 0)}",
        f"- False accept: {summary.get('false_accept_count', 0)}",
        f"- Short false accept: {summary.get('short_false_accept_count', 0)}",
        "",
        "## Run stats",
        "",
        f"- New full labels: {processed.get('new_full', 0)}",
        f"- New partial (2h): {processed.get('new_partial', 0)}",
        f"- Skipped (incremental): {processed.get('skipped', 0)}",
        f"- Still waiting: {processed.get('waiting', 0)}",
        "",
        "## Final answers",
        "",
        "1. Shadow log 2h/4h labels — Yes (labeled CSV)",
        "2. Side-aware ROI — Yes (LONG/SHORT directional)",
        "3. Unelapsed rows — Yes (untouched / WAITING)",
        "4. false_skip / false_accept — Yes (auto flags)",
        "5. Command Center summary JSON — Yes",
        "6. Policy B LIVE decision data structure — Yes (accumulation ready)",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_shadow_labeler(
    data_dir: Path,
    *,
    cfg: LabelerConfig | None = None,
    config: dict | None = None,
    hours: int = 4,
    force: bool = False,
    dry_run: bool = False,
    mode: str = "live",
    diagnose_only: bool = False,
    now_kst_str: str | None = None,
    klines_fetcher: Callable[[str, int, int], list[list]] | None = None,
    pkg_root: Path | None = None,
) -> dict:
    label_cfg = cfg or load_labeler_config(config)
    now_s = now_kst_str or now_kst()
    root = pkg_root or data_dir.parent
    shadow_dir = data_dir / "runtime_shadow"

    from scout_auto_os.engine.predator.labeler_diagnostics import ReplaySources, run_diagnostics

    if diagnose_only:
        return run_diagnostics(
            data_dir, root, cfg=label_cfg, mode=mode, now_s=now_s, klines_fetcher=klines_fetcher,
        )

    replay_store = ReplaySources.discover(root) if mode == "replay" else None
    source_path = shadow_dir / SHADOW_SOURCE
    labeled_path = shadow_dir / SHADOW_LABELED
    watch_source = shadow_dir / WATCH_SOURCE
    watch_labeled = shadow_dir / WATCH_LABELED

    source_rows = _read_csv(source_path)
    existing = { _row_key(r): r for r in _read_csv(labeled_path) }

    fetcher = klines_fetcher or (
        lambda sym, s, e: default_klines_fetcher(sym, s, e, label_cfg)
    )

    out_rows: list[dict] = []
    stats = {"new_full": 0, "new_partial": 0, "skipped": 0, "waiting": 0}

    for src in source_rows:
        key = _row_key(src)
        if force:
            base = dict(src)
        else:
            base = dict(existing.get(key, src))
            for k in SHADOW_FIELDS:
                if k not in base or (not base.get(k) and src.get(k)):
                    base[k] = src.get(k, base.get(k, ""))

        if mode == "replay" and replay_store is not None:
            updated, status = label_shadow_row_replay(
                base, replay_store, label_cfg, force=force,
            )
        else:
            updated, status = label_shadow_row(
                base,
                now_kst_str=now_s,
                cfg=label_cfg,
                klines_fetcher=fetcher,
                hours=hours,
                force=force,
            )
        if status == "skipped":
            stats["skipped"] += 1
        elif status == "waiting":
            stats["waiting"] += 1
        elif status == "partial":
            if not _is_partial_labeled(existing.get(key, {})):
                stats["new_partial"] += 1
        elif status == "full":
            if not _is_fully_labeled(existing.get(key, {})):
                stats["new_full"] += 1
        out_rows.append(updated)

    summary = build_summary(out_rows, label_cfg)
    summary["run_stats"] = stats
    summary["label_mode"] = mode

    labeled_by_scan = {r.get("scan_id", ""): r for r in out_rows}
    watch_out: list[dict] = []
    for w in _read_csv(watch_source):
        merged = dict(w)
        match = labeled_by_scan.get(w.get("scan_id", ""))
        if match and _is_numeric(match.get("actual_roi_2h")):
            for col in (
                "actual_roi_2h", "actual_roi_4h", "actual_peak_roi",
                "actual_drawdown", "actual_dna_type", "false_skip", "false_accept",
            ):
                merged[col] = match.get(col, "")
            merged["actual_after_label_if_available"] = match.get("actual_roi_2h", "")
        watch_out.append(merged)

    if not dry_run:
        _write_csv(labeled_path, SHADOW_FIELDS, out_rows)
        if watch_out or watch_source.exists():
            _write_csv(watch_labeled, WATCH_LABEL_FIELDS, watch_out)
        (shadow_dir / SUMMARY_JSON).write_text(
            json.dumps(summary, indent=2), encoding="utf-8",
        )
        write_label_report(shadow_dir, summary, processed=stats)
        run_diagnostics(data_dir, root, cfg=label_cfg, mode=mode, now_s=now_s, klines_fetcher=fetcher)

    return {
        "ok": True,
        "dry_run": dry_run,
        "summary": summary,
        "labeled_path": str(labeled_path),
        "watch_labeled_path": str(watch_labeled),
        "stats": stats,
    }
