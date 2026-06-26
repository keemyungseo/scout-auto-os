"""Shadow labeler diagnostics — timestamp, kline, and replay join analysis."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scout_auto_os.engine.predator.shadow_labeler import (
    HORIZON_2H_MIN,
    HORIZON_4H_MIN,
    LabelerConfig,
    _is_fully_labeled,
    _is_numeric,
    _kst_to_ms,
    _parse_kst,
    default_klines_fetcher,
    effective_scan_time,
)
from scout_auto_os.engine.research.zero_base.runner import load_forward_klines
from scout_auto_os.storage.db import now_kst

KST = timezone(timedelta(hours=9))

DIAG_FIELDS = (
    "scan_id", "symbol", "side", "timestamp", "effective_timestamp",
    "entry_price", "policy_b_decision",
    "diagnosis", "label_source", "can_label_2h", "can_label_4h",
    "kline_count", "join_match", "error_reason",
)

LOCAL_SEARCH_DIRS = (
    "data/trade_dna",
    "data/replay",
    "data/candles",
    "data/cache",
    "data/binance",
    "data/history",
    "research_bundle/forward",
)

FORWARD_KLINE_CANDIDATES = (
    "research_bundle/forward/forward_klines_15m.jsonl",
    "research_bundle/forward/forward_klines_15m_last54.jsonl",
)

TRADE_DNA_CANDIDATES = (
    "data/trade_dna/trade_cluster.csv",
    "data/trade_dna/value_prediction.csv",
    "data/trade_dna/position_size_simulation.csv",
)


@dataclass
class ReplaySources:
    pkg_root: Path
    cluster: dict[str, dict] = field(default_factory=dict)
    value: dict[str, dict] = field(default_factory=dict)
    forward: dict[tuple[str, str], list] = field(default_factory=dict)
    discovered_paths: list[str] = field(default_factory=list)

    @classmethod
    def discover(cls, pkg_root: Path) -> ReplaySources:
        store = cls(pkg_root=pkg_root)
        for rel in TRADE_DNA_CANDIDATES:
            path = pkg_root / rel
            if not path.exists():
                continue
            store.discovered_paths.append(str(path))
            rows = _read_csv(path)
            if "trade_cluster" in rel:
                store.cluster = {r["trade_key"]: r for r in rows if r.get("trade_key")}
            elif "value_prediction" in rel:
                store.value = {r["trade_key"]: r for r in rows if r.get("trade_key")}
        for rel in FORWARD_KLINE_CANDIDATES:
            path = pkg_root / rel
            if path.exists():
                store.discovered_paths.append(str(path))
                store.forward.update(load_forward_klines(path))
        return store


def trade_key_from_row(row: dict) -> str:
    scan_id = row.get("scan_id", "")
    if scan_id.count("|") >= 2:
        parts = scan_id.split("|")
        return f"{parts[0]}|{parts[1].upper()}|{parts[2].lower()}"
    eff = effective_scan_time(row)
    return f"{eff}|{row.get('symbol', '').upper()}|{row.get('side', 'LONG').lower()}"


def labels_from_cluster(cluster_row: dict) -> dict:
    rois = [
        float(cluster_row[k]) for k in cluster_row
        if k.startswith("roi_") and cluster_row.get(k) not in ("", None)
    ]
    signed_dd = round(min(rois), 4) if rois else -round(float(cluster_row.get("max_drawdown", 0)), 4)
    return {
        "actual_roi_2h": round(float(cluster_row.get("final_roi_2h", 0)), 4),
        "actual_roi_4h": round(float(cluster_row.get("final_roi_4h", 0)), 4),
        "actual_peak_roi": round(float(cluster_row.get("peak_roi", 0)), 4),
        "actual_drawdown": signed_dd,
        "actual_dna_type": cluster_row.get("trade_type_id", ""),
    }


def labels_from_value(value_row: dict, cluster_row: dict | None) -> dict | None:
    if not value_row and not cluster_row:
        return None
    c = cluster_row or {}
    roi_2h = value_row.get("actual_expected_roi") or c.get("final_roi_2h")
    if roi_2h in ("", None):
        return None
    labels = labels_from_cluster(c) if c else {}
    labels["actual_roi_2h"] = round(float(roi_2h), 4)
    if c.get("final_roi_4h") not in ("", None):
        labels["actual_roi_4h"] = round(float(c["final_roi_4h"]), 4)
    peak = value_row.get("actual_expected_peak_roi") or c.get("peak_roi")
    if peak not in ("", None):
        labels["actual_peak_roi"] = round(float(peak), 4)
    dna = c.get("trade_type_id") or value_row.get("trade_type_id", "")
    if dna:
        labels["actual_dna_type"] = dna
    return labels


def resolve_replay_labels(row: dict, store: ReplaySources) -> tuple[dict | None, str, str]:
    """Return (labels, label_source, join_match)."""
    key = trade_key_from_row(row)
    scan_kst = key.split("|")[0]
    symbol = key.split("|")[1]

    fwd = store.forward.get((scan_kst, symbol))
    if fwd and len(fwd) >= 2:
        from scout_auto_os.engine.predator.shadow_labeler import compute_side_aware_labels
        labels = compute_side_aware_labels(fwd, row.get("side", "LONG"), hours=4)
        if labels.get("actual_roi_2h") is not None:
            return labels, "local_forward_klines", key

    if key in store.cluster:
        return labels_from_cluster(store.cluster[key]), "trade_cluster", key

    if key in store.value:
        joined = labels_from_value(store.value[key], store.cluster.get(key))
        if joined:
            return joined, "value_prediction", key

    return None, "label_unavailable", key


def classify_timestamp(row: dict, now_s: str) -> tuple[str, str, dict]:
    """Return (primary_diagnosis, error_reason, timing_info)."""
    raw = row.get("timestamp", "")
    eff = effective_scan_time(row)
    info: dict = {
        "raw_timestamp": raw,
        "effective_timestamp": eff,
        "timezone": "KST",
    }
    try:
        eff_dt = _parse_kst(eff) if eff else None
        raw_dt = _parse_kst(raw) if raw else None
        now_dt = _parse_kst(now_s)
    except ValueError:
        return "PARSE_ERROR", f"cannot parse effective={eff!r}", info

    if eff_dt is None:
        return "PARSE_ERROR", "missing effective timestamp", info

    info["parsed_effective"] = eff_dt.isoformat()
    info["elapsed_minutes"] = round((now_dt - eff_dt).total_seconds() / 60, 2)
    info["can_label_2h"] = info["elapsed_minutes"] >= HORIZON_2H_MIN
    info["can_label_4h"] = info["elapsed_minutes"] >= HORIZON_4H_MIN

    if raw and raw != eff:
        info["timestamp_mismatch"] = True
        info["raw_elapsed_minutes"] = round((now_dt - raw_dt).total_seconds() / 60, 2) if raw_dt else None

    if info["elapsed_minutes"] < 0:
        return "FUTURE_TIMESTAMP", f"effective {eff} is in the future", info
    if info["elapsed_minutes"] < HORIZON_2H_MIN:
        return "TOO_RECENT", f"only {info['elapsed_minutes']:.1f}m elapsed", info
    return "OK", "", info


def diagnose_row(
    row: dict,
    *,
    now_s: str,
    mode: str,
    store: ReplaySources,
    cfg: LabelerConfig,
    klines_fetcher=None,
) -> dict:
    sym = row.get("symbol", "")
    join_match = trade_key_from_row(row)
    label_source = ""
    kline_count = 0
    error = ""
    entry_price = row.get("entry_price", "")
    can_2h = 0
    can_4h = 0

    if not sym or not sym.endswith("USDT"):
        return {
            "scan_id": row.get("scan_id", ""),
            "symbol": sym,
            "side": row.get("side", ""),
            "timestamp": row.get("timestamp", ""),
            "effective_timestamp": effective_scan_time(row),
            "entry_price": entry_price,
            "policy_b_decision": row.get("policy_b_decision", ""),
            "diagnosis": "SYMBOL_FORMAT_ERROR",
            "label_source": "",
            "can_label_2h": 0,
            "can_label_4h": 0,
            "kline_count": 0,
            "join_match": join_match,
            "error_reason": f"symbol={sym!r}",
        }

    eff = effective_scan_time(row)
    ts_diag, error, timing = classify_timestamp(row, now_s)
    replay_labels, label_source, join_match = resolve_replay_labels(row, store)
    join_hit = replay_labels is not None

    fwd = store.forward.get((eff, sym.upper()))
    if not entry_price and fwd:
        entry_price = str(fwd[0][1])

    diagnosis = ts_diag
    if timing.get("timestamp_mismatch") and timing.get("can_label_2h"):
        diagnosis = "TIMESTAMP_MISMATCH"
        error = (
            f"row.timestamp={row.get('timestamp')} vs scan_id={eff}; "
            "labeler must use effective scan time"
        )

    if mode == "replay":
        if join_hit:
            diagnosis = "REPLAY_JOIN_OK"
            error = ""
        else:
            diagnosis = "LABEL_UNAVAILABLE"
            error = f"no local join for {join_match}"
            label_source = "label_unavailable"

    if mode == "live" and ts_diag == "TOO_RECENT":
        diagnosis = "TOO_RECENT"

    if klines_fetcher and eff and sym and mode == "live":
        try:
            start_ms = _kst_to_ms(eff)
            end_ms = start_ms + HORIZON_4H_MIN * 60_000 + 900_000
            kl = klines_fetcher(sym, start_ms, end_ms)
            kline_count = len(kl)
            if kline_count == 0 and diagnosis in ("OK", "TIMESTAMP_MISMATCH"):
                diagnosis = "KLINE_EMPTY"
                error = error or "Binance returned 0 candles"
        except OSError as exc:
            kline_count = 0
            diagnosis = "KLINE_EMPTY"
            error = str(exc)

    if not entry_price and diagnosis not in ("REPLAY_JOIN_OK", "LABEL_UNAVAILABLE"):
        if diagnosis == "OK":
            diagnosis = "MISSING_ENTRY_PRICE"

    can_2h = int(bool(timing.get("can_label_2h") or join_hit or kline_count > 0))
    can_4h = int(bool(timing.get("can_label_4h") or join_hit or kline_count > 8))

    return {
        "scan_id": row.get("scan_id", ""),
        "symbol": sym,
        "side": row.get("side", ""),
        "timestamp": row.get("timestamp", ""),
        "effective_timestamp": eff,
        "entry_price": entry_price,
        "policy_b_decision": row.get("policy_b_decision", ""),
        "diagnosis": diagnosis,
        "label_source": label_source,
        "can_label_2h": can_2h,
        "can_label_4h": can_4h,
        "kline_count": kline_count,
        "join_match": join_match,
        "error_reason": error,
    }


def probe_klines_sample(
    rows: list[dict],
    cfg: LabelerConfig,
    *,
    limit: int = 20,
    klines_fetcher=None,
) -> list[dict]:
    fetcher = klines_fetcher or (lambda s, a, b: default_klines_fetcher(s, a, b, cfg))
    probes: list[dict] = []
    for row in rows[:limit]:
        eff = effective_scan_time(row)
        sym = row.get("symbol", "")
        start_ms = _kst_to_ms(eff) if eff else 0
        end_ms = start_ms + HORIZON_4H_MIN * 60_000 + 900_000
        reason = ""
        candles: list = []
        try:
            candles = fetcher(sym, start_ms, end_ms) if eff and sym else []
        except OSError as exc:
            reason = str(exc)
        first_t = last_t = ""
        if candles:
            first_t = datetime.fromtimestamp(int(candles[0][0]) / 1000, tz=KST).strftime("%Y-%m-%d %H:%M:%S")
            last_t = datetime.fromtimestamp(int(candles[-1][0]) / 1000, tz=KST).strftime("%Y-%m-%d %H:%M:%S")
        probes.append({
            "symbol": sym,
            "start_time": eff,
            "end_time": datetime.fromtimestamp(end_ms / 1000, tz=KST).strftime("%Y-%m-%d %H:%M:%S"),
            "requested_interval": cfg.kline_interval,
            "returned_candles": len(candles),
            "first_candle_time": first_t,
            "last_candle_time": last_t,
            "reason_if_empty": reason or ("no candles" if not candles else ""),
        })
    return probes


def run_diagnostics(
    data_dir: Path,
    pkg_root: Path,
    *,
    cfg: LabelerConfig | None = None,
    mode: str = "live",
    now_s: str | None = None,
    klines_fetcher=None,
    sample_limit: int = 20,
) -> dict:
    from scout_auto_os.engine.predator.shadow_labeler import load_labeler_config

    label_cfg = cfg or load_labeler_config()
    now_str = now_s or now_kst()
    shadow_dir = data_dir / "runtime_shadow"
    source_rows = _read_csv(shadow_dir / "value_gate_runtime_shadow.csv")
    labeled_rows = _read_csv(shadow_dir / "value_gate_runtime_shadow_labeled.csv")
    store = ReplaySources.discover(pkg_root)

    diag_rows = [
        diagnose_row(
            r, now_s=now_str, mode=mode, store=store, cfg=label_cfg, klines_fetcher=klines_fetcher,
        )
        for r in (labeled_rows or source_rows)
    ]
    probes = probe_klines_sample(source_rows, label_cfg, limit=sample_limit, klines_fetcher=klines_fetcher)

    counts: dict[str, int] = {}
    for d in diag_rows:
        counts[d["diagnosis"]] = counts.get(d["diagnosis"], 0) + 1

    replay_join_ok = sum(1 for d in diag_rows if d["diagnosis"] == "REPLAY_JOIN_OK")
    summary = {
        "generated_at": now_str,
        "mode": mode,
        "total_rows": len(diag_rows),
        "diagnosis_counts": counts,
        "replay_join_ok": replay_join_ok,
        "timestamp_mismatch_rows": sum(
            1 for r in source_rows
            if effective_scan_time(r) != r.get("timestamp", "")
        ),
        "kline_probe_sample_size": len(probes),
        "kline_probe_empty": sum(1 for p in probes if p["returned_candles"] == 0),
        "discovered_local_paths": store.discovered_paths,
        "labeled_2h_rows": sum(1 for r in labeled_rows if _is_numeric(r.get("actual_roi_2h"))),
        "labeled_4h_rows": sum(1 for r in labeled_rows if _is_fully_labeled(r)),
    }

    diag_csv = shadow_dir / "labeler_diagnostics.csv"
    _write_csv(diag_csv, DIAG_FIELDS, diag_rows)
    summary_path = shadow_dir / "labeler_diagnostics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report_path = shadow_dir / "labeler_diagnostics_report.md"
    _write_report(report_path, summary, probes, diag_rows)

    return {
        "ok": True,
        "summary": summary,
        "diagnostics_csv": str(diag_csv),
        "probes": probes,
    }


def _write_report(path: Path, summary: dict, probes: list[dict], diag_rows: list[dict]) -> None:
    lines = [
        "# Shadow Labeler Diagnostics Report",
        "",
        f"**Generated:** {summary.get('generated_at')}",
        f"**Mode:** {summary.get('mode')}",
        "",
        "## Root cause (replay 157 WAITING)",
        "",
        (
            f"- Timestamp mismatch rows: **{summary.get('timestamp_mismatch_rows', 0)}** "
            "(row `timestamp` = backfill wall clock; `scan_id` holds true scan_kst)"
        ),
        f"- Diagnosis counts: `{summary.get('diagnosis_counts')}`",
        f"- Replay join available: **{summary.get('replay_join_ok', 0)}** / {summary.get('total_rows', 0)}",
        "",
        "## Kline probe (sample)",
        "",
    ]
    for p in probes[:5]:
        lines.append(
            f"- {p['symbol']} {p['start_time']} → candles={p['returned_candles']} "
            f"({p['reason_if_empty'] or 'ok'})"
        )
    lines.extend([
        "",
        "## Final answers",
        "",
        "1. **WAITING cause:** backfill wrote `now_kst()` to `timestamp`; labeler used that (TOO_RECENT/FUTURE) instead of `scan_id` scan time.",
        "2. **Primary issue:** timestamp mismatch + live-mode elapsed check; Binance klines work when effective time is used.",
        "3. **Replay mode:** join via `trade_cluster` / `value_prediction` / forward jsonl.",
        "4. **Live mode:** unchanged WAITING for unelapsed rows.",
        "5. **Policy B dataset:** replay mode fills labeled CSV from local trade_dna.",
        "6. **Command Center summary:** valid after replay label run.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})
