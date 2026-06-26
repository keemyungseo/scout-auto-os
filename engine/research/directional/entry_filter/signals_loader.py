"""Load direction champion signals for threshold optimization."""

from __future__ import annotations

import csv
from pathlib import Path

from scout_auto_os.engine.research.directional.dna.collector import numeric_feature_keys
from scout_auto_os.engine.research.directional.entry_filter.analyzer import split_winner_loser
from scout_auto_os.engine.research.directional.entry_filter.collector import (
    collect_direction_champion_signals,
    filter_scans_last_months,
)
from scout_auto_os.engine.research.directional.entry_filter.constants import (
    CHAMPION_TOP_K,
    LOOKBACK_MONTHS,
)
from scout_auto_os.engine.research.zero_base.runner import load_candidates_jsonl, load_forward_klines


def load_signals_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for row in csv.DictReader(path.open(encoding="utf-8")):
        features = {
            k[5:]: float(row[k])
            for k in row
            if k.startswith("feat_") and row[k] not in ("", None)
        }
        rows.append({
            "direction": row["direction"],
            "engine": row.get("engine", ""),
            "scan_time_kst": row.get("scan_time_kst", ""),
            "symbol": row.get("symbol", ""),
            "return_30m": float(row.get("return_30m", 0)),
            "return_1h": float(row.get("return_1h", 0)),
            "return_2h": float(row.get("return_2h", 0)),
            "return_4h": float(row.get("return_4h", 0)),
            "features": features,
        })
    return rows


def label_signals(signals: list[dict]) -> tuple[list[dict], dict]:
    """Attach cohort label: winner | loser | middle."""
    winners, losers, meta = split_winner_loser(signals)
    winner_ids = {(s["scan_time_kst"], s["symbol"]) for s in winners}
    loser_ids = {(s["scan_time_kst"], s["symbol"]) for s in losers}
    labeled: list[dict] = []
    for s in signals:
        key = (s["scan_time_kst"], s["symbol"])
        if key in winner_ids:
            cohort = "winner"
        elif key in loser_ids:
            cohort = "loser"
        else:
            cohort = "middle"
        labeled.append({**s, "cohort": cohort})
    return labeled, meta


def resolve_signals(
    data_dir: Path,
    candidates_path: Path,
    forward_path: Path,
    lookback_months: int = LOOKBACK_MONTHS,
    top_k: int = CHAMPION_TOP_K,
) -> tuple[list[dict], list[dict], list[str]]:
    csv_path = data_dir / "zero_base" / "direction_champion_signals.csv"
    all_rows = load_signals_csv(csv_path)
    if all_rows:
        long_s = [r for r in all_rows if r["direction"] == "long"]
        short_s = [r for r in all_rows if r["direction"] == "short"]
        if long_s:
            keys = numeric_feature_keys(long_s[0]["features"])
            return long_s, short_s, keys

    by_scan = load_candidates_jsonl(candidates_path)
    fwd = load_forward_klines(forward_path)
    scans = filter_scans_last_months(sorted(by_scan.keys()), lookback_months)
    long_s, short_s = collect_direction_champion_signals(by_scan, fwd, scans, top_k=top_k)
    keys = numeric_feature_keys(long_s[0]["features"]) if long_s else []
    return long_s, short_s, keys


def load_dna_feature_sets(data_dir: Path, pkg_root: Path) -> dict[str, list[str] | set[str]]:
    """Winner DNA, common DNA — ordered importance lists + sets."""
    out: dict[str, list[str] | set[str]] = {
        "long": [],
        "short": [],
        "common": set(),
        "long_set": set(),
        "short_set": set(),
    }
    zb = data_dir / "zero_base"
    rb = pkg_root / "research_bundle" / "reports"

    def _read_imp(path: Path) -> list[str]:
        if not path.exists():
            return []
        rows: list[tuple[int, str]] = []
        with path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if "feature" not in row:
                    continue
                rank = int(row.get("importance_rank", 999))
                rows.append((rank, row["feature"]))
        rows.sort(key=lambda x: x[0])
        return [f for _, f in rows]

    long_path = zb / "entry_dna_feature_importance_long.csv"
    if not long_path.exists():
        long_path = rb / "entry_dna_feature_importance_long_v1.csv"
    short_path = zb / "entry_dna_feature_importance_short.csv"
    if not short_path.exists():
        short_path = rb / "entry_dna_feature_importance_short_v1.csv"
    common_path = zb / "entry_dna_common.csv"
    if not common_path.exists():
        common_path = rb / "entry_dna_common_v1.csv"

    out["long"] = _read_imp(long_path)
    out["short"] = _read_imp(short_path)
    out["long_set"] = set(out["long"])
    out["short_set"] = set(out["short"])
    if common_path.exists():
        with common_path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                out["common"].add(row["feature"])
    return out
