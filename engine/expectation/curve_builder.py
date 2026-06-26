"""Research-based expected progress curves — Long/Short independent."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class ExpectedPath:
    path_id: str
    thesis_id: str
    position_id: str
    symbol: str
    side: str
    expected_return: float
    expected_horizon: int
    expected_success_probability: float
    expected_peak_window_min: int
    expected_hold_profile: str
    expected_volatility: float
    expected_drawdown: float
    expected_progress_curve: list[dict]
    curve_version: str
    thesis_version: int = 1
    research_sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ExpectedPath":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _read_lifecycle_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _curve_version(sources: list[str]) -> str:
    h = hashlib.sha256("|".join(sorted(sources)).encode()).hexdigest()
    return f"rv_{h[:10]}"


def _load_short_research_curve(data_dir: Path) -> tuple[list[dict], dict, list[str]]:
    path = data_dir / "short_execution" / "lifecycle_aggregate.csv"
    rows = _read_lifecycle_csv(path)
    if not rows:
        raise FileNotFoundError(f"short lifecycle research missing: {path}")
    curve = [
        {"minute": int(r["checkpoint_min"]), "roi_pct": float(r["avg_roi_pct"])}
        for r in rows
    ]
    meta = {
        "peak_window_min": 90,
        "volatility": statistics.mean(float(r["avg_mae_pct"]) for r in rows),
        "drawdown": max(float(r["avg_mae_pct"]) for r in rows),
        "hold_profile": "peak_60_120m",
    }
    dist = _read_lifecycle_csv(data_dir / "short_execution" / "holding_distribution.csv")
    if dist:
        weighted_peak = 0.0
        total = sum(int(r["count"]) for r in dist)
        for r in dist:
            bucket = r["peak_bucket"]
            mid = _bucket_mid(bucket)
            weighted_peak += mid * int(r["count"])
        if total:
            meta["peak_window_min"] = int(round(weighted_peak / total))
    return curve, meta, [str(path)]


def _bucket_mid(label: str) -> int:
    if "+" in label:
        return 210
    lo, hi = label.replace("m", "").split("-")
    return (int(lo) + int(hi)) // 2


def _load_long_research_curve(data_dir: Path) -> tuple[list[dict], dict, list[str]]:
    short_path = data_dir / "short_execution" / "lifecycle_aggregate.csv"
    short_rows = _read_lifecycle_csv(short_path)
    sources = [str(short_path)]

    meta_path = data_dir / "constitution_validation" / "constitution_validation_meta.json"
    long_final = 5.2608
    if meta_path.exists():
        meta_json = json.loads(meta_path.read_text(encoding="utf-8"))
        long_final = float(meta_json.get("blind", {}).get("avg_return_2h", long_final))
        sources.append(str(meta_path))

    cluster_path = data_dir / "signal_lifecycle" / "lifecycle_cluster.csv"
    peak_window = 120
    volatility = 2.5
    if cluster_path.exists():
        sources.append(str(cluster_path))
        with cluster_path.open(encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if r.get("direction") == "long"]
        if rows:
            total = sum(int(r["signal_count"]) for r in rows)
            peak_window = int(round(
                sum(float(r["avg_peak_time_min"]) * int(r["signal_count"]) for r in rows) / max(total, 1),
            ))
            volatility = round(
                sum(abs(float(r["avg_mae_pct"])) * int(r["signal_count"]) for r in rows) / max(total, 1), 4,
            )

    if not short_rows:
        raise FileNotFoundError("cannot build long curve without short lifecycle shape")

    ref_min = 120
    ref_roi = next((float(r["avg_roi_pct"]) for r in short_rows if int(r["checkpoint_min"]) == ref_min), 1.0)
    scale = long_final / ref_roi if abs(ref_roi) > 1e-6 else 1.0

    curve = [
        {"minute": int(r["checkpoint_min"]), "roi_pct": round(float(r["avg_roi_pct"]) * scale, 4)}
        for r in short_rows
    ]
    meta = {
        "peak_window_min": peak_window,
        "volatility": volatility,
        "drawdown": round(volatility * 1.2, 4),
        "hold_profile": "constitution_long_shape",
    }
    return curve, meta, sources


def load_research_curve(side: str, data_dir: Path) -> tuple[list[dict], dict, list[str]]:
    side = side.upper()
    if side == "SHORT":
        return _load_short_research_curve(data_dir)
    return _load_long_research_curve(data_dir)


def scale_curve_to_thesis(
    base_curve: list[dict],
    expected_return: float,
    expected_horizon: int,
) -> list[dict]:
    if not base_curve:
        return [{"minute": 0, "roi_pct": 0.0}]
    horizon_pt = min(base_curve, key=lambda p: abs(p["minute"] - expected_horizon))
    ref = float(horizon_pt["roi_pct"]) or 1.0
    factor = expected_return / ref if abs(ref) > 1e-6 else 1.0
    scaled = [{"minute": 0, "roi_pct": 0.0}]
    for p in base_curve:
        if p["minute"] == 0:
            continue
        scaled.append({
            "minute": int(p["minute"]),
            "roi_pct": round(float(p["roi_pct"]) * factor, 4),
        })
    if scaled[-1]["minute"] != expected_horizon:
        scaled.append({"minute": expected_horizon, "roi_pct": round(expected_return, 4)})
    return sorted(scaled, key=lambda x: x["minute"])


def build_expected_path(
    thesis_id: str,
    position_id: str,
    symbol: str,
    side: str,
    expected_return: float,
    expected_horizon: int,
    success_probability: float,
    data_dir: Path,
    thesis_version: int = 1,
) -> ExpectedPath:
    base, meta, sources = load_research_curve(side, data_dir)
    curve = scale_curve_to_thesis(base, expected_return, expected_horizon)
    return ExpectedPath(
        path_id=f"ep_{uuid.uuid4().hex[:12]}",
        thesis_id=thesis_id,
        position_id=position_id,
        symbol=symbol,
        side=side.upper(),
        expected_return=expected_return,
        expected_horizon=expected_horizon,
        expected_success_probability=success_probability,
        expected_peak_window_min=int(meta["peak_window_min"]),
        expected_hold_profile=str(meta["hold_profile"]),
        expected_volatility=float(meta["volatility"]),
        expected_drawdown=float(meta["drawdown"]),
        expected_progress_curve=curve,
        curve_version=_curve_version(sources),
        thesis_version=thesis_version,
        research_sources=sources,
    )


class ExpectedPathStore:
    def __init__(self, data_dir: Path) -> None:
        self.dir = data_dir / "expectation"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "expected_path.jsonl"
        self._by_thesis: dict[str, ExpectedPath] = {}
        self._by_position: dict[str, ExpectedPath] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                ep = ExpectedPath.from_dict(json.loads(line))
                self._by_thesis[ep.thesis_id] = ep
                self._by_position[ep.position_id] = ep
            except (json.JSONDecodeError, TypeError):
                continue

    def append(self, ep: ExpectedPath) -> None:
        self._by_thesis[ep.thesis_id] = ep
        self._by_position[ep.position_id] = ep
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(ep.to_dict(), ensure_ascii=False) + "\n")

    def get_by_thesis(self, thesis_id: str) -> ExpectedPath | None:
        return self._by_thesis.get(thesis_id)

    def get_by_position(self, position_id: str) -> ExpectedPath | None:
        return self._by_position.get(position_id)
