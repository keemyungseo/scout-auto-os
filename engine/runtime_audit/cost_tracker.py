"""Observe-only runtime cost tracking — no trading logic changes."""

from __future__ import annotations

import csv
import json
import os
import threading
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

_GLOBAL: "CostTracker | None" = None


@dataclass
class TickSample:
    scope: str
    module: str
    cpu_ms: float
    memory_delta_kb: float
    bar_fetches: int
    db_reads: int
    db_writes: int
    disk_writes: int
    duplicate_calcs: int
    positions_reviewed: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class ModuleCostAgg:
    module: str
    sample_count: int = 0
    total_cpu_ms: float = 0.0
    avg_cpu_ms: float = 0.0
    max_cpu_ms: float = 0.0
    total_bar_fetches: int = 0
    total_db_reads: int = 0
    total_db_writes: int = 0
    total_disk_writes: int = 0
    total_duplicate_calcs: int = 0
    memory_peak_kb: float = 0.0


class CostTracker:
    """Non-invasive cost probe. Writes tick samples + rolling aggregates."""

    def __init__(self, out_dir: Path, enabled: bool = True) -> None:
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.enabled = enabled
        self._lock = threading.Lock()
        self._samples: list[TickSample] = []
        self._tick_path = self.out_dir / "cost_tick.jsonl"
        self._bar_fetches = 0
        self._db_reads = 0
        self._db_writes = 0
        self._disk_writes = 0
        self._dup_calcs = 0
        self._positions_reviewed = 0
        self._active_module = ""

    def record_bar_fetch(self, n: int = 1) -> None:
        if self.enabled:
            self._bar_fetches += n

    def record_db_read(self, n: int = 1) -> None:
        if self.enabled:
            self._db_reads += n

    def record_db_write(self, n: int = 1) -> None:
        if self.enabled:
            self._db_writes += n

    def record_disk_write(self, n: int = 1) -> None:
        if self.enabled:
            self._disk_writes += n

    def record_duplicate_calc(self, n: int = 1) -> None:
        if self.enabled:
            self._dup_calcs += n

    def record_positions_reviewed(self, n: int) -> None:
        if self.enabled:
            self._positions_reviewed += n

    @contextmanager
    def tick(self, scope: str, module: str = "") -> Iterator[None]:
        if not self.enabled:
            yield
            return
        mod = module or scope
        mem_before = 0
        tracemalloc_started = False
        if not tracemalloc.is_tracing():
            tracemalloc.start()
            tracemalloc_started = True
        mem_before = tracemalloc.get_traced_memory()[0]

        self._bar_fetches = 0
        self._db_reads = 0
        self._db_writes = 0
        self._disk_writes = 0
        self._dup_calcs = 0
        self._positions_reviewed = 0
        self._active_module = mod
        t0 = time.perf_counter()
        try:
            yield
        finally:
            cpu_ms = (time.perf_counter() - t0) * 1000
            mem_after = tracemalloc.get_traced_memory()[0]
            mem_delta_kb = max(0.0, (mem_after - mem_before) / 1024)
            if tracemalloc_started:
                tracemalloc.stop()
            sample = TickSample(
                scope=scope,
                module=mod,
                cpu_ms=round(cpu_ms, 3),
                memory_delta_kb=round(mem_delta_kb, 2),
                bar_fetches=self._bar_fetches,
                db_reads=self._db_reads,
                db_writes=self._db_writes,
                disk_writes=self._disk_writes,
                duplicate_calcs=self._dup_calcs,
                positions_reviewed=self._positions_reviewed,
            )
            with self._lock:
                self._samples.append(sample)
                self._append_tick(sample)
            self._active_module = ""

    def _append_tick(self, sample: TickSample) -> None:
        row = {
            "ts": sample.timestamp,
            "scope": sample.scope,
            "module": sample.module,
            "cpu_ms": sample.cpu_ms,
            "memory_delta_kb": sample.memory_delta_kb,
            "bar_fetches": sample.bar_fetches,
            "db_reads": sample.db_reads,
            "db_writes": sample.db_writes,
            "disk_writes": sample.disk_writes,
            "duplicate_calcs": sample.duplicate_calcs,
            "positions_reviewed": sample.positions_reviewed,
        }
        with self._tick_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def aggregate_by_module(self) -> list[ModuleCostAgg]:
        buckets: dict[str, ModuleCostAgg] = {}
        with self._lock:
            samples = list(self._samples)
        for s in samples:
            b = buckets.setdefault(s.module, ModuleCostAgg(module=s.module))
            b.sample_count += 1
            b.total_cpu_ms += s.cpu_ms
            b.max_cpu_ms = max(b.max_cpu_ms, s.cpu_ms)
            b.total_bar_fetches += s.bar_fetches
            b.total_db_reads += s.db_reads
            b.total_db_writes += s.db_writes
            b.total_disk_writes += s.disk_writes
            b.total_duplicate_calcs += s.duplicate_calcs
            b.memory_peak_kb = max(b.memory_peak_kb, s.memory_delta_kb)
        for b in buckets.values():
            b.avg_cpu_ms = round(b.total_cpu_ms / b.sample_count, 3) if b.sample_count else 0.0
        return sorted(buckets.values(), key=lambda x: -x.total_cpu_ms)

    def write_module_cost_csv(self, path: Path) -> None:
        rows = self.aggregate_by_module()
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "module", "sample_count", "avg_cpu_ms", "max_cpu_ms", "total_cpu_ms",
            "total_bar_fetches", "total_db_reads", "total_db_writes",
            "total_disk_writes", "total_duplicate_calcs", "memory_peak_kb",
        ]
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow({
                    "module": r.module,
                    "sample_count": r.sample_count,
                    "avg_cpu_ms": r.avg_cpu_ms,
                    "max_cpu_ms": r.max_cpu_ms,
                    "total_cpu_ms": round(r.total_cpu_ms, 3),
                    "total_bar_fetches": r.total_bar_fetches,
                    "total_db_reads": r.total_db_reads,
                    "total_db_writes": r.total_db_writes,
                    "total_disk_writes": r.total_disk_writes,
                    "total_duplicate_calcs": r.total_duplicate_calcs,
                    "memory_peak_kb": r.memory_peak_kb,
                })

    def load_ticks_from_disk(self) -> None:
        if not self._tick_path.exists():
            return
        with self._lock:
            self._samples.clear()
        for line in self._tick_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                d = json.loads(line)
                self._samples.append(TickSample(
                    scope=d["scope"],
                    module=d["module"],
                    cpu_ms=float(d["cpu_ms"]),
                    memory_delta_kb=float(d.get("memory_delta_kb", 0)),
                    bar_fetches=int(d.get("bar_fetches", 0)),
                    db_reads=int(d.get("db_reads", 0)),
                    db_writes=int(d.get("db_writes", 0)),
                    disk_writes=int(d.get("disk_writes", 0)),
                    duplicate_calcs=int(d.get("duplicate_calcs", 0)),
                    positions_reviewed=int(d.get("positions_reviewed", 0)),
                    timestamp=float(d.get("ts", 0)),
                ))
            except (json.JSONDecodeError, KeyError, TypeError):
                continue

    @staticmethod
    def estimate_from_registry() -> list[dict]:
        """Fallback cost profile when no live ticks recorded."""
        from scout_auto_os.engine.runtime_audit.module_registry import MODULES
        rows = []
        for spec in MODULES.values():
            rows.append({
                "module": spec.module_id,
                "sample_count": 0,
                "avg_cpu_ms": spec.est_cpu_ms_per_tick,
                "max_cpu_ms": spec.est_cpu_ms_per_tick * 2,
                "total_cpu_ms": spec.est_cpu_ms_per_tick,
                "total_bar_fetches": int(spec.est_bar_fetches_per_tick),
                "total_db_reads": int(spec.est_db_ops_per_tick),
                "total_db_writes": 0,
                "total_disk_writes": 0,
                "total_duplicate_calcs": 1 if spec.duplicate_risk == "high" else 0,
                "memory_peak_kb": 0,
                "source": "registry_estimate",
            })
        return rows


def get_cost_tracker(out_dir: Path | None = None, enabled: bool = True) -> CostTracker:
    global _GLOBAL
    if _GLOBAL is None:
        base = out_dir or Path(os.environ.get("SCOUT_DATA_DIR", "scout_auto_os/data")) / "runtime_audit"
        _GLOBAL = CostTracker(base, enabled=enabled)
    return _GLOBAL


def reset_cost_tracker() -> None:
    global _GLOBAL
    _GLOBAL = None
