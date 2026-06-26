"""Runtime mode plan — LIVE_CORE / LIVE_SHADOW / RESEARCH."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from scout_auto_os.engine.runtime_audit.module_registry import MODULES, ModuleSpec


class RuntimeMode(str, Enum):
    LIVE_CORE = "LIVE_CORE"
    LIVE_SHADOW = "LIVE_SHADOW"
    RESEARCH = "RESEARCH"


class ModuleStatus(str, Enum):
    ENABLED_LIVE = "enabled_live"
    ENABLED_SHADOW = "enabled_shadow"
    ENABLED_RESEARCH_ONLY = "enabled_research_only"
    DISABLED_CANDIDATE = "disabled_candidate"


MODE_MODULES: dict[RuntimeMode, tuple[str, ...]] = {
    RuntimeMode.LIVE_CORE: (
        "a6_search",
        "ranking_engine",
        "trade_thesis",
        "position_evaluation",
        "exit_engine",
        "portfolio_slots",
        "manual_guard",
    ),
    RuntimeMode.LIVE_SHADOW: (
        "expectation",
        "memory_logging",
        "expected_ev",
        "review_layer",
    ),
    RuntimeMode.RESEARCH: (
        "research_engine",
    ),
}


@dataclass
class ModuleModeRow:
    module_id: str
    name: str
    layer: str
    runtime_mode: str
    status: str
    critical: bool
    config_keys: str
    notes: str = ""


def _config_enabled(config: dict, key: str) -> bool | None:
    parts = key.split(".")
    cur: Any = config
    for p in parts:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    if isinstance(cur, bool):
        return cur
    return bool(cur)


def resolve_module_status(spec: ModuleSpec, config: dict) -> ModuleStatus:
    if spec.critical:
        return ModuleStatus.ENABLED_LIVE

    for mode, ids in MODE_MODULES.items():
        if spec.module_id in ids:
            if mode == RuntimeMode.LIVE_CORE:
                for ck in spec.config_keys:
                    val = _config_enabled(config, ck)
                    if val is False:
                        return ModuleStatus.DISABLED_CANDIDATE
                return ModuleStatus.ENABLED_LIVE
            if mode == RuntimeMode.LIVE_SHADOW:
                for ck in spec.config_keys:
                    val = _config_enabled(config, ck)
                    if val is False:
                        return ModuleStatus.DISABLED_CANDIDATE
                return ModuleStatus.ENABLED_SHADOW
            return ModuleStatus.ENABLED_RESEARCH_ONLY

    if spec.default_mode == "RESEARCH":
        return ModuleStatus.ENABLED_RESEARCH_ONLY
    if spec.default_mode == "LIVE_SHADOW":
        return ModuleStatus.ENABLED_SHADOW
    return ModuleStatus.ENABLED_LIVE


def resolve_runtime_mode(spec: ModuleSpec, status: ModuleStatus) -> RuntimeMode:
    if spec.critical or status == ModuleStatus.ENABLED_LIVE:
        return RuntimeMode.LIVE_CORE
    if status == ModuleStatus.ENABLED_SHADOW:
        return RuntimeMode.LIVE_SHADOW
    if status == ModuleStatus.ENABLED_RESEARCH_ONLY:
        return RuntimeMode.RESEARCH
    return RuntimeMode.LIVE_SHADOW


def build_mode_plan(config: dict) -> list[ModuleModeRow]:
    rows: list[ModuleModeRow] = []
    for spec in MODULES.values():
        status = resolve_module_status(spec, config)
        mode = resolve_runtime_mode(spec, status)
        notes = ""
        if spec.module_id == "research_engine":
            notes = "Must not run during LIVE trading loop"
        if spec.module_id == "manual_guard":
            notes = "CRITICAL — performance-independent safety"
        rows.append(ModuleModeRow(
            module_id=spec.module_id,
            name=spec.name,
            layer=spec.layer,
            runtime_mode=mode.value,
            status=status.value,
            critical=spec.critical,
            config_keys=",".join(spec.config_keys),
            notes=notes,
        ))
    return rows


def live_core_modules(config: dict | None = None) -> list[str]:
    if config is None:
        return list(MODE_MODULES[RuntimeMode.LIVE_CORE])
    plan = build_mode_plan(config)
    return [r.module_id for r in plan if r.runtime_mode == RuntimeMode.LIVE_CORE.value]


def research_allowed(config: dict) -> bool:
    return bool(config.get("research", {}).get("enabled", False))
