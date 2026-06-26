"""Runtime module registry — identity, mode, and cost profile for audit."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Verdict = Literal["KEEP", "SHADOW", "DISABLE", "CRITICAL"]


@dataclass(frozen=True)
class ModuleSpec:
    module_id: str
    name: str
    layer: str
    description: str
    config_keys: tuple[str, ...] = ()
    critical: bool = False
    default_mode: str = "LIVE_CORE"
    est_cpu_ms_per_tick: float = 0.0
    est_bar_fetches_per_tick: float = 0.0
    est_db_ops_per_tick: float = 0.0
    duplicate_risk: str = "low"
    performance_measurable: bool = True


MODULES: dict[str, ModuleSpec] = {
    "a6_search": ModuleSpec(
        module_id="a6_search",
        name="A6 Frozen Search",
        layer="search",
        description="MarketWatcher A6_frozen scan → top5 candidates",
        config_keys=("long_engine.enabled", "long_engine.search_formula"),
        critical=False,
        default_mode="LIVE_CORE",
        est_cpu_ms_per_tick=120.0,
        est_bar_fetches_per_tick=0.0,
        est_db_ops_per_tick=2.0,
        duplicate_risk="low",
    ),
    "ranking_engine": ModuleSpec(
        module_id="ranking_engine",
        name="Ranking Engine",
        layer="search",
        description="Portfolio constitution ranking (Long/Short frozen weights)",
        config_keys=("portfolio_engine.enabled",),
        critical=False,
        default_mode="LIVE_CORE",
        est_cpu_ms_per_tick=80.0,
        est_bar_fetches_per_tick=0.0,
        est_db_ops_per_tick=0.0,
        duplicate_risk="low",
    ),
    "trade_thesis": ModuleSpec(
        module_id="trade_thesis",
        name="Trade Thesis",
        layer="position",
        description="Entry thesis + expected path seed on create_position",
        config_keys=("position_evaluation.enabled",),
        critical=False,
        default_mode="LIVE_CORE",
        est_cpu_ms_per_tick=2.0,
        est_bar_fetches_per_tick=0.0,
        est_db_ops_per_tick=1.0,
        duplicate_risk="low",
    ),
    "expectation": ModuleSpec(
        module_id="expectation",
        name="Expectation Engine",
        layer="position",
        description="Curve progress, expectation score, thesis state machine",
        config_keys=("expectation.enabled",),
        critical=False,
        default_mode="LIVE_SHADOW",
        est_cpu_ms_per_tick=8.0,
        est_bar_fetches_per_tick=0.0,
        est_db_ops_per_tick=0.0,
        duplicate_risk="medium",
    ),
    "position_evaluation": ModuleSpec(
        module_id="position_evaluation",
        name="Position Evaluation",
        layer="position",
        description="Validity, exit pressure, decision merge with state exit",
        config_keys=("position_evaluation.enabled", "position_evaluation.override_state_exit"),
        critical=False,
        default_mode="LIVE_CORE",
        est_cpu_ms_per_tick=12.0,
        est_bar_fetches_per_tick=0.0,
        est_db_ops_per_tick=0.0,
        duplicate_risk="medium",
    ),
    "exit_engine": ModuleSpec(
        module_id="exit_engine",
        name="Exit Engine",
        layer="position",
        description="StateExitEngine alive-based exit + protective SL",
        config_keys=("state_engine.review_interval_sec", "state_engine.protective_sl_pct"),
        critical=False,
        default_mode="LIVE_CORE",
        est_cpu_ms_per_tick=6.0,
        est_bar_fetches_per_tick=0.0,
        est_db_ops_per_tick=0.0,
        duplicate_risk="medium",
    ),
    "portfolio_slots": ModuleSpec(
        module_id="portfolio_slots",
        name="Portfolio Slot Manager",
        layer="portfolio",
        description="Long3/Short3 slot book, replacement margin",
        config_keys=("portfolio_engine.enabled", "position.max_long_slots", "position.max_short_slots"),
        critical=False,
        default_mode="LIVE_CORE",
        est_cpu_ms_per_tick=40.0,
        est_bar_fetches_per_tick=0.0,
        est_db_ops_per_tick=1.0,
        duplicate_risk="low",
    ),
    "manual_guard": ModuleSpec(
        module_id="manual_guard",
        name="Manual Guard",
        layer="portfolio",
        description="Manual position protection, entry block, locked symbols",
        config_keys=(),
        critical=True,
        default_mode="LIVE_CORE",
        est_cpu_ms_per_tick=1.0,
        est_bar_fetches_per_tick=0.0,
        est_db_ops_per_tick=1.0,
        duplicate_risk="low",
        performance_measurable=False,
    ),
    "memory_logging": ModuleSpec(
        module_id="memory_logging",
        name="Memory / Logging",
        layer="audit",
        description="Review CSV, thesis jsonl, expectation logs, engine_events",
        config_keys=("runtime_audit.enabled",),
        critical=False,
        default_mode="LIVE_SHADOW",
        est_cpu_ms_per_tick=5.0,
        est_bar_fetches_per_tick=0.0,
        est_db_ops_per_tick=0.0,
        duplicate_risk="high",
    ),
    "expected_ev": ModuleSpec(
        module_id="expected_ev",
        name="Expected EV (R010)",
        layer="position",
        description="compute_live_ev during update_prices — display/entry filter",
        config_keys=("entry_quality.block_negative_ev",),
        critical=False,
        default_mode="LIVE_SHADOW",
        est_cpu_ms_per_tick=15.0,
        est_bar_fetches_per_tick=0.0,
        est_db_ops_per_tick=0.0,
        duplicate_risk="high",
    ),
    "review_layer": ModuleSpec(
        module_id="review_layer",
        name="Review Layer V1.2",
        layer="audit",
        description="Missed winners, scan audit, learning snapshot",
        config_keys=(),
        critical=False,
        default_mode="LIVE_SHADOW",
        est_cpu_ms_per_tick=10.0,
        est_bar_fetches_per_tick=0.0,
        est_db_ops_per_tick=2.0,
        duplicate_risk="low",
        performance_measurable=False,
    ),
    "research_engine": ModuleSpec(
        module_id="research_engine",
        name="Research Engine",
        layer="research",
        description="Background formula/feature/state leagues",
        config_keys=("research.enabled",),
        critical=False,
        default_mode="RESEARCH",
        est_cpu_ms_per_tick=500.0,
        est_bar_fetches_per_tick=50.0,
        est_db_ops_per_tick=10.0,
        duplicate_risk="low",
        performance_measurable=False,
    ),
}


ABLATION_SCENARIOS: dict[str, dict] = {
    "baseline": {
        "label": "Baseline: A6 frozen + slot2",
        "modules": ("a6_search", "portfolio_slots"),
        "slots_long": 2,
        "slots_short": 0,
        "exit_mode": "hold_2h",
    },
    "A": {
        "label": "A6 only",
        "modules": ("a6_search",),
        "slots_long": 1,
        "slots_short": 0,
        "exit_mode": "hold_2h",
    },
    "B": {
        "label": "A6 + Ranking",
        "modules": ("a6_search", "ranking_engine"),
        "slots_long": 2,
        "slots_short": 0,
        "exit_mode": "hold_2h",
    },
    "C": {
        "label": "A6 + Ranking + Thesis",
        "modules": ("a6_search", "ranking_engine", "trade_thesis"),
        "slots_long": 2,
        "slots_short": 0,
        "exit_mode": "hold_2h",
    },
    "D": {
        "label": "A6 + Ranking + Thesis + Expectation",
        "modules": ("a6_search", "ranking_engine", "trade_thesis", "expectation"),
        "slots_long": 2,
        "slots_short": 0,
        "exit_mode": "expectation_proxy",
    },
    "E": {
        "label": "A6 + Ranking + Thesis + Expectation + PE",
        "modules": (
            "a6_search", "ranking_engine", "trade_thesis", "expectation", "position_evaluation",
        ),
        "slots_long": 2,
        "slots_short": 0,
        "exit_mode": "pe_proxy",
    },
    "F": {
        "label": "A6 + Ranking + Thesis + Expectation + Exit",
        "modules": ("a6_search", "ranking_engine", "trade_thesis", "expectation", "exit_engine"),
        "slots_long": 2,
        "slots_short": 0,
        "exit_mode": "state_exit",
    },
    "G": {
        "label": "Full Core",
        "modules": (
            "a6_search", "ranking_engine", "trade_thesis", "expectation",
            "position_evaluation", "exit_engine", "portfolio_slots", "manual_guard",
        ),
        "slots_long": 3,
        "slots_short": 3,
        "exit_mode": "full_exit",
    },
}


def module_ids() -> list[str]:
    return list(MODULES.keys())


def scenario_ids() -> list[str]:
    return list(ABLATION_SCENARIOS.keys())
