"""Performance gate — KEEP / SHADOW / DISABLE / CRITICAL verdicts."""

from __future__ import annotations

from dataclasses import dataclass

from scout_auto_os.engine.runtime_audit.module_registry import MODULES, Verdict


@dataclass
class GateResult:
    module_id: str
    verdict: Verdict
    status: str
    roi_lift_pct: float
    latency_increase_pct: float
    mdd_improvement: float
    false_exit_delta: int
    duplicate_calc_count: int
    efficiency_roi_per_cpu: float
    reason: str


GATE_RULES = {
    "min_roi_lift_pct": 3.0,
    "max_latency_increase_pct": 20.0,
    "min_mdd_improvement": 0.0,
    "max_false_exit_increase": 0,
}


def _safe_div(a: float, b: float) -> float:
    return round(a / b, 6) if b else 0.0


def evaluate_module(
    module_id: str,
    perf: dict,
    cost: dict,
    baseline_perf: dict,
    baseline_cost: dict,
) -> GateResult:
    spec = MODULES.get(module_id)
    if spec and spec.critical:
        return GateResult(
            module_id=module_id,
            verdict="CRITICAL",
            status="enabled_live",
            roi_lift_pct=0.0,
            latency_increase_pct=0.0,
            mdd_improvement=0.0,
            false_exit_delta=0,
            duplicate_calc_count=int(cost.get("total_duplicate_calcs", 0)),
            efficiency_roi_per_cpu=0.0,
            reason="Safety-critical — performance-independent KEEP",
        )

    roi_lift = float(perf.get("roi_lift_pct", 0))
    avg_roi = float(perf.get("avg_roi", 0))
    base_roi = float(baseline_perf.get("avg_roi", 0))
    if roi_lift == 0 and base_roi:
        roi_lift = round((avg_roi - base_roi) / abs(base_roi) * 100, 2)

    cpu = float(cost.get("avg_cpu_ms", cost.get("est_cpu_ms", 0)))
    base_cpu = float(baseline_cost.get("avg_cpu_ms", baseline_cost.get("est_cpu_ms", 1)))
    latency_inc = round((cpu - base_cpu) / base_cpu * 100, 2) if base_cpu else 0.0

    mdd = float(perf.get("mdd", 0))
    base_mdd = float(baseline_perf.get("mdd", 0))
    mdd_improvement = round(base_mdd - mdd, 4)

    false_exit = int(perf.get("false_exit_count", 0))
    base_false = int(baseline_perf.get("false_exit_count", 0))
    false_delta = false_exit - base_false

    dup = int(cost.get("total_duplicate_calcs", 0))
    eff = _safe_div(roi_lift, cpu) if cpu else 0.0

    verdict: Verdict = "KEEP"
    status = "enabled_live"
    reasons: list[str] = []

    if roi_lift < GATE_RULES["min_roi_lift_pct"] and module_id not in ("a6_search", "exit_engine"):
        if spec and spec.default_mode == "LIVE_SHADOW":
            verdict = "SHADOW"
            status = "enabled_shadow"
            reasons.append(f"roi_lift {roi_lift}% < {GATE_RULES['min_roi_lift_pct']}%")
        elif spec and spec.default_mode == "RESEARCH":
            verdict = "DISABLE"
            status = "enabled_research_only"
            reasons.append("research-only module")
        else:
            verdict = "SHADOW"
            status = "enabled_shadow"
            reasons.append(f"roi_lift {roi_lift}% below gate")

    if latency_inc > GATE_RULES["max_latency_increase_pct"] and roi_lift < GATE_RULES["min_roi_lift_pct"]:
        verdict = "DISABLE"
        status = "disabled_candidate"
        reasons.append(f"latency +{latency_inc}% without roi lift")

    if dup >= 2 and roi_lift < GATE_RULES["min_roi_lift_pct"]:
        if verdict != "CRITICAL":
            verdict = "DISABLE"
            status = "disabled_candidate"
            reasons.append(f"duplicate_calcs={dup}")

    if mdd_improvement <= GATE_RULES["min_mdd_improvement"] and roi_lift < GATE_RULES["min_roi_lift_pct"]:
        if module_id in ("expectation", "expected_ev", "memory_logging", "review_layer"):
            verdict = "SHADOW"
            status = "enabled_shadow"
            reasons.append("no mdd improvement — shadow only")

    if false_delta > GATE_RULES["max_false_exit_increase"] and roi_lift < 5:
        verdict = "SHADOW"
        status = "enabled_shadow"
        reasons.append(f"false_exit increased by {false_delta}")

    if roi_lift >= GATE_RULES["min_roi_lift_pct"] and (mdd_improvement > 0 or avg_roi > base_roi):
        verdict = "KEEP"
        status = "enabled_live"
        reasons.append(f"roi_lift {roi_lift}% confirmed")

    if module_id == "a6_search":
        verdict = "KEEP"
        status = "enabled_live"
        reasons = ["Core search — frozen baseline"]

    if module_id == "ranking_engine" and roi_lift >= 0:
        verdict = "KEEP"
        status = "enabled_live"
        reasons.append("Ranking improves or preserves selection quality")

    return GateResult(
        module_id=module_id,
        verdict=verdict,
        status=status,
        roi_lift_pct=roi_lift,
        latency_increase_pct=latency_inc,
        mdd_improvement=mdd_improvement,
        false_exit_delta=false_delta,
        duplicate_calc_count=dup,
        efficiency_roi_per_cpu=eff,
        reason="; ".join(reasons) or "within gate thresholds",
    )


def evaluate_all(
    module_perf: dict[str, dict],
    module_cost: dict[str, dict],
    baseline_module: str = "a6_search",
) -> list[GateResult]:
    baseline_perf = module_perf.get(baseline_module, {})
    baseline_cost = module_cost.get(baseline_module, {"avg_cpu_ms": 120.0})
    results: list[GateResult] = []
    for mid in MODULES:
        perf = module_perf.get(mid, {})
        cost = module_cost.get(mid, {"avg_cpu_ms": MODULES[mid].est_cpu_ms_per_tick})
        results.append(evaluate_module(mid, perf, cost, baseline_perf, baseline_cost))
    return results
