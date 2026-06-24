"""
Scout Mission — shared convergence criteria for Season2 research.

Research may diverge. Operational output must converge.
Every discovery is evaluated against four core questions before promotion to Scout logic.
"""

from __future__ import annotations

# What the Scout ultimately exists to understand
LIFECYCLE_TARGETS = (
    "market_situations",
    "trend_birth",
    "trend_growth",
    "trend_exhaustion",
    "fake_trends",
    "genuine_trends",
    "regime_changes",
    "behaviour_lifecycle",
)

# Four convergence gates — must improve at least one to become core Scout logic
CONVERGENCE_CRITERIA = (
    "early_trend_detection",
    "real_vs_fake_trend_discrimination",
    "trend_persistence_estimation",  # 1h / 2h / 6h / 12h / 24h
    "relative_ranking_between_candidates",
)

PERSISTENCE_HORIZONS = ("1h", "2h", "6h", "12h", "24h")

RESEARCH_TIERS = (
    "core",       # passed convergence gate — eligible for operational engine
    "background", # valid research, not yet tied to convergence
    "hypothesis", # exploratory, insufficient evidence
)

# Map common research outputs to which convergence criteria they may support
RESEARCH_TO_CONVERGENCE: dict[str, tuple[str, ...]] = {
    "situation_archetype": ("early_trend_detection", "real_vs_fake_trend_discrimination"),
    "participant_state": ("early_trend_detection", "real_vs_fake_trend_discrimination", "trend_persistence_estimation"),
    "behaviour_grammar": ("real_vs_fake_trend_discrimination", "trend_persistence_estimation"),
    "supply_label": ("trend_persistence_estimation", "relative_ranking_between_candidates"),
    "market_memory": ("trend_persistence_estimation", "early_trend_detection"),
    "situation_evolution": ("trend_persistence_estimation", "early_trend_detection"),
    "situation_health": ("real_vs_fake_trend_discrimination", "trend_persistence_estimation"),
    "situation_pressure": ("real_vs_fake_trend_discrimination", "trend_persistence_estimation"),
    "functional_role": ("early_trend_detection",),
    "ecology_regime": ("real_vs_fake_trend_discrimination", "trend_persistence_estimation"),
    "interaction_mining": ("relative_ranking_between_candidates", "early_trend_detection"),
}


def evaluate_convergence(
    research_kind: str,
    improves: list[str] | None = None,
    sample_size: int = 0,
    confidence: str = "hypothesis",
) -> dict:
    """
    Classify whether a finding belongs in core Scout logic or background research.

    Returns tier, matched criteria, and promotion note.
  Never forces promotion — Unknown / background is always valid.
    """
    improves = improves or []
    valid_improves = [c for c in improves if c in CONVERGENCE_CRITERIA]

    if sample_size < 6 or confidence == "unknown":
        return {
            "research_kind": research_kind,
            "tier": "hypothesis",
            "convergence_criteria_met": [],
            "lifecycle_relevance": list(RESEARCH_TO_CONVERGENCE.get(research_kind, ())),
            "promotion_note": "insufficient_history_or_confidence",
            "operational": False,
        }

    if valid_improves:
        return {
            "research_kind": research_kind,
            "tier": "core" if confidence in ("high", "medium") and sample_size >= 12 else "hypothesis",
            "convergence_criteria_met": valid_improves,
            "lifecycle_relevance": list(RESEARCH_TO_CONVERGENCE.get(research_kind, ())),
            "promotion_note": "convergence_gate_passed" if confidence == "high" else "conditional_promotion",
            "operational": confidence in ("high", "medium") and len(valid_improves) >= 1,
        }

    potential = RESEARCH_TO_CONVERGENCE.get(research_kind, ())
    return {
        "research_kind": research_kind,
        "tier": "background",
        "convergence_criteria_met": [],
        "lifecycle_relevance": list(potential),
        "promotion_note": "valid_research_not_yet_linked_to_convergence — keep as background",
        "operational": False,
    }


def mission_summary_lines() -> list[str]:
    """Standard footer for research reports."""
    return [
        "--- Scout Mission (convergence) ---",
        " Purpose: Situation Evaluation Engine + Early Trend Detection Scout",
        " Lifecycle: birth -> growth -> exhaustion | genuine vs fake | regime change",
        " Core gates: early detection | real vs fake | persistence 1h-24h | relative rank",
        " Research may diverge; operational output must converge",
        " Unknown preferred over false certainty",
    ]
