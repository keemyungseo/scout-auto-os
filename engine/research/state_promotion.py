"""State Formula promotion gates — Research only, never auto-applies to LIVE."""

from __future__ import annotations

MIN_SAMPLES_HYPOTHESIS = 30
MIN_SAMPLES_VERIFICATION = 100
MIN_WIN_RATE_CANDIDATE = 45.0
TOP_RANK_CANDIDATE = 3


def promotion_tier(row: dict, rank: int, blind_pass: bool = False) -> str:
    """
    hypothesis → verification_needed → state_candidate (requires user approval for LIVE).
    blind_pass must be True for state_candidate even when stats look good.
    """
    n = int(row.get("sample_count") or 0)
    if n < MIN_SAMPLES_HYPOTHESIS:
        return "hypothesis"
    if n < MIN_SAMPLES_VERIFICATION:
        return "verification_needed" if rank <= 10 else "hypothesis"
    win = float(row.get("win_rate") or 0)
    if (
        rank <= TOP_RANK_CANDIDATE
        and win >= MIN_WIN_RATE_CANDIDATE
        and blind_pass
    ):
        return "state_candidate"
    if rank <= 10:
        return "verification_needed"
    return "background"


def live_apply_allowed(tier: str, user_approved: bool = False) -> bool:
    """LIVE State Engine may only change weights when explicitly approved."""
    return tier == "state_candidate" and user_approved
