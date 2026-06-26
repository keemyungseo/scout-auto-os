"""Collect Top5 pass candidates and apply execution ranking."""

from __future__ import annotations

from scout_auto_os.engine.portfolio.diversification import diversify_select
from scout_auto_os.engine.portfolio.engine import PortfolioEngine
from scout_auto_os.engine.portfolio.scoring import build_pass_candidates
from scout_auto_os.engine.research.directional.evaluation import to_long_metrics, to_short_metrics
from scout_auto_os.engine.research.execution_research.constants import TOP2_SIZE, TOP5_SIZE
from scout_auto_os.engine.research.execution_research.observation import compute_observation_features, execution_score
from scout_auto_os.engine.research.zero_base.forward_eval import compute_forward_metrics


def top5_pass_candidates(
    rows: list[dict],
    scan_kst: str,
    rules_engine: PortfolioEngine,
) -> tuple[list[dict], list[dict]]:
    long_c, short_c = build_pass_candidates(rows, scan_kst, rules_engine.rules, scan_kst)
    long_c.sort(key=lambda x: x["entry_score"], reverse=True)
    short_c.sort(key=lambda x: x["entry_score"], reverse=True)
    return long_c[:TOP5_SIZE], short_c[:TOP5_SIZE]


def rank_by_execution(
    candidates: list[dict],
    direction: str,
    fwd: dict,
    scan_kst: str,
    weights: dict[str, float] | None = None,
) -> list[dict]:
    scored: list[dict] = []
    for c in candidates:
        klines = fwd.get((scan_kst, c["symbol"]))
        obs = compute_observation_features(klines or [], direction, c.get("features") or {})
        if not obs:
            continue
        ex_score = execution_score(obs, weights)
        scored.append({**c, **obs, "execution_score": ex_score})
    scored.sort(key=lambda x: x["execution_score"], reverse=True)
    return scored


def pick_top2_entry_score(candidates: list[dict]) -> list[dict]:
    return diversify_select(sorted(candidates, key=lambda x: x["entry_score"], reverse=True), TOP2_SIZE)


def pick_top2_execution(scored: list[dict]) -> list[dict]:
    return diversify_select(scored, TOP2_SIZE)


def eval_return_2h(klines: list, direction: str) -> float:
    raw = compute_forward_metrics(klines)
    if not raw:
        return 0.0
    if direction == "short":
        m = to_short_metrics(raw)
        return float(m.get("short_return_2h", -float(m.get("return_2h", 0))))
    return float(raw.get("return_2h", 0))
