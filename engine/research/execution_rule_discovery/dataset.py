"""Collect Top5 execution dataset per scan."""

from __future__ import annotations

from scout_auto_os.engine.portfolio.engine import PortfolioEngine
from scout_auto_os.engine.research.execution_research.observation import execution_score
from scout_auto_os.engine.research.execution_research.selector import (
    eval_return_2h,
    top5_pass_candidates,
)
from scout_auto_os.engine.research.execution_rule_discovery.constants import TOP2_SIZE, TOP5_SIZE
from scout_auto_os.engine.research.execution_rule_discovery.features import (
    build_candidate_record,
    enrich_top5_ranks,
)


def collect_execution_groups(
    by_scan: dict,
    fwd: dict,
    scans: list[str],
    engine: PortfolioEngine,
) -> list[list[dict]]:
    """Each group = Top5 PASS candidates at one scan x direction."""
    groups: list[list[dict]] = []

    for scan_kst in scans:
        rows = [{"symbol": r["symbol"], "features": r["features"]} for r in by_scan.get(scan_kst, [])]
        long5, short5 = top5_pass_candidates(rows, scan_kst, engine)

        for direction, pool in (("long", long5), ("short", short5)):
            if len(pool) < TOP2_SIZE:
                continue
            pool = pool[:TOP5_SIZE]
            best_entry = max(float(c.get("entry_score", 0)) for c in pool)
            pre_records: list[dict] = []
            for i, c in enumerate(sorted(pool, key=lambda x: -float(x.get("entry_score", 0)))):
                klines = fwd.get((scan_kst, c["symbol"]))
                r2h = eval_return_2h(klines or [], direction)
                pre_records.append(
                    build_candidate_record(
                        c, direction, scan_kst, klines or [], r2h,
                        top5_rank=i + 1, top5_size=len(pool),
                        gap_to_best=best_entry - float(c.get("entry_score", 0)),
                        rank_obs=0.0, rank_entry=0.0,
                    ),
                )
            for r in pre_records:
                obs = r["features"]
                r["features"]["execution_score"] = execution_score(obs)
            enrich_top5_ranks(pre_records)
            groups.append(pre_records)

    return groups
