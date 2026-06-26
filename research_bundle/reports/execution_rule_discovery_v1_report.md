# Execution Rule Discovery V1

Data-driven execution rules — **execution layer only**.
Search, Entry Rule V2, Entry Score unchanged.

## Method

- Features: search-time (entry score, margins, top5 rank) + **first observation bar only**
- Rule mining: threshold, ratio, diff, top5 rank, AND/OR/NOT
- Train groups: 11 | Blind groups: 15
- Selection: blind avg 2h return vs Execution Score Top2 baseline

## Blind comparison

| Strategy | Direction | Trades | Avg 2h | Win% | Lift vs Exec Score | Lift vs Entry Top2 |
|----------|-----------|--------|--------|------|--------------------|--------------------|
| top2_entry_score | combined | 26 | 3.2529 | 53.85 |  |  |
| top2_execution_score | combined | 26 | 2.7002 | 50.0 |  |  |
| top2_discovered_rule | long | 12 | 14.3148 | 58.33 | 430.14 | 340.06 |

## Top discovered rules (blind)

- **long** `(direction_confidence >= 0.7824 OR rank_obs_return_top5 >= 0.125)` — blind avg=14.3148 lift=430.14%
- **long** `(direction_confidence >= 0.7824 OR rank_obs_return_top5 >= 0.25)` — blind avg=14.3148 lift=430.14%
- **long** `(direction_confidence >= 0.8912 OR rank_obs_return_top5 >= 0.125)` — blind avg=14.3148 lift=430.14%
- **long** `(direction_confidence >= 0.8912 OR rank_obs_return_top5 >= 0.25)` — blind avg=14.3148 lift=430.14%
- **long** `(obs_low_pct >= -2.914 OR rank_obs_return_top5 >= 0.125)` — blind avg=14.3148 lift=430.14%
- **long** `(obs_low_pct >= -2.914 OR rank_obs_return_top5 >= 0.25)` — blind avg=14.3148 lift=430.14%
- **long** `(execution_score >= -0.2779 OR rank_obs_return_top5 >= 0.125)` — blind avg=14.3148 lift=430.14%
- **long** `(execution_score >= -0.2779 OR rank_obs_return_top5 >= 0.25)` — blind avg=14.3148 lift=430.14%
- **long** `((obs_return_pct - gap_to_best_entry) >= -5.877 OR rank_obs_return_top` — blind avg=14.3148 lift=430.14%
- **long** `((obs_return_pct - gap_to_best_entry) >= -5.877 OR rank_obs_return_top` — blind avg=14.3148 lift=430.14%

## Recommendation

**Decision:** Needs further validation

- Rule beats Execution Score on blind (14.3148 vs 2.7002) but sample is small (n=12) — extend blind window before LIVE

Baseline Execution Score blind avg: **2.7002**
Best rule blind avg: **14.3148**

_Generated 2026-06-25T14:21:04.873874+09:00_
**Convergence tier:** core | relative_ranking_between_candidates
--- Scout Mission (convergence) ---
 Purpose: Situation Evaluation Engine + Early Trend Detection Scout
 Lifecycle: birth -> growth -> exhaustion | genuine vs fake | regime change
 Core gates: early detection | real vs fake | persistence 1h-24h | relative rank
 Research may diverge; operational output must converge
 Unknown preferred over false certainty