# Scan Cadence Optimizer V1

LIVE scan interval comparison — **no new features, rules, or prediction models**.

## Data constraint

- Base universe snapshots: **every 120m** (180 scans)
- Sub-120m cadences replay portfolio on synthetic ticks using **latest available snapshot** (causal, no lookahead)
- Forward returns: **evaluation only**

## Interval performance

| Interval | Scans | Trades | Pass/day | Long avg | Short avg | Combined | Win% | MDD | Replace | Hold(min) | Occupancy | Turnover | Ret/trade | Ret/day | Ret/turnover |
|----------|-------|--------|----------|----------|-----------|----------|------|-----|---------|-----------|-----------|----------|-----------|---------|--------------|
| 1m | 21600 | 14 | 1256.0 | 7.5135 | 9.1755 | 8.3445 | 64.29 | -12.1142 | 16 | 1.0 | 0.9889 | 0.001 | 8.3445 | 7.7882 | 5.3101 |
| 3m | 7200 | 14 | 418.6667 | 7.5135 | 9.1755 | 8.3445 | 64.29 | -12.1142 | 16 | 3.0 | 0.9889 | 0.0031 | 8.3445 | 7.7882 | 5.3101 |
| 5m | 4320 | 14 | 251.2 | 7.5135 | 9.1755 | 8.3445 | 64.29 | -12.1142 | 16 | 5.0 | 0.9889 | 0.0051 | 8.3445 | 7.7882 | 5.3101 |
| 10m | 2160 | 14 | 125.6 | 7.5135 | 9.1755 | 8.3445 | 64.29 | -12.1142 | 16 | 10.0 | 0.9889 | 0.0102 | 8.3445 | 7.7882 | 5.3101 |
| 15m | 1440 | 14 | 83.7333 | 9.4762 | 17.1414 | 13.3088 | 71.43 | -13.4261 | 16 | 15.0 | 0.9889 | 0.0153 | 13.3088 | 12.4216 | 8.4693 |
| 30m | 720 | 14 | 41.8667 | 13.437 | 15.4837 | 14.4604 | 78.57 | -12.1219 | 16 | 30.0 | 0.9889 | 0.0306 | 14.4604 | 13.4963 | 9.202 |
| 60m | 360 | 14 | 20.9333 | 19.2459 | 15.8844 | 17.5651 | 78.57 | -13.3652 | 16 | 60.0 | 0.9889 | 0.0611 | 17.5651 | 16.3941 | 11.1778 |
| 120m | 180 | 14 | 10.4667 | 20.7527 | 11.8809 | 16.3168 | 78.57 | -32.1855 | 16 | 120.0 | 0.9889 | 0.1222 | 16.3168 | 15.229 | 10.3834 |

## Rankings

1. **Highest return/day:** 60m
2. **Most stable (return/turnover, low churn):** 60m
3. **Best turnover efficiency:** 60m

## Noise analysis

- **1m:** High churn (turnover/scan=0.00), reselect=1; avg return 8.3445% — likely noise-dominated
- **3m:** High churn (turnover/scan=0.00), reselect=1; avg return 8.3445% — likely noise-dominated
- **5m:** High churn (turnover/scan=0.01), reselect=1; avg return 8.3445% — likely noise-dominated
- **10m:** Moderate turnover=0.01; balance of refresh vs stability
- **15m:** Moderate turnover=0.02; balance of refresh vs stability
- **30m:** turnover/scan=0.03, avg return=14.4604%
- **60m:** Low turnover=0.06; fewer opportunities, longer holds
- **120m:** Low turnover=0.12; fewer opportunities, longer holds

## LIVE recommendation

```
primary_scan_interval = 60 minutes
candidate_refresh_interval = 1 minutes
portfolio_rebalance_interval = 60 minutes
```

**Reason:** Interval 60m: return/day=16.3941 return/turnover=11.1778 turnover/scan=0.0611

## Intervals to deprioritize

- **1m:** return/turnover=5.3101 — High churn (turnover/scan=0.00), reselect=1; avg return 8.3445% — likely noise-dominated
- **3m:** return/turnover=5.3101 — High churn (turnover/scan=0.00), reselect=1; avg return 8.3445% — likely noise-dominated

Probabilistic — validate on new regimes before LIVE cadence change.

_Generated 2026-06-25T14:10:17.927051+09:00_
**Convergence tier:** core | trend_persistence_estimation, relative_ranking_between_candidates
--- Scout Mission (convergence) ---
 Purpose: Situation Evaluation Engine + Early Trend Detection Scout
 Lifecycle: birth -> growth -> exhaustion | genuine vs fake | regime change
 Core gates: early detection | real vs fake | persistence 1h-24h | relative rank
 Research may diverge; operational output must converge
 Unknown preferred over false certainty