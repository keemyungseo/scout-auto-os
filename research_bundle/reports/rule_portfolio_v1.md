# Rule Portfolio Engine V1

Frozen: Direction Champion, Entry Rule V2, Entry Score, Execution Score baseline.

- Rules in library: **563** (incl. universal baseline)
- Long mined rules: **277**
- Scans: 180 | Execution groups: 26

## Can specialized rules outperform a universal execution rule?

**Yes (hypothesis)** — regime-routed specialized rules beat universal Execution Score on this window

| Strategy | Avg 2h | Trades |
|----------|--------|--------|
| Regime router (specialized) | 2.8559 | 45 |
| Universal Execution Score | 1.6756 | 45 |
| Lift | 70.44% | routes=2 |

## Universal baseline

- Execution Score long avg: **1.6756%** (trades=45, precision=100.0%)

## Top specialized rules (long)

- `EX_OR_E_direction_confidence_gte_0.7824_E_rank_obs_return_to` — avg=5.8078%, best=bull, worst=sideway, tags=candidate|momentum_hybrid|recommended|rejected
- `EX_OR_E_direction_confidence_gte_0.7824_E_rank_obs_return_to` — avg=5.8078%, best=bull, worst=sideway, tags=candidate|momentum_hybrid
- `EX_OR_E_direction_confidence_gte_0.8912_E_rank_obs_return_to` — avg=5.8078%, best=bull, worst=sideway, tags=candidate|momentum_hybrid

## Clusters

| Cluster | Rules | Mean avg 2h | Top rule | Regime affinity |
|---------|-------|-------------|----------|-----------------|
| bull_trend | 210 | 3.7515 | EX_OR_E_direction_confidence_gte_0.7824_ | bull |
| execution_aligned | 8 | 3.5048 | EX_4 | bull |
| high_atr | 1 | 4.3405 | EX_24 | bull |
| low_atr | 1 | 4.3405 | EX_23 | bull |
| mixed | 14 | 1.959 | EX_NOT_E_obs_body_pct_gte_9.53875 | sideway |
| reversal | 8 | 4.2473 | EX_3 | bull |
| sideways | 5 | 0.2731 | EX_25 | sideway |
| strong_breakout | 316 | 1.357 | EX_OR_E_obs_low_pct_gte_-2.914012_E_rank | sideway |

## Portfolio design

At runtime the engine should:
1. Classify scan regime + volatility band (scan-time only)
2. Select rule from `rule_metadata.csv` matching preferred regime
3. Fall back to `execution_score_v1` when sample_size < threshold or avoid_regime matches

## Caveats

- Small calendar window — cluster and regime labels are **hypothesis-level**
- Generalization test **REJECT**ed the top discovered rule on bear/sideway negatives
- Regime router uses in-sample best-rule-per-regime (upper-bound estimate, not blind)

_Generated 2026-06-25T14:49:26.974726+09:00_