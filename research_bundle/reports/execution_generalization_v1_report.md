# Execution Rule Generalization Test V1

**Frozen rule — no tuning, no new discovery.**

- Rule: `(direction_confidence >= 0.7824 OR rank_obs_return_top5 >= 0.125)`
- Direction: **long**
- Scans: 180 | Groups: 12

## Overall (all periods)

| Metric | Discovered Rule | Execution Score |
|--------|-----------------|-----------------|
| Avg 2h | 5.8078 | 3.9396 |
| Win% | 47.62 | 42.86 |
| Trades | 21 | 21 |
| MDD | -74.4342 | -85.1916 |
| Sharpe | 1.1449 | 0.7404 |
| Return/day | 17.4234 | 11.8187 |

## Split stability

- Folds beating Execution Score: **15/15** (1.0)
- Monthly stability (std): 0.0

### Fold summary

| Split | Fold | Rule avg | Base avg | Rule win% | Beats base | Trades |
|-------|------|----------|----------|-----------|------------|--------|
| monthly | 2026-06 | 5.8078 | 3.9396 | 47.62 | True | 21 |
| weekly | 2026-W22 | -4.467 | -6.0037 | 42.86 | True | 7 |
| weekly | 2026-W23 | 15.6903 | 12.8428 | 60.0 | True | 10 |
| weekly | 2026-W24 | -0.9178 | -0.9178 | 25.0 | True | 4 |
| walk_forward | wf_2026-06-01 | -4.467 | -6.0037 | 42.86 | True | 7 |
| walk_forward | wf_2026-06-08 | 15.6903 | 12.8428 | 60.0 | True | 10 |
| walk_forward | wf_2026-06-15 | -0.9178 | -0.9178 | 25.0 | True | 4 |
| expanding | exp_1 | -2.1862 | -2.1862 | 50.0 | True | 4 |
| expanding | exp_2 | -4.467 | -6.0037 | 42.86 | True | 7 |
| expanding | exp_3 | 7.3903 | 5.0825 | 52.94 | True | 17 |
| expanding | exp_4 | 5.8078 | 3.9396 | 47.62 | True | 21 |
| leave_one_out | loo_2026-W22 | 10.9452 | 8.9112 | 50.0 | True | 14 |
| leave_one_out | loo_2026-W23 | -3.1764 | -4.1543 | 36.36 | True | 11 |
| leave_one_out | loo_2026-W24 | 7.3903 | 5.0825 | 52.94 | True | 17 |
| temporal_blind | blind_holdout | 14.3148 | 11.9418 | 58.33 | True | 12 |

## Decision

**REJECT**

Rule beats baseline in aggregate but fails fold/regime robustness gates

Rejection / caution reasons:
- negative in 2 regimes: ['bear', 'sideway']

Probabilistic — small calendar coverage limits regime conclusions.

_Generated 2026-06-25T14:28:19.236335+09:00_