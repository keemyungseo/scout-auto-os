# Target Discovery Engine V1

Label research - same CatBoost ranker and snapshot features; training target only changes.

- Baseline label (max_up_4h): avg 2h **4.7902**
- Best label: **Return + drawdown (2h)** avg 2h **5.2608**
- Candidates tested: **32**

## 1. Best blind label

Best blind label: **Return + drawdown (2h)** (`return_minus_dd`) avg 2h **5.2608%**

## 2. Better than return-only labels?

**YES hypothesis** - non-return labels beat baseline on blind avg 2h.

## 3. Generalization

**Hypothesis pass** - best label beats baseline on blind holdout.

## 4. LIVE applicability

**Shadow only** - label swap requires retraining pipeline; not wired to LIVE.

## 5. Meaningful vs Ranking Engine?

**YES hypothesis** - lift 9.82% (z~10.8017) vs Ranking V1 baseline label.

## 6. Learnability

Highest baseline rank correlation: `baseline_max_up_4h`, `max_up_6h`, `max_up_12h`.

## Label ranking (blind avg 2h)

| Rank | Label | Category | Avg 2h | Top2 | NDCG5 | P@5 | Sharpe | vs baseline |
|------|-------|----------|--------|------|-------|-----|--------|-------------|
| 1 | return_minus_dd | risk_adjusted | 5.2608 | 8.5783 | 0.7414 | 0.4509 | 11.149 | 9.82% |
| 2 | drawdown_resilience | risk_adjusted | 5.2608 | 8.5783 | 0.7414 | 0.4509 | 11.149 | 9.82% |
| 3 | return_2h | return | 5.0314 | 8.2223 | 0.8261 | 0.5382 | 10.1092 | 5.04% |
| 4 | return_4h | return | 4.8957 | 8.0263 | 0.9234 | 0.5782 | 9.6724 | 2.2% |
| 5 | avg_return_multi | return | 4.8474 | 8.2508 | 0.8455 | 0.5564 | 9.5798 | 1.19% |
| 6 | hit_3pct | binary | 4.8109 | 7.2017 | 0.6687 | 0.4509 | 10.3422 | 0.43% |
| 7 | label_success_2h | binary | 4.8109 | 7.2017 | 0.6687 | 0.4509 | 10.3422 | 0.43% |
| 8 | breakout_success | binary | 4.8038 | 6.3854 | 0.6448 | 0.4582 | 10.5924 | 0.28% |
| 9 | baseline_max_up_4h | baseline | 4.7902 | 7.5366 | 0.9664 | 0.5927 | 9.1936 | 0.0% |
| 10 | peak_efficiency | efficiency | 4.7899 | 6.7669 | 0.6506 | 0.4145 | 10.7536 | -0.01% |
| 11 | return_1h | return | 4.6376 | 7.5966 | 0.5844 | 0.3673 | 9.9799 | -3.19% |
| 12 | return_6h | return | 4.612 | 8.0487 | 0.8853 | 0.5709 | 8.8588 | -3.72% |
| 13 | momentum_persist | persistence | 4.6074 | 7.5966 | 0.5764 | 0.3564 | 9.8869 | -3.82% |
| 14 | max_up_6h | mfe | 4.5729 | 7.6036 | 0.9405 | 0.5818 | 8.6094 | -4.54% |
| 15 | max_up_2h | mfe | 4.4558 | 7.5423 | 0.8336 | 0.5345 | 8.4968 | -6.98% |
| 16 | mfe_2h | mfe | 4.4558 | 7.5423 | 0.8336 | 0.5345 | 8.4968 | -6.98% |
| 17 | intrabar_sharpe | risk_adjusted | 4.3708 | 7.1886 | 0.6292 | 0.4109 | 9.649 | -8.76% |
| 18 | return_12h | return | 4.3675 | 6.2893 | 0.725 | 0.5091 | 8.3549 | -8.82% |
| 19 | max_return_multi | return | 4.272 | 8.326 | 0.8996 | 0.5564 | 8.0265 | -10.82% |
| 20 | max_up_12h | mfe | 4.211 | 7.8626 | 0.9032 | 0.5564 | 7.8117 | -12.09% |
| 21 | return_per_risk | risk_adjusted | 4.2009 | 5.3737 | 0.4348 | 0.3018 | 10.2405 | -12.3% |
| 22 | return_30m | return | 3.9613 | 6.0408 | 0.4926 | 0.3018 | 8.6771 | -17.3% |
| 23 | intrabar_sortino | risk_adjusted | 3.9469 | 5.8965 | 0.6192 | 0.4109 | 9.6306 | -17.6% |
| 24 | time_to_3pct_score | timing | 3.8897 | 5.3349 | 0.546 | 0.4 | 8.4329 | -18.8% |
| 25 | max_up_1h | mfe | 3.8021 | 6.8634 | 0.6604 | 0.4436 | 6.8439 | -20.63% |

Probabilistic - ~15 day calendar; labels from forward klines for training only.
Blind trading metrics always use realized 2h return.

_Generated 2026-06-26T08:21:08.776553+09:00_