# Short Constitution Research V1

Independent short search constitution - not a sign flip of Long.

- Direction: **short** | Model: **catboost_ranker** | Samples: **6536**
- Best label: **risk_adjusted_short** avg 2h **3.5443%**
- Long constitution (frozen): avg 2h **5.2608%**
- Leak check: **FAIL**

## 1. Independent feature structure?

**PARTIAL** - top-15 overlap 66.67% (long-only: 5, short-only: 5).

## 2. Best short label

**Risk adjusted short** (`risk_adjusted_short`) blind avg 2h **3.5443%**

## 3. Reuse long features?

**Reuse with relabeling** - high overlap suggests shared cross-sectional signals.

## 4. Sufficient data signal?

**YES hypothesis** - short blind 3.5443% vs long 5.2608% on 15d window.

## 5. Completeness score

**50/100** - research completeness (not LIVE readiness).

## 6. Next steps

Shadow short ranker parallel to long; regime-gated activation only. | Re-run blind after infrastructure reaches 90d calendar.

## Label ranking (blind)

| Rank | Label | Avg 2h | Sharpe | NDCG5 | vs baseline |
|------|-------|--------|--------|-------|-------------|
| 1 | risk_adjusted_short | 3.5443 | 10.463 | 0.4397 | 99.05% |
| 2 | return_short_1h | 3.5225 | 10.7093 | 0.4715 | 97.83% |
| 3 | return_short_2h | 3.4756 | 9.025 | 0.4514 | 95.19% |
| 4 | return_plus_dd | 3.2367 | 8.7548 | 0.4659 | 81.78% |
| 5 | drawup_resilience | 3.2367 | 8.7548 | 0.4659 | 81.78% |
| 6 | distribution_success | 3.1739 | 8.7678 | 0.5251 | 78.25% |
| 7 | hit_short_3pct | 3.1228 | 8.1286 | 0.4631 | 75.38% |
| 8 | capitulation_fade | 2.9323 | 10.5777 | 0.5123 | 64.68% |
| 9 | return_per_risk_short | 2.8895 | 9.3269 | 0.465 | 62.28% |
| 10 | intrabar_sharpe_short | 2.7793 | 9.0037 | 0.4305 | 56.09% |
| 11 | return_short_4h | 2.6274 | 5.8351 | 0.4809 | 47.56% |
| 12 | mae_short_2h | 2.2513 | 9.7095 | 0.4715 | 26.43% |
| 13 | max_up_adverse_2h | 2.2513 | 9.7095 | 0.4715 | 26.43% |
| 14 | return_short_30m | 2.1523 | 5.9774 | 0.5026 | 20.87% |
| 15 | baseline_max_down_2h | 1.7806 | 3.3298 | 0.4786 | 0.0% |

## Short regime performance (best label)

| Regime | Scans | Avg 2h | Win% |
|--------|-------|--------|------|
| Mixed | 1 | 3.5092 | 40.0 |
| Mixed | 1 | 6.1233 | 60.0 |
| Mixed | 1 | 4.3123 | 60.0 |
| Mixed | 1 | 6.7579 | 100.0 |
| Mixed | 1 | 2.4703 | 60.0 |
| Mixed | 1 | 1.5615 | 40.0 |
| Mixed | 1 | 3.5097 | 40.0 |
| Mixed | 1 | 2.6425 | 40.0 |
| Mixed | 1 | 6.5842 | 60.0 |
| Mixed | 1 | 3.205 | 60.0 |
| Mixed | 1 | 0.8752 | 0.0 |
| Mixed | 1 | 8.1196 | 60.0 |

Probabilistic - 15d calendar; no price targets.

_Generated 2026-06-26T08:54:40.221990+09:00_