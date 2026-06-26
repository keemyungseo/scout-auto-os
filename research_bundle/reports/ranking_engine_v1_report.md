# Ranking Engine V1 — Search AI

Rule-free ranking from full feature matrix. Random seed fixed.

- Samples: **6536** | Features: **170**
- Blind scans: **55** | Models trained: **7**
- Best model: **ridge_return**

## Decision

### 1. Ranking Engine vs Formula League V2 (blind)
**Hypothesis YES** — Ranking vs Formula V2 avg 2h 4.8367% vs 4.6707% (NDCG 0.6413 vs 0.5844).

### 2. Ranking Engine vs Execution Engine proxy (blind)
**Hypothesis YES** — Ranking vs Execution proxy avg 2h 4.8367% vs 2.4294% (NDCG 0.6413 vs 0.4879).

### 3. Can Search Formula be fully replaced?
**Hypothesis partial** — ranking beats A6 on blind avg 2h; requires extended walk-forward before config change.

### 4. Top ranking features

- `ctx_rank_5m_release` — combined 0.256231
- `ctx_rank_5m_compression` — combined 0.040069
- `direction_confidence` — combined 0.036598
- `entry_score` — combined 0.028418
- `dna_2h_previous_volume_ratio` — combined 0.026868
- `a6_formula_score` — combined 0.016514
- `cluster_top_score` — combined 0.014345
- `dna_15m_current_range_pct` — combined 0.013924
- `meta_derived_atr_expansion` — combined 0.011044
- `dna_5m_release` — combined 0.010314

### 5. LIVE applicability
**HYPOTHESIS** — candidate for shadow mode only.

### 6. Blockers if not LIVE
Short blind window (~15 days); observation features need causal LIVE pipeline; no multi-regime validation; model not registered in live scanner.

## Blind comparison (avg 2h / NDCG@5 / P@5)

| Strategy | Avg 2h | Win% | NDCG5 | P@5 | Sharpe |
|----------|--------|------|-------|-----|--------|
| current_search_a6 | 3.0968 | 43.01 | 0.6414 | 0.4182 | 5.2426 |
| entry_score_top5 | 3.9123 | 49.26 | 0.6134 | 0.3891 | 7.083 |
| formula_league_v2 | 4.6707 | 50.37 | 0.5844 | 0.3673 | 10.0815 |
| execution_score_proxy | 2.4294 | 34.56 | 0.4879 | 0.3382 | 4.3914 |
| lightgbm_ranker | 4.577 | 52.21 | 0.9614 | 0.5964 | 8.5121 |
| xgboost_ranker | 4.5025 | 53.31 | 0.9531 | 0.5818 | 8.4831 |
| catboost_ranker | 4.7902 | 52.94 | 0.9664 | 0.5927 | 9.1936 |
| random_forest | 3.9646 | 45.22 | 0.5246 | 0.3382 | 8.2187 |
| extra_trees | 4.5107 | 47.43 | 0.6061 | 0.3855 | 9.3247 |
| logistic_top3 | 4.2709 | 51.84 | 0.8782 | 0.5745 | 7.9894 |
| ridge_return | 4.8367 | 52.94 | 0.6413 | 0.4 | 10.5306 |

## Model comparison (blind)

Best ranking model avg 2h: **4.8367** vs A6 **3.0968**

Probabilistic — short calendar window; no price targets.

_Generated 2026-06-25T16:06:05.200328+09:00_