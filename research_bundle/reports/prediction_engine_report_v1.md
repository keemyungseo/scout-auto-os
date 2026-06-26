# Prediction Engine Report (V1)

## Blind comparison (same top-k, same scans)

| Method | Direction | Samples | avg2h | win% | trap% | score |
|--------|-----------|---------|-------|------|-------|-------|
| RANDOM | long | 272 | 2.0219 | 30.51 | 4.04 | 12.1 |
| RANDOM | short | 272 | -1.0369 | 11.76 | 43.01 | -8.28 |
| ZERO_BASE_CHAMPION | long | 272 | 0.2757 | 16.54 | 2.21 | 6.27 |
| ZERO_BASE_CHAMPION | short | 272 | -4.6707 | 5.51 | 77.57 | -24.45 |
| DIRECTION_CHAMPION | long | 272 | 4.6662 | 50.37 | 4.78 | 22.68 |
| DIRECTION_CHAMPION | short | 272 | 3.0669 | 39.34 | 13.97 | 15.09 |
| PREDICTION_ENGINE | long | 272 | -0.6825 | 9.19 | 0.37 | 3.14 |
| PREDICTION_ENGINE | short | 272 | 0.792 | 18.01 | 26.1 | 1.16 |
| CLUSTER_CHAMPION | long | 272 | 4.2685 | 49.63 | 6.99 | 22.06 |
| CLUSTER_CHAMPION | short | 272 | -0.4856 | 20.96 | 44.12 | -4.57 |

## 3+3 slot simulation (prediction engine)
- Long slots: 3 | picks=164 | avg2h=-0.8506%
- Short slots: 3 | picks=164 | avg2h=1.5034%
- Combined avg2h: **0.3264%** (n=328)

## Baselines compared
- Zero-Base Champion: A6 (long) / MOMENTUM proxy (short)
- Direction Champion: LONG_CONTINUATION / SHORT_CONTINUATION
- Cluster Champion: best blind cluster formula per direction
- Prediction Engine: rank by long_score / short_score

_Generated 2026-06-25T11:53:15.521714+09:00_

**Convergence tier:** core | criteria: relative_ranking_between_candidates, trend_persistence_estimation
--- Scout Mission (convergence) ---
 Purpose: Situation Evaluation Engine + Early Trend Detection Scout
 Lifecycle: birth -> growth -> exhaustion | genuine vs fake | regime change
 Core gates: early detection | real vs fake | persistence 1h-24h | relative rank
 Research may diverge; operational output must converge
 Unknown preferred over false certainty