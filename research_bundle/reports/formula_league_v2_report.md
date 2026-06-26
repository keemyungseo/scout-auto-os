# Formula League V2 — Search Formula Evolution

**Frozen:** Direction Champion, Entry V2, Execution stack, Portfolio, Router.
**Research:** Search Formula only.

- Formulas generated: **1993**
- Blind scans: **55** / 180
- Survivors: **797**

## Final question: Blind lift vs A6 baseline (Search only)

**Hypothesis YES** — best survivor `SF_LIN_1h_current_return_pct_1_50` avg 2h **4.9725%** vs A6 **3.1328%** (lift **58.72%**). Probabilistic — small calendar window.

## A6 frozen baseline (blind)

| Metric | Value |
|--------|-------|
| Avg 2h | 3.1328 |
| Win rate 2h | 43.33 |
| Hit top3 rate | 41.85 |
| Sharpe-like | 5.2694 |
| Stability | 1.4375 |

## Top formulas (blind generalization score)

| Formula | Avg 2h | Hit top3% | Win 2h | Stability | Gen score |
|---------|--------|-----------|--------|-----------|-----------|
| `SF_NOT_SF_2h_current_body_pct_gte_20.609` | 3.3569 | 35.56 | 45.93 | 0.8266 | 9.3629 |
| `SF_ATOM_40` | 3.5893 | 39.63 | 45.56 | 1.0133 | 9.2391 |
| `SF_ATOM_75` | 3.9822 | 38.52 | 47.04 | 1.0371 | 9.0617 |
| `SF_NOT_SF_2h_current_body_pct_gte_27.479` | 3.7946 | 37.78 | 48.15 | 1.092 | 9.0552 |
| `SF_NOT_SF_2h_previous_ma20_distance_pct_` | 4.4562 | 36.67 | 48.89 | 1.5431 | 8.7974 |
| `SF_ATOM_64` | 3.7192 | 39.26 | 48.15 | 1.2255 | 8.6611 |
| `SF_NOT_SF_1h_current_return_pct_gte_18.0` | 3.3171 | 36.3 | 47.04 | 1.0783 | 8.4625 |
| `SF_ATOM_41` | 4.1954 | 39.63 | 49.63 | 1.3717 | 8.443 |
| `SF_ATOM_60` | 3.789 | 38.89 | 47.41 | 1.1264 | 8.3297 |
| `SF_ATOM_56` | 4.3036 | 40.0 | 51.11 | 1.4834 | 8.2869 |
| `SF_ATOM_50` | 3.9468 | 38.89 | 46.67 | 1.2513 | 8.2304 |
| `SF_NOT_SF_15m_current_ma20_distance_pct_` | 4.1065 | 38.89 | 49.26 | 1.403 | 8.0378 |
| `SF_ATOM_44` | 4.0254 | 39.63 | 48.15 | 1.3873 | 8.018 |
| `SF_AND_SF_2h_previous_ma20_distance_pct_` | 4.0968 | 39.26 | 48.89 | 1.4151 | 7.9744 |
| `SF_ATOM_31` | 4.1207 | 39.63 | 49.26 | 1.4464 | 7.9731 |

## LIVE-ready candidates

- `SF_LIN_1h_current_return_pct_1_50` — round win rate 1.0, gen score 17.6311
- `SF_LIN_1h_current_return_pct_1_00` — round win rate 1.0, gen score 17.577
- `SF_LIN_1h_current_return_pct_2_00` — round win rate 1.0, gen score 17.5582
- `SF_LIN_1h_current_return_pct_0_50` — round win rate 1.0, gen score 17.5215
- `SF_LIN_1h_current_return_pct_3_00` — round win rate 1.0, gen score 17.3642

## Survivor system

Rounds: temporal blind, weekly, monthly, regime, volatility.
Survival = beat A6 in >=55% of blind rounds with positive generalization score.

## DNA summary

Top survivor features: 1h_current_return_pct, 1h_current_ma20_distance_pct, 15m_current_return_pct, 1h_current_range_pct, 15m_current_ma20_dist

Generalization > raw average return.

_Generated 2026-06-25T15:18:44.749458+09:00_