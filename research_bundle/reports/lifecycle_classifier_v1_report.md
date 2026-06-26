# Lifecycle Classifier V1 Report

Entry-time **lifecycle type classification** only.
No price prediction, return regression, or expected return.

## Method

- **Input:** scan-time features (DNA, pattern, cluster score, rule margin, entry score, etc.)
- **Target:** post-hoc lifecycle label from Signal Lifecycle Engine (used only as training label)
- **Model:** multinomial logistic regression with class balancing (numpy)
- **Split:** temporal train/validation by scan time

## Sample

| Field | Value |
|-------|-------|
| Date range | 2026-06-01 .. 2026-06-15 |
| Total signals | 1794 |
| Train scans | 125 |
| Validation scans | 55 |
| Feature dimensions | 97 |
| Long classes (train) | 8 |
| Short classes (train) | 8 |

## Validation metrics

### LONG

- Accuracy: **0.1949** | Top-2 accuracy: **0.3713**
- Macro F1: **0.1766** | Weighted F1: **0.2188**
- Macro precision: 0.1909 | Macro recall: 0.201

| Label | Support | Precision | Recall | F1 |
|-------|---------|-----------|--------|-----|
| Continuous Trend | 11 | 0.0 | 0.0 | 0.0 |
| Delayed Breakout | 23 | 0.0625 | 0.1304 | 0.0845 |
| Fake Breakout | 91 | 0.4074 | 0.2418 | 0.3034 |
| Immediate Explosion | 9 | 0.2083 | 0.5556 | 0.303 |
| Late Runner | 71 | 0.3333 | 0.1831 | 0.2364 |
| Slow Trend | 35 | 0.2069 | 0.1714 | 0.1875 |
| Unclassified | 32 | 0.1176 | 0.125 | 0.1212 |

**Fake vs Continuous binary view** (pred=Fake Breakout): precision=0.4074 recall=0.2418 f1=0.3034

### SHORT

- Accuracy: **0.1765** | Top-2 accuracy: **0.4154**
- Macro F1: **0.131** | Weighted F1: **0.1957**
- Macro precision: 0.1558 | Macro recall: 0.1577

| Label | Support | Precision | Recall | F1 |
|-------|---------|-----------|--------|-----|
| Continuous Trend | 12 | 0.0 | 0.0 | 0.0 |
| Delayed Breakout | 19 | 0.125 | 0.3684 | 0.1867 |
| Fake Breakout | 94 | 0.4571 | 0.1702 | 0.2481 |
| Immediate Explosion | 6 | 0.0476 | 0.1667 | 0.0741 |
| Late Runner | 73 | 0.3333 | 0.1644 | 0.2202 |
| Slow Trend | 34 | 0.0833 | 0.0588 | 0.069 |
| Unclassified | 30 | 0.2 | 0.3333 | 0.25 |
| V-Reversal | 4 | 0.0 | 0.0 | 0.0 |

**Fake vs Continuous binary view** (pred=Fake Breakout): precision=0.4571 recall=0.1702 f1=0.2481

## Holding-strategy readiness (probabilistic)

Can entry-time features support **different holding strategies** by lifecycle type?

- **Long:** insufficient — entry features do not yet separate lifecycle types reliably
- **Short:** insufficient — entry features do not yet separate lifecycle types reliably

Interpretation stays non-operational until out-of-sample validation on new regimes.
Unknown is valid when macro F1 remains near majority-class baseline.

## Probability output columns

Per signal: `prob_continuous`, `prob_slow`, `prob_late_runner`, `prob_delayed`, `prob_explosion`, `prob_fake`, `prob_dead`

_Generated 2026-06-25T13:34:41.174365+09:00_
**Convergence tier:** core | real_vs_fake_trend_discrimination, trend_persistence_estimation
--- Scout Mission (convergence) ---
 Purpose: Situation Evaluation Engine + Early Trend Detection Scout
 Lifecycle: birth -> growth -> exhaustion | genuine vs fake | regime change
 Core gates: early detection | real vs fake | persistence 1h-24h | relative rank
 Research may diverge; operational output must converge
 Unknown preferred over false certainty