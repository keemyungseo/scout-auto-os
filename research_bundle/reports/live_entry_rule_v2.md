# LIVE Entry Rule V2

Optimized from V1 thresholds — **more recall, precision maintained**.
No new features. No ML. No prediction.

Generated: 2026-06-25T12:19:48.979720+09:00
Signals: long=897 short=897
Rule trees tested per direction: 29

## Long

| Metric | V1 (ABCD) | **V2 (selected)** |
|--------|-----------|-------------------|
| Rule | `(A AND B AND C AND D)` | `(B AND D)` |
| Pass count | 11 | **23** |
| Precision | 1.0 | **0.913** |
| Recall | 0.0615 | **0.1173** |
| F1 | 0.1158 | **0.2079** |
| Pass/day | 0.7333 | **1.5333** |
| 2h Avg | 28.8611% | **26.3633%** |
| 4h Avg | 28.7559% | **28.4172%** |

## Short

| Metric | V1 (ABCD) | **V2 (selected)** |
|--------|-----------|-------------------|
| Rule | `(A AND B AND C AND D)` | `((A AND B AND D) OR C)` |
| Pass count | 11 | **37** |
| Precision | 1.0 | **0.9189** |
| Recall | 0.0615 | **0.1899** |
| F1 | 0.1158 | **0.3148** |
| Pass/day | 0.7333 | **2.4667** |
| 2h Avg | 23.7559% | **18.6786%** |
| 4h Avg | 25.9786% | **18.4476%** |

## Selection criteria
- Precision ≥ 90% of V1 full-AND baseline
- Maximize recall + pass frequency (target ~3 passes/day)
- LIVE score = 0.45×recall + 0.30×freq + 0.25×precision_ratio

**Convergence tier:** core
--- Scout Mission (convergence) ---
 Purpose: Situation Evaluation Engine + Early Trend Detection Scout
 Lifecycle: birth -> growth -> exhaustion | genuine vs fake | regime change
 Core gates: early detection | real vs fake | persistence 1h-24h | relative rank
 Research may diverge; operational output must converge
 Unknown preferred over false certainty