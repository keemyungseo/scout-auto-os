# Signal Lifecycle Engine V1

Research-only lifecycle analysis of Direction Champion signals.
No rules, thresholds, or predictions were modified.

## Method

- Resolution: **15m bars** (forward bundle; target was 5m — finer data not in bundle)
- Minimum track: **6h**; preferred: **12h** when bars available
- Labels emerge from measured trajectory shape (MFE/MAE, peak timing, 2h vs 6h return shift)
- Confidence: descriptive cohort statistics only

## Sample

| Field | Value |
|-------|-------|
| Date range | 2026-06-01 .. 2026-06-15 |
| Scans | 180 |
| Long signals | 897 |
| Short signals | 897 |
| Timeline rows | 87906 |

## 2h vs 6h evaluation shift (all signals)

### LONG

- Sustained winner (>=3% at 2h and 6h): 39.35%
- Winner at 2h, fade by 6h: 14.83%
- Under 3% at 2h, runner by 6h: 12.04%
- Avg return shift (6h - 2h): 0.4441%

### SHORT

- Sustained winner (>=3% at 2h and 6h): 39.02%
- Winner at 2h, fade by 6h: 12.6%
- Under 3% at 2h, runner by 6h: 12.71%
- Avg return shift (6h - 2h): -0.5075%

## Lifecycle type summary

### LONG

| Label | N | Avg MFE | Avg peak (min) | Avg 2h | Avg 6h | 2h win% | 6h win% | Fade 2h->6h% |
|-------|---|---------|----------------|--------|--------|---------|---------|--------------|
| Fake Breakout | 321 | 11.4486 | 158.4112 | 2.9276 | -1.2388 | 40.19 | 20.87 | 24.92 |
| Late Runner | 232 | 28.4934 | 651.9181 | 6.3031 | 10.9372 | 65.09 | 76.72 | 6.9 |
| Slow Trend | 139 | 21.8859 | 290.9353 | 8.8326 | 12.4742 | 83.45 | 81.29 | 12.23 |
| Unclassified | 76 | 10.7782 | 74.2105 | 5.0968 | 2.4501 | 50.0 | 32.89 | 17.11 |
| Delayed Breakout | 56 | 11.064 | 496.875 | 1.0613 | 3.4369 | 7.14 | 50.0 | 0.0 |
| Continuous Trend | 40 | 25.5905 | 469.875 | 7.8821 | 13.5637 | 85.0 | 95.0 | 0.0 |
| Immediate Explosion | 28 | 11.876 | 31.6071 | 4.4998 | 2.6887 | 50.0 | 28.57 | 25.0 |
| V-Reversal | 5 | 15.7895 | 591.0 | -0.9946 | 8.2144 | 0.0 | 80.0 | 0.0 |

### SHORT

| Label | N | Avg MFE | Avg peak (min) | Avg 2h | Avg 6h | 2h win% | 6h win% | Fade 2h->6h% |
|-------|---|---------|----------------|--------|--------|---------|---------|--------------|
| Fake Breakout | 279 | 8.3793 | 152.2581 | 1.8657 | -3.2784 | 36.56 | 19.71 | 23.66 |
| Late Runner | 264 | 15.9755 | 655.2841 | 4.6126 | 6.5148 | 60.61 | 70.45 | 6.44 |
| Slow Trend | 171 | 14.2254 | 284.5614 | 6.9858 | 9.2395 | 78.95 | 82.46 | 8.77 |
| Unclassified | 68 | 7.508 | 67.7206 | 1.2839 | -3.9153 | 30.88 | 20.59 | 10.29 |
| Delayed Breakout | 49 | 10.8312 | 529.2857 | 0.6745 | 3.7667 | 8.16 | 40.82 | 4.08 |
| Continuous Trend | 39 | 19.707 | 475.3846 | 6.0881 | 13.0358 | 74.36 | 87.18 | 2.56 |
| Immediate Explosion | 18 | 8.6157 | 32.5 | 3.7281 | 1.8928 | 50.0 | 33.33 | 27.78 |
| V-Reversal | 9 | 19.3131 | 546.6667 | 1.5021 | 7.7818 | 33.33 | 88.89 | 0.0 |

## What SCOUT finds vs misses (empirical)

Interpretation stays probabilistic — correlation with champion ranking, not causation.

### LONG — relatively strong lifecycle shapes
- **Continuous Trend** (n=40): 6h success 95.0%, avg MFE 25.5905%
- **Slow Trend** (n=139): 6h success 81.3%, avg MFE 21.8859%
- **V-Reversal** (n=5): 6h success 80.0%, avg MFE 15.7895%

### LONG — weak / trap-prone shapes
- **Fake Breakout** (n=321): fade after 2h 24.9%, avg 6h -1.2388%
- **Immediate Explosion** (n=28): fade after 2h 25.0%, avg 6h 2.6887%

### SHORT — relatively strong lifecycle shapes
- **V-Reversal** (n=9): 6h success 88.9%, avg MFE 19.3131%
- **Continuous Trend** (n=39): 6h success 87.2%, avg MFE 19.707%
- **Slow Trend** (n=171): 6h success 82.5%, avg MFE 14.2254%

### SHORT — weak / trap-prone shapes
- **Fake Breakout** (n=279): fade after 2h 23.7%, avg 6h -3.2784%
- **Immediate Explosion** (n=18): fade after 2h 27.8%, avg 6h 1.8928%

## Mission convergence

**Convergence tier:** core | trend_persistence_estimation, real_vs_fake_trend_discrimination

--- Scout Mission (convergence) ---
 Purpose: Situation Evaluation Engine + Early Trend Detection Scout
 Lifecycle: birth -> growth -> exhaustion | genuine vs fake | regime change
 Core gates: early detection | real vs fake | persistence 1h-24h | relative rank
 Research may diverge; operational output must converge
 Unknown preferred over false certainty
