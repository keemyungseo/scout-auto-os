# Cluster Prediction Report (V1)

## Mission
- Scan-time P(cluster) from DNA cluster formulas (features only, no future data).
- Expected return = Σ P(cluster) × train cluster avg2h.
- Research / Lab only — does not modify LIVE logic.

## Configuration
- Validation scans: 180
- Blind scans: 55
- Long formulas: 16
- Short formulas: 15
- Best long cluster (blind): LONG_V_REVERSAL_A
- Best short cluster (blind): SHORT_V_REVERSAL_A

## Example predictions (latest scan sample)

### BSBUSDT @ 2026-06-15 22:00:00
- Long score **55.7** | expected **+2.34%** | cluster `LONG_CONTINUATION_D` **19.43%**
- Short score **53.4** | expected **+1.90%** | cluster `SHORT_V_REVERSAL_B` **23.45%**
- Recommend **LONG** | confidence **51.1%** | holding ~120.0m

### EVAAUSDT @ 2026-06-15 22:00:00
- Long score **59.4** | expected **+2.84%** | cluster `LONG_CONTINUATION_D` **20.51%**
- Short score **43.8** | expected **+0.57%** | cluster `SHORT_V_REVERSAL_B` **22.01%**
- Recommend **LONG** | confidence **57.6%** | holding ~120.0m

### GUAUSDT @ 2026-06-15 22:00:00
- Long score **60.2** | expected **+2.88%** | cluster `LONG_CONTINUATION_D` **22.84%**
- Short score **45.4** | expected **+0.55%** | cluster `SHORT_V_REVERSAL_B` **29.69%**
- Recommend **LONG** | confidence **57.0%** | holding ~120.0m

### ALCHUSDT @ 2026-06-15 22:00:00
- Long score **66.3** | expected **+3.96%** | cluster `LONG_BASE_REVERSAL_D` **16.35%**
- Short score **50.9** | expected **+1.86%** | cluster `SHORT_TOP_REVERSAL_D` **13.03%**
- Recommend **LONG** | confidence **56.6%** | holding ~120.0m

### EIGENUSDT @ 2026-06-15 22:00:00
- Long score **59.8** | expected **+2.91%** | cluster `LONG_CONTINUATION_D` **20.28%**
- Short score **52.9** | expected **+2.03%** | cluster `SHORT_V_REVERSAL_B` **16.75%**
- Recommend **LONG** | confidence **53.1%** | holding ~120.0m

**Convergence tier:** core | criteria: relative_ranking_between_candidates, trend_persistence_estimation
--- Scout Mission (convergence) ---
 Purpose: Situation Evaluation Engine + Early Trend Detection Scout
 Lifecycle: birth -> growth -> exhaustion | genuine vs fake | regime change
 Core gates: early detection | real vs fake | persistence 1h-24h | relative rank
 Research may diverge; operational output must converge
 Unknown preferred over false certainty