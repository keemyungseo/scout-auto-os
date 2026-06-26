# Online State Machine V1

Post-entry **online state estimation** — no entry prediction, no price forecast.
State updates causally as each bar closes (simulated on forward bundle for research labels).

## Method

- Update cadence: **15m bars** (bundle resolution; design target was 5m)
- Features at each step: body, range, volume, ATR, momentum, slope, MFE, MAE, drawdown, acceleration
- State rules are empirical thresholds on **observable-to-date** dynamics only

## Sample

| Field | Value |
|-------|-------|
| Date range | 2026-06-01 .. 2026-06-15 |
| Scans | 180 |
| Long signals | 897 |
| Short signals | 897 |
| Timeline rows | 87906 |
| Transitions | 46401 |

## Top transitions — LONG

- PULLBACK -> ACCELERATION: P=52.52% n=2556 avgDur=27.31min avgRet=16.1274%
- ACCELERATION -> PULLBACK: P=34.87% n=2382 avgDur=18.07min avgRet=14.7953%
- ACCELERATION -> HEALTHY_TREND: P=25.01% n=1709 avgDur=19.08min avgRet=13.7519%
- ACCELERATION -> WEAK_TREND: P=17.87% n=1221 avgDur=18.53min avgRet=7.4547%
- PULLBACK -> EXHAUSTION: P=23.69% n=1153 avgDur=27.93min avgRet=7.4307%

## State statistics — LONG

| State | Obs | Avg return | Avg duration (min) | Top next |
|-------|-----|------------|------------------|----------|
| REVERSAL | 10723 | -6.2407 | 128.9 | FAKE_BREAKOUT |
| PULLBACK | 8675 | 12.9294 | 26.11 | ACCELERATION |
| ACCELERATION | 8303 | 11.5562 | 17.96 | PULLBACK |
| FAKE_BREAKOUT | 6987 | 2.5178 | 43.33 | REVERSAL |
| HEALTHY_TREND | 2736 | 12.2124 | 17.89 | ACCELERATION |
| EXHAUSTION | 2727 | 6.2653 | 19.54 | PULLBACK |
| WEAK_TREND | 2460 | 6.5044 | 18.68 | ACCELERATION |
| EARLY_BREAKOUT | 1326 | 2.8024 | 16.48 | ACCELERATION |
| DEAD | 16 | -0.0422 | 30.0 | ACCELERATION |

## Top transitions — SHORT

- PULLBACK -> ACCELERATION: P=50.91% n=2284 avgDur=30.38min avgRet=11.0795%
- ACCELERATION -> PULLBACK: P=28.9% n=2036 avgDur=17.73min avgRet=10.2228%
- ACCELERATION -> HEALTHY_TREND: P=26.85% n=1891 avgDur=18.58min avgRet=9.3541%
- ACCELERATION -> WEAK_TREND: P=25.72% n=1812 avgDur=18.82min avgRet=7.1169%
- WEAK_TREND -> ACCELERATION: P=58.17% n=1788 avgDur=18.41min avgRet=7.4635%

## State statistics — SHORT

| State | Obs | Avg return | Avg duration (min) | Top next |
|-------|-----|------------|------------------|----------|
| REVERSAL | 10414 | -8.7071 | 120.56 | FAKE_BREAKOUT |
| PULLBACK | 8638 | 9.5717 | 28.17 | ACCELERATION |
| ACCELERATION | 8483 | 8.3723 | 17.75 | PULLBACK |
| FAKE_BREAKOUT | 5295 | 1.8213 | 36.93 | REVERSAL |
| WEAK_TREND | 4071 | 6.1017 | 19.57 | ACCELERATION |
| HEALTHY_TREND | 3417 | 8.8496 | 18.76 | WEAK_TREND |
| EXHAUSTION | 2425 | 4.9997 | 19.21 | PULLBACK |
| EARLY_BREAKOUT | 1178 | 2.0831 | 16.21 | ACCELERATION |
| DEAD | 32 | -0.3077 | 28.24 | ACCELERATION |

## State transition diagram — LONG

```mermaid
flowchart TD
  subgraph LONG["LONG state machine"]
    s0["ACCELERATION"]
    s1["DEAD"]
    s2["EARLY BREAKOUT"]
    s3["EXHAUSTION"]
    s4["EXIT"]
    s5["FAKE BREAKOUT"]
    s6["HEALTHY TREND"]
    s7["PULLBACK"]
    s8["REVERSAL"]
    s9["WEAK TREND"]
    s2 -->|"64% avgRet +4.5%"| s0
    s1 -->|"62% avgRet +3.7%"| s0
    s9 -->|"58% avgRet +8.3%"| s0
    s8 -->|"57% avgRet +0.9%"| s5
    s7 -->|"53% avgRet +16.1%"| s0
    s5 -->|"38% avgRet -1.4%"| s8
    s3 -->|"37% avgRet +7.9%"| s7
    s6 -->|"37% avgRet +15.5%"| s0
    s0 -->|"35% avgRet +14.8%"| s7
    s5 -->|"32% avgRet +5.0%"| s0
    s3 -->|"30% avgRet +7.3%"| s0
    s6 -->|"29% avgRet +13.8%"| s7
    s8 -->|"25% avgRet -8.3%"| s4
    s3 -->|"25% avgRet +3.5%"| s5
    s0 -->|"25% avgRet +13.8%"| s6
    s1 -->|"25% avgRet +0.7%"| s6
    s6 -->|"24% avgRet +7.8%"| s9
    s7 -->|"24% avgRet +7.4%"| s3
    s5 -->|"22% avgRet +5.1%"| s7
    s9 -->|"22% avgRet +8.2%"| s7
    s0 -->|"18% avgRet +7.5%"| s9
    s2 -->|"14% avgRet +2.4%"| s9
    s8 -->|"13% avgRet +2.5%"| s0
    s1 -->|"12% avgRet -2.3%"| s8
    s7 -->|"12% avgRet +3.4%"| s5
    s0 -->|"9% avgRet +5.2%"| s3
    s9 -->|"7% avgRet +4.7%"| s6
    s5 -->|"7% avgRet +3.4%"| s4
    s9 -->|"6% avgRet +2.2%"| s3
    s0 -->|"6% avgRet +2.5%"| s5
    s7 -->|"6% avgRet +9.2%"| s6
    s2 -->|"6% avgRet +5.4%"| s7
    s2 -->|"5% avgRet -1.2%"| s8
    s6 -->|"5% avgRet +3.2%"| s3
  end
```

## State transition diagram — SHORT

```mermaid
flowchart TD
  subgraph SHORT["SHORT state machine"]
    s0["ACCELERATION"]
    s1["DEAD"]
    s2["EARLY BREAKOUT"]
    s3["EXHAUSTION"]
    s4["EXIT"]
    s5["FAKE BREAKOUT"]
    s6["HEALTHY TREND"]
    s7["PULLBACK"]
    s8["REVERSAL"]
    s9["WEAK TREND"]
    s1 -->|"65% avgRet +1.3%"| s0
    s2 -->|"59% avgRet +3.8%"| s0
    s9 -->|"58% avgRet +7.5%"| s0
    s8 -->|"53% avgRet +0.8%"| s5
    s7 -->|"51% avgRet +11.1%"| s0
    s5 -->|"42% avgRet -1.3%"| s8
    s6 -->|"36% avgRet +7.1%"| s9
    s3 -->|"34% avgRet +6.3%"| s7
    s6 -->|"34% avgRet +9.7%"| s0
    s3 -->|"31% avgRet +5.9%"| s0
    s5 -->|"30% avgRet +4.2%"| s0
    s0 -->|"29% avgRet +10.2%"| s7
    s0 -->|"27% avgRet +9.4%"| s6
    s0 -->|"26% avgRet +7.1%"| s9
    s3 -->|"25% avgRet +2.6%"| s5
    s7 -->|"24% avgRet +5.6%"| s3
    s9 -->|"22% avgRet +7.6%"| s7
    s8 -->|"21% avgRet -12.6%"| s4
    s6 -->|"21% avgRet +10.5%"| s7
    s2 -->|"21% avgRet +1.8%"| s9
    s5 -->|"20% avgRet +3.8%"| s7
    s8 -->|"15% avgRet +3.3%"| s0
    s1 -->|"12% avgRet +0.6%"| s6
    s1 -->|"12% avgRet -2.0%"| s8
    s1 -->|"12% avgRet -0.1%"| s9
    s7 -->|"9% avgRet +2.4%"| s5
    s7 -->|"9% avgRet +8.1%"| s6
    s9 -->|"8% avgRet +5.3%"| s6
    s0 -->|"8% avgRet +4.3%"| s3
    s2 -->|"6% avgRet -3.7%"| s8
    s5 -->|"5% avgRet +2.1%"| s4
    s2 -->|"5% avgRet +5.1%"| s7
    s8 -->|"5% avgRet +1.1%"| s7
    s0 -->|"5% avgRet +1.9%"| s5
  end
```

## Interpretation

- Transition probabilities describe **historical cohort behaviour**, not forecasts.
- Suitable for **holding-policy hypotheses** (extend in HEALTHY/ACCELERATION, tighten in EXHAUSTION/FAKE).
- Operational use requires live bar feed at intended cadence.

_Generated 2026-06-25T13:39:36.218883+09:00_
**Convergence tier:** core | trend_persistence_estimation, real_vs_fake_trend_discrimination
--- Scout Mission (convergence) ---
 Purpose: Situation Evaluation Engine + Early Trend Detection Scout
 Lifecycle: birth -> growth -> exhaustion | genuine vs fake | regime change
 Core gates: early detection | real vs fake | persistence 1h-24h | relative rank
 Research may diverge; operational output must converge
 Unknown preferred over false certainty