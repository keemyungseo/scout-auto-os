# SCOUT Zero-Base Validation V1 Report

Generated: 2026-06-25 11:23:33 KST
Train: before 2026-06-01 (0 scans)
Blind Validation: June+ (180 scans, 5m grid)
Random: 100 draws, 95% CI [0.1415, 0.2172]

## ① Engine Rankings
- #1 **FORMULA_LEAGUE** avg2h=5.1456% win=56.86% PF=13.58 sig=✓ [verification_needed]
- #2 **MOMENTUM** avg2h=5.1456% win=56.86% PF=13.58 sig=✓ [verification_needed]
- #3 **RELATIVE_STRENGTH** avg2h=5.1456% win=56.86% PF=13.58 sig=✓ [verification_needed]
- #4 **STATE_LEAGUE** avg2h=4.6435% win=53.4% PF=7.39 sig=✓ [verification_needed]
- #5 **BREAKOUT** avg2h=3.9064% win=52.84% PF=3.82 sig=✓ [verification_needed]
- #6 **RANGE_EXPANSION** avg2h=0.8774% win=36.68% PF=1.27 sig=✓ [verification_needed]
- #7 **FEATURE_LEAGUE** avg2h=2.9404% win=32.99% PF=8.81 sig=✓ [champion_candidate]
- #8 **PULLBACK** avg2h=2.1728% win=33.22% PF=3.3 sig=✓ [verification_needed]
- #9 **VWAP** avg2h=1.6107% win=31.66% PF=1.87 sig=✓ [verification_needed]
- #10 **COMPRESSION** avg2h=0.7097% win=30.43% PF=1.27 sig=✗ [verification_needed]
- #11 **RANDOM** avg2h=0.0604% win=17.61% PF=1.04 sig=✗ [verification_needed]
- #12 **A6** avg2h=0.1028% win=16.5% PF=1.07 sig=✗ [verification_needed]

## ② A6 Rank
- A6 rank: **#12** / 12
- A6 avg2h=0.1028% win=16.5% trap=2.34%

## ③ vs Random
- Random avg2h=0.2395% (std=2.5888)
- Random 95% CI: [0.1415, 0.2172]
- FORMULA_LEAGUE: delta=4.9061% significant=True
- MOMENTUM: delta=4.9061% significant=True
- RELATIVE_STRENGTH: delta=4.9061% significant=True
- STATE_LEAGUE: delta=4.404% significant=True
- BREAKOUT: delta=3.6669% significant=True
- RANGE_EXPANSION: delta=0.6379% significant=True
- FEATURE_LEAGUE: delta=2.7009% significant=True
- PULLBACK: delta=1.9333% significant=True

## ④ Champion Candidates
- **FEATURE_LEAGUE** n=879 avg2h=2.9404%

## ⑤ Failure Reasons
- **RANDOM**: avg_return_2h below random baseline; median_return_2h not positive; negative avg in regimes: bear
- **A6**: avg_return_2h below random baseline; negative avg in regimes: bear
- **FORMULA_LEAGUE**: trap_rate not lower than random
- **STATE_LEAGUE**: trap_rate not lower than random
- **MOMENTUM**: trap_rate not lower than random
- **BREAKOUT**: trap_rate not lower than random
- **COMPRESSION**: trap_rate not lower than random; negative avg in regimes: bear
- **RANGE_EXPANSION**: trap_rate not lower than random; negative avg in regimes: bear
- **VWAP**: trap_rate not lower than random
- **PULLBACK**: trap_rate not lower than random
- **RELATIVE_STRENGTH**: trap_rate not lower than random

## ⑥ Regime Win Rates (validation)
- A6: sideway=17.4%, bull=25.2%, bear=9.8%
- BREAKOUT: sideway=56.0%, bull=63.2%, bear=41.1%
- COMPRESSION: sideway=33.1%, bull=38.1%, bear=21.1%
- FEATURE_LEAGUE: sideway=34.6%, bull=29.7%, bear=32.0%
- FORMULA_LEAGUE: sideway=61.0%, bull=65.8%, bear=44.2%
- MOMENTUM: sideway=61.0%, bull=65.8%, bear=44.2%
- PULLBACK: sideway=33.8%, bull=45.8%, bear=24.9%
- RANDOM: sideway=20.3%, bull=22.6%, bear=9.8%
- RANGE_EXPANSION: sideway=38.4%, bull=48.4%, bear=26.8%
- RELATIVE_STRENGTH: sideway=61.0%, bull=65.8%, bear=44.2%
- STATE_LEAGUE: sideway=57.4%, bull=64.5%, bear=39.6%
- VWAP: sideway=35.2%, bull=39.4%, bear=20.8%

## ⑦ Most Promising Direction
- FORMULA_LEAGUE: avg2h=5.1456% big_winner=79.71% (hypothesis tier)
- MOMENTUM: avg2h=5.1456% big_winner=79.71% (hypothesis tier)
- RELATIVE_STRENGTH: avg2h=5.1456% big_winner=79.71% (hypothesis tier)

## ⑧ Next Research Priority
1. Obtain May train data for true blind calibration (current bundle: train=0)
2. Re-test champion candidates with trap_rate gate on expanded sample
3. Cross-validate top momentum/breakout engines on July+ live research forward
4. State League replay with position evolution (V1.5) on same validation window

*Research only. No LIVE changes. Correlation ≠ causation. Probabilistic labels.*