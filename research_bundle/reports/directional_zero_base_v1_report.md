# SCOUT Directional Zero-Base V1 Report

Generated: 2026-06-25 11:35:20 KST
Validation scans: 180

## Long TOP Engines
- #1 **LONG_CONTINUATION** avg2h=5.0931% win=56.3% PF=13.83 slot=True [champion_candidate]
- #2 **LONG_V_REVERSAL** avg2h=1.7575% win=29.65% PF=2.59 slot=False [hypothesis]
- #3 **LONG_ACCELERATION** avg2h=1.4867% win=30.21% PF=1.87 slot=False [hypothesis]
- #4 **LONG_BASE_REVERSAL** avg2h=0.8452% win=18.62% PF=1.89 slot=False [hypothesis]
- #5 **RANDOM_LONG** avg2h=None% win=None% PF=None slot=False [hypothesis]

## Short TOP Engines
- #1 **SHORT_CONTINUATION** short_avg2h=3.8598% win=50.84% PF=7.96 slot=True [champion_candidate]
- #2 **SHORT_V_REVERSAL** short_avg2h=1.0626% win=27.42% PF=1.71 slot=False [hypothesis]
- #3 **SHORT_ACCELERATION** short_avg2h=0.773% win=28.21% PF=1.41 slot=False [hypothesis]
- #4 **RANDOM_SHORT** short_avg2h=None% win=None% PF=None slot=False [hypothesis]
- #5 **SHORT_TOP_REVERSAL** short_avg2h=-0.2564% win=12.93% PF=0.82 slot=False [hypothesis]

## vs Random Baseline
- Long Random: avg2h=0.2735% win=17.61%
- Short Random: short_avg2h=-0.2735% win=16.29%
- A6 Long (baseline only): avg2h=0.1028% win=16.5%
- Long LONG_CONTINUATION vs Random: delta=4.8196%
- Long LONG_V_REVERSAL vs Random: delta=1.4840%
- Long LONG_ACCELERATION vs Random: delta=1.2132%
- Short SHORT_CONTINUATION vs Random: delta=4.1333%
- Short SHORT_V_REVERSAL vs Random: delta=1.3361%
- Short SHORT_ACCELERATION vs Random: delta=1.0465%

## Pattern Performance
- DOWN_BASE_UP (long): n=968 long_avg2h=2.7 short_avg2h=-3.0965
- DOWN_UP (long): n=913 long_avg2h=3.731 short_avg2h=-1.515
- UP_CONTINUATION (long): n=475 long_avg2h=3.0215 short_avg2h=-2.5781
- UP_ACCELERATION (long): n=1054 long_avg2h=5.0046 short_avg2h=-3.7957
- UP_BASE_DOWN (short): n=1025 long_avg2h=-2.1263 short_avg2h=2.3227
- UP_DOWN (short): n=1037 long_avg2h=-0.0866 short_avg2h=2.3518
- DOWN_CONTINUATION (short): n=442 long_avg2h=-1.8475 short_avg2h=3.0763
- DOWN_ACCELERATION (short): n=735 long_avg2h=-2.4788 short_avg2h=4.6456

## 3 Long + 3 Short Slot Simulation
- Long slots filled: 1/3 engines=['LONG_CONTINUATION']
- Short slots filled: 1/3 engines=['SHORT_CONTINUATION']
- Empty long slots: 2 | empty short slots: 2
- Combined avg2h: 4.4765% (n=1794)
- Long leg: 5.0931% | Short leg: 3.8598%

## Quality Gate (empty slot if fail)
- min_samples=50 min_win=35.0% min_avg2h=0.3% must beat random

*Research/Lab only. No LIVE order changes. Long/Short evaluated independently.*