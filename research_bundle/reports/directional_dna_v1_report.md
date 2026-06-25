# SCOUT Directional DNA Discovery V1 Report

Generated: 2026-06-25 11:46:17 KST
Validation scans: 180 | Train: 125 | Blind: 55

## Pattern DNA Summary
- **LONG_BASE_REVERSAL**: n=897 success=18.62% sig_features=48 clusters=4
  top DNA: 1h_current_return_pct, 2h_current_close_position, 1h_current_close_position, 2h_current_return_pct, 1h_current_body_pct
- **LONG_V_REVERSAL**: n=897 success=29.65% sig_features=45 clusters=4
  top DNA: 1h_current_return_pct, 1h_current_close_position, 2h_current_close_position, 1h_current_body_pct, 30m_current_return_pct
- **LONG_CONTINUATION**: n=897 success=56.3% sig_features=43 clusters=4
  top DNA: 1h_current_return_pct, 1h_current_body_pct, 1h_current_range_pct, 2h_current_range_pct, 2h_current_body_pct
- **LONG_ACCELERATION**: n=897 success=30.21% sig_features=39 clusters=4
  top DNA: 1h_current_return_pct, 1h_current_close_position, 2h_current_close_position, 2h_current_body_pct, 2h_current_return_pct
- **SHORT_TOP_REVERSAL**: n=897 success=14.27% sig_features=49 clusters=4
  top DNA: 1h_current_return_pct, 1h_current_close_position, 2h_current_close_position, 2h_current_return_pct, 2h_current_body_pct
- **SHORT_V_REVERSAL**: n=897 success=15.27% sig_features=44 clusters=4
  top DNA: 2h_current_close_position, 1h_current_return_pct, 1h_current_close_position, 2h_current_return_pct, 2h_current_body_pct
- **SHORT_CONTINUATION**: n=897 success=5.57% sig_features=40 clusters=4
  top DNA: 2h_current_close_position, 1h_current_close_position, 2h_current_ma20_distance_pct, 1h_current_ma20_distance_pct, 1h_current_return_pct
- **SHORT_ACCELERATION**: n=897 success=19.62% sig_features=41 clusters=4
  top DNA: 2h_current_close_position, 1h_current_return_pct, 1h_current_close_position, 30m_current_return_pct, 15m_current_return_pct

## Global Feature Importance TOP20
- #1 `2h_current_close_position` effect=1.4375 delta=0.334 p=0.0 sig=True
- #2 `1h_current_return_pct` effect=1.4075 delta=6.184 p=0.0 sig=True
- #1 `1h_current_return_pct` effect=1.3843 delta=4.6807 p=0.0 sig=True
- #3 `1h_current_close_position` effect=1.3449 delta=0.3253 p=0.0 sig=True
- #1 `2h_current_close_position` effect=1.3034 delta=0.2964 p=0.0 sig=True
- #1 `1h_current_return_pct` effect=1.2429 delta=4.8701 p=0.0 sig=True
- #2 `2h_current_close_position` effect=1.2336 delta=0.299 p=0.0 sig=True
- #2 `1h_current_return_pct` effect=1.214 delta=5.5236 p=0.0 sig=True
- #1 `1h_current_return_pct` effect=1.1757 delta=4.6124 p=0.0 sig=True
- #2 `1h_current_close_position` effect=1.163 delta=0.2878 p=0.0 sig=True
- #1 `1h_current_return_pct` effect=1.1518 delta=5.9281 p=0.0 sig=True
- #3 `1h_current_close_position` effect=1.1382 delta=0.2797 p=0.0 sig=True
- #1 `2h_current_close_position` effect=1.1313 delta=0.2531 p=0.0 sig=True
- #3 `2h_current_close_position` effect=1.1301 delta=0.2674 p=0.0 sig=True
- #2 `1h_current_close_position` effect=1.1275 delta=0.2398 p=0.0 sig=True
- #3 `1h_current_close_position` effect=1.1215 delta=0.2736 p=0.0 sig=True
- #2 `1h_current_close_position` effect=1.0688 delta=0.2643 p=0.0 sig=True
- #4 `2h_current_return_pct` effect=1.0533 delta=5.43 p=0.0 sig=True
- #4 `2h_current_return_pct` effect=1.0469 delta=4.5352 p=0.0 sig=True
- #2 `1h_current_close_position` effect=1.0261 delta=0.2529 p=0.0 sig=True

## Cluster Performance
- LONG_BASE_REVERSAL_A: n=64 avg2h=-2.9806% win=1.56% PF=0.07 trap=4.69%
- LONG_BASE_REVERSAL_B: n=283 avg2h=-0.5506% win=2.47% PF=0.44 trap=0.0%
- LONG_BASE_REVERSAL_C: n=39 avg2h=7.7707% win=89.74% PF=11.18 trap=7.69%
- LONG_BASE_REVERSAL_D: n=239 avg2h=2.0127% win=29.29% PF=7.63 trap=0.84%
- LONG_V_REVERSAL_A: n=266 avg2h=-1.1636% win=7.89% PF=0.44 trap=7.14%
- LONG_V_REVERSAL_B: n=11 avg2h=17.0287% win=90.91% PF=19.49 trap=9.09%
- LONG_V_REVERSAL_C: n=243 avg2h=2.1214% win=31.28% PF=8.15 trap=1.23%
- LONG_V_REVERSAL_D: n=105 avg2h=5.5532% win=76.19% PF=8.7 trap=9.52%
- LONG_CONTINUATION_A: n=17 avg2h=19.9171% win=88.24% PF=16.77 trap=11.76%
- LONG_CONTINUATION_B: n=140 avg2h=8.7722% win=77.14% PF=27.33 trap=5.71%
- LONG_CONTINUATION_C: n=134 avg2h=2.0807% win=35.07% PF=7.03 trap=1.49%
- LONG_CONTINUATION_D: n=334 avg2h=4.3527% win=59.28% PF=15.77 trap=4.19%
- LONG_ACCELERATION_A: n=109 avg2h=5.2482% win=63.3% PF=4.67 trap=11.01%
- LONG_ACCELERATION_B: n=248 avg2h=1.7624% win=31.05% PF=3.83 trap=4.84%
- LONG_ACCELERATION_C: n=23 avg2h=21.6376% win=91.3% PF=24.18 trap=8.7%
- LONG_ACCELERATION_D: n=245 avg2h=-2.6914% win=6.53% PF=0.17 trap=8.57%
- SHORT_TOP_REVERSAL_B: n=229 avg2h=2.2681% win=30.13% PF=13.48 trap=3.49%
- SHORT_TOP_REVERSAL_C: n=320 avg2h=-0.6282% win=6.25% PF=0.5 trap=33.12%
- SHORT_TOP_REVERSAL_D: n=71 avg2h=-3.9082% win=8.45% PF=0.13 trap=87.32%
- SHORT_V_REVERSAL_A: n=326 avg2h=2.4144% win=38.34% PF=7.87 trap=7.98%
- SHORT_V_REVERSAL_B: n=222 avg2h=-1.4077% win=10.36% PF=0.43 trap=46.85%
- SHORT_V_REVERSAL_C: n=28 avg2h=-1.4489% win=25.0% PF=0.78 trap=64.29%
- SHORT_V_REVERSAL_D: n=49 avg2h=7.5559% win=75.51% PF=12.48 trap=8.16%
- SHORT_CONTINUATION_A: n=352 avg2h=4.9136% win=65.91% PF=38.62 trap=3.12%
- SHORT_CONTINUATION_B: n=161 avg2h=2.8051% win=38.51% PF=7.07 trap=10.56%
- SHORT_CONTINUATION_C: n=77 avg2h=2.1038% win=37.66% PF=2.14 trap=22.08%
- SHORT_CONTINUATION_D: n=35 avg2h=8.1384% win=74.29% PF=7.07 trap=20.0%
- SHORT_ACCELERATION_A: n=35 avg2h=12.4734% win=97.14% PF=138.34 trap=2.86%
- SHORT_ACCELERATION_B: n=311 avg2h=2.4432% win=40.19% PF=5.04 trap=12.86%
- SHORT_ACCELERATION_C: n=50 avg2h=1.3202% win=20.0% PF=5.1 trap=6.0%
- SHORT_ACCELERATION_D: n=229 avg2h=-3.1588% win=8.3% PF=0.21 trap=68.56%

## Blind Validation vs Random / Pattern Champion
- LONG_BASE_REVERSAL / RANDOM: avg2h=2.0219% Δrandom=0.0 Δchampion=0
- LONG_BASE_REVERSAL / PATTERN_CHAMPION: avg2h=1.179% Δrandom=-0.8429 Δchampion=0
- LONG_BASE_REVERSAL / LONG_BASE_REVERSAL_A: avg2h=1.1293% Δrandom=-0.8926 Δchampion=-0.0497
- LONG_BASE_REVERSAL / LONG_BASE_REVERSAL_B: avg2h=1.0503% Δrandom=-0.9716 Δchampion=-0.1287
- LONG_BASE_REVERSAL / LONG_BASE_REVERSAL_C: avg2h=0.21% Δrandom=-1.8119 Δchampion=-0.969
- LONG_BASE_REVERSAL / LONG_BASE_REVERSAL_D: avg2h=3.054% Δrandom=1.0321 Δchampion=1.875
- LONG_V_REVERSAL / RANDOM: avg2h=2.0219% Δrandom=0.0 Δchampion=0
- LONG_V_REVERSAL / PATTERN_CHAMPION: avg2h=2.2061% Δrandom=0.1842 Δchampion=0
- LONG_V_REVERSAL / LONG_V_REVERSAL_A: avg2h=4.2685% Δrandom=2.2466 Δchampion=2.0624
- LONG_V_REVERSAL / LONG_V_REVERSAL_B: avg2h=-0.6598% Δrandom=-2.6817 Δchampion=-2.8659
- LONG_V_REVERSAL / LONG_V_REVERSAL_C: avg2h=2.0941% Δrandom=0.0722 Δchampion=-0.112
- LONG_V_REVERSAL / LONG_V_REVERSAL_D: avg2h=-0.058% Δrandom=-2.0799 Δchampion=-2.2641
- LONG_CONTINUATION / RANDOM: avg2h=2.0219% Δrandom=0.0 Δchampion=0
- LONG_CONTINUATION / PATTERN_CHAMPION: avg2h=4.6662% Δrandom=2.6443 Δchampion=0
- LONG_CONTINUATION / LONG_CONTINUATION_A: avg2h=-1.0594% Δrandom=-3.0813 Δchampion=-5.7256
- LONG_CONTINUATION / LONG_CONTINUATION_B: avg2h=0.6352% Δrandom=-1.3867 Δchampion=-4.031
- LONG_CONTINUATION / LONG_CONTINUATION_C: avg2h=2.9397% Δrandom=0.9178 Δchampion=-1.7265
- LONG_CONTINUATION / LONG_CONTINUATION_D: avg2h=2.4535% Δrandom=0.4316 Δchampion=-2.2127
- LONG_ACCELERATION / RANDOM: avg2h=2.0219% Δrandom=0.0 Δchampion=0
- LONG_ACCELERATION / PATTERN_CHAMPION: avg2h=1.7873% Δrandom=-0.2346 Δchampion=0
- LONG_ACCELERATION / LONG_ACCELERATION_A: avg2h=0.0529% Δrandom=-1.969 Δchampion=-1.7344
- LONG_ACCELERATION / LONG_ACCELERATION_B: avg2h=2.6486% Δrandom=0.6267 Δchampion=0.8613
- LONG_ACCELERATION / LONG_ACCELERATION_C: avg2h=-0.9917% Δrandom=-3.0136 Δchampion=-2.779
- LONG_ACCELERATION / LONG_ACCELERATION_D: avg2h=0.655% Δrandom=-1.3669 Δchampion=-1.1323
- SHORT_TOP_REVERSAL / RANDOM: avg2h=-2.0219% Δrandom=0.0 Δchampion=0
- SHORT_TOP_REVERSAL / PATTERN_CHAMPION: avg2h=-0.6294% Δrandom=1.3925 Δchampion=0
- SHORT_TOP_REVERSAL / SHORT_TOP_REVERSAL_B: avg2h=-1.645% Δrandom=0.3769 Δchampion=-1.0156
- SHORT_TOP_REVERSAL / SHORT_TOP_REVERSAL_C: avg2h=-2.3461% Δrandom=-0.3242 Δchampion=-1.7167
- SHORT_TOP_REVERSAL / SHORT_TOP_REVERSAL_D: avg2h=-2.5891% Δrandom=-0.5672 Δchampion=-1.9597
- SHORT_V_REVERSAL / RANDOM: avg2h=-2.0219% Δrandom=0.0 Δchampion=0
- SHORT_V_REVERSAL / PATTERN_CHAMPION: avg2h=0.5475% Δrandom=2.5694 Δchampion=0
- SHORT_V_REVERSAL / SHORT_V_REVERSAL_A: avg2h=-0.4856% Δrandom=1.5363 Δchampion=-1.0331
- SHORT_V_REVERSAL / SHORT_V_REVERSAL_B: avg2h=-2.7841% Δrandom=-0.7622 Δchampion=-3.3316
- SHORT_V_REVERSAL / SHORT_V_REVERSAL_C: avg2h=-4.3542% Δrandom=-2.3323 Δchampion=-4.9017
- SHORT_V_REVERSAL / SHORT_V_REVERSAL_D: avg2h=-0.5951% Δrandom=1.4268 Δchampion=-1.1426
- SHORT_CONTINUATION / RANDOM: avg2h=-2.0219% Δrandom=0.0 Δchampion=0
- SHORT_CONTINUATION / PATTERN_CHAMPION: avg2h=3.0669% Δrandom=5.0888 Δchampion=0
- SHORT_CONTINUATION / SHORT_CONTINUATION_A: avg2h=-1.3031% Δrandom=0.7188 Δchampion=-4.37
- SHORT_CONTINUATION / SHORT_CONTINUATION_B: avg2h=-0.7336% Δrandom=1.2883 Δchampion=-3.8005
- SHORT_CONTINUATION / SHORT_CONTINUATION_C: avg2h=-4.2545% Δrandom=-2.2326 Δchampion=-7.3214
- SHORT_CONTINUATION / SHORT_CONTINUATION_D: avg2h=-1.7543% Δrandom=0.2676 Δchampion=-4.8212
- SHORT_ACCELERATION / RANDOM: avg2h=-2.0219% Δrandom=0.0 Δchampion=0
- SHORT_ACCELERATION / PATTERN_CHAMPION: avg2h=0.5675% Δrandom=2.5894 Δchampion=0
- SHORT_ACCELERATION / SHORT_ACCELERATION_A: avg2h=-2.4554% Δrandom=-0.4335 Δchampion=-3.0229
- SHORT_ACCELERATION / SHORT_ACCELERATION_B: avg2h=-0.8193% Δrandom=1.2026 Δchampion=-1.3868
- SHORT_ACCELERATION / SHORT_ACCELERATION_C: avg2h=-2.0592% Δrandom=-0.0373 Δchampion=-2.6267
- SHORT_ACCELERATION / SHORT_ACCELERATION_D: avg2h=-3.8396% Δrandom=-1.8177 Δchampion=-4.4071

## LIVE Cluster Candidates (research only — not auto-applied)
- `LONG_BASE_REVERSAL_D` engine=LONG_BASE_REVERSAL blind_avg2h=3.054% tier=verification_needed reason=blind beat champion and random
- `LONG_V_REVERSAL_A` engine=LONG_V_REVERSAL blind_avg2h=4.2685% tier=verification_needed reason=blind beat champion and random
- `LONG_ACCELERATION_B` engine=LONG_ACCELERATION blind_avg2h=2.6486% tier=verification_needed reason=blind beat champion and random

## Principles
- No future data in features or clustering train
- No manual feature selection — all ranked by statistics
- Long/Short researched independently
- Zero-base: cluster formulas compete on blind holdout only

*Research/Lab only. LIVE engines unchanged.*