# SCOUT Research Infrastructure V1 — Dashboard

Automatic blind dataset builder — frozen constitution, no new research.

## Dataset status

- Version: **scout_constitution_v1**
- Calendar: **2026-06-01** to **2026-06-15** (15d)
- Scans: **180** | Samples: **6536** | Labeled: **6536**
- Label coverage: **100.0%**

## Validation windows

| Window | Coverage | Scans | Ready |
|--------|----------|-------|-------|
| 15d | 100.0% | 180 | YES |
| 30d | 50.0% | 180 | NO |
| 60d | 25.0% | 180 | NO |
| 90d | 16.67% | 180 | NO |
| 180d | 8.33% | 180 | NO |

## Regime coverage

| Axis | Regime | Scans | Sufficient |
|------|--------|-------|------------|
| bull_bear_sideways | bear | 53 | YES |
| bull_bear_sideways | bull | 31 | YES |
| bull_bear_sideways | sideway | 96 | YES |
| volatility | high_volatility | 172 | YES |
| volatility | low_volatility | 8 | NO (+2) |
| structure | breakout | 1 | NO (+9) |
| structure | compression | 163 | YES |
| structure | neutral | 16 | YES |
| dynamics | mixed | 134 | YES |
| dynamics | rotation | 13 | YES |
| dynamics | trend | 33 | YES |
| ecology | Bear | 28 | YES |
| ecology | Bottom | 1 | NO (+9) |
| ecology | Breakout | 1 | NO (+9) |
| ecology | Bull | 48 | YES |
| ecology | Capitulation | 11 | YES |
| ecology | Sideway | 52 | YES |
| ecology | Strong_Bull | 18 | YES |
| ecology | Unknown | 21 | YES |

## Integrity

- Quality: **PASS - all integrity checks passed**
- Duplicates: 0 | Leak features: 0 | Label errors: 0

## Final questions

1. 90d validation possible? **NO** - 15d available, need 75d more for 90d blind validation.
2. Regime gaps? **4 insufficient buckets** - weakest: volatility/low_volatility (need +2), structure/breakout (need +9), ecology/Bottom (need +9), ecology/Breakout (need +9).
3. Days to LIVE validation? **75 calendar days** (+ regime coverage) before LIVE validation at 90d target. Recommend 75d shadow accumulation minimum.
4. Auto accumulation? **YES** - SQLite history DB + append_scan API + forward_labeler can grow dataset without new research code. Run runner after each scan batch.

_Updated 2026-06-26T08:33:16.721998+09:00_