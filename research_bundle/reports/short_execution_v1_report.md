# Short Execution Research V1

Execution / Exit / Portfolio — frozen Long + Short constitutions.

- Short blind picks: **164** | Calendar: **15d**
- Frozen short label: **risk_adjusted_short**

## 1. Best exit strategies (blind TOP5)

| Rank | Rule | Avg return | Sharpe | PF | Avg hold |
|------|------|------------|--------|-----|----------|
| 1 | roi_trail5 | 5.3166% | 8.8492 | 14.7603 | 205.1m |
| 2 | peak_drop3 | 5.2772% | 9.4126 | 20.4741 | 179.6m |
| 3 | high_break2 | 5.2121% | 8.1984 | 10.2197 | 212.6m |
| 4 | roi_trail3 | 5.1711% | 9.2801 | 15.6768 | 190.1m |
| 5 | momentum_weak | 5.1144% | 9.4399 | 56.1416 | 107.9m |

## 2. Recommended holding time

- Median peak ROI at **127.5m**
- 32.32% peaks before 1h | 50.0% before 2h
- Avg profit at 20%/50%/80% of hold: 3.8991% / 4.6387% / 4.3362%

## 3. Exit constitution recommendation

- **Primary:** `roi_trail5`
- **Secondary:** `peak_drop3`

## 4. Early vs late vs dynamic

| Strategy | Avg return | Sharpe | MDD |
|----------|------------|--------|-----|
| hold_2h_constitution | 4.5213% | 8.9581 | -15.5745 |
| hold_4h | 4.6213% | 6.2446 | -42.3743 |
| dynamic_best | 5.3166% | 8.8492 | -12.029 |

## 5. Portfolio Long3 + Short3

- Combined avg 2h: **5.7589%** | Sharpe **13.7411**
- Scan-level long/short corr: **0.0928**
- Simultaneous loss scans: **0.0%**

## 6. Live trading issues TOP10

- **[CRITICAL]** StateExitEngine protective SL is long-oriented only — SL uses entry*(1-pct) and bars[-1].l — wrong direction for SHORT positions
- **[CRITICAL]** Alive score / state_snapshot has no side parameter — R008 state_snapshot is long-biased; short positions get inverted hold/exit signals
- **[HIGH]** alive_score >= hold_alive (70) blocks exit beyond 2h target — HEI +30% ROI pattern: strong alive score returns should_exit=False indefinitely
- **[HIGH]** No short-specific ROI take-profit in StateExitEngine — Primary exit is alive-score only; no TP at favorable ROI for short
- **[MEDIUM]** MET long hold — min_hold 30m then alive hold if score high — Positions with persistent trend_alive stay open past hold_target_minutes=120
- **[INFO]** manual_lock bypass verified in check_exits — manual_lock / auto_manage=False / source=MANUAL skip exit — WLD safe if flagged
- **[MEDIUM]** Review interval 30m may delay exit after signal decay — maybe_review only logs on interval unless should_exit immediate
- **[LOW]** Emergency risk guard separate from state exit — floor only — Risk guard runs first but is emergency floor not profit capture
- **[MEDIUM]** Short execution research shows peak before 2h on many picks — Fixed 2h constitution hold leaves trailing gap vs roi_trail / tp rules
- **[HIGH]** Live log paths empty in repo — audit incomplete without trades.db — Deploy environment must export position_review.csv for MET/HEI case study

## Pre-LIVE blockers

- StateExitEngine protective SL is long-oriented only
- Alive score / state_snapshot has no side parameter
- alive_score >= hold_alive (70) blocks exit beyond 2h target
- No short-specific ROI take-profit in StateExitEngine
- Live log paths empty in repo — audit incomplete without trades.db

Probabilistic — 15d calendar; no price targets.

_Generated 2026-06-26T09:08:29.230699+09:00_