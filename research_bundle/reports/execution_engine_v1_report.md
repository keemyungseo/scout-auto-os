# Execution Engine V1

Top5 observe → **Execution Score** → Top2 execute.
Search rules unchanged — execution layer only.

## Method

- Pipeline: Direction Champion → Entry Rule V2 → Entry Score → Top5 PASS
- Observation: first **15m** bar after scan (bundle resolution)
- Execution features: obs return, volume, VWAP dev, ATR expansion, breakout, false-breakout penalty
- Weights tuned on train scans (125), blind on 55
- Forward 2h return: **evaluation only**

## Blind comparison

| Strategy | Direction | Trades | Avg 2h | Win% | Lift vs Top5 | Lift vs Entry Top2 |
|----------|-----------|--------|--------|------|--------------|-------------------|
| top5_all | long | 15 | 6.9971 | 46.67 |  |  |
| top2_entry_score | long | 14 | 8.7159 | 50.0 |  |  |
| top2_execution | long | 14 | 9.6908 | 50.0 | 38.5 | 11.19 |
| top5_all | short | 17 | -3.2239 | 52.94 |  |  |
| top2_entry_score | short | 16 | -4.1201 | 50.0 |  |  |
| top2_execution | short | 16 | -4.4921 | 50.0 | -39.34 | -9.03 |

## Combined blind (long + short)

- Top5 all avg 2h: **1.5672**
- Top2 entry-score avg 2h: **1.87**
- Top2 execution avg 2h: **2.1266**
- Execution lift vs Top5: **35.69%**
- Execution lift vs entry Top2: **13.72%**

## Interpretation

Positive lift on blind suggests execution timing adds value beyond search ranking.
Unknown / negative lift is valid — keep Entry Top2 until more observation data exists.

_Generated 2026-06-25T14:16:17.718738+09:00_
**Convergence tier:** core | relative_ranking_between_candidates, trend_persistence_estimation
--- Scout Mission (convergence) ---
 Purpose: Situation Evaluation Engine + Early Trend Detection Scout
 Lifecycle: birth -> growth -> exhaustion | genuine vs fake | regime change
 Core gates: early detection | real vs fake | persistence 1h-24h | relative rank
 Research may diverge; operational output must converge
 Unknown preferred over false certainty