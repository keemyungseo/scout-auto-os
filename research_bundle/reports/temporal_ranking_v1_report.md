# Temporal Ranking Engine V1

Time-series search AI — snapshot + leak-safe scan history temporal features.

- Baseline (Ranking V1 snapshot CatBoost): avg 2h **4.7902**
- Best temporal model: **catboost_ranker** seq=12 avg 2h **4.6368** (snapshot baseline retained unless temporal beats it)
- Leak check: **PASS**

## 1. Did temporal features improve blind performance?

**NO proven lift** - snapshot baseline 4.7902% vs temporal 4.6368%

## 2. Best time window (sequence length)

Best sequence length: **12** scans (~24h history at 2h cadence).

## Sequence length comparison

| Seq len | Model | Avg 2h | Top2 | NDCG5 | P@5 | Sharpe |
|---------|-------|--------|------|-------|-----|--------|
| 3 | catboost_ranker | 4.5702 | 7.5561 | 0.9588 | 0.5891 | 8.593 |
| 3 | lightgbm_ranker | 4.4722 | 7.7798 | 0.9624 | 0.5891 | 8.2997 |
| 3 | xgboost_ranker | 4.4989 | 7.7322 | 0.954 | 0.5818 | 8.5086 |
| 6 | catboost_ranker | 4.6215 | 7.5327 | 0.9574 | 0.5855 | 8.8497 |
| 6 | lightgbm_ranker | 4.4794 | 7.7637 | 0.957 | 0.5818 | 8.4351 |
| 6 | xgboost_ranker | 4.4004 | 7.6493 | 0.9496 | 0.5745 | 8.2303 |
| 9 | catboost_ranker | 4.6282 | 7.491 | 0.9576 | 0.5818 | 8.8308 |
| 9 | lightgbm_ranker | 4.2796 | 7.7637 | 0.957 | 0.5855 | 7.8395 |
| 9 | xgboost_ranker | 4.559 | 7.96 | 0.9536 | 0.5818 | 8.6719 |
| 12 | catboost_ranker | 4.6368 | 7.6991 | 0.9595 | 0.5891 | 8.8539 |
| 12 | lightgbm_ranker | 4.4212 | 7.7637 | 0.9572 | 0.5818 | 8.2386 |
| 12 | xgboost_ranker | 4.29 | 7.8169 | 0.9521 | 0.5782 | 7.8585 |
| 24 | catboost_ranker | 4.5513 | 7.4973 | 0.9606 | 0.5891 | 8.5566 |
| 24 | lightgbm_ranker | 4.337 | 7.7637 | 0.9571 | 0.5855 | 7.9593 |
| 24 | xgboost_ranker | 4.4679 | 7.8126 | 0.9556 | 0.5855 | 8.2802 |

## 3. Delta vs absolute importance

Temporal delta+accel features account for **7.2%** of combined importance. Delta-dominant bases listed in report.

### Top delta-dominant features

- `dna_1h_current_return_pct` — delta 100.0% vs absolute

## 4. Statistically meaningful vs Ranking V1?

Approx lift -3.2% (z~-3.521). Not significant on short window.

## 5. LIVE applicability

**REJECT for LIVE** - no blind improvement vs snapshot ranker.

## 6. Generalization

**Fail / insufficient** — temporal does not beat snapshot on blind or leak check failed.

Probabilistic — ~15 day calendar; no price targets.

_Generated 2026-06-26T08:05:33.580833+09:00_