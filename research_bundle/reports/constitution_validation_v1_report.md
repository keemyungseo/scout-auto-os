# Constitution Validation V1 — Final Blind Validation

Frozen SCOUT Constitution — no new feature, rule, label, model, or tuning.

## Frozen stack

- **Features:** Ranking Engine V1 (170 snapshot features)
- **Model:** CatBoost Ranker (seed=42, iterations=200, lr=0.05)
- **Label:** `return_minus_dd` (Target Discovery winner)

## Calendar coverage

- Period: **2026-06-01** to **2026-06-15**
- Days: **15** (target 90d: **FAIL**)
- Scans: **180** | Samples: **6536**

## Blind holdout (30%)

| Metric | Value |
|--------|-------|
| Avg 2h | 5.2608% |
| Top2 | 8.5783% |
| Top5 | 5.2608% |
| Win% | 54.78% |
| Sharpe | 11.149 |
| Sortino | 39.599 |
| MDD | -34.9749% |
| Profit Factor | 13.866 |
| NDCG@5 | 0.7587 |
| P@5 | 0.4945 |
| Rank Corr | 0.6413 |

## Final conclusions

### 1. Long-term blind persistence

**Partial YES** - blind avg 2h **5.2608%**, walk-forward **5.2608%** on **15d** (2026-06-01 to 2026-06-15). Does NOT meet 3-month target.

### 2. LIVE confidence tier

**MEDIUM-LOW** (medium-low) - Sharpe 11.149, PF 13.866, NDCG@5 0.7587. Recommend shadow LIVE minimum 30d before capital.

### 3. Core engine confirmation

**YES for research core, NO for LIVE core** - constitution hypothesis-validated on short window.

### 4. Biggest remaining risk

**Calendar length** (15d vs 90d target) and **regime gaps** (0 weak regime buckets). All regime buckets positive on available sample.

### 5. Pre-LIVE blockers

1) Extend Binance history to 90d+  2) Persist model artifact + version pin  3) LIVE scan-history for label retrain pipeline  4) Regime-negative monitoring  5) NDCG/P@5 trade-off vs return_minus_dd label

Probabilistic — correlation is not causation; no price targets.

_Generated 2026-06-26T08:28:44.378140+09:00_