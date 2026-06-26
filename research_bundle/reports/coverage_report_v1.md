# Coverage Report — Reject Analysis V1

## Funnel (Direction Champion → Rule → Portfolio → Fill)

| Stage | Count | Survival % |
|-------|-------|------------|
| Direction Champion candidates | 1794 | 100% |
| Entry Rule V2 PASS | 85 | **4.74%** |
| Portfolio PASS / Replacement | 14 | **0.78%** |

- Scans analyzed: 180 (2h interval)
- Near-pass (Almost Pass) candidates: **7**

## Rule Reject — condition failure share

| Feature group | Fail % of rule failures |
|---------------|-------------------------|
| Momentum | **40.53%** |
| Body | **33.33%** |
| Range | **26.14%** |

## Reject tier (rule stage)
- **Near Pass**: ≤1 failed condition, gap ≤10%
- **Medium Reject**: ≤2 failed, gap ≤40%
- **Impossible Reject**: 3+ failed or large gap

## Portfolio reject breakdown

| Reason | Count |
|--------|-------|
| Low_Score | 71 |
| Replacement | 8 |
| PASS | 6 |
| Already_Occupied | 6 |

## Bottleneck (PASS increase potential)

- **Primary stage:** `entry_rule_v2`
- **Candidates lost at stage:** 1709 (95.26%)
- **Top rule blocker:** Momentum (40.53% of condition failures)
- **Recommendation:** Primary coverage gap at Entry Rule V2 — Body/Range/Momentum thresholds block ~95% of champion picks
- **Priority:** high

## Notes
- Rules and thresholds **not modified** — coverage analysis only.
- No prediction / ML.

--- Scout Mission (convergence) ---
 Purpose: Situation Evaluation Engine + Early Trend Detection Scout
 Lifecycle: birth -> growth -> exhaustion | genuine vs fake | regime change
 Core gates: early detection | real vs fake | persistence 1h-24h | relative rank
 Research may diverge; operational output must converge
 Unknown preferred over false certainty