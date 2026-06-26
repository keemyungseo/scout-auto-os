# Rule Discovery Engine V1

Automatic operational entry rule search — **scan-time features only**.
No lifecycle, clustering, pattern discovery, or price prediction.

## Bottleneck context

- Direction Champion candidates: **1794**
- Entry Rule V2 pass (full sample): **60** (3.34%)

## Method

- Candidate generation: threshold, ratio, diff, window-increase, scan rank, AND/OR/NOT
- Thresholds mined on **train scans only**; metrics on **blind scans**
- Train/blind split: 125 / 55 scans (temporal)
- Primary objective: maximize **pass_per_day** subject to precision >= V2 - 2%
- Secondary: maximize avg 2h return

## Current Entry Rule V2 — blind validation

| Direction | Pass | Precision | Recall | Pass/day | Avg 2h | Avg 4h | Coverage |
|-----------|------|-----------|--------|----------|--------|--------|----------|
| Long | 8 | 1.0 | 0.1569 | 1.6 | 27.979 | 32.2433 | 2.94% |
| Short | 10 | 0.9 | 0.1957 | 2.0 | 17.1915 | 17.7112 | 3.68% |

## Top discovered rules — LONG (blind)

| Rank | Rule | Pass | Prec | Recall | Lift | Pass/day | Avg2h | Avg4h | Cov% | Floor |
|------|------|------|------|--------|------|----------|-------|-------|------|-------|
| 13 | `(1h_current_body_pct >= 18.24 OR 1h_current_range_pct >` | 10 | 1.0 | 0.1961 | 5.3333 | 2.0 | 27.1955 | 33.0943 | 3.68 | True |
| 14 | `(1h_current_body_pct >= 14.62 AND 2h_current_body_pct >` | 10 | 1.0 | 0.1961 | 5.3333 | 2.0 | 26.1137 | 32.8077 | 3.68 | True |
| 15 | `(1h_current_return_pct >= 15.03 AND 2h_current_body_pct` | 10 | 1.0 | 0.1961 | 5.3333 | 2.0 | 26.1137 | 32.8077 | 3.68 | True |
| 16 | `((15m_current_body_pct - 1h_current_body_pct) <= -10.12` | 10 | 1.0 | 0.1961 | 5.3333 | 2.0 | 26.1137 | 32.8077 | 3.68 | True |
| 17 | `(1h_current_body_pct >= 14.62 AND 1h_current_return_pct` | 10 | 1.0 | 0.1961 | 5.3333 | 2.0 | 26.1137 | 32.8077 | 3.68 | True |
| 18 | `(1h_current_body_pct >= 14.62 AND 1h_current_body_pct >` | 10 | 1.0 | 0.1961 | 5.3333 | 2.0 | 26.1137 | 32.8077 | 3.68 | True |
| 19 | `(1h_current_body_pct >= 14.62 AND 1h_current_return_pct` | 10 | 1.0 | 0.1961 | 5.3333 | 2.0 | 26.1137 | 32.8077 | 3.68 | True |
| 20 | `(1h_current_body_pct >= 14.62 AND 1h_current_body_pct >` | 10 | 1.0 | 0.1961 | 5.3333 | 2.0 | 26.1137 | 32.8077 | 3.68 | True |
| 9 | `(1h_current_body_pct >= 14.62 AND 2h_current_return_pct` | 10 | 1.0 | 0.1961 | 5.3333 | 2.0 | 26.1137 | 32.8077 | 3.68 | True |
| 10 | `(1h_current_body_pct >= 14.62 AND (15m_current_body_pct` | 10 | 1.0 | 0.1961 | 5.3333 | 2.0 | 26.1137 | 32.8077 | 3.68 | True |

## Top discovered rules — SHORT (blind)

| Rank | Rule | Pass | Prec | Recall | Lift | Pass/day | Avg2h | Avg4h | Cov% | Floor |
|------|------|------|------|--------|------|----------|-------|-------|------|-------|
| 1 | `(1h_current_body_pct >= 10.96 AND 1h_current_return_pct` | 14 | 0.9286 | 0.2826 | 5.4907 | 2.8 | 15.952 | 17.4416 | 5.15 | True |
| 2 | `(1h_current_body_pct >= 10.96 AND 1h_current_return_pct` | 14 | 0.9286 | 0.2826 | 5.4907 | 2.8 | 15.952 | 17.4416 | 5.15 | True |
| 3 | `(2h_current_return_pct <= -18.51 OR 30m_current_body_pc` | 13 | 0.9231 | 0.2609 | 5.4582 | 2.6 | 16.4274 | 16.1916 | 4.78 | True |
| 4 | `(1h_current_return_pct <= -14.72 OR 30m_current_body_pc` | 13 | 0.9231 | 0.2609 | 5.4582 | 2.6 | 16.3046 | 17.924 | 4.78 | True |
| 5 | `(1h_current_body_pct >= 10.96 AND 1h_current_return_pct` | 11 | 0.9091 | 0.2174 | 5.3755 | 2.2 | 17.1626 | 18.7446 | 4.04 | True |
| 6 | `(1h_current_body_pct >= 10.96 AND 1h_current_return_pct` | 11 | 0.9091 | 0.2174 | 5.3755 | 2.2 | 16.1234 | 16.221 | 4.04 | True |
| 7 | `(30m_current_return_pct <= -9.65 OR 2h_current_return_p` | 11 | 0.9091 | 0.2174 | 5.3755 | 2.2 | 15.9148 | 15.8437 | 4.04 | True |
| 8 | `(1h_current_return_pct <= -14.72 OR 30m_current_return_` | 11 | 0.9091 | 0.2174 | 5.3755 | 2.2 | 15.7697 | 17.8911 | 4.04 | True |
| 9 | `30m_current_body_pct >= 9.48` | 11 | 0.9091 | 0.2174 | 5.3755 | 2.2 | 14.8466 | 16.7566 | 4.04 | True |
| 10 | `(30m_current_body_pct >= 9.48 AND 30m_current_body_pct ` | 11 | 0.9091 | 0.2174 | 5.3755 | 2.2 | 14.8466 | 16.7566 | 4.04 | True |

## Recommendation

**Decision:** Adopt

- Reason: Long: Adopt. Short: Adopt. Candidate beats V2 pass/day (2.0 vs 1.6) with precision 1.0
- Mode: replace_v2
- Rule: `(1h_current_body_pct >= 18.24 OR 1h_current_range_pct >= 24.01)`

## Interpretation

- Precision floor is **V2 blind precision - 2%** — not a guarantee on future regimes.
- Hybrid `(V2 OR candidate)` trades coverage for precision — validate before LIVE.
- Reject/Needs-validation outcomes are valid — do not force rule promotion.

_Generated 2026-06-25T13:48:59.433628+09:00_
**Convergence tier:** core | relative_ranking_between_candidates, early_trend_detection
--- Scout Mission (convergence) ---
 Purpose: Situation Evaluation Engine + Early Trend Detection Scout
 Lifecycle: birth -> growth -> exhaustion | genuine vs fake | regime change
 Core gates: early detection | real vs fake | persistence 1h-24h | relative rank
 Research may diverge; operational output must converge
 Unknown preferred over false certainty