# Direction Champion Entry DNA Report (V1)

## Purpose
- **Not prediction** — distinguish good vs bad **entries** after Direction Champion selection.
- Basis for **Entry Filter Engine V1** (research only; LIVE not modified).

## Data window
- Lookback: **6 months** (from latest scan in bundle)
- Date range: 2026-06-01 → 2026-06-15
- Scans used: 180
- Champion engines: `LONG_CONTINUATION` / `SHORT_CONTINUATION`
- Picks per scan: top **5** (direction champion standard)

## Signal counts
- Long signals: **897** | Short signals: **897**

### Long winner/loser split (by return_2h)
- Winners (top 20%): n=179 threshold≥8.8551%
- Losers (bottom 20%): n=179 threshold≤0.774%
- Median return_2h: 3.6379%

### Short winner/loser split (by short return_2h)
- Winners: n=179 threshold≥6.7249%
- Losers: n=179 threshold≤0.6072%
- Median return_2h: 3.059%

## Long Winner DNA (feature importance)
- #1* `1h_current_return_pct` effect=1.3617 Δ=6.557 (W=9.3812 L=2.8242) ↑ winner higher
- #2* `1h_current_body_pct` effect=1.3264 Δ=6.268 (W=9.3812 L=3.1132) ↑ winner higher
- #3* `1h_current_range_pct` effect=1.0721 Δ=7.9378 (W=13.9972 L=6.0594) ↑ winner higher
- #4* `2h_current_body_pct` effect=0.896 Δ=6.8765 (W=12.0459 L=5.1694) ↑ winner higher
- #5* `2h_current_range_pct` effect=0.8623 Δ=9.1036 (W=18.4979 L=9.3942) ↑ winner higher
- #6* `2h_current_return_pct` effect=0.8498 Δ=7.0264 (W=11.5193 L=4.4929) ↑ winner higher
- #7* `1h_current_ma20_distance_pct` effect=0.7333 Δ=10.2427 (W=19.8905 L=9.6478) ↑ winner higher
- #8* `2h_current_ma20_distance_pct` effect=0.6655 Δ=13.5024 (W=26.1883 L=12.6859) ↑ winner higher
- #9* `1h_current_close_position` effect=0.6606 Δ=0.1294 (W=0.8033 L=0.6739) ↑ winner higher
- #10* `30m_current_range_pct` effect=0.6503 Δ=4.3157 (W=9.0531 L=4.7374) ↑ winner higher
- #11* `30m_current_return_pct` effect=0.6425 Δ=2.9166 (W=5.0596 L=2.143) ↑ winner higher
- #12* `30m_current_body_pct` effect=0.6323 Δ=2.7372 (W=5.2531 L=2.516) ↑ winner higher
- #13* `30m_previous_range_pct` effect=0.5576 Δ=1.9443 (W=5.8365 L=3.8922) ↑ winner higher
- #14* `30m_current_ma20_distance_pct` effect=0.5358 Δ=5.3508 (W=11.7811 L=6.4303) ↑ winner higher
- #15* `15m_previous_range_pct` effect=0.5328 Δ=1.4405 (W=4.2172 L=2.7766) ↑ winner higher

Winner favors **higher**: `1h_current_return_pct`, `1h_current_body_pct`, `1h_current_range_pct`, `2h_current_body_pct`, `2h_current_range_pct`, `2h_current_return_pct`, `1h_current_ma20_distance_pct`, `2h_current_ma20_distance_pct`
Winner favors **lower**: 

## Short Winner DNA
- #1* `1h_current_return_pct` effect=-1.2478 Δ=-6.561 (W=-8.1284 L=-1.5674) ↓ winner lower
- #2* `1h_current_body_pct` effect=1.0121 Δ=4.9351 (W=8.1284 L=3.1933) ↑ winner higher
- #3* `2h_current_return_pct` effect=-0.8614 Δ=-7.5051 (W=-7.2518 L=0.2533) ↓ winner lower
- #4* `2h_current_close_position` effect=-0.8431 Δ=-0.1659 (W=0.1849 L=0.3509) ↓ winner lower
- #5* `1h_current_range_pct` effect=0.7728 Δ=5.6162 (W=12.9558 L=7.3396) ↑ winner higher
- #6* `30m_current_return_pct` effect=-0.7247 Δ=-3.5849 (W=-4.5337 L=-0.9488) ↓ winner lower
- #7* `1h_current_close_position` effect=-0.7232 Δ=-0.1397 (W=0.1954 L=0.3351) ↓ winner lower
- #8* `15m_current_return_pct` effect=-0.7103 Δ=-2.4177 (W=-2.8359 L=-0.4183) ↓ winner lower
- #9* `30m_current_body_pct` effect=0.6885 Δ=2.9633 (W=5.2866 L=2.3232) ↑ winner higher
- #10* `2h_current_range_pct` effect=0.6213 Δ=6.2482 (W=17.5086 L=11.2604) ↑ winner higher
- #11* `15m_current_body_pct` effect=0.581 Δ=1.6564 (W=3.4513 L=1.7949) ↑ winner higher
- #12* `2h_current_body_pct` effect=0.5808 Δ=4.0634 (W=9.044 L=4.9806) ↑ winner higher
- #13* `30m_current_range_pct` effect=0.5662 Δ=3.4109 (W=8.8941 L=5.4831) ↑ winner higher
- #14* `15m_current_range_pct` effect=0.5639 Δ=2.3619 (W=6.2918 L=3.93) ↑ winner higher
- #15* `15m_current_close_position` effect=-0.5214 Δ=-0.14 (W=0.3292 L=0.4692) ↓ winner lower

Winner favors **higher**: `1h_current_body_pct`, `1h_current_range_pct`, `30m_current_body_pct`, `2h_current_range_pct`, `15m_current_body_pct`, `2h_current_body_pct`, `30m_current_range_pct`, `15m_current_range_pct`
Winner favors **lower**: `1h_current_return_pct`, `2h_current_return_pct`, `2h_current_close_position`, `30m_current_return_pct`, `1h_current_close_position`, `15m_current_return_pct`, `15m_current_close_position`

## Common DNA (both directions)
- #1 `1h_current_return_pct` combined_effect=1.3047 long=1.3617 short=-1.2478 (opposite sign)
- #2 `1h_current_body_pct` combined_effect=1.1692 long=1.3264 short=1.0121 (same sign)
- #3 `1h_current_range_pct` combined_effect=0.9224 long=1.0721 short=0.7728 (same sign)
- #4 `2h_current_return_pct` combined_effect=0.8556 long=0.8498 short=-0.8614 (opposite sign)
- #5 `2h_current_range_pct` combined_effect=0.7418 long=0.8623 short=0.6213 (same sign)
- #6 `2h_current_body_pct` combined_effect=0.7384 long=0.896 short=0.5808 (same sign)
- #7 `1h_current_close_position` combined_effect=0.6919 long=0.6606 short=-0.7232 (opposite sign)
- #8 `30m_current_return_pct` combined_effect=0.6836 long=0.6425 short=-0.7247 (opposite sign)
- #9 `30m_current_body_pct` combined_effect=0.6604 long=0.6323 short=0.6885 (same sign)
- #10 `30m_current_range_pct` combined_effect=0.6082 long=0.6503 short=0.5662 (same sign)
- #11 `15m_current_range_pct` combined_effect=0.5445 long=0.5251 short=0.5639 (same sign)
- #12 `15m_current_body_pct` combined_effect=0.5436 long=0.5063 short=0.581 (same sign)

## Entry Filter V1 hypotheses (probabilistic)
- Filter candidates where winner-DNA features align (direction-specific).
- Reject entries matching loser-DNA profile (high trap correlation — verify empirically).
- Long and Short filters remain **independent**.

_Generated 2026-06-25T12:04:20.430016+09:00_
**Convergence tier:** core | real_vs_fake_trend_discrimination, relative_ranking_between_candidates
--- Scout Mission (convergence) ---
 Purpose: Situation Evaluation Engine + Early Trend Detection Scout
 Lifecycle: birth -> growth -> exhaustion | genuine vs fake | regime change
 Core gates: early detection | real vs fake | persistence 1h-24h | relative rank
 Research may diverge; operational output must converge
 Unknown preferred over false certainty