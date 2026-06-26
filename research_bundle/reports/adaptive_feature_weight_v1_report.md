# Adaptive Feature Weight Engine V1

Conditional importance analysis on **frozen** CatBoost Ranker — no model retrain.

- Features: **170** | Conditions: **14**
- Train scans: 125 | Blind scans: 55

## Final question: Does conditional weight improve blind performance?

**NO proven lift** — uniform frozen ranker remains competitive (avg 2h 4.7902% vs adaptive 4.2089%). Conditional importance is descriptive, not yet operational.

## Uniform vs Adaptive (blind)

| Metric | Uniform | Adaptive | Delta |
|--------|---------|----------|-------|
| Avg 2h | 4.7902 | 4.2089 | -0.5813 |
| Top2 avg | 7.5366 | 7.416 | -0.1206 |
| Top5 avg | 4.7902 | 4.2089 | -0.5813 |
| NDCG@5 | 0.9664 | 0.9344 | -0.032 |
| P@5 | 0.5927 | 0.5564 | -0.0363 |
| Precision top3 | 59.93 | 56.25 | -3.68 |
| Sharpe | 9.1936 | 8.0186 | -1.175 |

## Conditional patterns (data-driven)

- **bear_leader**: top feature `ctx_rank_5m_compression` — condition-specific feature shift
- **breakout**: top feature `direction_confidence` — volume and release rank rise
- **bull_leader**: top feature `ctx_rank_5m_compression` — direction confidence and trend features
- **compression**: top feature `ctx_rank_5m_compression` — compression and body features dominate
- **high_volatility**: top feature `ctx_rank_5m_compression` — range/ATR proxies dominate
- **low_volatility**: top feature `ctx_rank_5m_compression` — condition-specific feature shift
- **momentum**: top feature `ctx_rank_5m_compression` — momentum and return rank dominate
- **range_expansion**: top feature `ctx_rank_5m_compression` — condition-specific feature shift

## Top conditional features

- [low_volatility] `ctx_rank_5m_release` combined=0.267763 (n=3208)
- [reversal] `ctx_rank_5m_release` combined=0.26771 (n=1631)
- [strong_trend] `ctx_rank_5m_release` combined=0.263295 (n=2770)
- [sideway] `ctx_rank_5m_release` combined=0.262873 (n=2227)
- [volume_decline] `ctx_rank_5m_release` combined=0.262736 (n=2370)
- [high_volatility] `ctx_rank_5m_release` combined=0.256778 (n=2243)
- [weak_trend] `ctx_rank_5m_release` combined=0.255754 (n=2588)
- [momentum] `ctx_rank_5m_release` combined=0.251333 (n=2522)
- [bull_leader] `ctx_rank_5m_release` combined=0.250873 (n=1164)
- [range_expansion] `ctx_rank_5m_release` combined=0.249211 (n=3021)
- [volume_surge] `ctx_rank_5m_release` combined=0.24918 (n=2987)
- [compression] `ctx_rank_5m_release` combined=0.248837 (n=3341)
- [bear_leader] `ctx_rank_5m_release` combined=0.242252 (n=1932)
- [breakout] `ctx_rank_5m_release` combined=0.226975 (n=1132)
- [strong_trend] `ctx_rank_5m_compression` combined=0.04101 (n=2770)

## When / why features matter

In `bear_leader`, `ctx_rank_5m_compression` leads (condition-specific feature shift); In `breakout`, `direction_confidence` leads (volume and release rank rise); In `bull_leader`, `ctx_rank_5m_compression` leads (direction confidence and trend features); In `compression`, `ctx_rank_5m_compression` leads (compression and body features dominate); In `high_volatility`, `ctx_rank_5m_compression` leads (range/ATR proxies dominate); In `low_volatility`, `ctx_rank_5m_compression` leads (condition-specific feature shift)

Probabilistic — short calendar window.

_Generated 2026-06-26T07:51:10.036624+09:00_