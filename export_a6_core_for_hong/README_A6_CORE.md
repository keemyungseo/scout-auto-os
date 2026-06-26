# A6 Core Export Package (for external Cursor projects)

Standalone extraction of **A6 frozen search/ranking logic** from Scout Project v2.
No Auto OS, dashboard, execution, or live trading code included.

---

## Package contents

| File | Role | Original source |
|------|------|-----------------|
| `a6_common.py` | Shared utils (ohlcv, ema, percentile) | phase16, phase19, universe |
| `a6_feature_core.py` | Feature extraction + Pattern B | phase19, phase13, phase16 |
| `a6_state_core.py` | State classification + profile | phase20 |
| `a6_score_core.py` | A6 formula scoring | phase22, phase23 |
| `a6_pipeline_example.py` | train → search → rank → eval | blind_test_b001 |

---

## Function name mapping

| Export name | Original file | Original function |
|-------------|---------------|-------------------|
| `extract_dna_features_from_klines` | scout_phase19_winner_ranking_dna.py | `extract_dna_features` |
| `extract_dna_features_from_dataframes` | (new wrapper) | — |
| `pattern_b_pass` | scout_phase19 (inline filter) | Pattern B block |
| `window_seq` | scout_phase13_5m_sequence_ignition.py | `window_seq` |
| `build_thresholds` | scout_phase20_winner_state_ranking.py | `build_thresholds` |
| `classify_5m/15m/30m/1h/2h` | scout_phase20 | same names |
| `state_match_score` | scout_phase20 | `state_match_score` |
| `build_profile` | scout_phase20 | `build_profile` |
| `within_scan_pct` | scout_phase22_search_formula_evolution.py | `within_scan_pct` |
| `bonus_a5_raw` | scout_phase22 | `bonus_a5_raw` |
| `build_train_stats` | scout_phase22 | `build_train_stats` |
| `formula_scores_a6` | scout_phase23_search_formula_league.py | `formula_scores_a6` |
| `rank_scan_candidates` | scout_blind_test_b001.py | scan ranking block |

---

## Required input data format

### DataFrame (recommended for external projects)

Each timeframe needs a pandas DataFrame with columns:

```
open_time, open, high, low, close, volume
```

- `open_time`: candle open time in **milliseconds** (int) or datetime
- `open/high/low/close/volume`: float

Provide a dict keyed by timeframe:

```python
dfs = {
    "5m": df_5m,
    "15m": df_15m,
    "30m": df_30m,   # optional; falls back to 15m
    "1h": df_1h,     # optional
    "2h": df_2h,     # optional
}
```

### Kline list (Binance format)

```python
[open_time_ms, open, high, low, close, volume, ...]
```

Slice all series to `end_ms` (scan timestamp) before scoring.

---

## A6 calculation order

```
1. extract_dna_features     → Pattern B filter → feature dict
2. build_thresholds         → winner-pool percentiles (p25/p50/p75/p90)
3. annotate                 → states + transitions per candidate
4. build_profile            → empirical lift tables (winner vs all)
5. build_train_stats        → ig_a2, ig_a5, expansion_metric_w
6. state_match_score        → base score (Phase20)
7. formula_scores_a6        → A6 = base + ig_a2*b2 + ig_a5*b5
8. rank descending by A6
```

---

## Pattern B meaning (frozen filter)

A symbol passes Pattern B only if ALL hold on the **15m anchor candle**:

| Rule | Threshold |
|------|-----------|
| Price range | `0.05 <= price <= 400` USDT |
| MACD signal | `macd_sig >= -0.0016` |
| 15m range | `range_pct >= 1.4768%` |

Failed candidates are excluded before ranking (not penalized).

---

## State classification meaning

Each candidate gets states on 5 timeframes:

| TF | States |
|----|--------|
| 5m | Quiet, Normal, SequenceStrong, MomentumStrong, Release |
| 15m | Weak, VolumeSupport, Expansion |
| 30m | Compression, Neutral, Expansion |
| 1h | Flat, ExpansionStart, Expansion, Acceleration |
| 2h | Flat, TrendAlive, StrongTrend, OverExtended |

Thresholds are **data-derived** from winner pool (top-3 per historical scan), not hand-tuned.

`state_match_score` = sum of log(lift) for states + 0.5×transitions + cluster combo.

---

## formula_scores_a6 usage

```python
from export_a6_core_for_hong.a6_state_core import annotate, build_profile, build_thresholds, state_match_score, winner_loser_sets
from export_a6_core_for_hong.a6_score_core import build_train_stats, formula_scores_a6

# train_rows: list of {scan_kst, symbol, features, outcome_rank}
th = build_thresholds(winner_feats)
annotated = annotate(train_rows, th)
profile = build_profile(winners, annotated)
stats = build_train_stats(annotated, train_by_scan, th)

# at scan time:
row = annotate([{"symbol": "ABCUSDT", "features": feats}], th)[0]
base = state_match_score(row["states"], row["transitions"], profile)
a6 = formula_scores_a6(row, peers, base, th, stats)["A6"]
```

---

## Minimal execution example

```bash
cd "Project solo"
pip install -r export_a6_core_for_hong/requirements_a6_core.txt
python -m export_a6_core_for_hong.a6_pipeline_example
```

With real data:

```python
from export_a6_core_for_hong.a6_feature_core import extract_dna_features_from_dataframes
from export_a6_core_for_hong.a6_pipeline_example import (
    load_train_from_jsonl, rank_scan_at_timestamp,
)

train, train_by = load_train_from_jsonl("candidates.jsonl", cutoff_kst="2026-06-16 17:00:00")
top5 = rank_scan_at_timestamp(symbol_klines, "2026-06-16 17:00:00", train, train_by)
```

---

## Connecting to external Cursor project

1. Copy folder `export_a6_core_for_hong/` into your project root
2. Add project root to `PYTHONPATH` or install as local package
3. Feed your own Binance OHLCV dataframes (no API keys in this package)
4. Build training JSONL from historical scans with `outcome_rank` labels
5. Call `rank_scan_at_timestamp()` at each scan interval

Training data format (JSONL line):

```json
{
  "scan_kst": "2026-06-15 17:00:00",
  "symbol": "ABCUSDT",
  "features": { "...": 0.0 },
  "outcome_rank": 1,
  "max_up_4h": 12.5
}
```

---

## Remaining dependencies

This package is **stdlib + pandas/numpy only** for core logic.

Not included (you must provide):

- Binance data fetch (use your own REST/WS)
- Universe symbol list
- Historical `candidates.jsonl` for training (or build your own)
- Forward evaluation klines (for backtest only)

No scipy required for frozen A6 path.

---

## Frozen formula (production)

```
A6 = state_match_score
   + ig_a2 × within_scan_pct(1h_current_range_pct)
   + ig_a5 × expansion_metric_weighted_score
```

Do not modify Pattern B, state boundaries, or A6 weights without re-running Phase22/23 validation.
