# A6 external validation bundle

## 1. Training data — `candidates.jsonl`

| | |
|--|--|
| Rows | 6,536 |
| Scans | 180 (`2026-06-01 00:00` ~ `2026-06-15 22:00`) |
| Path | `research_bundle/seed/candidates.jsonl` |
| Raw download | https://raw.githubusercontent.com/keemyungseo/scout-auto-os/main/research_bundle/seed/candidates.jsonl |

```bash
curl -L -o candidates.jsonl \
  https://raw.githubusercontent.com/keemyungseo/scout-auto-os/main/research_bundle/seed/candidates.jsonl
```

Each line: `{scan_kst, symbol, features, max_up_4h, outcome_rank}` — 69 feature keys (5m/15m/30m/1h/2h).

## 2. R005 LOO hit-rate report

| File | Description |
|------|-------------|
| `reports/r005_loo_hit_report/r005_loo_hit_report.txt` | Summary (179 LOO scans) |
| `reports/r005_loo_hit_report/per_scan_hit_rates.csv` | Per-scan Top2/5/7 metrics |
| `reports/r005_loo_hit_report/aggregate_summary.csv` | Tier aggregates |
| `reports/r005_loo_hit_report/scout_research_r005_loo_hit_report.py` | Regenerate script |

**Key results (Top5, 179 scans):**

- Actual Top-N overlap: **52.1%**
- max_up_4h ≥ 3%: **81.8%** avg per scan
- max_up_4h ≥ 4%: **73.9%** avg per scan
- final_4h > 0: **63.8%** avg per scan

Re-run (from full project root with `logs/phase19_winner_dna/`):

```bash
python scout_research_r005_loo_hit_report.py
```
