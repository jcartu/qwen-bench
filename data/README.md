# `data/` — merged machine-readable result tables

Each file here is a CSV exported from one of the indexed studies, named:

```
{YYYY-MM}-{slug}.csv
```

These are **canonical pointers**. The authoritative source remains the study's
own repo. If a CSV here ever drifts from its origin, treat the study repo as
ground truth and file an issue.

## Current files

| File | Source study | Rows |
|------|--------------|------|
| `2026-05-day1-sprint.csv` | [`qwen36-27b-blackwell-inference-study`](https://github.com/jcartu/qwen36-27b-blackwell-inference-study) | 328 |
| `2026-05-day2-stress-validation.csv` | [`qwen36-27b-blackwell-stress-validation`](https://github.com/jcartu/qwen36-27b-blackwell-stress-validation) | 5 |

## Schema notes

The two studies use slightly different column conventions because they
measure different things:

### `2026-05-day1-sprint.csv` — throughput grid
```
experiment, build, cell, aggregate_tps, ttft_avg_ms, ttft_p99_ms,
itl_avg_ms, per_user_tps, spec_accept_rate, server_utilization
```

### `2026-05-day2-stress-validation.csv` — correctness + summary
```
config, gates, humaneval_pass, humaneval_total, humaneval_pct,
mbpp_pass, mbpp_total, mbpp_pct, humaneval_empty, humaneval_test_fail,
mbpp_empty, mbpp_test_fail, humaneval_eff_tps, mbpp_eff_tps,
humaneval_p95_s, mbpp_p95_s
```

A **future automated merger** will produce a normalized `master.csv` joining
the two on `(config, hardware)` keys. For now, query each separately.

## Reading from Python

```python
import pandas as pd
day1 = pd.read_csv("data/2026-05-day1-sprint.csv")
day2 = pd.read_csv("data/2026-05-day2-stress-validation.csv")

# Throughput records
print(day1.nlargest(5, "aggregate_tps")[["experiment", "build", "cell", "aggregate_tps"]])

# Correctness records
print(day2.sort_values("humaneval_pct", ascending=False))
```
