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
| `2026-05-dflash-v2-sweep.csv` | [`qwen-bench-2026-05-dflash-v2-sweep`](https://github.com/jcartu/qwen-bench-2026-05-dflash-v2-sweep) | 15 |
| `2026-05-11-v2-followup.csv` | [`qwen-bench-2026-05-11-v2-followup`](https://github.com/jcartu/qwen-bench-2026-05-11-v2-followup) | 4 |
| `2026-05-12-v3-suite.csv` | [`qwen-bench-2026-05-12-v3-suite`](https://github.com/jcartu/qwen-bench-2026-05-12-v3-suite) | 4 |
| `2026-05-15-single-user-thinking-budget.csv` | [`studies/2026-05-15-single-user-thinking-budget`](../studies/2026-05-15-single-user-thinking-budget/) | 11 |

> The `2026-05-12-v3-suite.csv` `production_status` column captures the post-bench leak-probe verdict per config (DEPLOYED-PRODUCTION-SOTA, DO-NOT-DEPLOY-think-token-leak, benchmark-only). Full per-trial leak-probe artifacts (JSONL + summary.json) live under [`leak-runs/`](https://github.com/jcartu/qwen-bench-2026-05-12-v3-suite/tree/main/leak-runs) in the study repo — they are not benchmark grid measurements and are kept outside this CSV by design.

## Schema notes

The two studies use slightly different column conventions because they
measure different things:

### `2026-05-day1-sprint.csv` — throughput grid
```
experiment, build, cell, aggregate_tps, ttft_avg_ms, ttft_p99_ms,
itl_avg_ms, per_user_tps, spec_accept_rate, server_utilization
```

### `2026-05-15-single-user-thinking-budget.csv` — client-side thinking budget
```
phase, condition, config, thinking_enabled, thinking_token_budget, concurrency,
n, pass_rate, stuck_rate, wall_s, p50_latency_s, p95_latency_s,
throughput_req_per_s, max_reasoning_chars, notes
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
thinking = pd.read_csv("data/2026-05-15-single-user-thinking-budget.csv")

# Throughput records
print(day1.nlargest(5, "aggregate_tps")[["experiment", "build", "cell", "aggregate_tps"]])

# Correctness records
print(day2.sort_values("humaneval_pct", ascending=False))

# Thinking-budget winner
print(thinking.sort_values(["stuck_rate", "p95_latency_s"], na_position="last").head())
```

| `2026-05-15-thinking-budget-generalization.csv` | 2026-05-15 | Cross-domain `thinking_token_budget=2048` vs unbounded on Qwen3.6-27B FP8+MTP=3 on `repne/vllm:v3`. GPQA Diamond + GSM-Plus 2k + MMLU-Pro 1.4k + GPQA budget sweep at tb={1024,4096,8192}. 3,598 trials. | [study](../studies/2026-05-15-thinking-budget-generalization/) |
