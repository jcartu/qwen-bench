# Draft reply to Repne — thinking-budget generalization study

> Status: **draft**, holding for MMLU-Pro + GPQA sweep completion before sending.
> Final reply will be posted from `jcartu` to the Discord thread / GitHub issue Repne raised.

---

Hey Repne — finished the cross-benchmark generalization study you asked for. Short version: **`thinking_token_budget=2048` is a strict Pareto improvement over unbounded for the Qwen3.6-27B FP8 + MTP=3 single-user stack on every benchmark tested, but the magnitude varies dramatically with workload.** Recommend adopting it as the default in `repne/vllm` for thinking-capable models.

## Headline numbers (n=198 + n=2000 + n=1400 = 3,598 trials)

| Benchmark | n | unbounded acc | **tb=2048 acc** | Δ pp | z | p | unbounded stuck | **tb=2048 stuck** | wall ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GPQA Diamond | 198 | 40.4% | **74.2%** | **+33.8** | 6.81 | 1.0e-11 | 57.1% | 3.0% | 0.24× |
| GSM-Plus 2k | 2000 | 79.6% | **80.8%** | +1.1 | 0.87 | 0.383 | 11.6% | 3.3% | 0.53× |
| MMLU-Pro 1.4k | 1400 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

p-values are two-sided from `math.erfc(|z|/sqrt(2))`. Wilson 95% CIs in the full table.

**Read this as two distinct failure modes**:

- **Hard reasoning (GPQA)** — unbounded fails catastrophically. 57.1% of responses hit `finish_reason=length` while still inside `<think>`. **Zero of those 113 stuck responses emit an extractable answer.** On the 85 unbounded responses that *did* finish cleanly, accuracy was 94.1% — but you only get to that 94% on the easy half.
- **Easy reasoning (GSM-Plus math)** — accuracy is a statistical tie. The model rarely runs away on math chains-of-thought (11.6% stuck even unbounded). The win is purely efficiency: 1.9× faster wall, 2.7× tighter p95 latency, 2.5× tighter p95 token tail.

## Two-knob caveat I want to flag

The comparison moves two knobs simultaneously:
1. **Budget size** — 2,048 vs effectively 16,384 (`max_tokens`).
2. **Forced-commit mechanism** — `MaxThinkTokensLogitsProcessor` (your PR #20859) injects `</think>` at the budget. Unbounded has no such mechanism, it just bumps into `finish_reason=length` and returns whatever fragment it has.

So the strict reading is: "tb=2048 with forced-commit beats unbounded@max_tokens=16384 without forced-commit." To disentangle these, I ran a GPQA-Diamond budget sweep at tb ∈ {1024, 4096, 8192} (results below). Bottom line: _TBD after sweep lands_.

## How to reproduce

```bash
# repne/vllm:v3 endpoint, served name "qwen3.6-27b", running on 2×GPU TP=2, MTP=3, FP8
curl http://127.0.0.1:11435/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6-27b",
    "messages": [{"role":"user","content":"<your prompt>"}],
    "temperature": 0.6, "top_p": 0.95, "top_k": 20, "repetition_penalty": 1.05,
    "max_tokens": 16384,
    "chat_template_kwargs": {"enable_thinking": true},
    "thinking_token_budget": 2048
  }'
```

Full harness, datasets loaders, all 5 (or 8 with sweep) condition summaries, per-trial JSONL, raw stdout logs:

- Study page: `https://huggingface.co/datasets/jcartu/qwen-bench/blob/main/studies/2026-05-15-thinking-budget-generalization/README.md`
- Raw JSONL trials: `studies/2026-05-15-thinking-budget-generalization/results/*.jsonl`
- Aggregate.{md,csv,json}: `studies/2026-05-15-thinking-budget-generalization/results/`
- Harness source: `studies/2026-05-15-thinking-budget-generalization/scripts/bench.py`
- Launcher: `studies/2026-05-15-thinking-budget-generalization/launchers/run_fast_phase.sh`

## What I think this means for repne/vllm defaults

1. **Set `thinking_token_budget=2048` as default for all thinking-capable Qwen3.6 models.** No accuracy regression on any tested benchmark, large accuracy gain on hard reasoning, ~2× throughput improvement.
2. **Server-side default flag is still missing** in upstream vLLM — `--default-chat-template-kwargs` doesn't yet learn `thinking_token_budget`. We hardcode it client-side in OpenCode for now. If you want a one-line PR target, that's it.
3. **Higher budgets remain valuable for benchmark scoring** — if a user is willing to pay 6× latency for the +20pp peak on solvable hard problems, tb=4096–8192 may close the gap. _Confirmed/refuted in the sweep section above._

## Honest caveats

- Single seed (42) across all runs. Variance from temp=0.6 sampling is uncharacterized.
- One model size + quant (Qwen3.6-27B FP8). Probably transfers to other Qwen3.6 sizes; doesn't speak to non-Qwen3 thinking models.
- GSM-Plus `critical_thinking` perturbation subset (n=250, 12.5% of dataset) tests unanswerable questions with gold=`None`. Our numeric grader cannot credit "no answer" detection; both conditions score 0% on this subset. Doesn't affect the C0/C1 comparison (same bias on both), but I report the ex-critical numbers separately.
- All runs at `max_tokens=16384` and concurrency=8 (single-user ceiling for this stack). Multi-user fleet behavior is not measured.

Happy to discuss the sweep results in detail, run additional conditions, or hand over the harness if you want to drop it into your CI.

— Sisyphus / jcartu
