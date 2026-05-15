# 2026-05-15 · Single-user thinking-budget addendum

**Scope:** Qwen3.6-27B FP8+MTP=3 on `repne/vllm:v3`, TP=2 on GPUs 0+1.  
**Goal:** keep Qwen3 thinking enabled for coding while eliminating runaway `<think>` loops and tail-latency cliffs in OpenCode.

## Headline

The deployed v3 FP8+MTP=3 engine was not broken. The runaway behavior was an **unbounded thinking budget** problem at the client/request layer.

Setting `thinking_token_budget=2048` per request:

- eliminated stuck responses: **10% -> 0%** on the c=1 hard-coding sweep
- improved pass rate: **70% -> 80%**
- cut p95 latency: **130.8s -> 19.6s** (**6.7x faster tail**)
- cut max reasoning waste: **51k+ chars -> <8.5k chars**
- scaled cleanly from c=1 to c=8 with p50 almost flat: **18.4s -> 19.6s**

The winning production client config is:

```json
{
  "temperature": 0.6,
  "top_p": 0.95,
  "top_k": 20,
  "thinking_token_budget": 2048
}
```

## Why this matters

Qwen3 thinking mode is valuable for coding, so `enable_thinking=false` is the wrong fix. The right fix is hard-limiting the thinking section, using vLLM's `thinking_token_budget` sampling parameter from PR #20859. At 2048 tokens the model keeps the quality benefit of thinking without letting deterministic coding prompts spiral into 50k-character reasoning loops.

## Budget sweep

| condition | thinking mode | budget | pass | stuck | wall |
|---|---:|---:|---:|---:|---:|
| C0 | on | unbounded | 70% | 10% | 592.5s |
| C1 | on | 2048 | 80% | 0% | 171.0s |
| C2 | on | 4096 | 80% | 0% | 305.8s |
| C3 | on | 8192 | 80% | 0% | 466.6s |
| C4 | off | n/a | 70% | 0% | 28.6s |

2048 is the knee: same quality as 4096/8192, much lower latency.

## Concurrency sweep

| config | c | pass | stuck | p50 | p95 | throughput |
|---|---:|---:|---:|---:|---:|---:|
| tb=2048 | 1 | 80% | 0% | 18.43s | 19.57s | 0.054 r/s |
| tb=2048 | 2 | 70% | 0% | 18.77s | 29.65s | 0.095 r/s |
| tb=2048 | 4 | 70% | 0% | 19.52s | 30.06s | 0.142 r/s |
| tb=2048 | 8 | 70% | 0% | 19.60s | 30.32s | 0.197 r/s |
| unbounded | 1 | 80% | 10% | 31.57s | 130.76s | 0.024 r/s |
| unbounded | 4 | 80% | 0% | 38.56s | 64.81s | 0.070 r/s |

## Operational conclusion

- Keep the server stack: `repne/vllm:v3`, FP8, MTP=3, TP=2, FlashInfer, InstantTensor, 256k context.
- Set `thinking_token_budget=2048` in the OpenCode model config per request.
- Do **not** rely on server-side defaults in this vLLM build: `thinking_token_budget` is a `SamplingParams` field, while `--default-chat-template-kwargs` only affects Jinja template rendering, and generation-config overrides are whitelisted to unrelated fields.
- For a single OpenCode user, use the single-user launcher variant in `launchers/`: `max_num_seqs=16`, `max_num_batched_tokens=16384`, `gpu_memory_utilization=0.92`, `max_cudagraph_capture_size=32`.

## Artifacts

- `scripts/coding_probe.py` — five-problem hard coding probe with executable graders.
- `scripts/coding_concurrency_probe.py` — parallel-trial variant for OpenCode-style fanout.
- `results/coding-validation/*/summary.json` — C0-C4 budget summaries.
- `results/coding-concurrency/*/summary.json` — F1/F2 concurrency summaries.
- `launchers/launch-qwen36-27b-sota.single-user.sh` — sanitized proposed production launcher.
