# Reply to Repne — thinking-budget generalization study

Hey Repne — finished the cross-benchmark generalization study you asked for.

**Short version**: `thinking_token_budget=2048` is a strict Pareto improvement over unbounded thinking for the Qwen3.6-27B FP8 + MTP=3 single-user stack on every benchmark tested, with very large gains on hard reasoning workloads and a pure efficiency win on easy ones. **Recommend adopting it as the default in `repne/vllm` for thinking-capable models.**

## Headline numbers (n = 198 + 2,000 + 1,400 = 3,598 trials)

| Benchmark      |     n | unbounded acc | **tb=2048 acc** |     Δ pp |    z |       p | unbounded stuck | **tb=2048 stuck** | wall ratio |
|----------------|------:|--------------:|----------------:|---------:|-----:|--------:|----------------:|------------------:|-----------:|
| GPQA Diamond   |   198 |         40.4% |       **74.2%** | **+33.8**| 6.81 | 1.0e-11 |           57.1% |              3.0% |      0.24× |
| GSM-Plus 2k    | 2,000 |         79.6% |       **80.8%** |     +1.1 | 0.87 | 0.383   |           11.6% |              3.3% |      0.53× |
| MMLU-Pro 1.4k  | 1,400 |         73.1% |       **82.6%** | **+9.5** | 6.06 | 1.4e-09 |           16.5% |              1.3% |      0.32× |

p-values are two-sided from `math.erfc(|z|/sqrt(2))`. Wilson 95% CIs in the full aggregate table.

**Read this as three distinct workload regimes**:

- **Hard reasoning (GPQA Diamond)** — unbounded fails catastrophically. 57.1% of responses hit `finish_reason=length` while still inside `<think>`. **Zero of those 113 stuck responses emit an extractable answer.** On the 85 unbounded responses that *did* finish cleanly, accuracy was 94.1% — but you only get to that 94% on the easy half. tb=2048 forces commit on all 198, lands at 74.2%.
- **Mixed hard/easy (MMLU-Pro)** — similar mechanism, smaller magnitude. Unbounded stuck rate 16.5%; non-stuck accuracy is actually 87.6% (slightly above tb=2048's 82.6%), but the aggregate is dragged down 9.5 pp by the 231 length-cap failures. Cleanest illustration that the dominant axis is **forced-commit**, not "thinking quality".
- **Easy reasoning (GSM-Plus math)** — accuracy is a statistical tie. The model rarely runs away on math chains-of-thought (11.6% stuck even unbounded, and many of those still produce a parseable final answer before the cap). The win here is purely efficiency: 1.9× faster wall, 2.7× tighter p95 latency, 2.5× tighter p95 token tail.

## Two-knob disentanglement — GPQA budget sweep (n = 198 each)

The headline comparison moves two knobs simultaneously: budget size (2,048 vs ~16,384) **and** the forced-commit mechanism (`MaxThinkTokensLogitsProcessor` from your PR #20859 vs none). To isolate which one matters I ran a GPQA-Diamond sweep across four budgets, all with forced-commit active, all `max_tokens=16384`:

| Condition       | tb     | accuracy   | stuck rate | p50 latency |
|-----------------|-------:|-----------:|-----------:|------------:|
| sweep           | 1,024  |     77.3%  |       2.5% |       20.5s |
| **fast-phase**  | 2,048  |     74.2%  |       3.0% |       31.2s |
| sweep           | 4,096  |     78.3%  |       3.5% |       54.1s |
| sweep           | 8,192  |     78.3%  |       2.5% |      110.5s |
| **fast-phase**  | none*  |     40.4%  |      57.1% |      196.9s |

*"none" = `max_tokens=16384` with no `MaxThinkTokensLogitsProcessor`, i.e. the unbounded condition.

**Verdict**: all four bounded conditions sit in a tight 74–78% plateau (pairwise differences within sampling noise at n=198; e.g. tb=2048 vs tb=8192 is z≈1.0, p≈0.32). The unbounded condition is the only outlier and it crashes 33.8 pp below the lowest bounded point. **The dominant effect is forced-commit, not budget size.** Once the logits processor is in place to inject `</think>` at the budget, even 1,024 tokens of think suffices on GPQA. Below ~1k you'd probably see degradation; above ~2k you're just paying latency for no accuracy gain.

This also means **the practical default is robust**: tb=2048 isn't on a knife-edge. tb=1024 through tb=8192 all work; 2,048 is just the sweet spot for latency.

## How to reproduce

```bash
# repne/vllm:v3 endpoint, served name "qwen3.6-27b", 2× GPU TP=2, MTP=3, FP8
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

Full harness, dataset loaders, all 9 condition summaries, per-trial JSONL, raw stdout logs:

- Study page: <https://github.com/jcartu/qwen-bench/blob/main/studies/2026-05-15-thinking-budget-generalization/README.md>
- Aggregate `{md,csv,json}`: <https://github.com/jcartu/qwen-bench/tree/main/studies/2026-05-15-thinking-budget-generalization/results>
- Raw per-trial JSONL (9 files, one per condition): `studies/2026-05-15-thinking-budget-generalization/results/raw/`
- Harness source: `studies/2026-05-15-thinking-budget-generalization/scripts/bench.py`
- Launcher: `studies/2026-05-15-thinking-budget-generalization/scripts/run_fast_phase.sh`

Commit: `b50b3db` on `jcartu/qwen-bench@main`.

## What I think this means for `repne/vllm` defaults

1. **Set `thinking_token_budget=2048` as the default for all thinking-capable Qwen3.6 models.** No accuracy regression on any tested benchmark, +9.5 to +33.8 pp gain on hard reasoning, ~2–3× throughput improvement.
2. **Server-side default flag is still missing in upstream vLLM** — `--default-chat-template-kwargs` doesn't yet learn `thinking_token_budget`. We hardcode it client-side in OpenCode for now. If you want a one-line PR target, that's it.
3. **Higher budgets don't recover accuracy** — the sweep refutes my own earlier hypothesis that 4k–8k might close the gap. The plateau is real. Save the latency.

## Honest caveats

- Single seed (`SEED=42`) across all runs. Variance from `temperature=0.6` sampling is uncharacterized; would need ≥3 seeds to put error bars on the within-condition spread.
- One model size + quant (Qwen3.6-27B FP8 + MTP=3). Probably transfers to other Qwen3.6 sizes; doesn't speak to non-Qwen3 thinking models.
- GSM-Plus `critical_thinking` perturbation subset (n=250, 12.5% of dataset) tests unanswerable questions with gold=`None`. The numeric grader cannot credit "no answer" detection; both conditions score 0% on this subset. Doesn't affect the tb=2048 vs unbounded comparison (same bias on both), but it pulls the headline number down ~10 pp relative to ex-critical_thinking.
- All runs at `max_tokens=16384` and `concurrency=8` (single-user ceiling for this stack). Multi-user fleet behavior is not measured.

Happy to discuss the sweep results in detail, run additional conditions (other Qwen3.6 sizes, AIME, other thinking models), or hand the harness over for your CI.

— Sisyphus / jcartu
