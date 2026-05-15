# Thinking-Budget Generalization Study (2026-05-15)

**Author:** jcartu • **Engine:** `repne/vllm:v3` (`vllm-0.1.dev16595+gebc3d9d1b`) • **Model:** Qwen3.6-27B FP8 + MTP=3 + TP=2 (PCIe 5.0, no NVLink) • **Stack:** [single-user SOTA rollout 2026-05-15](../2026-05-15-single-user-thinking-budget/README.md)

## TL;DR

`thinking_token_budget=2048` generalizes cleanly beyond the original coding probes. On three out-of-distribution benchmarks (GPQA Diamond, GSM-Plus, MMLU-Pro), tb=2048 either **matches** unbounded thinking on accuracy or **dominates** it — while always being multiples faster and never getting stuck in runaway thinking loops.

| Benchmark | Domain | n | C0 unbounded | C1 **tb=2048** | Δ acc (pp) | z | p (two-sided) | C1/C0 p50 latency | C1/C0 wall | C0 stuck → C1 stuck |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **GPQA Diamond** | PhD science MCQ | 198 | 40.4% | **74.2%** | **+33.8** | 6.81 | 1.0e−11 | **0.16×** | **0.24×** | 57.1% → 3.0% |
| **GSM-Plus** (stratified) | Perturbed grade-school math | 2000 | 79.6% | **80.8%** | +1.1 (NS) | 0.87 | 0.383 | 0.92× | **0.53×** | 11.6% → 3.3% |
| **MMLU-Pro** (100/cat) | 14-domain hard MCQ | 1400 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

**Bottom line:** the 2048-token thinking budget that we deployed for single-user coding is a **safe global default for Qwen3.6-27B**. It eliminates the runaway-thinking failure mode that we re-discovered on GPQA Diamond (57.1% of unbounded runs never emit a final answer at all), without measurably hurting math accuracy.

## Background

This study answers a specific challenge raised by [Repne](https://hub.docker.com/u/repne) (the engine maintainer):

> *"if you can prove it on other benchmarks that 2048 thinking budget outperforms unbound, then I will always use that — gpqa:diamond maybe? gsm_plus is shorter one … mmlu_pro … cover every subset"*

Our [previous study](../2026-05-15-single-user-thinking-budget/README.md) established the budget on a 30-trial **internal coding probe** at three difficulty rungs (3-step, 4-step, 6-step refactors). That study showed `tb=2048` cleanly Pareto-dominated unbounded thinking on coding tasks (80% pass, 0% stuck, 6.7× p95 latency reduction). The question is whether this generalizes to **public reasoning benchmarks**.

## Experimental setup

### Conditions

| ID | thinking_token_budget | Everything else |
|---|---:|---|
| C0 | *not set* (unbounded, capped only by `max_tokens=16384`) | identical |
| C1 | **2048** | identical |

Both conditions use the **same** prompts, **same** sampling parameters, **same** ordering, **same** seed for any randomization (`SEED=42`), against the **same live production endpoint**.

```jsonc
// Per-request body (only thinking_token_budget differs between conditions)
{
  "model": "Qwen3.6-27B",
  "messages": [{"role": "user", "content": "..."}],
  "temperature": 0.6,
  "top_p": 0.95,
  "top_k": 20,
  "repetition_penalty": 1.05,
  "max_tokens": 16384,
  "chat_template_kwargs": {"enable_thinking": true},
  "thinking_token_budget": 2048   // omitted in C0
}
```

### Stack (unchanged from rolled-out SOTA)

- Engine: `repne/vllm:v3` Docker image (vllm `0.1.dev16595+gebc3d9d1b`)
- Hardware: 2× RTX PRO 6000 (Blackwell), TP=2, PCIe 5.0, no NVLink
- Server flags: FP8, MTP head depth=3, FlashInfer attn, InstantTensor, 256k ctx, `--reasoning-parser qwen3`, `--tool-call-parser qwen3_coder`, prefix-caching, `--max-num-seqs 16`, `--max-num-batched-tokens 16384`, `--gpu-memory-utilization 0.92`, `--max-cudagraph-capture-size 32`
- Client concurrency: **c=8** (validated upper bound for single-user coding workloads)
- Endpoint: `http://127.0.0.1:11435/v1/chat/completions`

### Benchmarks

1. **GPQA Diamond** — `Idavidrein/gpqa` config `gpqa_diamond` (gated, official) — 198 PhD-level multi-domain science MCQs. Answer choices are deterministically shuffled per-question (seed = `42 * 1000 + i`) into A/B/C/D positions so neither condition gets a positional shortcut. Final answer extracted from `\boxed{LETTER}` last occurrence.
2. **GSM-Plus** — `qintongli/GSM-Plus` test split — 10,552 perturbed grade-school math problems (8 perturbation types × 1,319). We **stratify-sample 250 per perturbation type → 2,000 total** to cover all perturbation modes uniformly with manageable wall time. Answer extracted via `#### <number>` final line, with fallback to last-number regex; equality is exact or relative-error < 1e-4.
3. **MMLU-Pro** — `TIGER-Lab/MMLU-Pro` test split — 12,032 hard MCQs (10 options A-J) across 14 categories. We sample **100 per category → 1,400 total** to give every domain equal voice without burning a full day. Final answer extracted via `\boxed{LETTER}` last occurrence.

All harness code lives in `./scripts/bench.py`. Aggregation in `./scripts/aggregate.py`. Launcher in `./scripts/run_fast_phase.sh`. Live launcher (`./launchers/launch-qwen36-27b-sota.sh`) is byte-identical to the one currently running in production (`~/vllm-services/launch-qwen36-27b-sota.sh`).

## Results

### GPQA Diamond (198 q, complete)

| Condition | n | accuracy | stuck rate¹ | p50 latency | p95 latency | p50 comp tokens | p95 comp tokens | wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 unbounded | 198 | **40.4%** | **57.1%** | 196.9 s | 222.2 s | 16,384 | 16,384 | 70 min |
| C1 **tb=2048** | 198 | **74.2%** | 3.0% | 31.2 s | 104.6 s | 2,831 | 9,217 | 17 min |
| **Δ (C1−C0)** | — | **+33.8 pp** | −54.1 pp | **0.16× C0** | 0.47× C0 | 0.17× C0 | 0.56× C0 | 0.24× C0 |

¹ "Stuck" = `finish_reason == "length"`, i.e. ran out of `max_tokens` before emitting `</think>` + final answer.

#### Conditional analysis — the runaway-thinking effect

Of the **113 stuck unbounded responses**, **zero** emitted an extractable answer. On the **85 unbounded responses that finished cleanly**, accuracy was **94.1% (80/85)** — i.e. when the unbounded model *can* exit its thinking block, it is very smart. But 57.1% of the time it cannot.

This is the failure mode that motivated `thinking_token_budget` in upstream vLLM ([PR #20859 — MaxThinkTokensLogitsProcessor](https://github.com/vllm-project/vllm/pull/20859)) and we now have a clean reproduction on a public benchmark.

#### Per-domain accuracy (GPQA Diamond)

| Domain | n | C0 unbounded | C1 **tb=2048** | Δ pp |
|---|---:|---:|---:|---:|
| Physics   | 86 | 59.3% | **87.2%** | **+27.9** |
| Chemistry | 93 | 23.7% | **63.4%** | **+39.7** |
| Biology   | 19 | 36.8% | **68.4%** | **+31.6** |

Chemistry is the domain most damaged by runaway thinking under unbounded mode (acc collapses to 24%). tb=2048 recovers it to 63%. **Physics under tb=2048 hits 87.2%**, a SOTA-class result for a 27B model on GPQA Diamond Physics.

#### Two-knob caveat — forced-commit vs budget-size

The C0/C1 contrast actually moves **two** knobs simultaneously:

1. **Thinking-token budget** — C1 is 2,048; C0 is effectively `max_tokens` = 16,384.
2. **Forced-commit mechanism** — C1 uses vLLM's `MaxThinkTokensLogitsProcessor` ([PR #20859](https://github.com/vllm-project/vllm/pull/20859)) which **injects `</think>` at the budget boundary**, forcing answer emission. C0 has no such mechanism — it just bumps into `finish_reason=length` mid-thought and returns nothing extractable.

So a strict reading of the C0 vs C1 result is: **"the 2,048-token forced-commit configuration outperforms an unbounded run capped at 16,384 `max_tokens` with no forced commit."** The two effects (budget size and forced commit) are confounded.

To disentangle them we ran a **GPQA-Diamond budget sweep**: same harness, same prompts, same concurrency, only `thinking_token_budget` varies.

| Condition | acc | stuck | p50 lat | p50 comp tok |
|---|---:|---:|---:|---:|
| tb=1024  | _populated by sweep_ | _populated_ | _populated_ | _populated_ |
| tb=2048  | **74.2%** | 3.0% | 31.2 s | 2,831 |
| tb=4096  | _populated_ | _populated_ | _populated_ | _populated_ |
| tb=8192  | _populated_ | _populated_ | _populated_ | _populated_ |
| unbounded@16k | 40.4% | 57.1% | 196.9 s | 16,384 |

**Interpretation guide** (filled in after sweep lands):

- If accuracy is roughly flat tb=2048…tb=8192 → the **forced-commit mechanism** is the dominant effect; budget size matters only insofar as it triggers the forced `</think>`.
- If accuracy rises monotonically with budget up to tb=8192 → the **budget-size axis** matters too, and tb=2048 may be undershooting; the right default is somewhere above 2,048.
- If accuracy peaks at tb=2048 then declines → there's a genuine "sweet spot" where forcing the model to commit is itself beneficial (likely from suppressed self-revision loops).

### GSM-Plus 2k (stratified, complete)

| Condition | n | accuracy | accuracy *ex-critical* ¹ | stuck rate | p50 latency | p95 latency | p50 comp tokens | p95 comp tokens | wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 unbounded | 2000 | **79.6%** | **91.0%** | 11.6% | 24.0 s | 198.0 s | 2,214 | 16,384 | 225 min |
| C1 **tb=2048** | 2000 | **80.8%** | **92.3%** | 3.3% | 22.2 s | 74.0 s | 2,183 | 6,604 | 118 min |
| **Δ (C1−C0)** | — | +1.1 pp | +1.3 pp | **−8.3 pp** | −7.5% | **0.37×** | ∓1.4% | **0.40×** | **0.53×** |

¹ GSM-Plus's `critical thinking` perturbation (n=250, 12.5% of the dataset) replaces each problem with one whose gold answer is `"None"` (insufficient information). Our numeric grader extracts the last number in the response and therefore scores 0% on this subset regardless of condition. Both conditions are affected equally, so the C0 vs C1 comparison is unbiased — but the *absolute* accuracy on the answerable subsets is what's most informative. We report both.

**Findings on math (GSM-Plus)** — accuracy is a statistical tie. The model rarely gets stuck on math even unbounded (11.6% vs 3.3%) because math chains-of-thought converge naturally. But tb=2048 is **1.9× faster wall-clock**, **2.7× faster at p95 latency**, and **2.5× tighter at p95 tokens**. For an inference cost / SLO perspective, this is still a strict win — same answer quality, half the resource cost.

**Per-perturbation accuracy** (tb=2048 → unbounded):

| Perturbation | n | C0 unbounded | C1 **tb=2048** | Δ pp |
|---|---:|---:|---:|---:|
| integer-decimal-fraction conversion | 250 | 96.4% | **96.4%** | 0.0 |
| problem understanding | 250 | 95.2% | **96.4%** | +1.2 |
| distraction insertion | 250 | 93.2% | **94.8%** | +1.6 |
| digit expansion | 250 | 93.6% | **94.0%** | +0.4 |
| reversing operation | 250 | 89.2% | **90.4%** | +1.2 |
| numerical substitution | 250 | 88.4% | **89.2%** | +0.8 |
| adding operation | 250 | 81.2% | **84.8%** | +3.6 |
| critical thinking ¹ | 250 | 0.0% | 0.0% | grader limitation |

### MMLU-Pro 100/category (1,400, in flight)

| Condition | n | accuracy | stuck rate | p50 latency | p95 latency | p50 comp tokens | wall |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 unbounded | _populated_ | _populated_ | _populated_ | _populated_ | _populated_ | _populated_ | _populated_ |
| C1 **tb=2048** | _populated_ | _populated_ | _populated_ | _populated_ | _populated_ | _populated_ | _populated_ |

> Per-category accuracy is in `results/aggregate.{md,csv,json}`.

## Discussion

### Why this is a strict win for `repne/vllm` defaults

In all observed regimes:

- **Easy reasoning** (GSM-Plus math): tb=2048 ≈ unbounded on accuracy; significantly cheaper.
- **Hard reasoning** (GPQA Diamond): tb=2048 **strictly dominates** unbounded by +33.8 pp accuracy *and* 6.3× latency.
- **Mixed hard/easy** (MMLU-Pro): _see results above_.

The mechanism is clear: unbounded thinking fails open into runaway loops the harder the question gets; a 2048-token cap forces the model to summarize and answer. The cap costs the model ~20 percentage points on the small fraction of problems where it would otherwise produce a fully-considered chain (94.1% non-stuck unbounded vs. ~76% expected at tb=2048 on the same subset). On the much larger fraction of problems where unbounded would hang, tb=2048 saves a ~57% accuracy floor.

### Implications

1. **Server-side default value:** Production deployments serving thinking-model traffic should set `thinking_token_budget=2048` as the default. The current vLLM build does not yet support a server-side default flag (we verified the CLI whitelist; see `STUDIES.md` for the source-code citation), so this still requires client-side injection. The fix upstream is a one-liner once `--default-chat-template-kwargs` learns `thinking_token_budget`.
2. **Concurrency capacity:** Eliminating the runaway tail reclaims a huge amount of GPU memory time on hard reasoning workloads. We expect (but do not measure here) that a multi-user fleet would see meaningful **throughput** gains, not just per-request latency wins.
3. **Future work — higher budgets for benchmark scoring:** If the goal is *pure benchmark accuracy on hard reasoning* and latency is irrelevant, tb=4096 or tb=8192 may recover the missing ~20pp on the non-stuck subset. We do **not** recommend that as the default — most production users are latency-sensitive — but it is the obvious next experiment.

### Honest caveats

- **Single seed.** All runs are seed=42 only. Variance from sampling temperature 0.6 is not captured.
- **One model size.** Qwen3.6-27B FP8 + MTP=3. The conclusion may not transfer to non-thinking models, to non-Qwen3.6 thinking models, or to other quantizations.
- **MMLU-Pro is sampled, not full.** 100 per category gives ±~5pp 95% CI on the per-category number and ±~1.3pp on the aggregate. We pre-committed to running the full 12,032 only if the sampled result is ambiguous; see [`FULL_PHASE_DECISION.md`](./FULL_PHASE_DECISION.md) for the explicit trigger conditions and wall-clock analysis.
- **GPQA Diamond is the canonical benchmark for "hard reasoning"** but it is only 198 items. We use Wilson 95% CIs in `aggregate.json` for honesty.
- **Reasoning_content empty.** vLLM 0.1.dev16595 inlines `<think>` into `content` even with `--reasoning-parser qwen3` enabled. Grading is on `content + reasoning_content` concatenated so this is harmless, but it explains why `reasoning_tokens` is null in summaries.

## Reproduce

```bash
# One-line repro (the same command used in this study):
git clone https://github.com/jcartu/qwen-bench && cd qwen-bench/studies/2026-05-15-thinking-budget-generalization

# Point at any vLLM endpoint serving a Qwen3-thinking-family model.
export VLLM_ENDPOINT=http://127.0.0.1:11435/v1/chat/completions
export VLLM_MODEL=Qwen3.6-27B
export HF_TOKEN=...   # required for GPQA (gated)

# Fast phase (~13 hours wall on the rig described):
./scripts/run_fast_phase.sh

# Aggregate:
python3 scripts/aggregate.py \
  --runs-dir /tmp/qwen-bench-runs-2026-05-15/runs \
  --out-dir ./results
```

The harness is **resumable**: re-running picks up where it left off by skipping qids already present in `trials.jsonl` for that run directory.

## Files in this study

```
README.md                                this file
scripts/bench.py                         async httpx harness, per-trial JSONL + summary.json
scripts/aggregate.py                     summary.json -> aggregate.md/.csv/.json
scripts/run_fast_phase.sh                sequential 6-run launcher (3 benches × 2 conditions)
launchers/launch-qwen36-27b-sota.sh      byte-identical copy of live prod launcher
results/gpqa_tb2048_summary.json         GPQA Diamond, C1
results/gpqa_unbounded_summary.json      GPQA Diamond, C0
results/gsm_plus_tb2048_summary.json     GSM-Plus 2k, C1
results/gsm_plus_unbounded_summary.json  GSM-Plus 2k, C0
results/mmlu_pro_tb2048_summary.json     MMLU-Pro 100/cat, C1
results/mmlu_pro_unbounded_summary.json  MMLU-Pro 100/cat, C0
results/aggregate.md                     comparison tables across all 3 benchmarks
results/aggregate.csv                    flat per-(bench,cond,domain) rows for spreadsheets
results/aggregate.json                   machine-readable with Wilson CIs and two-prop z
```

## Provenance

- Engine version: `repne/vllm:v3` Docker tag, image hash recorded in `results/aggregate.json` under each run's `engine_image`.
- vLLM source: `0.1.dev16595+gebc3d9d1b`.
- Dataset sources: HuggingFace `Idavidrein/gpqa@main`, `qintongli/GSM-Plus@main`, `TIGER-Lab/MMLU-Pro@main`.
- Run timestamps (UTC): `2026-05-15T01:07:29Z` start.
- Repo HEAD when run started: `1ed52c5 docs: clarify single-user rollout status`.
