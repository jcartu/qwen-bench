# Qwen-Bench · Studies Index

Chronological log of every benchmark study indexed by this hub. Newest first.

For headline numbers across all studies, see [SOTA.md](SOTA.md).
For the merged machine-readable result tables, see [`data/`](data/).

---

## 2026-05-15 · Thinking-budget generalization study (evening)

**Repo:** hub-native study at [`studies/2026-05-15-thinking-budget-generalization`](studies/2026-05-15-thinking-budget-generalization/)
**Trigger:** Repne (engine maintainer) asked whether the `thinking_token_budget=2048` default
from the morning's single-user rollout would generalize beyond coding probes to public reasoning
benchmarks before he adopts it as the `repne/vllm` default.
**Hardware:** 2× NVIDIA RTX PRO 6000 Blackwell (TP=2, GPUs 0+1; GPU 2 reserved for Hindsight, untouched)
**Server:** `repne/vllm:v3` (`vllm-0.1.dev16595+gebc3d9d1b`), `Qwen/Qwen3.6-27B-FP8`, MTP=3,
FlashInfer, InstantTensor, 256k context
**Datasets (all public):** `Idavidrein/gpqa@gpqa_diamond` (198 q, gated, accepted),
`qintongli/GSM-Plus` (2,000 stratified across 8 perturbation types), `TIGER-Lab/MMLU-Pro`
(1,400 = 100/category × 14 categories)
**Conditions per benchmark:** C0 unbounded thinking @ `max_tokens=16384`; C1 `thinking_token_budget=2048`.
GPQA also adds a budget sweep at tb ∈ {1024, 4096, 8192}.

### Abstract

`thinking_token_budget=2048` is a **strict Pareto improvement** over unbounded thinking on
Qwen3.6-27B FP8 + MTP=3 across every benchmark tested:

- **GPQA Diamond**: 40.4% → **74.2%** accuracy (+33.8 pp, z=6.81, p=1e-11); 196.9s → 31.2s p50 latency.
  57.1% of unbounded responses hit `finish_reason=length` inside `<think>` and emit zero answer.
- **GSM-Plus 2k**: 79.6% → **80.8%** accuracy (+1.1 pp, NS); but **1.9× faster wall-clock**,
  **2.7× tighter p95 latency**, **2.5× tighter p95 token tail**, stuck rate 11.6% → 3.3%.
- **MMLU-Pro**: 73.1% → **82.6%** accuracy (+9.5 pp, z=6.06, p=1.4e-09); 69.8s → 27.2s p50 latency; stuck rate 16.5% → 1.3%.

A GPQA budget sweep at tb ∈ {1024, 2048, 4096, 8192} shows accuracy peaks at tb=4096 (acc=78.3%) → there is a **genuine sweet spot** in budget choice, disambiguating
the "forced-commit-mechanism vs budget-size" confound inherent in the two-condition design.

The mechanism is the same one [PR #20859](https://github.com/vllm-project/vllm/pull/20859)
shipped (`MaxThinkTokensLogitsProcessor`): forcing `</think>` insertion at the budget boundary
guarantees answer emission, whereas unbounded responses just bump into `finish_reason=length`
and return mid-thought fragments. This is the same mechanism the **morning single-user
study** ([`2026-05-15-single-user-thinking-budget`](studies/2026-05-15-single-user-thinking-budget/))
established on 30 internal coding probes; the present study confirms it generalizes to public
reasoning benchmarks with N=3,598 trials.

**Recommendation:** `thinking_token_budget=2048` should be the default for thinking-capable
Qwen3.6 models in `repne/vllm`. Server-side default flag is missing in upstream vLLM
(`--default-chat-template-kwargs` doesn't yet learn `thinking_token_budget`) — currently
hardcoded client-side in OpenCode.

## 2026-05-15 · Single-user thinking-budget addendum — OpenCode production tuning

**Repo:** hub-native addendum at [`studies/2026-05-15-single-user-thinking-budget`](studies/2026-05-15-single-user-thinking-budget/)
**Trigger:** production OpenCode traffic showed runaway `<think>` reasoning loops on hard coding prompts despite the v3 FP8+MTP=3 engine being leak-free.
**Hardware:** 2× NVIDIA RTX PRO 6000 Blackwell (TP=2, GPUs 0+1; GPU 2 reserved for Hindsight, untouched)
**Server:** `repne/vllm:v3` (vllm-0.1.dev16595+gebc3d9d1b), `Qwen/Qwen3.6-27B-FP8`, MTP=3, FlashInfer, InstantTensor, 256k context
**Configs measured:** unbounded thinking, `thinking_token_budget` ∈ {2048, 4096, 8192}, and no-thinking floor; concurrency c ∈ {1,2,4,8} for the winning budget.
**Harness:** five hard coding tasks with executable graders plus a parallel-trial fanout variant.

### Abstract

This production addendum separates engine stability from client-side reasoning control. Rolling back between Repne v4 and v3 did not fix the issue: the runaway was a Qwen3 thinking-mode loop under unbounded request budgets, not a parser leak or vLLM engine regression. The vLLM `thinking_token_budget` sampling parameter hard-stops the thinking section by forcing the end-of-thinking token once the budget is exhausted.

### Headline result

- **Winning client config: `thinking_token_budget=2048`** with Qwen3 thinking-coding sampling (`temperature=0.6`, `top_p=0.95`, `top_k=20`).
- **Quality improved:** 70% → **80%** pass on the hard-coding budget sweep.
- **Stuck responses eliminated:** 10% → **0%**.
- **Tail latency collapsed:** c=1 p95 130.8s → **19.6s** (6.7× faster).
- **Fanout is safe:** p50 stayed essentially flat from c=1 to c=8 (18.4s → 19.6s), so OpenCode subagent parallelism does not need an artificial cap below normal fanout.
- **Server-side default is unavailable in this vLLM build:** `thinking_token_budget` must be sent per request; `--default-chat-template-kwargs` cannot set it.

### Records broken / clarifications

- First hub-indexed hard-coding benchmark for Qwen3 thinking-budget control.
- Clarifies that the production v3 FP8+MTP=3 stack remains the right engine choice; the single-user fix is client-side request budgeting plus a smaller production launcher profile.
- Establishes `2048` as the recommended OpenCode thinking budget for Qwen3.6-27B coding on this stack.

---

## 2026-05-12 · Qwen3.6-27B Tier-3 v3 full suite — `repne/vllm:v3` validation

**Repo:** [`qwen-bench-2026-05-12-v3-suite`](https://github.com/jcartu/qwen-bench-2026-05-12-v3-suite)
**Trigger:** Repne shipped `repne/vllm:v3`. Need to validate all 4 spec-decoding configs against v2 baselines before promoting to `:latest`.
**Hardware:** 2× NVIDIA RTX PRO 6000 Blackwell (TP=2, GPUs 0+1; GPU 2 reserved for Hindsight, untouched)
**Models:** `Qwen/Qwen3.6-27B` (BF16), `Qwen/Qwen3.6-27B-FP8`; drafter `z-lab/Qwen3.6-27B-DFlash`
**Server:** `repne/vllm:v3` (digest `fd2f7b567b19`, 29.7 GB)
**Configs measured:** BF16+DFlash N=8, FP8+DFlash N=8, FP8+MTP=3, FP8+MTP=5 (4 × full suite: gates + throughput sweep + prefill sweep + HE-164 + MBPP-257)
**Harness:** patched stress harness with `smart_glue_humaneval`, mt=16384, hardened launcher with per-phase log snapshots
**Wall time:** 3 h 32 min (03:59:19 → 07:31:32 MSK)

### Abstract
Full 4-configuration stress-validation suite on Repne's `:v3` image to determine whether to promote it as the new online SOTA. Mirrors the v2-followup Tier 2 methodology (mt=16384, patched harness) but adds FP8+DFlash N=8 and FP8+MTP=5 alongside the BF16+DFlash N=8 and FP8+MTP=3 baselines. Run 1 of Config 1 (BF16+DFlash) crashed mid-HumanEval with `EngineCore encountered an issue` (HTTP 500); launcher was hardened to capture per-phase docker logs; Run 2 of all 4 configs completed clean with zero engine errors.

### Headline result
- **`:v3` improves over `:v2` cleanly on every matching config**: BF16+DFlash N=8 HE 90.9 → **92.7 %** (+1.8 pp), FP8+MTP=3 HE 84.8 → **88.4 %** (+3.6 pp).
- **Production SOTA (deployed): `:v3` + FP8+MTP=3** — 88.4 % HE / 89.1 % MBPP / 369 tok/s peak / 98 tok/s single-user / 0 length-trunc / reasoning cleanly separated into the OpenAI `reasoning` field. Live since ~10:03 MSK 2026-05-12.
- **Benchmark-only, NOT production: `:v3` + FP8+MTP=5** — 93.3 % HE / 402 tok/s peak in the offline harness, but leaks raw `<think>...</think>` blocks into OpenAI `content` on production traffic. The +4.9 pp HE delta vs MTP=3 is harness-counting-noise (the harness scores leaked think blobs as code), not real downstream quality.
- **0 length_truncated across all 4 v3 configs on both HE and MBPP.** The `empty_response` drift seen on v2 FP8+MTP=3 at mt=16384 (7→15) is gone on v3.
- **MBPP record broken: 91.1 %** (BF16+DFlash N=8 on `:v3`) over prior 90.3 % on `:v2`.
- Run 1 BF16+DFlash crash filed for Repne as low-frequency transient (non-reproducible on rerun; stack trace lost on Run 1, hardened launcher in study repo will preserve it if it recurs).
- **Production-incident finding** during v3 rollout: `--speculative-config.use_local_argmax_reduction true` is **DFlash-only**, not TP-only — the in-model `Qwen3_5MTP` drafter does not implement `get_top_tokens()`, so the engine refuses to start an MTP config with that flag (`ValueError`). The bench harness already had this right; the production launcher was the one wrong. See [Production Incident](https://github.com/jcartu/qwen-bench-2026-05-12-v3-suite/blob/main/FINAL_REPORT.md#production-incident-2026-05-12-mtp--use_local_argmax_reduction-incompatibility).
- **Production-incident finding** (think-token leak validation): operator escalation "MTP=3 leaks too, just less so you need a longer test to validate" + "we need a more robust leak test which does extensive tool and function calling" prompted a permanent dual-mode leak probe (`harness/leak_probe.py`: 75-prompt chat corpus across 15 categories + 30-scenario tools corpus across 10 OpenAI-schema function tools; scans `content` AND every `tool_calls[*].function.{name, arguments}` for `<think`/`</think>`/`<reasoning`/`</reasoning>`). Runs: MTP=3 chat smoke 75/0, MTP=2 chat smoke 74/0, **MTP=3 chat extended @ T=0.7 — 300 trials, 0 leaks**, **MTP=3 tools @ T=0.7 — 120 trials (95 % real tool-call rate, 4 multi-tool responses), 0 leaks across all three surfaces**. Combined 420 trials, zero leaks; MTP=5 leak class is MTP=5-specific in our environment. See [`LEAK_DETECTION.md`](https://github.com/jcartu/qwen-bench-2026-05-12-v3-suite/blob/main/LEAK_DETECTION.md).

### Records broken / clarifications
- HumanEval pass@1 SOTA online — **deployed production**: 84.8 → **88.4 %** (FP8+MTP=3 on `:v3`)
- HumanEval pass@1 SOTA online — **benchmark harness only**: 90.9 → **93.3 %** (FP8+MTP=5 on `:v3`) — not deployable due to `<think>` leak
- MBPP pass@1 SOTA: 90.3 → **91.1 %** (BF16+DFlash N=8 on `:v3`)
- Peak throughput on a quality-validated config: 245 → **402 tok/s** (FP8+MTP=5 offline) / **369 tok/s** (FP8+MTP=3, production)
- Single-user throughput on a quality-validated config: ~69 → **101 tok/s** (FP8+MTP=5 offline) / **98 tok/s** (FP8+MTP=3, production)
- First clean 4-config full-suite at 0 length-truncated on Qwen3.6-27B
- **New permanent CI-gatable leak probe** (`harness/leak_probe.py`, dual-mode chat + tools) with 420-trial clean-run baseline on production config

**Full report**: [FINAL_REPORT.md](https://github.com/jcartu/qwen-bench-2026-05-12-v3-suite/blob/main/FINAL_REPORT.md). **Production-incident post-mortem**: [§ Production Incident](https://github.com/jcartu/qwen-bench-2026-05-12-v3-suite/blob/main/FINAL_REPORT.md#production-incident-2026-05-12-mtp--use_local_argmax_reduction-incompatibility). **Repne bug report draft**: [repne_reply_draft.md](https://github.com/jcartu/qwen-bench-2026-05-12-v3-suite/blob/main/repne_reply_draft.md).

---

## 2026-05-11 (post-mortem) · HumanEval harness bug — every prior HE score deflated 13–23 pp

**Repo:** [`qwen-bench-2026-05-11-v2-followup`](https://github.com/jcartu/qwen-bench-2026-05-11-v2-followup) (ADDENDUM + Tier 1 rerun + patched harness)
**Trigger:** Repne’s Discord hypotheses about why HE scores looked off on `repne/vllm:v2` (max_tokens=4096 truncations, timeouts, c=1 vs c=8, pass@5 capability ceiling).
**Hardware:** 2× NVIDIA RTX PRO 6000 Blackwell (TP=2, GPUs 0+1)
**Harness:** `llm-stress-harness` v2 — **patched** with `smart_glue_humaneval()` extraction + “complete the function” prompt

### Abstract
Investigating Repne’s hypotheses surfaced a real bug in the HE extraction harness: `extract_code()` was calling `.strip()` on body-only model responses, which left the first response line at column 0 instead of indented inside the function signature appended by the test driver. Result: `IndentationError`, classified as `test_fail`, no signal that the model was actually correct. The bug is consistent across every HE study in this hub. Offline rescore via `smart_glue_humaneval()` (re-indents, prepends signature if missing) corrects all 9 historical jsonls by +13 to +23 pp.

### Headline result
- **The model has been ~93–96 % on HumanEval all along**, not 70–79 %.
- **Corrected SOTA: FP8+MTP=5 = 95.7 %** (offline rescore of stress-validation jsonl).
- **Closest BF16 SOTA: BF16+DFlash N=8 mt=8192 = 95.1 %** (offline rescore of v2-followup quality-rerun jsonl).
- **Capability ceiling — pass@5 (any of 5, temp=0.8) = 96.95 %** (159/164) on FP8+MTP=3 `:latest` mt=8192.
- **Tier 1 online re-bench (patched harness, mt=8192)**: BF16+DFlash N=8 = 87.2 %, FP8+MTP=3 = 83.5 %. Lower than offline rescore because the new prompt produces longer responses → more `max_tokens` truncations at mt=8192. Genuine semantic-fail count is 6–8 per config, confirming the ~95 % true ceiling.
- Repne hypotheses falsified: there is no MTP regression, no `:v2` regression, and no concurrency degradation. The gap was a measurement bug, not a model bug.

### Records broken / restored
- New canonical HumanEval SOTA on Qwen3.6-27B = **95.7 %** (FP8+MTP=5, offline-rescored).
- New capability-ceiling reference = **96.95 %** pass@5.
- All prior HE numbers in `SOTA.md` retained with strikethrough — we don’t erase history.
- Official harness patched ([commit on llm-stress-harness](https://github.com/jcartu/llm-stress-harness)); reusable offline rescorer published in [`harness/rescore_humaneval.py`](https://github.com/jcartu/qwen-bench-2026-05-11-v2-followup/blob/main/harness/rescore_humaneval.py).

**Read the full bug analysis**: [ADDENDUM.md](https://github.com/jcartu/qwen-bench-2026-05-11-v2-followup/blob/main/ADDENDUM.md)

---
## 2026-05-11 (followup) · Where are the next big gains? FP8+MTP{3,5} on `repne/vllm:v2` + max_tokens=8192

**Repo:** [`qwen-bench-2026-05-11-v2-followup`](https://github.com/jcartu/qwen-bench-2026-05-11-v2-followup)
**Hardware:** 2× NVIDIA RTX PRO 6000 Blackwell (TP=2, GPUs 0+1)
**Models:** `Qwen/Qwen3.6-27B-FP8` + MTP head; `Qwen/Qwen3.6-27B` (BF16) + `z-lab/Qwen3.6-27B-DFlash`
**Server:** `repne/vllm:v2`
**Configs measured:** 2 speed (FP8+MTP=3, FP8+MTP=5) + 2 quality (BF16+DFlash n=8 @ mt=8192, FP8+MTP=3 @ mt=8192)
**Harness:** v2 four-phase suite, `--decode-warmup-seconds 20`, `--duration 60`, 60s post-ready settle

### Abstract
Follow-up to study #3 (BF16+DFlash sweep) asking two open questions: (1) where do the next throughput gains come from — deeper MTP, or have we plateaued? and (2) do the empty_response failures at max_tokens=4096 recover at max_tokens=8192? Measures FP8+MTP=3 and FP8+MTP=5 head-to-head against study #3's BF16+DFlash n=8 winner, and re-runs both winning configs at max_tokens=8192 for HumanEval + MBPP.

### Headline result (⚠️ HE numbers below are pre-patch; see harness-bug post-mortem above for corrected values)
- **Speed: FP8+MTP=3 is the new SOTA on `repne/vllm:v2`** at **245.32 tok/s** mean across 15 cells — **+29.1%** over study #3's BF16+DFlash n=8 winner (189.98 tok/s).
- **FP8+MTP=5 plateaus**: 246.59 tok/s (+0.5% over MTP=3) but acceptance rate collapses 56.7% → 35.3%. Deeper drafter doesn't pay.
- **Peak single-cell throughput: 445 tok/s** at c=4, ctx=16k — highest measured across any of the four studies.
- **Quality at max_tokens=8192**: FP8+MTP=3 HumanEval = ~~70.7 %~~ → **93.3 %** (corrected via [ADDENDUM](https://github.com/jcartu/qwen-bench-2026-05-11-v2-followup/blob/main/ADDENDUM.md)); MBPP = 86.8 % (unaffected by the HE bug). Mean completion tokens 2983 vs ~1500 at mt=4096 in study #2.
- **Empty-response recovery is partial**: doubling max_tokens trades early-truncation empties for over-thinking-budget-exhaustion empties. The real fix is dynamic stopping, not more tokens.

### Records broken
- First measurement of FP8+MTP=3 and FP8+MTP=5 on `repne/vllm:v2`.
- New aggregate decode throughput record: 245.32 tok/s (FP8+MTP=3).
- New peak single-cell throughput record: 445.39 tok/s.
- First HumanEval+MBPP measurement at max_tokens=8192 on `repne/vllm:v2`.


## 2026-05-11 · Qwen3.6-27B BF16+DFlash parameter sweep on `repne/vllm:v2`

**Repo:** [`qwen-bench-2026-05-dflash-v2-sweep`](https://github.com/jcartu/qwen-bench-2026-05-dflash-v2-sweep)
**Hardware:** 2× NVIDIA RTX PRO 6000 Blackwell (TP=2)
**Model:** `Qwen/Qwen3.6-27B` (BF16) + drafter `z-lab/Qwen3.6-27B-DFlash`
**Server:** `repne/vllm:v2` (sha `58d92a127a1a`, vLLM `0.1.dev16530+ged1130111.d20260510`)
**Configs measured:** 13 (9 Stage A buffer/graph + 4 Stage B num_speculative_tokens)
**Harness:** v2 four-phase suite, `--decode-warmup-seconds 20`, `--duration 60`, 60s post-ready settle

### Abstract
A two-stage 13-config sweep across `--max-num-batched-tokens` ×
`--max-cudagraph-capture-size` (3×3 grid at num_spec=8 fixed), then
`--speculative-config.num_speculative_tokens` ∈ {4, 8, 15, 16} at the
Stage A winner. Establishes which `vllm serve` flags actually move
decode throughput for BF16+DFlash, and quantifies the quality-measurement
noise floor at concurrency=8.

### Headline result
- **Buffer/graph axis is essentially flat**: all 9 Stage A configs land in 183.5–190.1 tok/s (Δ=3.6%)
- **Speculative-tokens axis is decisive**: n=4 → −62% (drafter accept 0.5%); n=8 → winner (accept 23.1%); n=15 → −7.2%; n=16 → −42.6% (sharp fastpath cliff)
- **Recommended config = Repne's published defaults**: `batched=32768 capture=256 num_spec=8` at **190.1 tok/s** aggregate decode
- **Quality noise floor at c=8**: identical config run twice produced HumanEval 58.5%/65.2%, MBPP 82.1%/79.8% — 6.7-point HE spread is run-to-run variance
- All 13 configs passed 4/4 server gates; 7h total wall time

### Records broken
- (None — confirms existing recommendations.) Establishes the first public 3×3 buffer/graph map for BF16+DFlash, and the first cliff curve for `num_speculative_tokens` on `repne/vllm:v2`.

---

## 2026-05-07 · Qwen3.6-27B FP8+DFlash characterization (no-gumbel)

**Repo:** [`qwen36-27b-blackwell-stress-validation`](https://github.com/jcartu/qwen36-27b-blackwell-stress-validation) §13 (addendum)
**Hardware:** 2× NVIDIA RTX PRO 6000 Blackwell (TP=2)
**Model:** `Qwen/Qwen3.6-27B-FP8` + drafter `z-lab/Qwen3.6-27B-DFlash`
**Configs measured:** FP8+DFlash N ∈ {7, 8, 15}, no-gumbel
**Harness:** `llm-stress-harness` four-phase suite, 60s decode duration, 20s warmup, 60s post-ready settle

### Abstract
Characterizes the previously-untested FP8 base + DFlash drafter pairing
across three draft lengths. Documents the `instanttensor` segfault that
occurs when FP8 base + BF16 drafter are loaded with `--load-format instanttensor`.

### Headline result
- **F (N=7):** HE 73.8 % · MBPP 87.5 % · 244.2 tok/s aggregate
- **G (N=8):** HE 74.4 % · MBPP 88.3 % · 240.4 tok/s aggregate
- **H (N=15):** HE 73.8 % · MBPP 87.2 % · 222.8 tok/s aggregate
- **FP8+DFlash N=8 wins long-context** (ctx ≥ 64k) by +5–13 % vs FP8+MTP=3
- New guidance: FP8+DFlash N=8 recommended for long-context coding-agent workloads

### Records broken
- (Long-context FP8 throughput; see SOTA.md)

---

## 2026-05-06 · Qwen3.6-27B BF16+DFlash characterization (no-gumbel)

**Repo:** [`qwen36-27b-blackwell-stress-validation`](https://github.com/jcartu/qwen36-27b-blackwell-stress-validation) §12 (addendum)
**Hardware:** 2× NVIDIA RTX PRO 6000 Blackwell (TP=2)
**Model:** `Qwen/Qwen3.6-27B` (BF16) + drafter `z-lab/Qwen3.6-27B-DFlash`
**Configs measured:** BF16+DFlash N ∈ {7, 8, 15}, no-gumbel
**Harness:** `llm-stress-harness` four-phase suite

### Abstract
Re-runs the three BF16+DFlash variants from the main study with the
deprecated `--speculative-config.draft_sample_method gumbel` flag removed
and an improved harness (60s decode duration, 20s warmup).

### Headline result
- **C2 (N=7):** HE 73.2 % · MBPP 88.7 %
- **D2 (N=8):** HE 70.7 % · MBPP **89.5 %** ⭐ (matches SOTA on MBPP)
- **E2 (N=15):** HE **78.0 %** · MBPP 88.3 %  (+8 HumanEval problems vs gumbel-on)
- DFlash N=15 with `gumbel` disabled becomes the new SOTA among BF16+DFlash configurations.

### Records broken
- BF16 HumanEval pass-rate (78.0 % vs prior 73.2 %)

---

## 2026-05-06 · Qwen3.6-27B stress-validation suite (5-config head-to-head)

**Repo:** [`qwen36-27b-blackwell-stress-validation`](https://github.com/jcartu/qwen36-27b-blackwell-stress-validation) §1–§11 (main study)
**Hardware:** 2× NVIDIA RTX PRO 6000 Blackwell (TP=2)
**Model:** `Qwen/Qwen3.6-27B-FP8`, `Qwen/Qwen3.6-27B` (BF16)
**Configs measured:** FP8+MTP={3,5}, BF16+DFlash={7,8,15}
**Harness:** `llm-stress-harness` four-phase suite

### Abstract
A controlled stress-validation of five Qwen3.6-27B inference configurations.
For each: (i) four functional gates, (ii) nine-cell throughput matrix
(c × ctx grid, N=2), (iii) HumanEval-164, (iv) MBPP-257-sanitized — the
correctness phases at concurrency=8 to simulate burst production load.

### Headline result
- **2,105 hard coding problems executed across all 5 configs · zero engine crashes · zero hangs · zero malformed JSON**
- **FP8+MTP=3 wins HumanEval** (130/164 = 79.3 %)
- **BF16+DFlash variants win MBPP** (230/257 = 89.5 %)
- **FP8+MTP=5 strictly dominated by FP8+MTP=3 on HumanEval** (-3.7 pp) despite +15 % raw tok/s
- Textbook demonstration that throughput-on-decode-bench does not predict end-to-end agentic correctness

### Records broken
- HumanEval pass-rate (79.3 %)
- MBPP pass-rate (89.5 %)
- "Hardest validation completed without engine failure"

---

## 2026-05-05 · Qwen3.6-27B 24-hour vLLM benchmark sprint (Day 1)

**Repo:** [`qwen36-27b-blackwell-inference-study`](https://github.com/jcartu/qwen36-27b-blackwell-inference-study) (8 experiments)
**Hardware:** 2× NVIDIA RTX PRO 6000 Blackwell (TP=2, SM120, 96 GB each, PCIe Gen5 x16)
**Model:** `Qwen/Qwen3.6-27B` across BF16, FP8 W8A8, NVFP4, GGUF Q8_0
**Configs measured:** MTP n ∈ {2, 3, 4, 5, 6}, DFlash n ∈ {7, 8, 15}, plus baselines
**Harness:** custom benchmark harness (precursor to `llm-stress-harness`) + AesSedai sliding-window perplexity

### Abstract
A full-day systematic head-to-head of the [`repne/vllm`](https://hub.docker.com/r/repne/vllm)
fork against upstream [vLLM v0.19.1 / v0.20.1](https://github.com/vllm-project/vllm),
plus llama.cpp Q8_0 GGUF, across four quantization schemes and three
speculative-decoding methods.

### Eight experiments
1. Morning newimage validation
2. Scheduler investigation (pessimal-pocket identification)
3. NVFP4-MTP experiment ([`qwen36-27b-nvfp4-mtp-experiment`](https://github.com/jcartu/qwen36-27b-nvfp4-mtp-experiment))
4. FP8+MTP=3 head-to-head ([`qwen36-27b-fp8-repne-vs-upstream`](https://github.com/jcartu/qwen36-27b-fp8-repne-vs-upstream))
5. BF16+DFlash head-to-head ([`qwen36-27b-bf16-dflash-repne-vs-upstream`](https://github.com/jcartu/qwen36-27b-bf16-dflash-repne-vs-upstream))
6. New-image revalidation ([`repne-dflash-newimage`](https://github.com/jcartu/repne-dflash-newimage))
7. Quality sprint (102K-token KLD probe vs BF16, triggered by community concerns about W8A8 activation quantization)
8. X1Y1 sprint — high-concurrency characterization (c ∈ {8, 16, 32})

### Headline result
- Optimal production config: **FP8 W8A8 + MTP=3** on Repne fork
- **Peak 2,083.7 tok/s aggregate at c=32 ctx=0**
- **Q8_0 GGUF KLD vs BF16 = 0.001828 nats** (well within noise floor of perplexity)
- **97.9 % top-token agreement** with BF16

### Records broken
- All throughput records (this is the source of c=8/16/32 SOTA cells)
- Quality probe established noise-floor reference point

---

## 2026-03-17 · Closing the Opus Gap (tool-calling on wafer-scale)

**Repo:** [`closing-the-opus-gap`](https://github.com/jcartu/closing-the-opus-gap)
**Hardware:** Cerebras wafer-scale (different hardware class)
**Model:** Qwen3 235B, GLM-4.7
**Harness:** custom (predates `llm-stress-harness`)

### Abstract
The earlier study that anchored this whole line of work. Tool-calling
optimization on wafer-scale hardware: 3,500+ API calls across 10 experimental
phases. Tested whether an open-weight model at $0.10/M tokens could match
Claude Opus at $15/M tokens (150× cost difference) for production agent
workloads.

### Headline result
- Open-weight models *can* match Claude Opus on tool-calling tasks
- Most "best practices" for tool descriptions don't matter; tool *count* and
  *prompt structure* dominate
- Full paper + blog included in repo

### Records broken
- N/A (different scope; this is methodologically the ancestor)

---

## How studies get added here

See [`README.md` § Adding a new study](README.md#adding-a-new-study).

The short version:
1. Open the study's own repo (any name; convention is `qwen-bench-{YYYY-MM}-{slug}` for new ones)
2. PR against this hub adding: a `## YYYY-MM-DD · Title` block here, an entry in `SOTA.md` if records changed, and the merged CSV in `data/`
3. Tag both repos with appropriate GitHub topics

---

*Last updated: 2026-05-15 (single-user thinking-budget addendum)*
