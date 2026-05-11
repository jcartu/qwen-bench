# Qwen-Bench · Studies Index

Chronological log of every benchmark study indexed by this hub. Newest first.

For headline numbers across all studies, see [SOTA.md](SOTA.md).
For the merged machine-readable result tables, see [`data/`](data/).

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

*Last updated: 2026-05-11 (HumanEval harness-bug post-mortem)*
