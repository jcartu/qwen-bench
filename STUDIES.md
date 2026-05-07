# Qwen-Bench · Studies Index

Chronological log of every benchmark study indexed by this hub. Newest first.

For headline numbers across all studies, see [SOTA.md](SOTA.md).
For the merged machine-readable result tables, see [`data/`](data/).

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

*Last updated: 2026-05-07*
