<div align="center">

<img src="docs/images/hub_hero.png" alt="qwen-bench — empirical inference characterization of Qwen models on Blackwell" width="100%" />

# `qwen-bench`

### Empirical inference characterization of Qwen models on NVIDIA Blackwell

[![Studies](https://img.shields.io/badge/studies-2_published-success?style=for-the-badge)](STUDIES.md)
[![SOTA](https://img.shields.io/badge/SOTA-tracker-blue?style=for-the-badge)](SOTA.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)
[![Topic: qwen](https://img.shields.io/badge/topic-qwen-orange?style=for-the-badge)](https://github.com/topics/qwen)
[![Topic: vllm](https://img.shields.io/badge/topic-vllm-orange?style=for-the-badge)](https://github.com/topics/vllm)
[![Topic: blackwell](https://img.shields.io/badge/topic-blackwell-orange?style=for-the-badge)](https://github.com/topics/blackwell)

> **An ongoing public benchmark series. Updated whenever new measurements are taken.**

[Current SOTA](#current-sota) ·
[Studies](#studies-chronological) ·
[Tools](#tools) ·
[How to read this](#how-to-read-this) ·
[Methodology](#methodology) ·
[Reproduce](#reproduce-a-result)

</div>

---

## What this is

This is the **index repository** for an ongoing series of inference benchmarks
measuring [Qwen](https://github.com/QwenLM) models on consumer- and
workstation-class NVIDIA Blackwell hardware (`RTX PRO 6000 Blackwell`,
`RTX 5090`, etc.) under realistic agent-style production load.

Each *study* is a separately-versioned satellite repo with raw data, configs,
logs, and a written report. This hub repo provides:

- 📊 **The current SOTA leaderboard** across all studies → [SOTA.md](SOTA.md)
- 📚 **A chronological index** of every study → [STUDIES.md](STUDIES.md)
- 📈 **Merged result CSVs** combining numbers from all studies → [`data/`](data/)
- 🛠️ **Pointers to the toolchain** that produced these results

The hub is the front door. The studies are the substance. The tools are reusable
primitives. None of these collapse into the others; each has its own permanent,
citable URL.

---

## Current SOTA

> Last updated: **2026-05-07** · Hardware: **2× NVIDIA RTX PRO 6000 Blackwell** (TP=2, SM120, 96 GB each, PCIe Gen5 x16)

### 🏆 Production-recommended config: **FP8 + MTP=3** (Repne fork)

The configuration that ships in production. Wins HumanEval, second-best on MBPP,
zero crashes across 2,105 hard problems.

### Throughput records (aggregate tok/s)

| Concurrency × Context | Tok/s | Config | Source |
|-----------------------|------:|--------|--------|
| c=1, ctx=0            | 117.1 | FP8+MTP=3 | [day1-sprint § exp 06](https://github.com/jcartu/qwen36-27b-blackwell-inference-study) |
| c=4, ctx=131k         | 350.5 | FP8+MTP=3 | [day1-sprint § exp 06](https://github.com/jcartu/qwen36-27b-blackwell-inference-study) |
| c=8, ctx=0            | 875.0 | FP8+MTP=3 | [day1-sprint § exp 08 X1](https://github.com/jcartu/qwen36-27b-blackwell-inference-study) |
| c=16, ctx=0           | 1,520.6 | FP8+MTP=3 | [day1-sprint § exp 08 X1](https://github.com/jcartu/qwen36-27b-blackwell-inference-study) |
| **c=32, ctx=0** ⭐    | **2,083.7** | FP8+MTP=3 | [day1-sprint § exp 08 X1](https://github.com/jcartu/qwen36-27b-blackwell-inference-study) |
| c=4, ctx=131k (long)  | best | FP8+DFlash=8 | [stress-validation § 13](https://github.com/jcartu/qwen36-27b-blackwell-stress-validation) |

### Correctness records (Qwen3.6-27B)

| Benchmark | Pass rate | Config | Source |
|-----------|----------:|--------|--------|
| HumanEval (164) | **79.3 %** (130/164) | FP8+MTP=3 | [stress-validation](https://github.com/jcartu/qwen36-27b-blackwell-stress-validation) |
| MBPP (257)      | **89.5 %** (230/257) | BF16+DFlash=8 | [stress-validation](https://github.com/jcartu/qwen36-27b-blackwell-stress-validation) |
| MBPP (257)      | **89.5 %** (230/257) | BF16+DFlash=15 (no-gumbel) | [stress-validation § 12](https://github.com/jcartu/qwen36-27b-blackwell-stress-validation) |
| HumanEval (164) | **78.0 %** | BF16+DFlash=15 (no-gumbel) | [stress-validation § 12](https://github.com/jcartu/qwen36-27b-blackwell-stress-validation) |

### Quality vs BF16 reference

| Quantization | KL-divergence vs BF16 | Top-token agreement | Source |
|--------------|----------------------:|--------------------:|--------|
| Q8_0 GGUF    | 0.001828 nats         | 97.9 %              | [day1-sprint § exp 07](https://github.com/jcartu/qwen36-27b-blackwell-inference-study) |
| FP8 W8A8     | (indistinguishable)   | (per Qwen card)     | [day1-sprint § exp 07](https://github.com/jcartu/qwen36-27b-blackwell-inference-study) |

> See [SOTA.md](SOTA.md) for the complete cross-study record book.

<div align="center">
<img src="docs/images/sota_leaderboard.png" alt="SOTA leaderboard visualization" width="80%" />
</div>

---

## Studies (chronological)

<div align="center">
<img src="docs/images/studies_index.png" alt="Studies timeline" width="100%" />
</div>

Each study is a self-contained satellite repo. Studies are listed newest-first.

### 📖 2026-05 · Stress-validation suite
**[`qwen36-27b-blackwell-stress-validation`](https://github.com/jcartu/qwen36-27b-blackwell-stress-validation)** · *5 configs × 4 phases × 2,105 hard problems · Zero crashes*

A controlled head-to-head of 5 speculative-decoding configurations
(`FP8+MTP={3,5}`, `BF16+DFlash={7,8,15}`) on `Qwen3.6-27B` across functional
gates, throughput matrix, HumanEval, and MBPP. Adds two addenda re-running
DFlash variants without the deprecated `gumbel` flag and characterizing the
previously-untested FP8+DFlash pairing across N ∈ {7, 8, 15}.

**Headline:** FP8+MTP=3 holds HumanEval pass-rate SOTA (79.3 %); BF16+DFlash variants
hold MBPP SOTA (89.5 %); FP8+DFlash N=8 wins long-context.

### 📖 2026-05 · 24-hour inference study (Day 1 sprint)
**[`qwen36-27b-blackwell-inference-study`](https://github.com/jcartu/qwen36-27b-blackwell-inference-study)** · *8 experiments · 1,200+ benchmark runs · 102K KLD probe*

A systematic 24-hour sprint comparing the [`repne/vllm`](https://hub.docker.com/r/repne/vllm)
fork against upstream vLLM v0.19.1 / v0.20.1, plus llama.cpp Q8_0 GGUF, across
four quantization schemes (BF16, FP8 W8A8, NVFP4, GGUF Q8_0) and three
speculative-decoding methods.

**Headline:** Peak 2,083 tok/s at c=32. Q8 GGUF KLD vs BF16 = 0.0018 (noise floor).

#### Satellite repos absorbed by this study
The following pairwise-comparison repos contain the same data as subsections
of `inference-study`. They remain published as independent citable artifacts
but the canonical home is the parent study:

- [`qwen36-27b-bf16-dflash-repne-vs-upstream`](https://github.com/jcartu/qwen36-27b-bf16-dflash-repne-vs-upstream) → exp 05
- [`qwen36-27b-fp8-repne-vs-upstream`](https://github.com/jcartu/qwen36-27b-fp8-repne-vs-upstream) → exp 04
- [`qwen36-27b-nvfp4-mtp-experiment`](https://github.com/jcartu/qwen36-27b-nvfp4-mtp-experiment) → exp 03
- [`repne-dflash-newimage`](https://github.com/jcartu/repne-dflash-newimage) → image-version regression sub-experiment

### 📖 2026-03 · Closing the Opus Gap (tool-calling study)
**[`closing-the-opus-gap`](https://github.com/jcartu/closing-the-opus-gap)** · *3,500 API calls · 10 phases · Qwen3 235B and GLM-4.7 on Cerebras*

The earlier study that kicked this whole line of work off. Tool-calling
optimization on wafer-scale hardware. Different model and different hardware
class, but methodologically the same family of work. Included here for context.

> See [STUDIES.md](STUDIES.md) for the full chronological index with abstracts.

---

## Tools

All studies above were instrumented with:

### 🛠️ [`llm-stress-harness`](https://github.com/jcartu/llm-stress-harness)
The diagnostic toolkit — generic, model-agnostic, OpenAI-compatible:

- `harness/stress_harness.py` — 322-line failure-taxonomic correctness probe
- `orchestrator/four_phase_harness.sh` — end-to-end 4-phase validation runner
- `launchers/launch_*.sh` — parametric vLLM launcher templates (FP8+MTP, FP8+DFlash, BF16+DFlash)
- `utils/wait_vllm_ready.sh` — readiness probe with post-ready settle window

The toolkit lives in its own repo so it can grow to other models and stay
reusable. Studies pin a specific commit of the toolkit for reproducibility.

---

## How to read this

```
                        ┌───────────────────────────┐
                        │   you are here            │
                        │   jcartu/qwen-bench       │
                        │   (the hub / index)       │
                        └────────┬──────────────────┘
                                 │
      ┌──────────────────────────┼──────────────────────────┐
      │                          │                          │
      ▼                          ▼                          ▼
┌──────────────┐        ┌────────────────┐         ┌────────────────┐
│  STUDIES     │        │  SOTA          │         │  data/         │
│  list of all │        │  rolling       │         │  merged CSVs   │
│  studies     │        │  leaderboard   │         │  (machine      │
│              │        │                │         │   readable)    │
└──────────────┘        └────────────────┘         └────────────────┘
      │
      ├──────► jcartu/qwen36-27b-blackwell-stress-validation
      ├──────► jcartu/qwen36-27b-blackwell-inference-study
      ├──────► jcartu/closing-the-opus-gap
      └──────► (future studies land here)
                              │
                              └──── instrumented with ────► jcartu/llm-stress-harness
```

**Three-question reader flow:**

1. *"What's the best config for X?"* → start at [SOTA.md](SOTA.md)
2. *"How was that measured?"* → click through to the linked study repo
3. *"Can I reproduce this?"* → see [`Reproduce a result`](#reproduce-a-result) below

---

## Methodology

All studies in this hub follow shared conventions so numbers compare across studies:

| Aspect | Convention |
|--------|-----------|
| **Determinism** | `temperature=0.0`, `top_p=1.0`, `seed=42` in every payload |
| **Hardware logging** | GPU model, driver, CUDA version, NCCL version, PCIe gen + lane count documented per study |
| **Engine version** | Specific vLLM image SHA + git rev recorded in every study |
| **Concurrency notation** | `c=N` means N concurrent in-flight requests; `ctx=K` means K-token prefix (in tokens, not chars) |
| **Throughput definition** | `aggregate_tps = Σ completion_tokens / wall_time` (the user-observed number) |
| **Failure classification** | Every request labeled with one of 7 failure modes — see [llm-stress-harness](https://github.com/jcartu/llm-stress-harness#the-failure-taxonomy) |
| **N (re-runs)** | Phase 2 throughput cells re-run with N≥2 reseeded runs unless noted |

### What we deliberately do *not* measure

- **Pass@k for k>1.** All correctness scores are single-shot at `t=0`.
- **Prefill-only / decode-only throughput.** Use `vllm bench` for that. These studies are end-to-end.
- **Reasoning-content quality.** We log `reasoning_chars` for diagnostics, not for grading.
- **Cross-model leaderboards.** This hub is *Qwen-only* by design. Cross-model is out of scope.

---

## Reproduce a result

```bash
# 1. Read the study's METHODOLOGY section
gh repo view jcartu/qwen36-27b-blackwell-stress-validation --web

# 2. Pull the toolkit at the version the study used
git clone https://github.com/jcartu/llm-stress-harness
cd llm-stress-harness && git checkout <commit-sha-from-study>

# 3. Launch the documented config
NUM_SPEC=3 ./launchers/launch_fp8_mtp.sh   # or whichever the study used

# 4. Run the same orchestrator
./orchestrator/four_phase_harness.sh <config-label> <kv-budget> ./out/

# 5. Diff your numbers against the published CSV
python3 -c "import pandas as pd; print(pd.read_csv('your_results.csv').compare(pd.read_csv('published.csv')))"
```

Numbers should land within **±2 %** at the same hardware tier. If not, file an
issue against the study repo with both CSVs.

---

## Adding a new study

When you publish a new benchmark study, the integration steps are:

1. **Create the study repo** as `qwen-bench-{YYYY-MM}-{slug}` (or any name; name convention is soft, links are what matter)
2. **Set GitHub topics:** `qwen`, `vllm`, `blackwell`, `benchmark`, plus study-specific tags
3. **Open a PR against this hub repo** that:
   - Adds an entry to [`STUDIES.md`](STUDIES.md) with abstract + headline
   - Updates [`SOTA.md`](SOTA.md) if any record was broken
   - Drops the study's `master.csv` into [`data/{YYYY-MM}-{slug}.csv`](data/)
   - Adds a `## 2026-MM · {Title}` block to the [Studies section](#studies-chronological) above
4. **Add a "← qwen-bench hub" badge** to the top of the new study's README

A future automated script (planned) will compute SOTA-record diffs from
the merged CSV and prevent regressions from being missed.

---

## Acknowledgments

The work indexed here builds on:

- [Qwen team](https://github.com/QwenLM) — open-weight model lineage
- [vLLM](https://github.com/vllm-project/vllm) — primary inference engine
- [Repne](https://hub.docker.com/r/repne/vllm) — vLLM fork with DFlash + advanced spec-decode support
- [HumanEval](https://github.com/openai/human-eval), [MBPP](https://github.com/google-research/google-research/tree/master/mbpp), [EvalPlus](https://github.com/evalplus/evalplus) — benchmark sources
- [AesSedai](https://github.com/AesSedai/llama.cpp/tree/perplexity-sliding-window) — sliding-window perplexity branch used in KLD probe

Hub illustrations generated with **Gemini 3.1 nano banana** (`gemini-3.1-flash-image-preview`).

---

## License

MIT. See [LICENSE](LICENSE).

Each satellite study repo is independently licensed; check the study's own LICENSE
file.

---

<div align="center">

**Maintained by [Josh Cartu](https://github.com/jcartu) · RASPUTIN AI Research Lab**

If you cite numbers from this hub, please link to the **specific study repo**
(not just this hub) so readers can verify the methodology.

</div>
