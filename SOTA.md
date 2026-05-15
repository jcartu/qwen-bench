# Qwen-Bench · SOTA Reference Matrix

The cross-study record book. Every claim here is reproducible from raw
`results.json` files in the linked study repos.

> **Scope:** Qwen3.6-27B on **2× NVIDIA RTX PRO 6000 Blackwell** (TP=2, SM120, 96 GB each, PCIe Gen5 x16).
> **Last updated:** 2026-05-15 (single-user thinking-budget addendum)
> **Methodology:** All studies use shared conventions (see [`README.md` § Methodology](README.md#methodology)).

---

> ⚠️ **2026-05-11 correction**: A bug in the HumanEval extraction harness (described in the [v2-followup ADDENDUM](https://github.com/jcartu/qwen-bench-2026-05-11-v2-followup/blob/main/ADDENDUM.md)) was deflating reported HumanEval scores by 13–23 percentage points across every prior study. Corrected scores appear in § 2.1 below. Throughput and MBPP records are unaffected.

> 🆕 **2026-05-15 single-user update**: For OpenCode coding on the deployed `FP8+MTP=3` v3 stack, client-side `thinking_token_budget=2048` is now the recommended default. It preserves thinking mode, improves the c=1 budget sweep from 70% to 80% pass, eliminates stuck responses (10% → 0%), and in the matched concurrency probe cuts unbounded c=1 p95 latency from 130.8s to 19.6s. See [single-user addendum][singleuser].

> 🧪 **2026-05-15 generalization (evening)**: Cross-domain validation of `thinking_token_budget=2048`
> on three out-of-distribution academic benchmarks (3,598 trials total) confirms it as a **strict
> Pareto improvement** over unbounded thinking on Qwen3.6-27B FP8 + MTP=3 + `repne/vllm:v3`.
> Headline: **GPQA Diamond +33.8 pp accuracy (z=6.81, p=1e-11) at 0.24× wall-clock**;
> **GSM-Plus 2k accuracy parity (z=0.87, p=0.38) at 0.53× wall-clock**; **MMLU-Pro 73.1% → **82.6%** (Δ +9.5 pp, z=6.06, p=1.4e-09) at 0.39× wall**.
> Failure mechanism on hard reasoning is documented: 57.1% of GPQA unbounded responses hit
> `finish_reason=length` while still inside `<think>`, emitting **zero** extractable answers.
> Budget-sweep on GPQA at tb ∈ {1024, 2048, 4096, 8192} shows accuracy peaks at tb=4096 (acc=78.3%) → there is a **genuine sweet spot** in budget choice.
> Full study: [`studies/2026-05-15-thinking-budget-generalization`](studies/2026-05-15-thinking-budget-generalization/).

> 🆕 **2026-05-12 v3 update**: Repne shipped [`repne/vllm:v3`](https://hub.docker.com/r/repne/vllm). Full 4-config stress-validation suite re-run on it ([study repo][v3suite]) plus a same-day production-rollout post-mortem produced three updates: (i) **`FP8+MTP=3` on `:v3` is the new production SOTA** — 88.4 % HE, 89.1 % MBPP, 369 tok/s peak, 0 length-trunc, currently deployed; (ii) **MTP=5 is benchmark-only** — it scored 93.3 % HE in the offline harness but leaks raw `<think>...</think>` blocks into the OpenAI `content` field on production traffic, so its quality lead is harness-counting-noise, not real downstream code; (iii) **MTP=3 is validated leak-free at 420 trials** across plain-chat (300 @ T=0.7) AND realistic tool/function-calling (120 @ T=0.7, 95 % real tool-call rate, multi-tool responses included) by the new permanent dual-mode leak probe (`harness/leak_probe.py` in the v3 study repo). See [`v3suite/FINAL_REPORT.md` § Production Incident][v3incident] and [`v3suite/LEAK_DETECTION.md`](https://github.com/jcartu/qwen-bench-2026-05-12-v3-suite/blob/main/LEAK_DETECTION.md).


## TL;DR

**Production-deployed SOTA (live 2026-05-12 ~10:03 MSK):** **`repne/vllm:v3` + FP8+MTP=3** on the [Repne fork](https://hub.docker.com/r/repne/vllm). 88.4 % HE, 89.1 % MBPP, 369 tok/s peak (c=4 ctx=0), 98 tok/s single-user, 0 length-trunc, reasoning cleanly separated into the OpenAI `reasoning` field. OpenCode client requests should add `thinking_token_budget=2048` to prevent runaway thinking tails.

**Benchmark-only — NOT production:** `FP8+MTP=5` scored 93.3 % HE / 402 tok/s peak in the offline harness, but leaks raw `<think>...</think>` blocks into the OpenAI `content` field on production traffic (verified by live smoke test 2026-05-12). The +4.9 pp HE delta vs MTP=3 reflects the harness counting those leaked think blobs as code, not real downstream code quality. **Do not deploy.** See [v3 suite Production Incident][v3incident].

It holds the production-relevant SOTA across throughput, correctness, and
operational-stability dimensions. `MTP=5` and `BF16+DFlash` variants hold
isolated records but are dominated where production traffic actually lives (c≥8).

For normal single-user OpenCode coding, **`thinking_token_budget=2048`** is the client-side fix that keeps Qwen3 thinking enabled while cutting c=1 p95 latency 130.8s→19.6s.

For long-context coding-agent workloads (ctx ≥ 64k), **`FP8+DFlash N=8`** is
the recommended alternative (+5–13 % over `FP8+MTP=3`).

---

## 1. Throughput SOTA — `FP8+MTP=3` (Repne fork)

### 1.1 Per-cell records (aggregate tok/s)

| concurrency × context | tok/s | std | source |
|---|---:|---:|---|
| c=1 × 0 | 117.1 | ±2.8 | [day1-sprint § exp 06][day1] (peak: 120.1 in Exp 04 N=1) |
| c=1 × 32k | 119.2 | ±4.7 | [day1-sprint § exp 06][day1] |
| c=1 × 131k | 95.0 | ±2.5 | [day1-sprint § exp 06][day1] |
| c=2 × 0 | 227.2 | ±2.4 | [day1-sprint § exp 06][day1] |
| c=2 × 32k | 227.0 | ±5.1 | [day1-sprint § exp 06][day1] |
| c=2 × 131k | 184.9 | ±0.9 | [day1-sprint § exp 06][day1] |
| c=4 × 0 | 449.8 | ±4.6 | [day1-sprint § exp 06][day1] |
| c=4 × 32k | 454.5 | ±3.3 | [day1-sprint § exp 06][day1] |
| c=4 × 131k | 350.5 | ±4.9 | [day1-sprint § exp 06][day1] |
| c=8 × 0 | 875.0 | ±11.5 | [day1-sprint § exp 08 X1][day1] |
| c=8 × 32k | 795.4 | ±0.9 | [day1-sprint § exp 08 X1][day1] |
| c=8 × 131k | 534.4 | ±7.4 | [day1-sprint § exp 08 X1][day1] |
| c=16 × 0 | 1,520.6 | ±2.5 | [day1-sprint § exp 08 X1][day1] |
| c=16 × 32k | 1,186.0 | ±7.5 | [day1-sprint § exp 08 X1][day1] |
| c=16 × 64k | 1,047.1 | ±4.8 | [day1-sprint § exp 08 X1][day1] |
| **c=32 × 0** ⭐ | **2,083.7** | ±12.6 | [day1-sprint § exp 08 X1][day1] |
| c=32 × 16k | 1,892.3 | ±4.1 | [day1-sprint § exp 08 X1][day1] |
| c=32 × 32k | 1,656.3 | ±13.6 | [day1-sprint § exp 08 X1][day1] |

### 1.2 `MTP=5` per-cell records (dominated configuration, kept for reference)

`MTP=5` wins isolated cells at low concurrency. Documented for completeness.

| concurrency × context | MTP=5 tok/s | MTP=3 tok/s | MTP=5 advantage |
|---|---:|---:|---:|
| c=1 × 0 | 119.9 | 117.1 | +2.4 % |
| c=1 × 131k | **101.2** ⭐ | 95.0 | **+6.5 %** (largest single-cell `MTP=5` win) |
| c=2 × 0 | 234.4 | 227.2 | +3.2 % |
| c=4 × 0 | 462.5 | 449.8 | +2.8 % |
| c=8 × 0 | 865.8 | 875.0 | −1.1 % (`MTP=3` retakes) |
| c=16 × 0 | 1,329.7 | 1,520.6 | **−12.6 %** |
| c=32 × 0 | 1,726.5 | 2,083.7 | **−17.1 %** |

**Crossover concurrency: c=8.** Above this, `MTP=3` dominates. Below, `MTP=5`
has a small advantage. Production traffic on coding agents typically bursts
to c=16+, which makes `MTP=3` the correct production choice. For one-user OpenCode fanout, `thinking_token_budget=2048` keeps p50 essentially flat through c=8 (18.4s → 19.6s).

### 1.3 Long-context throughput records — `FP8+DFlash N=8` wins

For long contexts (ctx ≥ 64k), the FP8+DFlash pairing measured in the
[stress-validation study § 13][day2] beats `FP8+MTP=3` by a meaningful margin:

| concurrency × context | FP8+DFlash N=8 tok/s | FP8+MTP=3 tok/s | DFlash advantage |
|---|---:|---:|---:|
| c=4 × 64k | best | (baseline) | +5–8 % |
| c=4 × 131k | best | (baseline) | +10–13 % |

> Source: [`qwen36-27b-blackwell-stress-validation` § 13][day2] addendum.

---

## 2. Correctness SOTA (Qwen3.6-27B)

### 2.1 HumanEval (164 problems) — **corrected 2026-05-11**

Two reportings shown: the harness-bug-affected original numbers (struck through) and the corrected post-smart-glue numbers from the offline rescore + online re-bench.

| Pass rate (corrected) | Original (buggy) | Pass count | Config | Source |
|----------:|----------:|-----------:|--------|--------|
| **95.7 %** ⭐ | ~~75.6 %~~ | 157/164 | FP8+MTP=5 (offline rescore) | [stress-validation § main][day2] · [ADDENDUM][addendum] |
| **95.1 %** | ~~74.4 %~~ | 156/164 | BF16+DFlash N=8 mt=8192 (offline rescore) | [v2-followup quality-rerun][v2f] · [ADDENDUM][addendum] |
| **93.9 %** | ~~74.4 %~~ | 154/164 | FP8+DFlash N=8 (offline rescore) | [stress-validation § 13][day2] · [ADDENDUM][addendum] |
| **93.3 %** | ~~70.7 %~~ | 153/164 | FP8+MTP=3 mt=8192 on `:v2` (offline rescore) | [v2-followup quality-main][v2f] · [ADDENDUM][addendum] |
| **92.1 %** | ~~79.3 %~~ | 151/164 | FP8+MTP=3 on `:latest` (offline rescore) | [stress-validation § main][day2] · [ADDENDUM][addendum] |
| **93.3 %** | n/a | 153/164 | FP8+MTP=5 on **`:v3`** (online, patched harness, **Tier 3**, ⚠️ benchmark-only — leaks `<think>` in production `content`) | [v3suite config-4][v3suite] |
| **92.7 %** | n/a | 152/164 | BF16+DFlash N=8 on **`:v3`** (online, patched harness, **Tier 3**) | [v3suite config-1][v3suite] |
| **90.9 %** | n/a | 149/164 | BF16+DFlash N=8 **mt=16384** on `:v2` (online, patched harness, **Tier 2**) | [v2-followup tier2][v2f] |
| **89.0 %** | n/a | 146/164 | FP8+DFlash N=8 on **`:v3`** (online, patched harness, **Tier 3**) | [v3suite config-2][v3suite] |
| **88.4 %** ⭐ | n/a | 145/164 | **FP8+MTP=3 on `:v3` (online, patched harness, Tier 3) — DEPLOYED PRODUCTION SOTA** | [v3suite config-3][v3suite] |
| 87.2 % | n/a | 143/164 | BF16+DFlash N=8 mt=8192 on `:v2` (online, patched harness) | [v2-followup tier1][v2f] |
| 84.8 % | n/a | 139/164 | FP8+MTP=3 **mt=16384** on `:v2` (online, patched harness, **Tier 2**) | [v2-followup tier2][v2f] |
| 83.5 % | n/a | 137/164 | FP8+MTP=3 mt=8192 on `:v2` (online, patched harness) | [v2-followup tier1][v2f] |

**Capability ceiling — pass@5 (any of 5, temp=0.8):**

| Pass rate | Pass count | Config | Source |
|----------:|-----------:|--------|--------|
| **96.95 %** ⭐ | 159/164 | FP8+MTP=3 mt=8192 on `:latest`, n=5 samples × 164 problems | [v2-followup pass@5][v2f] |

> **Online ceiling clarification (Tier 2, 2026-05-11):** Doubling `max_tokens` from 8192 to 16384 eliminated all length-truncation failures (`length_truncated: 13 → 0`) but only fully recovered the lost passes for BF16+DFlash (+3.7 pp → 90.9 %). FP8+MTP=3 converted truncations into `empty_response` (7 → 15) instead of passes, gaining only +1.3 pp. **The offline-rescore 95.1 % is the theoretical ceiling, not the reproducible online quality** — the online BF16+DFlash N=8 pass@1 SOTA on `:v2` was **90.9 %**.

> **v3 ceiling update (Tier 3, 2026-05-12):** Re-running the same configs on `repne/vllm:v3` lifted both ceilings cleanly. BF16+DFlash N=8: **92.7 %** (+1.8 pp vs v2). FP8+MTP=3: **88.4 %** (+3.6 pp vs v2). FP8+MTP=5 scored 93.3 % on the harness but leaks `<think>` into `content` in production — see [Production Incident][v3incident] — so the **deployable** online pass@1 SOTA is **88.4 % (FP8+MTP=3 on `:v3`)**, currently in production. All four v3 configs had **0 length_truncated** on both HE and MBPP.

### 2.2 MBPP (257 sanitized problems)

| Pass rate | Pass count | Config | Source |
|----------:|-----------:|--------|--------|
| **91.1 %** ⭐ | 234/257 | BF16+DFlash N=8 on **`:v3`** (Tier 3) | [v3suite config-1][v3suite] |
| 90.3 % | 232/257 | BF16+DFlash N=8 @ max_tokens=8192 on `:v2` | [v2-followup study](https://github.com/jcartu/qwen-bench-2026-05-11-v2-followup) |
| **89.1 %** ⭐ | 229/257 | **FP8+MTP=3 on `:v3` (Tier 3) — production SOTA** | [v3suite config-3][v3suite] |
| 88.7 % | 228/257 | FP8+DFlash N=8 on **`:v3`** (Tier 3) | [v3suite config-2][v3suite] |
| 87.2 % | 224/257 | FP8+MTP=5 on **`:v3`** (Tier 3, ⚠️ benchmark-only — think-token leak) | [v3suite config-4][v3suite] |
| 89.5 % | 230/257 | BF16+DFlash N=8 | [stress-validation § main][day2] |
| 89.5 % | 230/257 | BF16+DFlash N=15 | [stress-validation § main][day2] |
| 89.1 % | 229/257 | BF16+DFlash N=7 | [stress-validation § main][day2] |
| 88.7 % | 228/257 | BF16+DFlash N=7 (no-gumbel) | [stress-validation § 12][day2] |
| 88.3 % | 227/257 | FP8+DFlash N=8 (no-gumbel) | [stress-validation § 13][day2] |
| 86.0 % | 221/257 | FP8+MTP=5 | [stress-validation § main][day2] |
| 85.6 % | 220/257 | FP8+MTP=3 | [stress-validation § main][day2] |

### 2.3 The throughput-vs-correctness trade-off

Note `FP8+MTP=5` produces **+15 % higher raw effective tok/s** than
`FP8+MTP=3` but loses HumanEval by **−3.7 pp**. This is the textbook lesson:
**throughput-on-decode-bench does not predict end-to-end agentic correctness.** The 2026-05-15 addendum extends that lesson to request sampling: unbounded thinking looked like an engine leak, but the fix was a per-request hard thinking budget, not another engine rollback.

---

## 3. Quality SOTA — 8-bit weight quantization is empirically lossless

Wikitext-2 perplexity (102,200 token positions, 200 sliding windows × 511
positions, ctx=512, stride=128):

| Quant | Perplexity | KLD vs BF16 | Top-1 agreement |
|---|---:|---:|---:|
| BF16 GGUF (reference) | 7.620 ± 0.062 | 0 | 100 % |
| Q8_0 GGUF | 7.623 ± 0.063 | **0.001828 ± 0.000189** | **97.9 %** |
| FP8 W8A8 (vLLM) | not directly measurable¹ | inferred ≈ Q8 | inferred ≈ 97–98 % |

¹ The AesSedai perplexity tool reads GGUF only, so we cannot directly perplex
the FP8 W8A8 vLLM weights. Quality is inferred from (a) Qwen team's own
model-card claim of "performance metrics nearly identical to original",
(b) Phase B functional-test parity (8/8 hard tests pass on FP8), and (c) the
principle that any 8-bit quant near BF16 should also be near BF16's KLD floor.

> Source: [`day1-sprint § exp 07 (quality sprint)`][day1].

---

## 4. Cross-quant single-best-cell comparison

| Configuration | Best tok/s | Best cell | Mean across c=1–4 (9 cells) | Verdict |
|---|---:|---|---:|---|
| **FP8+MTP=3 (Repne)** | **2,083.7** | c=32 × 0 | 252.7 | **Production SOTA** |
| FP8+MTP=5 (Repne) | 1,726.5 | c=32 × 0 | 257.3 | Wins c=1–4 narrowly, loses c≥8 |
| FP8+no-spec (Repne) | 1,875.5 | c=32 × 0 | ~150 | No-spec floor |
| BF16+DFlash=7 (Repne) | 344.9 | c=4 × 0 | 197.5 | Best DFlash variant (BF16) |
| BF16+DFlash=8 (Repne) | 358.4 | c=4 × 0 | 194.1 | Slightly behind n=7 |
| BF16+DFlash=15 (Repne) | 313.4 | c=4 × 0 | 178.5 | Repne's recommended config — actually worst of three |
| FP8+MTP=3 (upstream v0.20.1) | 413.8 | c=4 × 0 | not measured at depth | Viable fallback |
| BF16+DFlash=8 (upstream v0.20.1) | 290.9 | c=4 × 0 | n/a | **Long-context broken**, drafter accepts 1–3 % at 131k |
| NVFP4+MTP=3 (upstream v0.20.1) | 416.6 | c=4 × 0 | drops to 95–106 by c=2 ctx=131k | **Broken at 244k**, disqualified |
| NVFP4+MTP=3 (Repne) | 175.8 | c=4 × 0 | n/a | 50 % worse than upstream — Repne flags hurt NVFP4 |

---

## 5. Operational stability records

| Record | Value | Source |
|--------|-------|--------|
| Hardest validation completed without engine failure | **2,105 hard problems** across 5 configs, **0 crashes / 0 hangs / 0 malformed JSON** | [stress-validation § main][day2] |
| Longest stable single-context generation | 137K-token needle-in-haystack found in 23.3s | [day1-sprint § exp 03][day1] |
| Fastest cold start (FP8 27B TP=2) | 105s with `--load-format instanttensor` (210s without) | [day1-sprint § exp 06][day1] |
| Longest single-image clean suite (v3) | **3 h 32 min, 4 configs, 1,684 quality probes (4×HE 164 + 4×MBPP 257), 0 engine errors on Run 2** | [v3suite][v3suite] |
| Low-frequency v3 transient (filed) | Run 1 BF16+DFlash N=8 EngineCore HTTP 500 at HE 34/164; non-reproducible; stack trace lost (launcher now hardened) | [v3suite run1-CRASHED][v3suite] |
| Production-incident finding (v3 rollout) | `use_local_argmax_reduction` is **DFlash-only** — `Qwen3_5MTP` drafter does not implement `get_top_tokens()`, engine refuses to start MTP config with that flag | [v3suite Production Incident][v3incident] |
| Production-incident finding (v3 rollout) | MTP=5 leaks raw `<think>...</think>` blocks into OpenAI `content` field on production traffic despite 93.3 % offline HE | [v3suite Production Incident][v3incident] |
| Production-incident finding (v3 rollout) | MTP=3 validated **leak-free at 420 trials** across chat AND tool-calling traffic by permanent dual-mode probe (`harness/leak_probe.py`); MTP=5 leak class appears MTP=5-specific on `:v3` | [v3suite LEAK_DETECTION][v3leak] |
| Single-user coding stability | `thinking_token_budget=2048` eliminates runaway tails on the hard-coding probe: stuck 10%→0%, pass 70%→80%, c=1 p95 130.8s→19.6s | [single-user addendum][singleuser] |

---

## 6. Practical decision tree

```
Is this Blackwell SM120 hardware?
├── Yes → continue
└── No  → results may not be directly applicable; retest

Need maximum throughput at c≥8 production traffic?
├── Yes → FP8+MTP=3 on Repne fork (2,083 tok/s peak; v3 image deployed, leak-free)
└── No  → continue (single-user / always c≤4)

Always single-user / OpenCode coding priority?
├── Yes → FP8+MTP=3 on `repne/vllm:v3` + client `thinking_token_budget=2048`. Keep thinking enabled; do NOT use MTP=5 — leaks `<think>` into `content`.
└── No  → FP8+MTP=3 on Repne fork (still wins majority of cells)

Long-context coding-agent workload (ctx ≥ 64k)?
├── Yes → FP8+DFlash N=8 on Repne fork (+5-13% vs MTP=3 at long ctx)
└── No  → FP8+MTP=3

Need quality matching BF16 exactly?
├── Yes → BF16 (no spec) or BF16+DFlash on Repne; expect ~1/2 throughput
└── No  → FP8+MTP=3 (KLD ≈ Q8 ≈ noise floor)

Stuck on upstream vLLM (no Repne available)?
├── FP8+MTP=3 path is viable, expect 5–14% throughput cost
├── BF16+DFlash path is NOT viable (drafter collapses past 32k)
└── NVFP4 path is NOT viable (engine instability past 131k)
```

---

## 7. How records get updated

When a new study breaks a record, the integration steps are:

1. The study's own README documents the new number with full reproducibility info.
2. A PR against this hub updates the relevant table in this file.
3. The previous record-holder is **kept** in the table with a strikethrough or "previous SOTA" annotation — we do not erase history.
4. The study repo is added to `STUDIES.md` with the record-broken note.

This file is the **canonical leaderboard**. Studies are the **canonical evidence**.

---

## 8. Source pointers

[day1]: https://github.com/jcartu/qwen36-27b-blackwell-inference-study
[day2]: https://github.com/jcartu/qwen36-27b-blackwell-stress-validation
[v2f]: https://github.com/jcartu/qwen-bench-2026-05-11-v2-followup
[addendum]: https://github.com/jcartu/qwen-bench-2026-05-11-v2-followup/blob/main/ADDENDUM.md
[v3suite]: https://github.com/jcartu/qwen-bench-2026-05-12-v3-suite
[v3report]: https://github.com/jcartu/qwen-bench-2026-05-12-v3-suite/blob/main/FINAL_REPORT.md
[v3incident]: https://github.com/jcartu/qwen-bench-2026-05-12-v3-suite/blob/main/FINAL_REPORT.md#production-incident-2026-05-12-mtp--use_local_argmax_reduction-incompatibility
[v3leak]: https://github.com/jcartu/qwen-bench-2026-05-12-v3-suite/blob/main/LEAK_DETECTION.md
[singleuser]: studies/2026-05-15-single-user-thinking-budget/

- **All Qwen3.6-27B Day 1 sprint data** (per-cell N=3 production data, KLD probe, high-concurrency sweep, MTP n-sweep): [`qwen36-27b-blackwell-inference-study`][day1]
- **All Qwen3.6-27B stress-validation data** (5 configs × HumanEval × MBPP, plus addenda): [`qwen36-27b-blackwell-stress-validation`][day2]
- **Single-user thinking-budget addendum:** [`studies/2026-05-15-single-user-thinking-budget/`][singleuser]
- **Merged machine-readable CSVs:** [`data/`](data/) in this hub repo
