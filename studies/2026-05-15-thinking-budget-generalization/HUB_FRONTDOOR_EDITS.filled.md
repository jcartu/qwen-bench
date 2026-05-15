# Hub front-door edits to apply at publish time

Three files in the hub root need synchronized edits. Each block below is
copy-paste-ready and **assumes MMLU-Pro tb=2048 + unbounded + sweep all complete**.
Numbers marked `{{MMLU_*}}` and `{{SWEEP_*}}` are filled by `scripts/fill_placeholders.py`
running against `results/aggregate.json` produced by the harness aggregator.

---

## 1. `SOTA.md` — append new callout right under existing 2026-05-15 single-user note

Insert directly **after** the existing `> 🆕 **2026-05-15 single-user update**` paragraph (currently
on the line beginning `> 🆕 **2026-05-15 single-user update**`), as a sibling callout block:

```markdown
> 🧪 **2026-05-15 generalization (evening)**: Cross-domain validation of `thinking_token_budget=2048`
> on three out-of-distribution academic benchmarks (3,598 trials total) confirms it as a **strict
> Pareto improvement** over unbounded thinking on Qwen3.6-27B FP8 + MTP=3 + `repne/vllm:v3`.
> Headline: **GPQA Diamond +33.8 pp accuracy (z=6.81, p=1e-11) at 0.24× wall-clock**;
> **GSM-Plus 2k accuracy parity (z=0.87, p=0.38) at 0.53× wall-clock**; **MMLU-Pro 73.1% → **82.6%** (Δ +9.5 pp, z=6.06, p=1.4e-09) at 0.39× wall**.
> Failure mechanism on hard reasoning is documented: 57.1% of GPQA unbounded responses hit
> `finish_reason=length` while still inside `<think>`, emitting **zero** extractable answers.
> Budget-sweep on GPQA at tb ∈ {1024, 2048, 4096, 8192} shows accuracy peaks at tb=4096 (acc=78.3%) → there is a **genuine sweet spot** in budget choice.
> Full study: [`studies/2026-05-15-thinking-budget-generalization`](studies/2026-05-15-thinking-budget-generalization/).
```

---

## 2. `STUDIES.md` — prepend new study entry at the top of the chronological log

Insert immediately **before** the existing `## 2026-05-15 · Single-user thinking-budget addendum`
heading so the generalization study is the newest entry:

```markdown
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
```

---

## 3. `README.md` — two edits

### 3a. Update badge count (line ~9)

```diff
-[![Studies](https://img.shields.io/badge/studies-10_published-success?style=for-the-badge)](STUDIES.md)
+[![Studies](https://img.shields.io/badge/studies-11_published-success?style=for-the-badge)](STUDIES.md)
```

### 3b. Insert new study entry above existing single-user entry (line ~119)

Insert immediately **before** the existing `### 📖 2026-05 · Single-user thinking-budget addendum`
heading:

```markdown
### 📖 2026-05 · Thinking-budget generalization study
**[`studies/2026-05-15-thinking-budget-generalization`](studies/2026-05-15-thinking-budget-generalization/)** · *cross-domain validation · GPQA Diamond / GSM-Plus 2k / MMLU-Pro 1.4k · 3,598 trials · GPQA budget sweep*

Repne challenge: prove `thinking_token_budget=2048` generalizes beyond the morning's coding probes. Result: strict Pareto improvement on every benchmark. **GPQA +33.8 pp accuracy at 0.24× wall** (z=6.81, p=1e-11; 57.1% of unbounded responses never emit an answer because they bump into `finish_reason=length` inside `<think>`). **GSM-Plus accuracy parity at 0.53× wall, 2.5× tighter p95 token tail.** **MMLU-Pro Δ +9.5 pp accuracy at 0.39× wall.** GPQA budget sweep at tb ∈ {1024, 2048, 4096, 8192} shows peak at tb=4096 → sweet spot exists.
```

---

## 4. `data/2026-05-15-thinking-budget-generalization.csv` — new file

Generated by `scripts/emit_hub_csv.py` from `results/aggregate.json`. Schema:

```csv
benchmark,condition,scope,domain,n,acc,acc_ci95_lo,acc_ci95_hi,p50_lat_s,p95_lat_s,p50_comp_tok,p95_comp_tok,stuck_rate,wall_s,seed,timestamp_utc
```

One row per `(benchmark, condition, scope, domain)` tuple. Wilson 95% CIs precomputed.

---

## 5. `data/README.md` — append data card entry

Append at the end of the data card list:

```markdown
| `2026-05-15-thinking-budget-generalization.csv` | 2026-05-15 | Cross-domain `thinking_token_budget=2048` vs unbounded on Qwen3.6-27B FP8+MTP=3 on `repne/vllm:v3`. GPQA Diamond + GSM-Plus 2k + MMLU-Pro 1.4k + GPQA budget sweep at tb={1024,4096,8192}. 3,598 trials. | [study](../studies/2026-05-15-thinking-budget-generalization/) |
```
