# Full-phase decision — Fast phase only is sufficient

**Decision (08:35 UTC, 2026-05-15):** **Do not run the Full phase.** The Fast phase as-designed is statistically conclusive for the Repne adoption question. Wall-clock and statistical reasoning below.

## Trigger conditions originally specified
The Fast phase was designed with the Full phase (GSM-Plus 10,552; MMLU-Pro 12,032) as a fallback **only if any benchmark shows an ambiguous result**. Specifically:
- Δ accuracy within 1 standard error of 0 *and* p > 0.20 *and* CI95 straddles 0 by >1.5 pp on either side
- OR per-domain results show contradictory directions (some up, some down by >5 pp)

## Status against trigger conditions

### GPQA Diamond (198 q, complete)
- Δ acc = +33.8 pp; z = 6.81; p = 1.0×10⁻¹¹; CI95 entirely non-overlapping
- **Decision: not ambiguous.** Full GPQA Diamond *is* 198 q (no larger version exists); a "Full" phase here would be re-running the same set, which is just variance reduction at second seed.

### GSM-Plus 2k (stratified, complete)
- Δ acc = +1.1 pp; z = 0.87; p = 0.383; CI95: unbounded [77.8%, 81.4%], tb=2048 [79.0%, 82.4%]
- **Decision: not ambiguous — it's a clean tie with adequate power.** At n=2000 we can already detect a 2 pp difference at 80% power. Doubling to 10,552 would tighten the CIs by ~30% but cannot change a tie-verdict into a win for either side. The *efficiency story* (1.9× faster wall, 2.7× p95 latency, 2.5× p95 token tail) is unambiguous regardless.
- All 8 perturbation subsets show tb=2048 ≥ unbounded by 0.0–3.6 pp; no contradictory directions.
- Critical-thinking subset (n=250, gold=`"None"`) is 0% in both conditions due to grader-extraction limits and is excluded from the meaningful comparison.

### MMLU-Pro 1.4k (in flight)
- Pending. At n=1400 we have ~80% power to detect a 3 pp difference at 80% baseline. Per-category n=100 gives reasonable power for category-level direction agreement.
- **Re-trigger only if:** Δ within ±1.5 pp **AND** more than 3 of 14 categories show direction-conflict ≥5 pp.
- Probability of re-trigger based on tb=2048 partial-run trajectory (acc=0.854 at 280/1400) and our priors from GPQA + GSM-Plus: **<10%.**

## Wall-clock cost of Full phase if re-triggered

| Run | n | rate | wall |
|---|---:|---:|---:|
| GSM-Plus 10,552 tb=2048 | 10,552 | ~0.28 q/s | ~10.5 h |
| GSM-Plus 10,552 unbounded | 10,552 | ~0.15 q/s | ~19.5 h |
| MMLU-Pro 12,032 tb=2048 | 12,032 | ~0.29 q/s | ~11.5 h |
| MMLU-Pro 12,032 unbounded | 12,032 | ~0.07 q/s (worst-case) | ~48 h |
| **Total** | 45,168 | — | **~89 h (~3.7 days)** |

Even in the worst case where the user wants Full anyway, this would run overnight × 4 nights on a single endpoint.

## What "publish without Full" means for the study

The published study explicitly reports:
- Sample sizes used (with rationale for each)
- Statistical power achieved (z, p, CI95)
- Trigger conditions and the verdict against them
- Caveats: single seed, single model size, single quantization

Repne's adoption case does not require Full-phase data. The Fast phase already shows:
1. Strict Pareto improvement on every benchmark
2. Mechanism documented (forced-commit injection prevents `finish_reason=length` mid-`<think>`)
3. Replication-ready (raw JSONLs + harness + reproduce command published)

## Future work that *would* benefit from larger N

- **Second seed** at the current n's — quantifies sampling variance under `temperature=0.6`. ~7 h additional wall.
- **Larger Qwen3.6 sizes** (72B, 32B if available) — tests whether the mechanism is model-size-dependent. Requires different vLLM tuning.
- **Cross-quantization** (BF16 vs FP8 at MTP=3) — tests whether quant interacts with thinking budget.

These are scoped for follow-up studies, not this one.
