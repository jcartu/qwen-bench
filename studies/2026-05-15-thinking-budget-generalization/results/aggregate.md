# Thinking Budget Generalization Results

Cross-domain validation of `thinking_token_budget=2048` vs unbounded on
the Qwen3.6-27B (FP8, TP=2, MTP=3, repne/vllm:v3) single-user SOTA stack.

- Endpoint: `http://127.0.0.1:11435/v1/chat/completions`
- Sampling: temperature=0.6, top_p=0.95, top_k=20, repetition_penalty=1.05
- Thinking: `chat_template_kwargs.enable_thinking=true`
- max_tokens: 16384, concurrency: 8

## Headline comparison

| Benchmark | n | C0 unbounded acc | C0 95% CI | C1 tb=2048 acc | C1 95% CI | Δ acc (pp) | two-prop z | p (two-sided) | C0 p50 lat (s) | C1 p50 lat (s) | C0 stuck | C1 stuck |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gpqa | 198/198 |  40.4% | [33.8,47.4] |  74.2% | [67.7,79.8] | +33.8 | 6.81 | 1.0e-11 | 196.9 | 31.2 |  57.1% |   3.0% |
| gsm_plus | 2000/2000 |  79.7% | [77.8,81.4] |  80.8% | [79.0,82.4] | +1.1 | 0.87 | 0.383 | 24.0 | 22.2 |  11.6% |   3.3% |
| mmlu_pro | 1400/1400 |  73.1% | [70.8,75.4] |  82.6% | [80.6,84.5] | +9.5 | 6.06 | 1.4e-09 | 69.8 | 27.2 |  16.5% |   1.3% |

## Per-domain accuracy

### gpqa

| Domain | n | C0 unbounded | C1 tb=2048 | Δ pp |
|---|---:|---:|---:|---:|
| Biology | 19 |  36.8% |  68.4% | +31.6 |
| Chemistry | 93 |  23.7% |  63.4% | +39.8 |
| Physics | 86 |  59.3% |  87.2% | +27.9 |

### gsm_plus

| Domain | n | C0 unbounded | C1 tb=2048 | Δ pp |
|---|---:|---:|---:|---:|
| adding operation | 250 |  81.2% |  84.8% | +3.6 |
| critical thinking | 250 |   0.0% |   0.0% | +0.0 |
| digit expansion | 250 |  93.6% |  94.0% | +0.4 |
| distraction insertion | 250 |  93.2% |  94.8% | +1.6 |
| integer-decimal-fraction conversion | 250 |  96.4% |  96.4% | +0.0 |
| numerical substitution | 250 |  88.4% |  89.2% | +0.8 |
| problem understanding | 250 |  95.2% |  96.4% | +1.2 |
| reversing operation | 250 |  89.2% |  90.4% | +1.2 |

### mmlu_pro

| Domain | n | C0 unbounded | C1 tb=2048 | Δ pp |
|---|---:|---:|---:|---:|
| biology | 100 |  88.0% |  92.0% | +4.0 |
| business | 100 |  80.0% |  84.0% | +4.0 |
| chemistry | 100 |  75.0% |  87.0% | +12.0 |
| computer science | 100 |  73.0% |  83.0% | +10.0 |
| economics | 100 |  81.0% |  88.0% | +7.0 |
| engineering | 100 |  49.0% |  69.0% | +20.0 |
| health | 100 |  68.0% |  73.0% | +5.0 |
| history | 100 |  70.0% |  81.0% | +11.0 |
| law | 100 |  54.0% |  68.0% | +14.0 |
| math | 100 |  76.0% |  91.0% | +15.0 |
| other | 100 |  79.0% |  87.0% | +8.0 |
| philosophy | 100 |  79.0% |  86.0% | +7.0 |
| physics | 100 |  72.0% |  88.0% | +16.0 |
| psychology | 100 |  80.0% |  80.0% | +0.0 |

## Latency / token deltas

| Benchmark | p50 lat ratio (C1/C0) | p95 lat ratio | p50 comp tok ratio | p95 comp tok ratio |
|---|---:|---:|---:|---:|
| gpqa | 0.16x | 0.47x | 0.17x | 0.56x |
| gsm_plus | 0.92x | 0.37x | 0.99x | 0.40x |
| mmlu_pro | 0.39x | 0.18x | 0.44x | 0.20x |

