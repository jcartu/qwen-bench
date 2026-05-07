# Contributing to `qwen-bench`

This document is the canonical reference for **how new studies enter the hub** and
**how existing URLs are kept stable**. It exists so that future-you (or anyone
else publishing a study) doesn't have to reverse-engineer the conventions from
the existing repos.

---

## TL;DR

1. Name new study repos `qwen-bench-{YYYY-MM}-{slug}`
2. Existing study repos are **never renamed retroactively** — their URLs are stable forever
3. Open a PR against this hub adding: STUDIES.md entry, SOTA.md update (if records broke), `data/{YYYY-MM}-{slug}.csv`, README block
4. Add the "← qwen-bench hub" badge to the new study's README

---

## Naming convention (for studies created from 2026-05 forward)

```
qwen-bench-{YYYY-MM}-{slug}
```

| Component | Rule | Example |
|---|---|---|
| Prefix | Always `qwen-bench-` | `qwen-bench-` |
| Date | `YYYY-MM` of the **first commit / first measurement** | `2026-06` |
| Slug | `kebab-case`, ≤ 4 words, describes the *primary intervention or question* | `mtp-tail-latency` |

**Full examples:**

```
qwen-bench-2026-06-mtp-tail-latency
qwen-bench-2026-07-blackwell-fp4-vs-fp8
qwen-bench-2026-08-long-context-rope-scaling
```

### Slug guidelines

- Describe **what makes this study different**, not what it tested in general
  - ✅ `mtp-tail-latency` — distinguishes it from prior MTP throughput study
  - ❌ `qwen-benchmark-results` — uninformative, redundant with prefix
- Prefer a noun phrase over a sentence
  - ✅ `bf16-dflash-vs-upstream`
  - ❌ `does-bf16-dflash-beat-upstream`
- If two studies share a slug, append `-v2`, `-v3`. Don't try to retroactively rename v1.

### Why this convention

- **Sortable**: `ls` / GitHub repo list orders chronologically by default
- **Greppable**: `gh repo list jcartu --search "qwen-bench-"` finds every study in the series
- **Self-dating**: future-you doesn't have to guess when something ran
- **Topic-aligned**: matches the `qwen` / `benchmark` / `blackwell` GitHub topics applied across the family

---

## URL stability guarantee (existing repos)

Studies published **before** the convention was adopted keep their original
names permanently:

| Original repo | Status |
|---|---|
| `qwen36-27b-blackwell-inference-study` | **Stable URL — never renamed** |
| `qwen36-27b-blackwell-stress-validation` | **Stable URL — never renamed** |
| `qwen36-27b-bf16-dflash-repne-vs-upstream` | **Stable URL — never renamed** |
| `qwen36-27b-fp8-repne-vs-upstream` | **Stable URL — never renamed** |
| `qwen36-27b-nvfp4-mtp-experiment` | **Stable URL — never renamed** |
| `repne-dflash-newimage` | **Stable URL — never renamed** |
| `closing-the-opus-gap` | **Stable URL — never renamed** |
| `llm-stress-harness` | **Stable URL — toolkit, naming convention does not apply** |

### Rationale

GitHub auto-redirects renamed repos *for HTML traffic only*. Citations in
**Discord, Slack, blog posts, papers, Twitter, and `git clone` commands**
become non-canonical the moment you rename, even though the redirect technically
works. Renaming for the sake of consistency would inflict permanent low-grade
breakage on every external reference for zero benefit to readers.

The hub's bidirectional badge graph (every study links back to the hub, hub
indexes every study) provides discoverability *without* requiring URL surgery.

---

## Per-study integration checklist

When you publish a new study, do these in order:

### 1. Create the study repo

```bash
gh repo create jcartu/qwen-bench-2026-MM-{slug} --public --description "..."
```

### 2. Add the hub badge to the study's README (line 1)

```markdown
[![← qwen-bench hub](https://img.shields.io/badge/%E2%86%90-qwen--bench_hub-blue?style=for-the-badge)](https://github.com/jcartu/qwen-bench)
```

### 3. Set GitHub topics on the study repo

```bash
gh api -X PUT repos/jcartu/qwen-bench-2026-MM-{slug}/topics \
  -f names[]=qwen -f names[]=qwen3 -f names[]=vllm \
  -f names[]=blackwell -f names[]=benchmark \
  -f names[]={study-specific-tag-1} -f names[]={study-specific-tag-2}
```

### 4. Open a PR against `qwen-bench` (this hub)

The PR should touch four things:

- `STUDIES.md` — append a new `## 2026-MM · {Title}` block with abstract,
  headline numbers, and links
- `SOTA.md` — **only if** any record was broken; otherwise leave alone
- `data/2026-MM-{slug}.csv` — the canonical results CSV from the study
- `README.md` — bump the `studies-N_published` shield count, and (optionally)
  add the study to the [Studies (chronological)](README.md#studies-chronological)
  section

### 5. After merge, verify

- Hub README renders the new shield count
- All links from STUDIES.md → study repo resolve (HTTP 200)
- Study repo's hub badge resolves back to this repo
- `data/2026-MM-{slug}.csv` is byte-identical to the source-of-truth CSV in the
  study repo (use `md5sum` on both)

---

## Toolkit repos (not studies)

`llm-stress-harness` is a **toolkit**, not a study, and is intentionally not
namespaced under `qwen-bench-`. If new toolkit repos appear (e.g. a SOTA-diff
linter, a results-merger CLI), they get descriptive names of their own and
are listed under [Tools](README.md#tools) in the hub README.

---

## Future automation (not yet implemented)

- **`scripts/check_sota.py`** — compares a new study's CSV against the merged
  hub data and flags any record-breaking row that didn't update `SOTA.md`
- **GitHub Action** on PR-open: runs the above check, posts a comment with the
  diff
- **GitHub Pages site** at `qwen-bench.jcartu.dev` rendering `SOTA.md` as a
  sortable HTML leaderboard

These are deferred until the third or fourth study lands; doing them now would
be premature optimization.

---

## Questions

Open an issue on this repo or ping `@jcartu` directly.
