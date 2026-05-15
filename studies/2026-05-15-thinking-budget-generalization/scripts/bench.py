#!/usr/bin/env python3
"""
Cross-domain thinking-budget generalization harness.

Compares two conditions on three benchmarks against a vLLM OpenAI-compatible
endpoint:
  C0 (unbounded):  thinking_token_budget NOT set
  C1 (tb2048):     thinking_token_budget=2048

Benchmarks:
  gpqa       hendrydong/gpqa_diamond_mc       (198, MC A-D, \\boxed{LETTER})
  gsm_plus   qintongli/GSM-Plus               (sampled, numeric, ####answer)
  mmlu_pro   TIGER-Lab/MMLU-Pro               (sampled per-category, MC A-J)

Writes per-trial JSONL and aggregated summary.json. Resumable: appends to
existing JSONL and skips already-graded question_ids.

Usage:
  python3 bench.py --benchmark gpqa --condition tb2048 --concurrency 4 \
      --out runs/gpqa/tb2048

Reproducibility:
  - Deterministic question order
  - Fixed sampling seed (42)
  - Same prompts across conditions, only thinking_token_budget differs
"""
from __future__ import annotations
import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import httpx

# -- env / defaults -----------------------------------------------------------

DEFAULT_ENDPOINT = os.environ.get("VLLM_ENDPOINT", "http://127.0.0.1:11435/v1/chat/completions")
DEFAULT_MODEL    = os.environ.get("VLLM_MODEL", "Qwen3.6-27B")
DEFAULT_MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "16384"))
DEFAULT_TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.6"))
DEFAULT_TOP_P      = float(os.environ.get("TOP_P", "0.95"))
DEFAULT_TOP_K      = int(os.environ.get("TOP_K", "20"))
DEFAULT_REPETITION_PENALTY = float(os.environ.get("REPETITION_PENALTY", "1.05"))
DEFAULT_REQUEST_TIMEOUT_S = float(os.environ.get("REQUEST_TIMEOUT_S", "600"))

SEED = 42

# -- dataset loaders ----------------------------------------------------------

def _hf_load(repo: str, split: str, name: Optional[str] = None):
    import datasets
    os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent.parent / "data_cache"))
    kwargs = {"name": name} if name else {}
    return datasets.load_dataset(repo, split=split, **kwargs)


def load_gpqa() -> List[Dict[str, Any]]:
    """Official Idavidrein/gpqa_diamond. Each row contains 'Correct Answer'
    + 3 'Incorrect Answer N' fields. We deterministically shuffle to A/B/C/D
    using a per-question seed so both conditions see the SAME prompt and the
    same gold letter."""
    ds = _hf_load("Idavidrein/gpqa", split="train", name="gpqa_diamond")
    items: List[Dict[str, Any]] = []
    for i, r in enumerate(ds):
        correct = (r.get("Correct Answer") or "").strip()
        incorrect = [
            (r.get("Incorrect Answer 1") or "").strip(),
            (r.get("Incorrect Answer 2") or "").strip(),
            (r.get("Incorrect Answer 3") or "").strip(),
        ]
        if not correct or any(not x for x in incorrect):
            continue
        choices = [correct] + incorrect
        # per-question deterministic shuffle
        rng = random.Random(SEED * 1000 + i)
        idx = list(range(4))
        rng.shuffle(idx)
        shuffled = [choices[j] for j in idx]
        gold_pos = idx.index(0)  # where the correct answer landed
        gold_letter = "ABCD"[gold_pos]
        opt_block = "\n".join(f"({chr(65 + j)}) {c}" for j, c in enumerate(shuffled))
        prompt = (
            f"{(r.get('Question') or '').strip()}\n\n"
            f"{opt_block}\n\n"
            "Please write your final answer in the form of "
            "\\boxed{A}, \\boxed{B}, \\boxed{C}, or \\boxed{D}."
        )
        items.append({
            "qid": f"gpqa-{i:04d}",
            "prompt": prompt,
            "gold": gold_letter,
            "domain": (r.get("High-level domain") or r.get("Subdomain") or "").strip(),
        })
    return items


def load_gsm_plus(limit: int = 2000) -> List[Dict[str, Any]]:
    """Stratified sample by perturbation_type for representativeness."""
    ds = _hf_load("qintongli/GSM-Plus", split="test")
    by_type: Dict[str, List[int]] = {}
    for i, r in enumerate(ds):
        by_type.setdefault(r["perturbation_type"], []).append(i)
    rng = random.Random(SEED)
    per_type = max(1, limit // len(by_type))
    picked: List[int] = []
    for k in sorted(by_type):
        idxs = list(by_type[k])
        rng.shuffle(idxs)
        picked.extend(idxs[:per_type])
    # trim/extend to exact limit
    picked = picked[:limit]
    picked.sort()  # deterministic order for both conditions
    items = []
    for i in picked:
        r = ds[int(i)]
        instr = (
            f"{r['question']}\n\n"
            "Solve this step by step. After your reasoning, give your final "
            "numerical answer on a line of the form: #### <answer>"
        )
        items.append({
            "qid": f"gsmplus-{i:05d}",
            "prompt": instr,
            "gold": str(r["answer"]).strip(),
            "domain": r["perturbation_type"],
        })
    return items


def load_mmlu_pro(per_category: int = 100) -> List[Dict[str, Any]]:
    """Sample fixed N per category (14 cats * N)."""
    ds = _hf_load("TIGER-Lab/MMLU-Pro", split="test")
    by_cat: Dict[str, List[int]] = {}
    for i, r in enumerate(ds):
        by_cat.setdefault(r["category"], []).append(i)
    rng = random.Random(SEED)
    picked: List[int] = []
    for k in sorted(by_cat):
        idxs = list(by_cat[k])
        rng.shuffle(idxs)
        picked.extend(idxs[:per_category])
    picked.sort()
    items = []
    for i in picked:
        r = ds[int(i)]
        opts: List[str] = r["options"]
        # MMLU-Pro supports up to 10 options labeled A-J
        letters = [chr(ord("A") + j) for j in range(len(opts))]
        opt_block = "\n".join(f"({L}) {o}" for L, o in zip(letters, opts))
        last_letter = letters[-1]
        instr = (
            f"{r['question']}\n\n"
            f"{opt_block}\n\n"
            f"Please write your final answer in the form of \\boxed{{X}} "
            f"where X is a letter from A to {last_letter}."
        )
        items.append({
            "qid": f"mmlu-{int(r['question_id']):05d}",
            "prompt": instr,
            "gold": r["answer"].strip().upper(),
            "domain": r["category"],
        })
    return items


LOADERS = {
    "gpqa":     ("gpqa_diamond_official", load_gpqa),
    "gsm_plus": ("gsm_plus_stratified", load_gsm_plus),
    "mmlu_pro": ("mmlu_pro_per_cat",    load_mmlu_pro),
}

# -- graders ------------------------------------------------------------------

_BOXED_LETTER = re.compile(r"\\boxed\{\s*([A-J])\s*\}")
_TRAILING_LETTER = re.compile(r"(?i)(?:^|\b)(?:answer|final answer)\s*[:\-]?\s*([A-J])\b")
_HASH_NUM = re.compile(r"####\s*(-?\d[\d,]*\.?\d*)")
_ANY_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def _normalize_num(s: str) -> Optional[float]:
    if s is None:
        return None
    s = s.replace(",", "").rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


def grade_mc(response: str, gold: str, max_letter: str = "J") -> Tuple[bool, Optional[str]]:
    if not response:
        return False, None
    # Prefer \boxed{X}, last occurrence
    matches = list(_BOXED_LETTER.finditer(response))
    pred: Optional[str] = None
    if matches:
        pred = matches[-1].group(1).upper()
    else:
        m2 = list(_TRAILING_LETTER.finditer(response))
        if m2:
            pred = m2[-1].group(1).upper()
    if pred is None:
        return False, None
    if pred > max_letter:
        return False, pred
    return pred == gold.upper(), pred


def grade_num(response: str, gold: str) -> Tuple[bool, Optional[str]]:
    if not response:
        return False, None
    pred_str: Optional[str] = None
    m = list(_HASH_NUM.finditer(response))
    if m:
        pred_str = m[-1].group(1)
    else:
        nums = _ANY_NUM.findall(response)
        if nums:
            pred_str = nums[-1]
    if pred_str is None:
        return False, None
    pred = _normalize_num(pred_str)
    gold_num = _normalize_num(gold)
    if pred is None or gold_num is None:
        return False, pred_str
    # Tolerance: exact for integers; 1e-4 relative for floats
    if abs(pred - gold_num) < 1e-4 or (gold_num != 0 and abs(pred - gold_num) / abs(gold_num) < 1e-4):
        return True, pred_str
    return False, pred_str


def grade(benchmark: str, response: str, gold: str) -> Tuple[bool, Optional[str]]:
    if benchmark == "gpqa":
        return grade_mc(response, gold, max_letter="D")
    if benchmark == "mmlu_pro":
        return grade_mc(response, gold, max_letter="J")
    if benchmark == "gsm_plus":
        return grade_num(response, gold)
    raise ValueError(benchmark)


# -- request runner -----------------------------------------------------------

@dataclass
class Trial:
    qid: str
    domain: str
    gold: str
    pred: Optional[str]
    correct: bool
    finish_reason: Optional[str]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    reasoning_tokens: Optional[int]
    total_tokens: Optional[int]
    latency_s: float
    err: Optional[str]
    # truncated for storage; full content saved separately if needed
    content_preview: str
    reasoning_preview: str


def _build_payload(model: str, prompt: str, thinking_budget: Optional[int], max_tokens: int) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": DEFAULT_TEMPERATURE,
        "top_p": DEFAULT_TOP_P,
        "top_k": DEFAULT_TOP_K,
        "repetition_penalty": DEFAULT_REPETITION_PENALTY,
        "max_tokens": max_tokens,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    if thinking_budget is not None:
        body["thinking_token_budget"] = thinking_budget
    return body


async def _one(
    client: httpx.AsyncClient,
    endpoint: str,
    model: str,
    item: Dict[str, Any],
    benchmark: str,
    thinking_budget: Optional[int],
    max_tokens: int,
    sem: asyncio.Semaphore,
) -> Trial:
    async with sem:
        t0 = time.perf_counter()
        body = _build_payload(model, item["prompt"], thinking_budget, max_tokens)
        err: Optional[str] = None
        content = ""
        reasoning = ""
        finish_reason = None
        usage: Dict[str, Any] = {}
        try:
            resp = await client.post(endpoint, json=body, timeout=DEFAULT_REQUEST_TIMEOUT_S)
            resp.raise_for_status()
            data = resp.json()
            choice = data["choices"][0]
            msg = choice.get("message", {})
            content = msg.get("content") or ""
            reasoning = msg.get("reasoning_content") or ""
            finish_reason = choice.get("finish_reason")
            usage = data.get("usage", {}) or {}
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:300]}"
        latency = time.perf_counter() - t0

        # Grade off content + reasoning concatenated (some models emit answer in either)
        graded_text = (content or "") + "\n" + (reasoning or "")
        correct, pred = grade(benchmark, graded_text, item["gold"])

        return Trial(
            qid=item["qid"],
            domain=item.get("domain", ""),
            gold=item["gold"],
            pred=pred,
            correct=bool(correct),
            finish_reason=finish_reason,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            reasoning_tokens=(
                usage.get("completion_tokens_details", {}).get("reasoning_tokens")
                if isinstance(usage.get("completion_tokens_details"), dict)
                else None
            ),
            total_tokens=usage.get("total_tokens"),
            latency_s=round(latency, 3),
            err=err,
            content_preview=(content or "")[:600],
            reasoning_preview=(reasoning or "")[:600],
        )


# -- summary ------------------------------------------------------------------

def _percentile(xs: List[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def _summarize(trials: List[Trial], benchmark: str, condition: str, run_meta: Dict[str, Any]) -> Dict[str, Any]:
    n = len(trials)
    ok = sum(1 for t in trials if t.correct)
    errs = [t for t in trials if t.err]
    finished = [t for t in trials if t.err is None]
    lats = [t.latency_s for t in finished]
    comp_tokens = [t.completion_tokens for t in finished if t.completion_tokens is not None]
    reas_tokens = [t.reasoning_tokens for t in finished if t.reasoning_tokens is not None]
    # bucketed accuracy by domain
    by_domain: Dict[str, Dict[str, int]] = {}
    for t in trials:
        d = by_domain.setdefault(t.domain or "(unknown)", {"n": 0, "ok": 0, "err": 0})
        d["n"] += 1
        if t.correct:
            d["ok"] += 1
        if t.err:
            d["err"] += 1
    for d, v in by_domain.items():
        v["acc"] = round(v["ok"] / max(v["n"], 1), 4)
    # stuck = finish_reason == 'length' (hit max_tokens)
    stuck = sum(1 for t in trials if t.finish_reason == "length")
    return {
        "benchmark": benchmark,
        "condition": condition,
        "n_trials": n,
        "n_errors": len(errs),
        "accuracy": round(ok / max(n, 1), 4),
        "stuck_rate": round(stuck / max(n, 1), 4),
        "latency": {
            "p50_s": round(_percentile(lats, 50), 3),
            "p95_s": round(_percentile(lats, 95), 3),
            "p99_s": round(_percentile(lats, 99), 3),
            "mean_s": round(sum(lats) / max(len(lats), 1), 3),
            "max_s": round(max(lats) if lats else 0.0, 3),
        },
        "completion_tokens": {
            "p50": round(_percentile(comp_tokens, 50), 1),
            "p95": round(_percentile(comp_tokens, 95), 1),
            "mean": round(sum(comp_tokens) / max(len(comp_tokens), 1), 1),
            "max": max(comp_tokens) if comp_tokens else 0,
        },
        "reasoning_tokens": {
            "p50": round(_percentile(reas_tokens, 50), 1) if reas_tokens else None,
            "p95": round(_percentile(reas_tokens, 95), 1) if reas_tokens else None,
            "mean": round(sum(reas_tokens) / max(len(reas_tokens), 1), 1) if reas_tokens else None,
            "max": max(reas_tokens) if reas_tokens else None,
        },
        "by_domain": by_domain,
        **run_meta,
    }


# -- main ---------------------------------------------------------------------

async def main_async(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    trials_path = out_dir / "trials.jsonl"
    summary_path = out_dir / "summary.json"

    print(f"[{args.benchmark}/{args.condition}] loading dataset...", flush=True)
    name, loader = LOADERS[args.benchmark]
    if args.benchmark == "gsm_plus":
        items = loader(limit=args.limit if args.limit else 2000)
    elif args.benchmark == "mmlu_pro":
        items = loader(per_category=args.per_category)
    else:
        items = loader()
    if args.limit and args.benchmark in ("gpqa",):
        items = items[: args.limit]
    print(f"[{args.benchmark}/{args.condition}] loaded {len(items)} items", flush=True)

    # resume: skip already-graded qids
    done: set[str] = set()
    if trials_path.exists():
        with trials_path.open() as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("err") is None:  # only skip successful
                        done.add(rec["qid"])
                except Exception:
                    pass
        if done:
            print(f"[{args.benchmark}/{args.condition}] resuming, {len(done)} already done", flush=True)
    pending = [it for it in items if it["qid"] not in done]
    print(f"[{args.benchmark}/{args.condition}] {len(pending)} to run, c={args.concurrency}", flush=True)

    tb = None if args.condition == "unbounded" else int(args.condition.replace("tb", ""))
    sem = asyncio.Semaphore(args.concurrency)
    timeout = httpx.Timeout(DEFAULT_REQUEST_TIMEOUT_S, connect=15.0)
    limits = httpx.Limits(max_connections=max(args.concurrency * 2, 8), max_keepalive_connections=args.concurrency)

    t_start = time.time()
    all_trials: List[Trial] = []
    # load existing for summary aggregation
    if trials_path.exists():
        with trials_path.open() as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    all_trials.append(Trial(**{k: rec.get(k) for k in Trial.__dataclass_fields__}))
                except Exception:
                    pass

    async with httpx.AsyncClient(timeout=timeout, limits=limits, http2=False) as client:
        tasks = [
            _one(client, args.endpoint, args.model, it, args.benchmark, tb, args.max_tokens, sem)
            for it in pending
        ]
        done_count = 0
        with trials_path.open("a") as f:
            for fut in asyncio.as_completed(tasks):
                t: Trial = await fut
                all_trials.append(t)
                f.write(json.dumps(asdict(t)) + "\n")
                f.flush()
                done_count += 1
                if done_count % max(1, len(tasks) // 20 or 1) == 0 or done_count == len(tasks):
                    acc_now = sum(1 for x in all_trials if x.correct) / max(len(all_trials), 1)
                    err_now = sum(1 for x in all_trials if x.err)
                    elapsed = time.time() - t_start
                    rate = done_count / max(elapsed, 1e-3)
                    eta = (len(tasks) - done_count) / max(rate, 1e-3)
                    print(
                        f"[{args.benchmark}/{args.condition}] {done_count}/{len(tasks)} "
                        f"acc={acc_now:.3f} errs={err_now} "
                        f"rate={rate:.2f}/s eta={eta/60:.1f}min",
                        flush=True,
                    )

    run_meta = {
        "endpoint": args.endpoint,
        "model": args.model,
        "thinking_token_budget": tb,
        "max_tokens": args.max_tokens,
        "temperature": DEFAULT_TEMPERATURE,
        "top_p": DEFAULT_TOP_P,
        "top_k": DEFAULT_TOP_K,
        "repetition_penalty": DEFAULT_REPETITION_PENALTY,
        "concurrency": args.concurrency,
        "dataset_name": name,
        "n_loaded": len(items),
        "elapsed_s": round(time.time() - t_start, 1),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_start)),
    }
    summary = _summarize(all_trials, args.benchmark, args.condition, run_meta)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", required=True, choices=list(LOADERS.keys()))
    p.add_argument("--condition", required=True, help="'unbounded' or 'tbN' e.g. tb2048")
    p.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=0, help="cap items (after sampling); 0=no cap")
    p.add_argument("--per-category", type=int, default=100, help="MMLU-Pro per-category sample")
    args = p.parse_args()
    if args.condition != "unbounded" and not re.fullmatch(r"tb\d+", args.condition):
        print(f"bad --condition '{args.condition}'", file=sys.stderr); return 2
    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
