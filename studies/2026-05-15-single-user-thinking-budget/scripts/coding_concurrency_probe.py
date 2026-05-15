#!/usr/bin/env python3
"""
Phase F: concurrency sweep for single-user coding workload.

Reuses the 5 hard problems & graders from coding_probe.py but runs C concurrent
trials at a time (rotating through problems). Mimics a real OpenCode session
with parallel subagent dispatch.

Per concurrency level reports:
  - overall pass rate
  - overall stuck rate
  - p50 / p95 latency
  - aggregate throughput (req/s)
  - median reasoning_len & content_len
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import importlib.util
from pathlib import Path

import aiohttp

# Import coding_probe.py from same dir
_here = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("coding_probe", _here / "coding_probe.py")
_cp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cp)

PROBLEMS = _cp.PROBLEMS
_extract_python = _cp._extract_python
_run_grader = _cp._run_grader


async def fire(session, endpoint, model, prompt, harness, max_tokens, name,
               trial_idx, body_overrides):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "seed": 5000 + trial_idx,
    }
    body.update(body_overrides)
    start = time.time()
    try:
        async with session.post(endpoint, json=body) as resp:
            data = await resp.json()
        latency = time.time() - start
        if "choices" not in data:
            return {"name": name, "trial": trial_idx,
                    "error": f"no choices: {str(data)[:300]}",
                    "latency_s": latency}
        msg = data["choices"][0]["message"]
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
        finish = data["choices"][0].get("finish_reason")
        stuck = (finish == "length") or (
            len(content.strip()) == 0 and len(reasoning) > 8000 and finish == "stop"
        )
        code = _extract_python(content)
        grade = _run_grader(code, harness)
        return {
            "name": name, "trial": trial_idx,
            "latency_s": round(latency, 2),
            "finish": finish,
            "content_len": len(content),
            "reasoning_len": len(reasoning),
            "stuck": stuck,
            "passed": grade["passed"],
            "n_pass": grade["n_pass"],
            "n_total": grade["n_total"],
            "detail": grade["detail"][:120],
        }
    except Exception as e:
        return {"name": name, "trial": trial_idx, "error": repr(e),
                "latency_s": time.time() - start}


async def run_concurrency_level(session, endpoint, model, problems, n_trials,
                                 max_tokens, overrides, c, label):
    """Run all (problem × trial) requests in waves of size C, in parallel.

    Returns list of result dicts and wall time.
    """
    # Build full request list (one per problem×trial)
    requests = []
    for pid, prompt, harness in problems:
        for t in range(n_trials):
            requests.append((pid, prompt, harness, t))

    results = []
    t0 = time.time()
    # Process in waves of c concurrent requests
    for wave_start in range(0, len(requests), c):
        wave = requests[wave_start: wave_start + c]
        coros = [
            fire(session, endpoint, model, prompt, harness, max_tokens, pid,
                 t, overrides)
            for (pid, prompt, harness, t) in wave
        ]
        wave_t0 = time.time()
        wave_results = await asyncio.gather(*coros)
        wave_wall = time.time() - wave_t0
        for r, (pid, _, _, t) in zip(wave_results, wave):
            r["wave"] = wave_start // c
            r["wave_wall_s"] = round(wave_wall, 2)
        results.extend(wave_results)
        # Compact per-wave log
        n_pass = sum(1 for r in wave_results if r.get("passed"))
        n_stuck = sum(1 for r in wave_results if r.get("stuck"))
        lats = sorted(r.get("latency_s", 0) for r in wave_results)
        print(f"[{label} c={c}] wave {wave_start//c+1}: "
              f"size={len(wave)} wall={wave_wall:.1f}s "
              f"pass={n_pass}/{len(wave)} stuck={n_stuck} "
              f"lat-min/med/max={lats[0]:.1f}/{lats[len(lats)//2]:.1f}/{lats[-1]:.1f}s",
              flush=True)
    wall = time.time() - t0
    return results, wall


def summarize(results, wall):
    ok = [r for r in results if "error" not in r]
    if not ok:
        return {"n": 0, "wall_s": round(wall, 1)}
    lats = sorted(r["latency_s"] for r in ok)
    rls = sorted(r["reasoning_len"] for r in ok)
    cls = sorted(r["content_len"] for r in ok)
    n = len(ok)
    p50 = lats[n // 2]
    p95 = lats[min(n - 1, int(n * 0.95))]
    return {
        "n": n,
        "n_errors": len(results) - len(ok),
        "wall_s": round(wall, 1),
        "pass_rate": round(sum(1 for r in ok if r.get("passed")) / n, 3),
        "stuck_rate": round(sum(1 for r in ok if r.get("stuck")) / n, 3),
        "p50_latency_s": p50,
        "p95_latency_s": p95,
        "max_latency_s": lats[-1],
        "median_reasoning_len": rls[n // 2],
        "max_reasoning_len": rls[-1],
        "median_content_len": cls[n // 2],
        "throughput_req_per_s": round(n / wall, 3),
    }


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=2, help="trials per problem")
    p.add_argument("--max-tokens", type=int, default=16384)
    p.add_argument("--timeout-s", type=int, default=600)
    p.add_argument("--concurrency", type=int, action="append", required=True,
                   help="Concurrency level (repeat for multiple)")
    p.add_argument("--enable-thinking", choices=["true", "false"], default=None)
    p.add_argument("--thinking-token-budget", type=int, default=None)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--presence-penalty", type=float, default=None)
    args = p.parse_args()

    overrides = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
    }
    if args.presence_penalty is not None:
        overrides["presence_penalty"] = args.presence_penalty
    if args.enable_thinking is not None:
        overrides["chat_template_kwargs"] = {
            "enable_thinking": args.enable_thinking == "true"
        }
    if args.thinking_token_budget is not None:
        overrides["thinking_token_budget"] = args.thinking_token_budget

    Path(args.out).mkdir(parents=True, exist_ok=True)
    timeout = aiohttp.ClientTimeout(total=args.timeout_s)
    all_summaries = {}
    all_trials = []

    async with aiohttp.ClientSession(timeout=timeout) as session:
        for c in args.concurrency:
            print(f"\n========================================================")
            print(f"[{args.label}] concurrency = {c}")
            print(f"========================================================")
            results, wall = await run_concurrency_level(
                session, args.endpoint, args.model, PROBLEMS, args.n,
                args.max_tokens, overrides, c, args.label,
            )
            for r in results:
                r["concurrency"] = c
            all_trials.extend(results)
            s = summarize(results, wall)
            all_summaries[f"c={c}"] = s
            print(f"\n[{args.label} c={c}] SUMMARY: pass={s.get('pass_rate',0)*100:.0f}% "
                  f"stuck={s.get('stuck_rate',0)*100:.0f}% "
                  f"p50={s.get('p50_latency_s',0)}s p95={s.get('p95_latency_s',0)}s "
                  f"wall={s.get('wall_s',0)}s thr={s.get('throughput_req_per_s',0)} req/s")

    # Dump everything
    with open(os.path.join(args.out, "trials.jsonl"), "w") as f:
        for r in all_trials:
            f.write(json.dumps(r) + "\n")
    out_summary = {
        "label": args.label,
        "overrides": overrides,
        "max_tokens": args.max_tokens,
        "by_concurrency": all_summaries,
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(out_summary, f, indent=2)
    print("\n=== AGGREGATE ===")
    print(json.dumps(out_summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
