#!/usr/bin/env python3
"""
Hard-coding validation probe for runaway-reasoning mitigation A/B.

5 real coding problems at increasing reasoning-depth requirement.
Each response is auto-graded by extracting the python code and running
it against deterministic test cases.

Concurrency = 1 (single-user OpenCode use case).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
import subprocess
import tempfile
from pathlib import Path

import aiohttp


# Each problem = (id, prompt, grader). Grader takes the generated python
# source string and returns dict(passed: bool, detail: str, n_pass: int, n_total: int).
PROBLEMS = []


def _extract_python(text: str) -> str:
    """Pull python code out of a chat response: prefer ```python ... ``` blocks."""
    if not text:
        return ""
    # First try fenced python block
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1)
    # Otherwise return the whole content (model may have emitted raw code)
    return text


def _run_grader(code: str, harness: str, timeout: int = 8) -> dict:
    """Execute code+harness in a subprocess. Harness prints PASS or FAIL lines.
    Returns dict(passed, n_pass, n_total, detail)."""
    full = code + "\n\n" + harness
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(full)
        path = f.name
    try:
        proc = subprocess.run(
            [sys.executable, path],
            capture_output=True, text=True, timeout=timeout,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        n_pass = out.count("PASS")
        n_fail = out.count("FAIL")
        n_total = n_pass + n_fail
        detail_lines = [ln for ln in out.splitlines() if ln.startswith(("PASS","FAIL"))]
        detail = " ".join(detail_lines[:6])
        if proc.returncode != 0 and n_total == 0:
            detail = f"runtime-err: {out.strip()[:300]}"
        return {
            "passed": n_pass == n_total and n_total > 0,
            "n_pass": n_pass, "n_total": n_total,
            "detail": detail,
        }
    except subprocess.TimeoutExpired:
        return {"passed": False, "n_pass": 0, "n_total": 0, "detail": "timeout"}
    except Exception as e:
        return {"passed": False, "n_pass": 0, "n_total": 0, "detail": f"exec-err: {e!r}"}
    finally:
        try: os.unlink(path)
        except: pass


# ---- Problem 1: zigzag matrix traversal (medium) ----
PROBLEMS.append(("zigzag", """\
Implement a Python function with this exact signature:

    def zigzag_traverse(matrix: list[list[int]]) -> list[int]

Given a rectangular m×n matrix of integers, return the elements traversed in a
zigzag pattern: left-to-right on row 0, right-to-left on row 1, left-to-right
on row 2, etc. Output ONLY a Python code block. Do not include test calls.
""", """
test_cases = [
    ([[1,2,3],[4,5,6],[7,8,9]], [1,2,3,6,5,4,7,8,9]),
    ([[1,2,3,4]], [1,2,3,4]),
    ([[1],[2],[3]], [1,2,3]),
    ([[1,2],[3,4],[5,6],[7,8]], [1,2,4,3,5,6,8,7]),
    ([[]], []),
]
for i,(m,expected) in enumerate(test_cases):
    try:
        got = zigzag_traverse(m)
        ok = got == expected
        print(f"{'PASS' if ok else 'FAIL'} case{i}: expected={expected} got={got}")
    except Exception as e:
        print(f"FAIL case{i}: {e!r}")
"""))


# ---- Problem 2: Minimum Window Substring (hard) ----
PROBLEMS.append(("min_window", """\
Implement a Python function with this exact signature:

    def min_window(s: str, t: str) -> str

Returns the minimum-length substring of `s` that contains every character of
`t` (counting multiplicities). If no such substring exists return "".
If multiple shortest substrings exist, return the leftmost. Output ONLY a
Python code block. Do not include test calls.
""", """
test_cases = [
    ("ADOBECODEBANC", "ABC", "BANC"),
    ("a", "a", "a"),
    ("a", "aa", ""),
    ("aab", "aab", "aab"),
    ("ab", "b", "b"),
    ("ADOBECODEBANCBANC", "ABC", "BANC"),
]
for i,(s,t,expected) in enumerate(test_cases):
    try:
        got = min_window(s, t)
        ok = got == expected
        print(f"{'PASS' if ok else 'FAIL'} case{i}: s={s!r} t={t!r} expected={expected!r} got={got!r}")
    except Exception as e:
        print(f"FAIL case{i}: {e!r}")
"""))


# ---- Problem 3: Median of Two Sorted Arrays (very hard, O(log) required) ----
PROBLEMS.append(("median_2sorted", """\
Implement a Python function with this exact signature:

    def find_median_sorted_arrays(nums1: list[int], nums2: list[int]) -> float

Returns the median of the two sorted arrays combined. Must run in
O(log(min(m, n))) time. Output ONLY a Python code block. Do not include test calls.
""", """
test_cases = [
    ([1,3], [2], 2.0),
    ([1,2], [3,4], 2.5),
    ([0,0], [0,0], 0.0),
    ([], [1], 1.0),
    ([2], [], 2.0),
    ([1,2,3,4,5], [6,7,8,9,10], 5.5),
    ([1,3,5,7,9], [2,4,6,8,10], 5.5),
]
for i,(a,b,expected) in enumerate(test_cases):
    try:
        got = find_median_sorted_arrays(a, b)
        ok = abs(got - expected) < 1e-9
        print(f"{'PASS' if ok else 'FAIL'} case{i}: a={a} b={b} expected={expected} got={got}")
    except Exception as e:
        print(f"FAIL case{i}: {e!r}")
"""))


# ---- Problem 4: LRU Cache (design, O(1) constraint) ----
PROBLEMS.append(("lru_cache", """\
Implement a Python class with this exact signature:

    class LRUCache:
        def __init__(self, capacity: int): ...
        def get(self, key: int) -> int: ...    # returns value or -1
        def put(self, key: int, value: int) -> None: ...

Both `get` and `put` must run in O(1) amortized time. When capacity is
exceeded the least-recently-used entry is evicted. Output ONLY a Python
code block. Do not include test calls.
""", """
c = LRUCache(2)
ops = [
    ('put', 1, 1, None),
    ('put', 2, 2, None),
    ('get', 1, None, 1),
    ('put', 3, 3, None),
    ('get', 2, None, -1),
    ('put', 4, 4, None),
    ('get', 1, None, -1),
    ('get', 3, None, 3),
    ('get', 4, None, 4),
]
for i,(op,k,v,expected) in enumerate(ops):
    try:
        if op == 'put':
            c.put(k, v)
            print(f"PASS case{i}: put({k},{v})")
        else:
            got = c.get(k)
            ok = got == expected
            print(f"{'PASS' if ok else 'FAIL'} case{i}: get({k})={got} expected={expected}")
    except Exception as e:
        print(f"FAIL case{i}: {e!r}")
"""))


# ---- Problem 5: Word Ladder II (extremely hard, BFS+DFS, multi-paragraph reasoning) ----
PROBLEMS.append(("word_ladder", """\
Implement a Python function with this exact signature:

    def ladder_length(beginWord: str, endWord: str, wordList: list[str]) -> int

Returns the length of the shortest transformation sequence from `beginWord`
to `endWord`, where each step changes one letter and the intermediate word
must be in `wordList`. Return 0 if no such sequence exists. `beginWord` does
not have to be in `wordList`. The length counts BOTH endpoints, so a direct
transformation has length 2. Output ONLY a Python code block. Do not include
test calls.
""", """
test_cases = [
    ("hit", "cog", ["hot","dot","dog","lot","log","cog"], 5),
    ("hit", "cog", ["hot","dot","dog","lot","log"], 0),
    ("a", "c", ["a","b","c"], 2),
    ("hot", "dog", ["hot","dog"], 0),
    ("hot", "dog", ["hot","dog","dot"], 3),
]
for i,(b,e,wl,expected) in enumerate(test_cases):
    try:
        got = ladder_length(b, e, wl)
        ok = got == expected
        print(f"{'PASS' if ok else 'FAIL'} case{i}: {b}->{e} expected={expected} got={got}")
    except Exception as e:
        print(f"FAIL case{i}: {e!r}")
"""))


async def trial(session, endpoint, model, prompt, harness, max_tokens, name,
                trial_idx, body_overrides):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "seed": 4000 + trial_idx,
    }
    body.update(body_overrides)
    start = time.time()
    try:
        async with session.post(endpoint, json=body) as resp:
            data = await resp.json()
        latency = time.time() - start
        if "choices" not in data:
            return {"name": name, "trial": trial_idx, "error": f"no choices: {str(data)[:300]}",
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
            "detail": grade["detail"][:160],
            "code_head": code[:120],
        }
    except Exception as e:
        return {"name": name, "trial": trial_idx, "error": repr(e),
                "latency_s": time.time() - start}


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=2, help="trials per problem")
    p.add_argument("--max-tokens", type=int, default=16384)
    p.add_argument("--timeout-s", type=int, default=300)
    # Mitigation knobs:
    p.add_argument("--enable-thinking", choices=["true","false"], default=None)
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

    timeout = aiohttp.ClientTimeout(total=args.timeout_s)
    results = []
    t0 = time.time()
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for pid, prompt, harness in PROBLEMS:
            for i in range(args.n):
                r = await trial(session, args.endpoint, args.model, prompt,
                                harness, args.max_tokens, pid, i, overrides)
                results.append(r)
                print(
                    f"[{args.label}] {pid:14s} t={i} "
                    f"pass={str(r.get('passed','?')):5s} "
                    f"({r.get('n_pass',0)}/{r.get('n_total',0)}) "
                    f"stuck={str(r.get('stuck','?')):5s} "
                    f"rl={r.get('reasoning_len',0):5d} "
                    f"cl={r.get('content_len',0):5d} "
                    f"lat={r.get('latency_s',0):.1f}s "
                    f"{r.get('detail','')[:80]}",
                    flush=True,
                )

    Path(args.out).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(args.out, "trials.jsonl"), "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    summary = {
        "label": args.label,
        "overrides": overrides,
        "max_tokens": args.max_tokens,
        "wall_s": round(time.time() - t0, 1),
        "by_problem": {},
    }
    for pid, _, _ in PROBLEMS:
        rows = [r for r in results if r.get("name") == pid and "error" not in r]
        if not rows:
            continue
        n_pass = sum(1 for r in rows if r.get("passed"))
        n_stuck = sum(1 for r in rows if r.get("stuck"))
        lats = sorted(r["latency_s"] for r in rows)
        rls  = sorted(r["reasoning_len"] for r in rows)
        summary["by_problem"][pid] = {
            "n": len(rows),
            "pass_rate": round(n_pass/len(rows), 3),
            "stuck_rate": round(n_stuck/len(rows), 3),
            "median_lat_s": lats[len(lats)//2],
            "median_reasoning_len": rls[len(rls)//2],
            "max_reasoning_len": max(rls),
        }
    summary["overall_pass_rate"] = round(
        sum(1 for r in results if r.get("passed")) / max(1, len(results)), 3
    )
    summary["overall_stuck_rate"] = round(
        sum(1 for r in results if r.get("stuck")) / max(1, len(results)), 3
    )
    summary["n_errors"] = sum(1 for r in results if "error" in r)

    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
