#!/usr/bin/env python3
"""
Aggregate all completed runs into a single comparison report.

Walks RUNS_DIR for any subdir matching <benchmark>/<condition>/summary.json
and emits:
  - aggregate.json    machine-readable, all runs + paired deltas
  - aggregate.md      human-readable comparison tables
  - aggregate.csv     flat per-(benchmark,condition,domain) rows
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def two_proportion_z(k1: int, n1: int, k2: int, n2: int) -> Optional[float]:
    if n1 == 0 or n2 == 0:
        return None
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    denom = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2)) if p not in (0.0, 1.0) else 0.0
    if denom == 0:
        return 0.0
    return (p1 - p2) / denom


def discover(root: Path) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Return {benchmark: {condition: summary_dict}}."""
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for sjson in sorted(root.glob("*/*/summary.json")):
        bench = sjson.parent.parent.name
        cond  = sjson.parent.name
        try:
            d = json.loads(sjson.read_text())
        except Exception:
            continue
        out.setdefault(bench, {})[cond] = d
    return out


def fmt_pct(x: float) -> str:
    return f"{x*100:5.1f}%"


def render_md(data: Dict[str, Dict[str, Dict[str, Any]]]) -> str:
    out: List[str] = []
    out.append("# Thinking Budget Generalization Results")
    out.append("")
    out.append("Cross-domain validation of `thinking_token_budget=2048` vs unbounded on")
    out.append("the Qwen3.6-27B (FP8, TP=2, MTP=3, repne/vllm:v3) single-user SOTA stack.")
    out.append("")
    out.append("- Endpoint: `http://127.0.0.1:11435/v1/chat/completions`")
    out.append("- Sampling: temperature=0.6, top_p=0.95, top_k=20, repetition_penalty=1.05")
    out.append("- Thinking: `chat_template_kwargs.enable_thinking=true`")
    out.append("- max_tokens: 16384, concurrency: 8")
    out.append("")
    out.append("## Headline comparison")
    out.append("")
    out.append("| Benchmark | n | C0 unbounded acc | C0 95% CI | C1 tb=2048 acc | C1 95% CI | Δ acc (pp) | two-prop z | p (two-sided) | C0 p50 lat (s) | C1 p50 lat (s) | C0 stuck | C1 stuck |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for bench, conds in sorted(data.items()):
        u = conds.get("unbounded"); t = conds.get("tb2048")
        if not u or not t:
            continue
        nu = u["n_trials"]; ku = round(u["accuracy"]*nu); nt = t["n_trials"]; kt = round(t["accuracy"]*nt)
        d_pp = (t["accuracy"] - u["accuracy"]) * 100
        z = two_proportion_z(kt, nt, ku, nu)
        z_str = f"{z:.2f}" if z is not None else "n/a"
        # two-sided p-value from z (normal approx, scipy-free)
        if z is None:
            p_str = "n/a"
        else:
            # erfc(|z|/sqrt(2)) gives two-sided p
            p_val = math.erfc(abs(z)/math.sqrt(2.0))
            p_str = f"{p_val:.1e}" if p_val < 1e-3 else f"{p_val:.3f}"
        u_lo, u_hi = wilson_ci(ku, nu); t_lo, t_hi = wilson_ci(kt, nt)
        out.append(
            f"| {bench} | {nu}/{nt} | {fmt_pct(u['accuracy'])} | [{u_lo*100:.1f},{u_hi*100:.1f}] | "
            f"{fmt_pct(t['accuracy'])} | [{t_lo*100:.1f},{t_hi*100:.1f}] | "
            f"{d_pp:+.1f} | {z_str} | {p_str} | "
            f"{u['latency']['p50_s']:.1f} | {t['latency']['p50_s']:.1f} | "
            f"{fmt_pct(u['stuck_rate'])} | {fmt_pct(t['stuck_rate'])} |"
        )
    out.append("")
    out.append("## Per-domain accuracy")
    out.append("")
    for bench, conds in sorted(data.items()):
        u = conds.get("unbounded"); t = conds.get("tb2048")
        if not u or not t:
            continue
        out.append(f"### {bench}")
        out.append("")
        out.append("| Domain | n | C0 unbounded | C1 tb=2048 | Δ pp |")
        out.append("|---|---:|---:|---:|---:|")
        domains = sorted(set(list(u.get("by_domain", {}).keys()) + list(t.get("by_domain", {}).keys())))
        for d in domains:
            du = u.get("by_domain", {}).get(d, {})
            dt = t.get("by_domain", {}).get(d, {})
            nu = du.get("n", 0); nt = dt.get("n", 0)
            au = du.get("acc", 0.0); at = dt.get("acc", 0.0)
            delta = (at - au) * 100
            n_display = nt if nt == nu else f"{nu}/{nt}"
            out.append(f"| {d} | {n_display} | {fmt_pct(au)} | {fmt_pct(at)} | {delta:+.1f} |")
        out.append("")
    out.append("## Latency / token deltas")
    out.append("")
    out.append("| Benchmark | p50 lat ratio (C1/C0) | p95 lat ratio | p50 comp tok ratio | p95 comp tok ratio |")
    out.append("|---|---:|---:|---:|---:|")
    for bench, conds in sorted(data.items()):
        u = conds.get("unbounded"); t = conds.get("tb2048")
        if not u or not t:
            continue
        def ratio(a, b): return f"{(a/b):.2f}x" if b > 0 else "n/a"
        out.append(
            f"| {bench} | "
            f"{ratio(t['latency']['p50_s'], u['latency']['p50_s'])} | "
            f"{ratio(t['latency']['p95_s'], u['latency']['p95_s'])} | "
            f"{ratio(t['completion_tokens']['p50'], u['completion_tokens']['p50'])} | "
            f"{ratio(t['completion_tokens']['p95'], u['completion_tokens']['p95'])} |"
        )
    out.append("")
    return "\n".join(out) + "\n"


def render_csv(data: Dict[str, Dict[str, Dict[str, Any]]], dst: Path) -> None:
    with dst.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "benchmark", "condition", "scope", "domain", "n", "acc",
            "p50_lat_s", "p95_lat_s", "p50_comp_tok", "p95_comp_tok",
            "stuck_rate", "n_errors",
        ])
        for bench, conds in sorted(data.items()):
            for cond, s in sorted(conds.items()):
                w.writerow([
                    bench, cond, "overall", "ALL", s.get("n_trials", 0),
                    f"{s.get('accuracy', 0.0):.4f}",
                    f"{s['latency']['p50_s']:.3f}", f"{s['latency']['p95_s']:.3f}",
                    f"{s['completion_tokens']['p50']:.1f}", f"{s['completion_tokens']['p95']:.1f}",
                    f"{s.get('stuck_rate', 0.0):.4f}", s.get("n_errors", 0),
                ])
                for dom, dv in (s.get("by_domain") or {}).items():
                    w.writerow([
                        bench, cond, "domain", dom, dv.get("n", 0),
                        f"{dv.get('acc', 0.0):.4f}",
                        "", "", "", "", "", "",
                    ])


def render_json(data: Dict[str, Dict[str, Dict[str, Any]]], dst: Path) -> None:
    out = {"runs": data, "pairs": {}}
    for bench, conds in data.items():
        u = conds.get("unbounded"); t = conds.get("tb2048")
        if not u or not t:
            continue
        nu = u["n_trials"]; ku = round(u["accuracy"]*nu); nt = t["n_trials"]; kt = round(t["accuracy"]*nt)
        z = two_proportion_z(kt, nt, ku, nu)
        out["pairs"][bench] = {
            "delta_acc_pp": (t["accuracy"] - u["accuracy"]) * 100,
            "two_prop_z":   z,
            "n_unbounded":  nu, "k_unbounded": ku,
            "n_tb2048":     nt, "k_tb2048":   kt,
            "unbounded_ci95": wilson_ci(ku, nu),
            "tb2048_ci95":    wilson_ci(kt, nt),
            "p50_lat_ratio":  t["latency"]["p50_s"] / u["latency"]["p50_s"] if u["latency"]["p50_s"] > 0 else None,
            "p95_lat_ratio":  t["latency"]["p95_s"] / u["latency"]["p95_s"] if u["latency"]["p95_s"] > 0 else None,
            "p50_tok_ratio":  t["completion_tokens"]["p50"] / u["completion_tokens"]["p50"] if u["completion_tokens"]["p50"] > 0 else None,
        }
    dst.write_text(json.dumps(out, indent=2) + "\n")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default="/tmp/qwen-bench-runs-2026-05-15/runs")
    p.add_argument("--out-dir",  default="/tmp/qwen-bench-runs-2026-05-15/runs")
    args = p.parse_args()
    root = Path(args.runs_dir); out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    data = discover(root)
    if not data:
        print("no runs discovered", file=sys.stderr); return 1
    (out / "aggregate.md").write_text(render_md(data))
    render_csv(data, out / "aggregate.csv")
    render_json(data, out / "aggregate.json")
    print(f"wrote {out/'aggregate.md'}, {out/'aggregate.csv'}, {out/'aggregate.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
