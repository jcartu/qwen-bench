#!/usr/bin/env python3
"""Fill {{MMLU_*}} and {{SWEEP_*}} placeholders in the front-door template.

Reads results/aggregate.json + gpqa_sweep summary files; emits filled markdown.

Placeholders handled:
    {{MMLU_DELTA_LINE}}      Short callout line for SOTA.md
    {{MMLU_HEADLINE_LINE}}   Full headline for STUDIES.md abstract
    {{MMLU_ONE_LINE}}        Compact one-liner for README.md entry
    {{SWEEP_VERDICT_LINE}}   Verdict from the GPQA budget sweep
    {{SWEEP_ONE_LINE}}       Compact verdict for README.md
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z*z/n
    centre = (p + z*z/(2*n)) / denom
    half = (z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def two_prop_z(k1, n1, k2, n2):
    p1 = k1/n1 if n1 else 0.0
    p2 = k2/n2 if n2 else 0.0
    p = (k1 + k2) / (n1 + n2) if (n1 + n2) > 0 else 0.0
    denom = math.sqrt(p*(1-p)*(1/n1 + 1/n2)) if p not in (0.0, 1.0) and n1 and n2 else 0.0
    return (p1 - p2) / denom if denom > 0 else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aggregate", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sweep-dir", default="/tmp/qwen-bench-runs-2026-05-15/runs/gpqa_sweep")
    args = ap.parse_args()

    data = json.loads(Path(args.aggregate).read_text())
    template = Path(args.template).read_text()

    subs = {}

    # ---- MMLU lines ----
    mm = data.get("runs", {}).get("mmlu_pro", {})
    mm_u = mm.get("unbounded"); mm_t = mm.get("tb2048")
    if mm_u and mm_t:
        nu = mm_u["n_trials"]; ku = round(mm_u["accuracy"]*nu)
        nt = mm_t["n_trials"]; kt = round(mm_t["accuracy"]*nt)
        z = two_prop_z(kt, nt, ku, nu)
        p = math.erfc(abs(z)/math.sqrt(2)) if z is not None else 1.0
        d_pp = (mm_t["accuracy"] - mm_u["accuracy"]) * 100
        lat_ratio = mm_t["latency"]["p50_s"] / mm_u["latency"]["p50_s"] if mm_u["latency"]["p50_s"] > 0 else None
        wall_ratio = mm_t["elapsed_s"] / mm_u["elapsed_s"] if mm_u.get("elapsed_s", 0) > 0 else None

        subs["MMLU_DELTA_LINE"] = (
            f"{mm_u['accuracy']*100:.1f}% \u2192 **{mm_t['accuracy']*100:.1f}%** "
            f"(\u0394 {d_pp:+.1f} pp, z={z:.2f}, p={p:.2g}) at "
            f"{lat_ratio:.2f}\u00d7 wall" if lat_ratio else f"{mm_t['accuracy']*100:.1f}%"
        )
        subs["MMLU_HEADLINE_LINE"] = (
            f"{mm_u['accuracy']*100:.1f}% \u2192 **{mm_t['accuracy']*100:.1f}%** accuracy "
            f"({d_pp:+.1f} pp, z={z:.2f}, p={p:.2g}); "
            f"{mm_u['latency']['p50_s']:.1f}s \u2192 {mm_t['latency']['p50_s']:.1f}s p50 latency; "
            f"stuck rate {mm_u['stuck_rate']*100:.1f}% \u2192 {mm_t['stuck_rate']*100:.1f}%."
        )
        subs["MMLU_ONE_LINE"] = (
            f"\u0394 {d_pp:+.1f} pp accuracy at {lat_ratio:.2f}\u00d7 wall" if lat_ratio else f"\u0394 {d_pp:+.1f} pp"
        )
    else:
        for k in ("MMLU_DELTA_LINE", "MMLU_HEADLINE_LINE", "MMLU_ONE_LINE"):
            subs[k] = "_pending_"

    # ---- Sweep lines ----
    sweep_root = Path(args.sweep_dir)
    sweep_pts = []
    EXPECTED_SWEEP_TBS = (1024, 4096, 8192)
    for tb in EXPECTED_SWEEP_TBS:
        sp = sweep_root / f"tb{tb}" / "summary.json"
        if sp.exists():
            d = json.loads(sp.read_text())
            sweep_pts.append((tb, d["accuracy"], d["stuck_rate"], d["latency"]["p50_s"]))
    # only emit a verdict if ALL expected sweep points landed; partial sweeps
    # would produce misleading monotonicity classifications
    sweep_complete = len(sweep_pts) == len(EXPECTED_SWEEP_TBS)
    if not sweep_complete:
        sweep_pts = []  # degrade to _pending_ rather than emit a partial verdict
    if sweep_pts:
        # add the existing tb=2048 point for context
        gp_t = data.get("runs", {}).get("gpqa", {}).get("tb2048")
        gp_u = data.get("runs", {}).get("gpqa", {}).get("unbounded")
        ref_pts = []
        if gp_t:
            ref_pts.append((2048, gp_t["accuracy"], gp_t["stuck_rate"], gp_t["latency"]["p50_s"]))
        all_pts = sorted(set(sweep_pts + ref_pts))
        accs = [p[1] for p in all_pts]
        # crude monotonicity classification
        if max(accs) - min(accs) < 0.03:
            verdict = "shows accuracy is essentially flat across budgets \u2192 the **forced-commit mechanism** dominates the result (budget size is secondary)"
            short = "shows flat accuracy \u2192 forced-commit mechanism dominates"
        elif accs == sorted(accs):  # monotonically increasing
            verdict = f"shows accuracy rises monotonically with budget (best at tb=8192, acc={sweep_pts[-1][1]*100:.1f}%) \u2192 the **budget-size axis is the dominant effect** and tb=2048 may be undershooting"
            short = "shows monotonic rise \u2192 budget-size axis dominates; tb=2048 may undershoot"
        else:
            best = max(all_pts, key=lambda x: x[1])
            verdict = f"shows accuracy peaks at tb={best[0]} (acc={best[1]*100:.1f}%) \u2192 there is a **genuine sweet spot** in budget choice"
            short = f"shows peak at tb={best[0]} \u2192 sweet spot exists"
        subs["SWEEP_VERDICT_LINE"] = verdict
        subs["SWEEP_ONE_LINE"] = short
    else:
        subs["SWEEP_VERDICT_LINE"] = "_pending_"
        subs["SWEEP_ONE_LINE"] = "_pending_"

    out = template
    for k, v in subs.items():
        out = out.replace("{{" + k + "}}", v)
    Path(args.out).write_text(out)
    print(f"wrote {args.out}")
    print(f"substitutions: {sorted(subs.keys())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
