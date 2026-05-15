#!/usr/bin/env python3
"""Generate the public flat CSV for /data from results/aggregate.json.

Schema:
    benchmark, condition, scope, domain, n, acc, acc_ci95_lo, acc_ci95_hi,
    p50_lat_s, p95_lat_s, p50_comp_tok, p95_comp_tok, stuck_rate, wall_s,
    seed, timestamp_utc
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path


def wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aggregate", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data = json.loads(Path(args.aggregate).read_text())
    rows = []
    for bench, conds in data.get("runs", {}).items():
        for cond, s in conds.items():
            n = s.get("n_trials", 0)
            acc = s.get("accuracy", 0.0)
            k = round(acc * n)
            lo, hi = wilson_ci(k, n)
            rows.append({
                "benchmark": bench,
                "condition": cond,
                "scope": "overall",
                "domain": "ALL",
                "n": n,
                "acc": f"{acc:.4f}",
                "acc_ci95_lo": f"{lo:.4f}",
                "acc_ci95_hi": f"{hi:.4f}",
                "p50_lat_s": f"{s['latency']['p50_s']:.3f}",
                "p95_lat_s": f"{s['latency']['p95_s']:.3f}",
                "p50_comp_tok": f"{s['completion_tokens']['p50']:.0f}",
                "p95_comp_tok": f"{s['completion_tokens']['p95']:.0f}",
                "stuck_rate": f"{s.get('stuck_rate', 0.0):.4f}",
                "wall_s": f"{s.get('elapsed_s', 0.0):.1f}",
                "seed": 42,
                "timestamp_utc": s.get("started_at", ""),
            })
            for dom, dv in (s.get("by_domain") or {}).items():
                dn = dv.get("n", 0)
                dk = dv.get("ok", round(dv.get("acc", 0.0) * dn))
                dlo, dhi = wilson_ci(dk, dn)
                rows.append({
                    "benchmark": bench,
                    "condition": cond,
                    "scope": "domain",
                    "domain": dom,
                    "n": dn,
                    "acc": f"{dv.get('acc', 0.0):.4f}",
                    "acc_ci95_lo": f"{dlo:.4f}",
                    "acc_ci95_hi": f"{dhi:.4f}",
                    "p50_lat_s": "",
                    "p95_lat_s": "",
                    "p50_comp_tok": "",
                    "p95_comp_tok": "",
                    "stuck_rate": "",
                    "wall_s": "",
                    "seed": 42,
                    "timestamp_utc": s.get("started_at", ""),
                })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
