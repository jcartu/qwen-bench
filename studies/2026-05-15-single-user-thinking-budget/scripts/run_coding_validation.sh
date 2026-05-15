#!/usr/bin/env bash
# Wider validation matrix: 5 hard coding problems × 2 trials × 5 conditions.
# Concurrency = 1 (single-user OpenCode workload).
set -euo pipefail

ROOT="/tmp/qwen-rollback-2026-05-14"
PROBE="$ROOT/scripts/coding_probe.py"
ENDPOINT="http://localhost:11435/v1/chat/completions"
MODEL="Qwen3.6-27B"
N=2
MAX=16384

OUT="$ROOT/coding-validation"
mkdir -p "$OUT"

run() {
    local id="$1"; shift
    local label="$1"; shift
    local extra=("$@")
    echo
    echo "===================================================================="
    echo "[$id] $label"
    echo "===================================================================="
    local odir="$OUT/$id"
    mkdir -p "$odir"
    python3 "$PROBE" \
        --endpoint "$ENDPOINT" \
        --model "$MODEL" \
        --label "$id" \
        --out "$odir" \
        --n "$N" \
        --max-tokens "$MAX" \
        "${extra[@]}" 2>&1 | tee "$odir/run.log"
    echo
    echo "--- $id condensed ---"
    python3 - <<PY
import json
d=json.load(open("$odir/summary.json"))
print(f"  overall_pass={d['overall_pass_rate']*100:.0f}%  overall_stuck={d['overall_stuck_rate']*100:.0f}%  wall={d['wall_s']}s")
for p,v in d['by_problem'].items():
    print(f"    {p:14s} pass={v['pass_rate']*100:5.0f}%  stuck={v['stuck_rate']*100:5.0f}%  lat={v['median_lat_s']}s  rl={v['median_reasoning_len']}")
PY
}

echo "Coding validation matrix start: $(date -Iseconds)"
echo "Endpoint: $ENDPOINT  Model: $MODEL  n=$N c=1 max_tokens=$MAX"

# C0: unbounded thinking (baseline — let runaway happen)
run C0 "unbounded thinking (qwen3-rec sampling)" \
    --temperature 0.6 --top-p 0.95 --top-k 20

# C1: thinking_token_budget=2048
run C1 "thinking_token_budget=2048" \
    --temperature 0.6 --top-p 0.95 --top-k 20 \
    --thinking-token-budget 2048

# C2: thinking_token_budget=4096
run C2 "thinking_token_budget=4096" \
    --temperature 0.6 --top-p 0.95 --top-k 20 \
    --thinking-token-budget 4096

# C3: thinking_token_budget=8192
run C3 "thinking_token_budget=8192" \
    --temperature 0.6 --top-p 0.95 --top-k 20 \
    --thinking-token-budget 8192

# C4: enable_thinking=false (sanity floor — measures non-think quality)
run C4 "enable_thinking=false (no-think floor)" \
    --temperature 0.7 --top-p 0.8 --top-k 20 \
    --presence-penalty 1.5 \
    --enable-thinking false

echo
echo "===================================================================="
echo "Coding validation complete: $(date -Iseconds)"
echo "===================================================================="
echo
echo "=== AGGREGATE TABLE (per condition) ==="
python3 - <<'PY'
import json, glob
rows = []
for path in sorted(glob.glob("/tmp/qwen-rollback-2026-05-14/coding-validation/*/summary.json")):
    d = json.load(open(path))
    rows.append(d)

print(f"{'id':<4}{'pass%':>7}{'stuck%':>8}{'wall':>7}  " +
      "  ".join(f"{p:>10}" for p in ('zigzag','min_window','median','lru','ladder')))
print("-"*100)
short = {"zigzag":"zigzag","min_window":"min_window","median_2sorted":"median",
         "lru_cache":"lru","word_ladder":"ladder"}
for r in rows:
    bp = r["by_problem"]
    cols = []
    for p in ('zigzag','min_window','median_2sorted','lru_cache','word_ladder'):
        if p in bp:
            cols.append(f"{bp[p]['pass_rate']*100:>4.0f}%/{bp[p]['median_lat_s']:>4.0f}s")
        else:
            cols.append("   ---   ")
    print(f"{r['label']:<4}{r['overall_pass_rate']*100:>6.0f}%{r['overall_stuck_rate']*100:>7.0f}%{r['wall_s']:>6.0f}s  " + "  ".join(cols))

print()
print("Legend: pass% / median latency per problem")
print("  zigzag   = matrix zigzag traversal (medium)")
print("  min_win  = minimum window substring (hard)")
print("  median   = median of 2 sorted arrays (very hard, O(log))")
print("  lru      = LRU cache O(1) design (medium-hard)")
print("  ladder   = word ladder BFS (very hard, multi-paragraph)")
PY
