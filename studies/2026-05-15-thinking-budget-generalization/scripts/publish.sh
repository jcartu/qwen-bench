#!/usr/bin/env bash
# Final publish step. Run after all 6 main runs + 3 sweep runs land.
#
# This script:
#   1. Runs the aggregator -> results/aggregate.{md,csv,json}
#   2. Copies raw trial JSONLs from /tmp into results/
#   3. Generates the public data CSV
#   4. Fills placeholders in HUB_FRONTDOOR_EDITS.md
#   5. Applies the front-door edits to SOTA.md, STUDIES.md, README.md, data/
#   6. Commits + pushes
#
# Idempotent: safe to re-run after partial completion.
set -euo pipefail

HUB=/tmp/opencode/qwen-bench
STUDY=$HUB/studies/2026-05-15-thinking-budget-generalization
RUNS=/tmp/qwen-bench-runs-2026-05-15
LOGS=$RUNS/logs

echo "==> [1/6] Running aggregator over all runs..."
python3 $RUNS/harness/aggregate.py \
    --runs-dir $RUNS/runs \
    --out-dir $STUDY/results

echo "==> [2/6] Copying raw JSONLs into study results/..."
mkdir -p $STUDY/results/raw
for bench in gpqa gsm_plus mmlu_pro; do
    for cond in unbounded tb2048; do
        src=$RUNS/runs/$bench/$cond/trials.jsonl
        if [[ -f "$src" ]]; then
            cp "$src" "$STUDY/results/raw/${bench}_${cond}.jsonl"
        fi
    done
done
# sweep
for tb in 1024 4096 8192; do
    src=$RUNS/runs/gpqa_sweep/tb${tb}/trials.jsonl
    if [[ -f "$src" ]]; then
        cp "$src" "$STUDY/results/raw/gpqa_sweep_tb${tb}.jsonl"
    fi
done

# also include the summary.json files for each run
mkdir -p $STUDY/results/summaries
for d in $RUNS/runs/*/*/summary.json $RUNS/runs/gpqa_sweep/*/summary.json; do
    if [[ -f "$d" ]]; then
        bench=$(basename $(dirname $(dirname $d)))
        cond=$(basename $(dirname $d))
        cp "$d" "$STUDY/results/summaries/${bench}_${cond}.json"
    fi
done

echo "==> [3/6] Generating public data CSV..."
python3 $STUDY/scripts/emit_hub_csv.py \
    --aggregate $STUDY/results/aggregate.json \
    --out $HUB/data/2026-05-15-thinking-budget-generalization.csv

echo "==> [4/6] Filling placeholders in front-door edits..."
python3 $STUDY/scripts/fill_placeholders.py \
    --aggregate $STUDY/results/aggregate.json \
    --template $STUDY/HUB_FRONTDOOR_EDITS.md \
    --out $STUDY/HUB_FRONTDOOR_EDITS.filled.md

echo "==> [5/6] Applying front-door edits..."
python3 $STUDY/scripts/apply_frontdoor.py \
    --hub-root $HUB \
    --filled $STUDY/HUB_FRONTDOOR_EDITS.filled.md

echo "==> [6/6] Final review prompts (manual):"
echo "  - $HUB/SOTA.md      (check new callout near top)"
echo "  - $HUB/STUDIES.md   (check newest-first ordering)"
echo "  - $HUB/README.md    (check badge bumped, study entry inserted)"
echo "  - $HUB/data/2026-05-15-thinking-budget-generalization.csv"
echo "  - $STUDY/README.md  (placeholders filled)"
echo
echo "When satisfied:"
echo "  cd $HUB && git add -A && git status"
echo "  git commit -F $STUDY/COMMIT_MSG.txt"
echo "  git push origin main"
