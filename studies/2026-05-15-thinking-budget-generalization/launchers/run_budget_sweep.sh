#!/usr/bin/env bash
# Budget sweep on GPQA-Diamond to disentangle "forced-commit" effect from "budget-size" effect.
# Runs sequentially after the main fast phase completes. ~17 min/run at c=8.
#
# Conditions added (tb2048 and unbounded already exist from fast phase):
#   tb1024  - tighter than current default
#   tb4096  - 2x default
#   tb8192  - 4x default; still well under 16k max_tokens so no stuck risk
#
# Result: 5-point budget curve on GPQA Diamond.

set -euo pipefail

cd /tmp/qwen-bench-runs-2026-05-15
ROOT=$PWD
LOG=$ROOT/logs/budget_sweep.log
mkdir -p $ROOT/runs/gpqa_sweep $ROOT/logs

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Budget sweep starting" | tee -a $LOG

for tb in 1024 4096 8192; do
    OUT=$ROOT/runs/gpqa_sweep/tb${tb}
    mkdir -p $OUT
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] === GPQA tb=${tb} ===" | tee -a $LOG
    python3 $ROOT/harness/bench.py \
        --benchmark gpqa \
        --condition tb${tb} \
        --concurrency 8 \
        --out $OUT \
        --limit 0 \
        2>&1 | tee -a $LOG
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] === done tb=${tb} ===" | tee -a $LOG
done

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Budget sweep complete" | tee -a $LOG

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Firing publish.sh" | tee -a $LOG
bash /tmp/opencode/qwen-bench/studies/2026-05-15-thinking-budget-generalization/scripts/publish.sh 2>&1 | tee -a $LOG
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] publish.sh complete (manual git push still required)" | tee -a $LOG
