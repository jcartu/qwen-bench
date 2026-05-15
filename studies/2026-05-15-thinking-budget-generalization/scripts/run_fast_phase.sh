#!/usr/bin/env bash
# Fast phase: GPQA Diamond (198) + GSM-Plus 2k + MMLU-Pro 100/subset (1400)
# Two conditions per benchmark: unbounded and tb2048.
# Sequential, c=8 (validated coding concurrency for Qwen3.6-27B SOTA).
set -euo pipefail

ROOT="/tmp/qwen-bench-runs-2026-05-15"
LOGDIR="${ROOT}/logs"
mkdir -p "${LOGDIR}"

export HF_TOKEN="$(cat "${HOME}/.cache/huggingface/token")"
export HF_HOME="${ROOT}/data_cache"
export PYTHONUNBUFFERED=1

C=8

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }

run() {
  local bench="$1" cond="$2"; shift 2
  local out="${ROOT}/runs/${bench}/${cond}"
  local log="${LOGDIR}/${bench}_${cond}.log"
  echo "[$(ts)] START  ${bench} ${cond}  out=${out}  log=${log}" | tee -a "${LOGDIR}/fast_phase.log"
  python3 "${ROOT}/harness/bench.py" \
    --benchmark "${bench}" \
    --condition "${cond}" \
    --concurrency "${C}" \
    --out "${out}" "$@" \
    >> "${log}" 2>&1
  local rc=$?
  echo "[$(ts)] DONE   ${bench} ${cond}  rc=${rc}" | tee -a "${LOGDIR}/fast_phase.log"
  return $rc
}

echo "[$(ts)] === FAST PHASE START (c=${C}) ===" | tee -a "${LOGDIR}/fast_phase.log"

# Run tb2048 first (faster, smoke-validated). Then unbounded.
run gpqa     tb2048
run gpqa     unbounded
run gsm_plus tb2048    --limit 2000
run gsm_plus unbounded --limit 2000
run mmlu_pro tb2048    --per-category 100
run mmlu_pro unbounded --per-category 100

echo "[$(ts)] === FAST PHASE COMPLETE ===" | tee -a "${LOGDIR}/fast_phase.log"
