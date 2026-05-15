#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# launch-qwen36-27b-sota.sh — SOTA 27B production server (SINGLE-USER TUNED)
#
# Configuration: FP8 + MTP=3 on Repne v3 fork, TP=2 across GPU 0 + GPU 1
# Reference:     https://github.com/jcartu/qwen-bench/blob/main/SOTA.md
# Canonical:     llm-stress-harness/launchers/launch_fp8_mtp.sh
#
# ── PROPOSED PATCH (Sisyphus 2026-05-15, see FINDINGS.md) ─────────────────────
# Changes from the live launcher (~/vllm-services/launch-qwen36-27b-sota.sh):
#
#   1. SECURITY: removed `-e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN` from the
#      docker run line. That env var was visible in `ps` (process cmdline)
#      and leaked the token to any process able to read /proc. The token
#      is already mounted via -v ${HOME}/.cache/huggingface; HuggingFace
#      libraries auto-read ~/.cache/huggingface/token when the env var is
#      absent. No functional impact.
#
#   2. SINGLE-USER TUNING (Phase F validated):
#        MAX_NUM_SEQS:     128   -> 16     (fanout never exceeds OpenCode parallel)
#        MAX_BATCHED:    32768   -> 16384  (matches reduced max-num-seqs)
#        GPU_MEM_UTIL:    0.85   -> 0.92   (more KV headroom; single user only)
#        max-cudagraph:    256   ->   32   (faster startup, ~30s saved)
#
#   3. NOTE on thinking_token_budget:
#      This vLLM build (0.1.dev16595+gebc3d9d1b) does NOT support a server-side
#      default for `thinking_token_budget` (verified via source read of
#      vllm/config/model.py:get_diff_sampling_param() whitelist and
#      vllm/entrypoints/openai/chat_completion/serving.py). Must be set
#      client-side per request. Already configured at
#      /home/josh/.config/opencode/opencode.jsonc (rasputin-27b) = 2048.
#
# Apply with:
#     cp ~/vllm-services/launch-qwen36-27b-sota.sh \
#        ~/vllm-services/launch-qwen36-27b-sota.sh.bak.$(date +%Y%m%dT%H%M%S)
#     cp /tmp/qwen-rollback-2026-05-14/launch-qwen36-27b-sota.proposed.sh \
#        ~/vllm-services/launch-qwen36-27b-sota.sh
#     systemctl --user restart vllm-qwen36-27b-sota.service
#
# Downtime: ~60-90s for cudagraph capture (reduced from ~120s thanks to #2).
# ── END PROPOSED PATCH ────────────────────────────────────────────────────────
#
# This is the production-recommended config after the 2026-05-12 v3 suite:
#   - FP8+MTP=3 on v3: 88.4% HE, 89.1% MBPP, 369 tok/s @ c=4 ctx=0
#   - MTP=5 produces ~25% stuck-thinking rate (cap-hits, empty content) vs MTP=3 at 8%. Verified 2026-05-13 via probe-mtp-think-leak.sh on dev16595 build (~/probe-mtp-think-leak.sh). NO <think>/</think> parser-state leakage observed at any MTP value (1/2/3/5) — PR #34668 fixed that. The remaining issue at MTP>=5 is degraded speculation accuracy on hard problems, causing the model to chase its reasoning longer before terminating. Not parser corruption.
#   - v3 requires local argmax reduction for TP=2 and FlashInfer spec attention
#
# Hardware pinning (intentional):
#   GPU 0 (PCIe x16 Gen5) — free, this server
#   GPU 1 (PCIe x16 Gen5) — free, this server
#   GPU 2 (PCIe x8  Gen5) — reserved for gpt-oss-120b, NOT touched
#
# Service management:
#   Run via the systemd user unit `vllm-qwen36-27b-sota.service`. This script
#   runs the container in the FOREGROUND so systemd can supervise it.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

CONTAINER_NAME="vllm-qwen36-27b-sota"
PORT="${PORT:-11435}"
MODEL="${MODEL:-Qwen/Qwen3.6-27B-FP8}"
SERVED_NAME_A="Qwen3.6-27B"
SERVED_NAME_B="qwen3.6-27b"
IMAGE="${IMAGE:-repne/vllm:v3}"
TP_SIZE="${TP_SIZE:-2}"
NUM_SPEC="${NUM_SPEC:-3}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
# Single-user tuning (was 128 / 32768 / 0.85 / 256):
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
MAX_BATCHED="${MAX_BATCHED:-16384}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
MAX_CUDAGRAPH_CAPTURE="${MAX_CUDAGRAPH_CAPTURE:-32}"

# Pin to GPU 0 + GPU 1 by UUID (most robust against PCI re-ordering).
# GPU 2 is explicitly excluded to keep gpt-oss-120b safe.
GPU_0_UUID="GPU-ba6334bc-6fec-5f2c-df75-a887bbca476e"
GPU_1_UUID="GPU-538bf008-7ff2-0d1d-69e9-20db81a00459"

# Token is mounted via -v below; HuggingFace libs auto-detect from the cache file.
# (We intentionally DO NOT pass HUGGING_FACE_HUB_TOKEN as an env var — that leaked
# the token to /proc/<pid>/cmdline visible in `ps`.)

# Stop any prior instance.
docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

# `exec` so systemd's PID is the docker client; on stop it sends SIGTERM correctly.
exec docker run \
  --name "${CONTAINER_NAME}" \
  --rm \
  --device "nvidia.com/gpu=${GPU_0_UUID}" \
  --device "nvidia.com/gpu=${GPU_1_UUID}" \
  --ipc=host \
  --shm-size=32g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --network host \
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  -v "${HOME}/.cache/vllm:/root/.cache/vllm" \
  -v "${HOME}/.cache/flashinfer:/root/.cache/flashinfer" \
  -v "${HOME}/.triton/cache:/root/.triton/cache" \
  -e "OMP_NUM_THREADS=16" \
  -e "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1" \
  -e "VLLM_WORKER_MULTIPROC_METHOD=spawn" \
  -e "VLLM_ALLREDUCE_USE_SYMM_MEM=0" \
  -e "NCCL_P2P_LEVEL=SYS" \
  -e "NCCL_NET_GDR_LEVEL=SYS" \
  -e "NCCL_MIN_NCHANNELS=8" \
  "${IMAGE}" \
    -O3 \
    --model "${MODEL}" \
    --served-model-name "${SERVED_NAME_A}" "${SERVED_NAME_B}" \
    --port "${PORT}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --max-num-batched-tokens "${MAX_BATCHED}" \
    --max-cudagraph-capture-size "${MAX_CUDAGRAPH_CAPTURE}" \
    --language-model-only \
    --enable-auto-tool-choice \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --enable-prefix-caching \
    --speculative-config.method mtp \
    --speculative-config.num_speculative_tokens "${NUM_SPEC}" \
    --attention-backend flashinfer \
    --load-format instanttensor \
    --default-chat-template-kwargs.preserve_thinking true
