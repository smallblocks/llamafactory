#!/usr/bin/env bash
# Runs INSIDE the container on the Spark. The orchestrator mounts a run dir at
# /workspace/run containing config.yaml + dataset_info.json + the dataset, then
# `docker run`s this. Logs go to stdout (the orchestrator tails them over SSH).
set -euo pipefail

RUN_DIR="${RUN_DIR:-/workspace/run}"
LF_CONFIG="${LF_CONFIG:-${RUN_DIR}/config.yaml}"

echo "== lf-train =="
echo "run dir : ${RUN_DIR}"
echo "config  : ${LF_CONFIG}"
nvidia-smi -L || { echo "!! no GPU visible — run with --gpus all"; exit 1; }

# Optional gated/private model pull auth. Gemma 4 is Apache-2.0 so this is
# usually unnecessary, but harmless if set.
if [[ -n "${HF_TOKEN:-}" ]]; then
  echo "== HF login =="
  hf auth login --token "${HF_TOKEN}" >/dev/null 2>&1 || \
    huggingface-cli login --token "${HF_TOKEN}" >/dev/null 2>&1 || true
fi

cd "${RUN_DIR}"

# Single node (default) vs. 2-node FSDP across both Sparks over ConnectX/200GbE.
# The orchestrator sets NNODES=2 + NODE_RANK + MASTER_ADDR to fan out.
if [[ "${NNODES:-1}" -gt 1 ]]; then
  echo "== distributed: NNODES=${NNODES} NODE_RANK=${NODE_RANK:-0} MASTER_ADDR=${MASTER_ADDR:-} =="
  export FORCE_TORCHRUN=1
  export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"   # 1 GB10 GPU per Spark
  export MASTER_PORT="${MASTER_PORT:-29500}"
fi

set -x
exec llamafactory-cli train "${LF_CONFIG}"
