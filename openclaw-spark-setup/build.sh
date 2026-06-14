#!/usr/bin/env bash
# Build the LLaMA-Factory Blackwell image on a DGX Spark.
# Run this once per Spark (or build once and `docker save | ssh spark2 docker load`).
set -euo pipefail

IMAGE="${IMAGE:-llamafactory-spark:latest}"
NGC_TAG="${NGC_TAG:-25.04-py3}"
LLAMAFACTORY_REF="${LLAMAFACTORY_REF:-main}"

cd "$(dirname "$0")"

echo ">> Building ${IMAGE} (NGC ${NGC_TAG}, LLaMA-Factory ${LLAMAFACTORY_REF})"
echo ">> arch: $(uname -m)  (expect aarch64 on a DGX Spark)"

docker build \
  --build-arg "NGC_TAG=${NGC_TAG}" \
  --build-arg "LLAMAFACTORY_REF=${LLAMAFACTORY_REF}" \
  -t "${IMAGE}" \
  .

echo ">> Done. Quick GPU check:"
docker run --rm --gpus all "${IMAGE}" \
  python -c "import torch; print('CUDA visible:', torch.cuda.is_available()); \
print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
