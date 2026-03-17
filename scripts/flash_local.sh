#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
FLASH_BIN="${PROJECT_ROOT}/.venv/bin/flash"

if [[ ! -x "${FLASH_BIN}" ]]; then
  echo "flash CLI not found at ${FLASH_BIN}. Install it with: pip install -e \".[dev,runpod]\"" >&2
  exit 1
fi

# Local Docker preview should not depend on Docker Hub `latest`, which can jump
# to a CUDA runtime newer than the host NVIDIA driver supports.
export FLASH_IMAGE_TAG="${FLASH_IMAGE_TAG:-1.0.1}"
export FLASH_GPU_IMAGE="${FLASH_GPU_IMAGE:-docker.io/runpod/flash:${FLASH_IMAGE_TAG}}"
export FLASH_CPU_IMAGE="${FLASH_CPU_IMAGE:-docker.io/runpod/flash-cpu:${FLASH_IMAGE_TAG}}"
export FLASH_LB_IMAGE="${FLASH_LB_IMAGE:-docker.io/runpod/flash-lb:${FLASH_IMAGE_TAG}}"
export FLASH_CPU_LB_IMAGE="${FLASH_CPU_LB_IMAGE:-docker.io/runpod/flash-lb-cpu:${FLASH_IMAGE_TAG}}"
export FLASH_PREVIEW_LB_PORT="${FLASH_PREVIEW_LB_PORT:-18000}"
export FLASH_BUILD_PYTHON_VERSION="${FLASH_BUILD_PYTHON_VERSION:-3.12}"

exec "${FLASH_BIN}" "$@"
