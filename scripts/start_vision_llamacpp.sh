#!/usr/bin/env bash
set -euo pipefail

HOST="${VISION_HOST:-127.0.0.1}"
PORT="${VISION_PORT:-8082}"
LLAMA_SERVER="${LLAMA_SERVER:-llama-server}"
LLAMA_GPU_LAYERS="${LLAMA_GPU_LAYERS:--1}"
LLAMA_REQUIRE_CUDA="${LLAMA_REQUIRE_CUDA:-0}"
LLAMA_CUDA_LIBRARY_PATH="${LLAMA_CUDA_LIBRARY_PATH:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ "${HOST}" != "127.0.0.1" && "${HOST}" != "localhost" ]]; then
  echo "Refusing to bind vision service to non-local host: ${HOST}" >&2
  exit 1
fi

eval "$("${PYTHON_BIN}" scripts/model_launch_info.py vision --shell)"
if [[ -n "${VISION_MODEL_PATH:-}" && "${VISION_MODEL_PATH}" != "${MANIFEST_MODEL_PATH}" ]]; then
  echo "Refusing VISION_MODEL_PATH override that differs from verified manifest path." >&2
  echo "Manifest path: ${MANIFEST_MODEL_PATH}" >&2
  echo "Override path: ${VISION_MODEL_PATH}" >&2
  exit 1
fi
if [[ -n "${VISION_MMPROJ_PATH:-}" && "${VISION_MMPROJ_PATH}" != "${MANIFEST_MMPROJ_PATH}" ]]; then
  echo "Refusing VISION_MMPROJ_PATH override that differs from verified manifest path." >&2
  echo "Manifest path: ${MANIFEST_MMPROJ_PATH}" >&2
  echo "Override path: ${VISION_MMPROJ_PATH}" >&2
  exit 1
fi
MODEL_PATH="${VISION_MODEL_PATH:-${MANIFEST_MODEL_PATH}}"
MMPROJ_PATH="${VISION_MMPROJ_PATH:-${MANIFEST_MMPROJ_PATH}}"
if [[ -n "${LLAMA_CUDA_LIBRARY_PATH}" ]]; then
  export LD_LIBRARY_PATH="${LLAMA_CUDA_LIBRARY_PATH}:${LD_LIBRARY_PATH:-}"
fi

if [[ "${LLAMA_REQUIRE_CUDA}" == "1" ]]; then
  "${PYTHON_BIN}" scripts/check_llamacpp_cuda.py --llama-server "${LLAMA_SERVER}" --require
fi

GPU_ARGS=()
if [[ -n "${LLAMA_GPU_LAYERS}" && "${LLAMA_GPU_LAYERS}" != "off" ]]; then
  GPU_ARGS+=(--n-gpu-layers "${LLAMA_GPU_LAYERS}")
fi

exec "${LLAMA_SERVER}" \
  --model "${MODEL_PATH}" \
  --mmproj "${MMPROJ_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  "${GPU_ARGS[@]}"
