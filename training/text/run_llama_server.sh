#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:-models/text/nemotron-router-summary-Q4_K_M.gguf}"
HOST="${LLAMA_HOST:-127.0.0.1}"
PORT="${LLAMA_PORT:-8081}"
LLAMA_SERVER="${LLAMA_SERVER:-llama-server}"

if [[ "${HOST}" != "127.0.0.1" && "${HOST}" != "localhost" ]]; then
  echo "Refusing to bind llama-server to non-local host: ${HOST}" >&2
  exit 1
fi

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "Missing GGUF model: ${MODEL_PATH}" >&2
  exit 1
fi

exec "${LLAMA_SERVER}" --model "${MODEL_PATH}" --host "${HOST}" --port "${PORT}"
