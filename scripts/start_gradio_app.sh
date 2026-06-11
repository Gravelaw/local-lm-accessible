#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

HOST="${GRADIO_HOST:-127.0.0.1}"
PORT="${GRADIO_PORT:-7860}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ "${HOST}" != "127.0.0.1" && "${HOST}" != "localhost" ]]; then
  echo "Refusing to bind Gradio app to non-local host: ${HOST}" >&2
  exit 1
fi

export GRADIO_SERVER_NAME="${HOST}"
export GRADIO_SERVER_PORT="${PORT}"
export GRADIO_ANALYTICS_ENABLED="${GRADIO_ANALYTICS_ENABLED:-False}"

exec "${PYTHON_BIN}" app.py
