#!/usr/bin/env bash
set -euo pipefail

HOST="${ASR_HOST:-127.0.0.1}"
PORT="${ASR_PORT:-8090}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ "${HOST}" != "127.0.0.1" && "${HOST}" != "localhost" ]]; then
  echo "Refusing to bind ASR service to non-local host: ${HOST}" >&2
  exit 1
fi

"${PYTHON_BIN}" scripts/verify_model_checksums.py --model asr >/dev/null
exec "${PYTHON_BIN}" -m uvicorn services.stt.parakeet_service:app --host "${HOST}" --port "${PORT}"
