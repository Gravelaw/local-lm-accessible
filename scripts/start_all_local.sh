#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

GATEWAY_HOST="${GATEWAY_HOST:-127.0.0.1}"
GATEWAY_PORT="${GATEWAY_PORT:-8000}"
PID_DIR="${PID_DIR:-.local_pids}"
LOG_DIR="${LOG_DIR:-.local_logs}"
START_TEXT="${START_TEXT:-1}"
START_VISION="${START_VISION:-1}"
START_ASR="${START_ASR:-1}"
START_OMNI="${START_OMNI:-0}"
STARTUP_TIMEOUT_SECONDS="${STARTUP_TIMEOUT_SECONDS:-45}"
STARTUP_POLL_SECONDS="${STARTUP_POLL_SECONDS:-0.5}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHON_BIN

if [[ "${GATEWAY_HOST}" != "127.0.0.1" && "${GATEWAY_HOST}" != "localhost" ]]; then
  echo "Refusing to bind gateway to non-local host: ${GATEWAY_HOST}" >&2
  exit 1
fi

mkdir -p "${PID_DIR}" "${LOG_DIR}"
"${PYTHON_BIN}" scripts/verify_model_checksums.py --manifest-only >/dev/null

if [[ "${START_TEXT}" == "1" ]]; then
  "${PYTHON_BIN}" scripts/verify_model_checksums.py --model text >/dev/null
fi

if [[ "${START_VISION}" == "1" ]]; then
  "${PYTHON_BIN}" scripts/verify_model_checksums.py --model vision >/dev/null
fi

if [[ "${START_ASR}" == "1" ]]; then
  "${PYTHON_BIN}" scripts/verify_model_checksums.py --model asr >/dev/null
fi

start_background() {
  local name="$1"
  shift
  echo "Starting ${name}: $*"
  if command -v setsid >/dev/null 2>&1; then
    setsid nohup "$@" >"${LOG_DIR}/${name}.log" 2>&1 &
  else
    nohup "$@" >"${LOG_DIR}/${name}.log" 2>&1 &
  fi
  local pid="$!"
  echo "${pid}" >"${PID_DIR}/${name}.pid"
  sleep 0.5
  if ! kill -0 "${pid}" >/dev/null 2>&1; then
    echo "${name} exited during startup. Log:" >&2
    sed -n '1,160p' "${LOG_DIR}/${name}.log" >&2
    exit 1
  fi
}

wait_ready() {
  local name="$1"
  local url="$2"
  shift 2
  echo "Waiting for ${name} readiness at ${url}"
  "${PYTHON_BIN}" scripts/wait_for_service.py \
    --name "${name}" \
    --url "${url}" \
    --timeout "${STARTUP_TIMEOUT_SECONDS}" \
    --interval "${STARTUP_POLL_SECONDS}" \
    "$@" >/dev/null
}

if [[ "${START_TEXT}" == "1" ]]; then
  start_background text scripts/start_text_llamacpp.sh
  wait_ready text "http://127.0.0.1:8081/health"
fi

if [[ "${START_VISION}" == "1" ]]; then
  start_background vision scripts/start_vision_llamacpp.sh
  wait_ready vision "http://127.0.0.1:8082/health"
fi

if [[ "${START_ASR}" == "1" ]]; then
  start_background asr scripts/start_asr_service.sh
  wait_ready asr "http://127.0.0.1:8090/health" --require-local-only
fi

if [[ "${START_OMNI}" == "1" ]]; then
  "${PYTHON_BIN}" scripts/verify_model_checksums.py --model omni >/dev/null
  echo "START_OMNI=1 requested, but no omni launcher is implemented yet." >&2
  exit 1
fi

start_background gateway "${PYTHON_BIN}" -m uvicorn services.gateway.app:app --host "${GATEWAY_HOST}" --port "${GATEWAY_PORT}"
wait_ready gateway "http://${GATEWAY_HOST}:${GATEWAY_PORT}/health" --require-local-only

echo "Local services requested. PID files are in ${PID_DIR}; logs are in ${LOG_DIR}."
echo "Run: ${PYTHON_BIN} scripts/healthcheck.py --gateway http://${GATEWAY_HOST}:${GATEWAY_PORT}"
echo "Run the Gradio UI separately with: ${PYTHON_BIN} scripts/start_gradio_app.sh"
