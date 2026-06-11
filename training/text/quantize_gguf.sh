#!/usr/bin/env bash
set -euo pipefail

INPUT_GGUF="${1:-models/text/nemotron-router-summary-f16.gguf}"
OUTPUT_DIR="${2:-models/text}"
LLAMA_QUANTIZE="${LLAMA_QUANTIZE:-llama-quantize}"

if [[ ! -f "${INPUT_GGUF}" ]]; then
  echo "Missing input GGUF: ${INPUT_GGUF}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
"${LLAMA_QUANTIZE}" "${INPUT_GGUF}" "${OUTPUT_DIR}/nemotron-router-summary-Q4_K_M.gguf" Q4_K_M
"${LLAMA_QUANTIZE}" "${INPUT_GGUF}" "${OUTPUT_DIR}/nemotron-router-summary-Q5_K_M.gguf" Q5_K_M
sha256sum "${OUTPUT_DIR}/nemotron-router-summary-Q4_K_M.gguf" >"${OUTPUT_DIR}/nemotron-router-summary-Q4_K_M.gguf.sha256"
sha256sum "${OUTPUT_DIR}/nemotron-router-summary-Q5_K_M.gguf" >"${OUTPUT_DIR}/nemotron-router-summary-Q5_K_M.gguf.sha256"
echo "wrote Q4_K_M and Q5_K_M GGUF files to ${OUTPUT_DIR}"
