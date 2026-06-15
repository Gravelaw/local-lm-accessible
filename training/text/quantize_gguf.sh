#!/usr/bin/env bash
set -euo pipefail

INPUT_GGUF="${1:-models/text/nemotron-router-summary-f16.gguf}"
OUTPUT_DIR="${2:-models/text}"
LLAMA_QUANTIZE="${LLAMA_QUANTIZE:-llama-quantize}"
MODEL_BASENAME="${MODEL_BASENAME:-nemotron-router-summary}"

if [[ ! -f "${INPUT_GGUF}" ]]; then
  echo "Missing input GGUF: ${INPUT_GGUF}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
Q4_PATH="${OUTPUT_DIR}/${MODEL_BASENAME}-Q4_K_M.gguf"
Q5_PATH="${OUTPUT_DIR}/${MODEL_BASENAME}-Q5_K_M.gguf"
"${LLAMA_QUANTIZE}" "${INPUT_GGUF}" "${Q4_PATH}" Q4_K_M
"${LLAMA_QUANTIZE}" "${INPUT_GGUF}" "${Q5_PATH}" Q5_K_M
sha256sum "${Q4_PATH}" >"${Q4_PATH}.sha256"
sha256sum "${Q5_PATH}" >"${Q5_PATH}.sha256"
echo "wrote Q4_K_M and Q5_K_M GGUF files to ${OUTPUT_DIR}"
