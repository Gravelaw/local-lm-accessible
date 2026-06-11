#!/usr/bin/env bash
set -euo pipefail

MERGED_MODEL_DIR="${1:-training/text/outputs/merged_nemotron}"
OUTFILE="${2:-models/text/nemotron-router-summary-f16.gguf}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-${HOME}/llama.cpp}"
CONVERT_SCRIPT="${CONVERT_SCRIPT:-${LLAMA_CPP_DIR}/convert_hf_to_gguf.py}"

if [[ ! -f "${CONVERT_SCRIPT}" ]]; then
  echo "Missing llama.cpp conversion script: ${CONVERT_SCRIPT}" >&2
  exit 1
fi

if [[ ! -d "${MERGED_MODEL_DIR}" ]]; then
  echo "Missing merged HF model directory: ${MERGED_MODEL_DIR}" >&2
  exit 1
fi

if [[ "${OUTFILE}" != *.gguf ]]; then
  echo "Output file must end with .gguf: ${OUTFILE}" >&2
  exit 1
fi

mkdir -p "$(dirname "${OUTFILE}")"
python3 "${CONVERT_SCRIPT}" "${MERGED_MODEL_DIR}" --outfile "${OUTFILE}" --outtype f16
echo "wrote ${OUTFILE}"
