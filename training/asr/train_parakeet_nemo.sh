#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-nvidia/parakeet-tdt-0.6b-v3}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-training/asr/manifests/train.jsonl}"
VAL_MANIFEST="${VAL_MANIFEST:-training/asr/manifests/val.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-training/asr/outputs/parakeet_tdt_0_6b_v3}"
MAX_STEPS="${MAX_STEPS:-100}"
DRY_RUN="${DRY_RUN:-1}"
REQUIRE_AUDIO_EXISTS="${REQUIRE_AUDIO_EXISTS:-0}"

if [[ "${ALLOW_REMOTE_AUDIO_UPLOAD:-0}" != "0" ]]; then
  echo "Remote audio upload is not allowed for local-lm ASR training." >&2
  exit 1
fi

if [[ ! -f "${TRAIN_MANIFEST}" ]]; then
  echo "Missing TRAIN_MANIFEST: ${TRAIN_MANIFEST}" >&2
  exit 1
fi

if [[ ! -f "${VAL_MANIFEST}" ]]; then
  echo "Missing VAL_MANIFEST: ${VAL_MANIFEST}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

args=(
  python3 training/asr/train_parakeet_nemo.py
  --model-name "${MODEL_NAME}"
  --train-manifest "${TRAIN_MANIFEST}"
  --val-manifest "${VAL_MANIFEST}"
  --output-dir "${OUTPUT_DIR}"
  --max-steps "${MAX_STEPS}"
)

if [[ "${DRY_RUN}" == "1" ]]; then
  args+=(--dry-run)
fi

if [[ "${REQUIRE_AUDIO_EXISTS}" == "1" ]]; then
  args+=(--require-audio-exists)
fi

exec "${args[@]}"
