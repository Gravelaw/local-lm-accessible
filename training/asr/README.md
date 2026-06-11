# ASR Training

This folder contains the eval-first and experimental Parakeet NeMo training path.
The target model is `nvidia/parakeet-tdt-0.6b-v3`.

Training is local-only. Scripts validate JSONL manifests before any NeMo command
is launched and reject `ALLOW_REMOTE_AUDIO_UPLOAD=1`.

## Dry Run

Dry-run is the default for the shell launcher:

```bash
TRAIN_MANIFEST=training/asr/sample_data/tiny_manifest.jsonl \
VAL_MANIFEST=training/asr/sample_data/tiny_manifest.jsonl \
bash training/asr/train_parakeet_nemo.sh
```

This validates metadata and prints the experimental NeMo command without requiring
audio files to exist.

## Local Artifact Staging

The Parakeet artifact is not downloaded automatically. Print the operator plan,
download only after explicit approval, then write the manifest checksum:

```bash
python3 scripts/download_models.py --model asr --print-plan
python3 scripts/download_models.py --model asr --download --allow-large-download
python3 scripts/verify_model_checksums.py --model asr --write-manifest-checksum
python3 scripts/verify_model_checksums.py --model asr
```

The release gate remains blocked until `models/manifest.json` has a non-empty
ASR checksum for the staged local artifact.

## Tiny Local Eval

Evaluate the tiny local manifest and write JSON/Markdown reports:

```bash
python3 training/asr/eval_wer.py \
  --manifest training/asr/sample_data/tiny_manifest.jsonl \
  --predictions training/asr/sample_data/tiny_predictions.json \
  --report-json reports/asr_eval.json \
  --report-md reports/asr_eval.md
```

The eval reports WER, CER, language detection accuracy, unsupported-language
detection, noisy-room WER, elderly-speaker WER, missing predictions, and explicit
unsupported-language failures.

## Experimental Training

Actual NeMo execution is opt-in and requires local audio files to exist:

```bash
DRY_RUN=0 \
REQUIRE_AUDIO_EXISTS=1 \
TRAIN_MANIFEST=training/asr/manifests/train.jsonl \
VAL_MANIFEST=training/asr/manifests/val.jsonl \
bash training/asr/train_parakeet_nemo.sh
```

Indian and Southeast Asian non-English ASR remains experimental unless local eval
results prove usable. See `README_LIMITATIONS.md` for supported language policy.
