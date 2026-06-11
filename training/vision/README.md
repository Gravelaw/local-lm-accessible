# Vision Training

This folder contains the local-first MiniCPM-V training scaffold for document OCR,
financial document extraction, handwritten notes, accessibility descriptions, and
image translation.

The v1 target model is `openbmb/MiniCPM-V-4.6`. Training must use LoRA or QLoRA,
keep logs local, and avoid cloud OCR, cloud inference, and remote telemetry. The
preferred execution backend is LLaMA-Factory, with SWIFT as a fallback for local
experiments.

## Data Format

Training records are JSONL objects validated by `training/vision/prepare_dataset.py`.
Each record must include:

- `image_path`
- `prompt`
- `expected_output`
- `region`
- `country`
- `language`
- `task`
- `source_type`
- `license`
- `pii_status`
- `modality`

Unknown licenses are rejected. Invoice, receipt, bill, and bank-statement records
must be synthetic, redacted, or explicit user opt-in. Bank statements are always
marked for human review.

## Dry Run And Command Planning

Validate the tiny local sample and write a dry-run summary:

```bash
python training/vision/train_minicpm_v_lora.py \
  --config training/vision/configs/minicpm_v_document_lora.yaml \
  --dry-run \
  --limit 6
```

Use `--require-images` only when the referenced image files are present locally.
The sample manifest intentionally stays lightweight and does not download images.

Running without `--dry-run` validates the train and eval manifests, enforces split
separation, and writes `local_backend_command_plan.json` in the configured output
directory. This command plan is non-executing by design; it records the local
LLaMA-Factory command and local-only environment defaults:

- `WANDB_DISABLED=true`
- `TRANSFORMERS_NO_ADVISORY_WARNINGS=1`
- `HF_HUB_OFFLINE=1`

The configured fallback backend is SWIFT. Use the command plan only after the
MiniCPM-V base model and required backend are staged locally.

## Current Limit

Full MiniCPM-V adapter execution remains hardware-dependent and is intentionally
not started by this scaffold. The current slice validates data, config,
local-only defaults, LoRA/QLoRA settings, split separation, metadata gates, and
the local backend command plan before any operator runs training.
