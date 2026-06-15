# HF Space And Model Operations

This runbook covers the files and commands needed to finish text model
fine-tuning, package the model, smoke-test llama.cpp, publish Hugging Face model
artifacts, and host the public Gradio Space.

## Modal Fine-Tuning And Packaging

Run from the repository root.

```bash
.venv/bin/modal run modal_workflows/local_lm_pipeline.py --action prepare_all_data --max-dataset-size-gb 10
.venv/bin/modal run modal_workflows/local_lm_pipeline.py --action finetune_text --no-dry-run
.venv/bin/modal run modal_workflows/local_lm_pipeline.py --action evaluate_text_adapter
.venv/bin/modal run modal_workflows/local_lm_pipeline.py --action run_text_adapter_packaging
.venv/bin/modal run modal_workflows/local_lm_pipeline.py --action smoke_test_packaged_gguf
```

The packaging job writes these Modal volume artifacts:

- `/models/text/local-lm-accessible-text-f16.gguf`
- `/models/text/local-lm-accessible-text-Q4_K_M.gguf`
- `/models/text/local-lm-accessible-text-Q5_K_M.gguf`
- `/reports/final_text_adapter_packaging_result.json`
- `/reports/final_text_gguf_smoke.json`

Download the selected local artifact when needed:

```bash
.venv/bin/modal volume get local-lm-data /models/text/local-lm-accessible-text-Q4_K_M.gguf models/text/local-lm-accessible-text-Q4_K_M.gguf --force
.venv/bin/modal volume get local-lm-data /reports/final_text_adapter_packaging_result.json reports/final_text_adapter_packaging_result.json --force
```

Update local release metadata after downloading the Q4 artifact:

```bash
.venv/bin/python scripts/update_text_model_release_manifest.py \
  --gguf-path models/text/local-lm-accessible-text-Q4_K_M.gguf \
  --model-id build-small-hackathon/local-lm-accessible-gguf
```

## Local llama.cpp Smoke Test

```bash
llama-cli \
  -m models/text/local-lm-accessible-text-Q4_K_M.gguf \
  -p "Return one short sentence explaining that local-lm runs locally." \
  -n 64 \
  --ctx-size 2048 \
  --temp 0.2 \
  --top-p 0.9 \
  --n-gpu-layers 99 \
  --no-display-prompt
```

## Publish Model Artifacts

Dry-run the guarded publish plan:

```bash
.venv/bin/modal run modal_workflows/local_lm_pipeline.py --action publish_hf_models
```

Publish to the hackathon organization:

```bash
.venv/bin/modal run modal_workflows/local_lm_pipeline.py \
  --action publish_hf_models \
  --publish-execute
```

If Hugging Face returns `403` for repo creation, ask an organization admin to
pre-create the two model repositories, then upload without the create step:

```bash
.venv/bin/modal run modal_workflows/local_lm_pipeline.py \
  --action publish_hf_models \
  --publish-execute \
  --publish-skip-create
```

Default target repositories:

- `build-small-hackathon/local-lm-accessible-text-lora`
- `build-small-hackathon/local-lm-accessible-gguf`

## Publish The Gradio Space

Build the upload bundle:

```bash
.venv/bin/python scripts/prepare_hf_space_bundle.py
```

Create or update the Space:

```bash
hf repos create build-small-hackathon/local-lm-accessible \
  --type space \
  --space-sdk gradio \
  --exist-ok

hf upload build-small-hackathon/local-lm-accessible \
  dist/hf_space_local_lm_accessible \
  --type space \
  --commit-message "Publish local-lm accessible Gradio Space"
```

If the create command returns `403`, ask an organization admin to create the
Gradio Space first, then run only the `hf upload` command.

The hosted Space is public-demo only. It must not ask users to upload private
documents. Private documents belong in laptop-local GGUF mode.

## Release Gate

```bash
.venv/bin/python scripts/verify_model_checksums.py --model text
.venv/bin/python scripts/release_gate.py
.venv/bin/python scripts/verify_dataset_locality.py
```
