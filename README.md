---
title: local-lm
sdk: gradio
sdk_version: 6.16.0
app_file: app.py
python_version: "3.11"
---

# local-lm

`local-lm` is a local-first small-model assistant for elderly users and low-vision users. v1 is designed to run without cloud inference, remote file upload, hosted OCR, external APIs, or default telemetry.

Hackathon track: Hugging Face Build Small Hackathon, Backyard AI. The app is built as a Gradio Space, keeps total deployed model parameters below 32B, and is designed to continue working offline after model download except for explicitly enabled optional web fetch.

Official hackathon Space:
https://build-small-hackathon-local-lm-accessible-for-elders.hf.space

## v1 Architecture

- `services/gateway/`: FastAPI local gateway, request schemas, local route selection, tool registry, and local model client definitions.
- `services/tools/`: Offline document, image, export, search, and safety utilities.
- `services/stt/`: Local speech-to-text service entry point.
- `configs/`: Local-only runtime, routing, training-mix, and safety policy defaults.
- `data/`: Dataset registry, schemas, synthetic/raw/processed data areas, and split outputs.
- `training/`: Local training/eval workspaces for Nemotron text LoRA, MiniCPM-V vision LoRA scaffolding, and Parakeet ASR eval-first experiments.
- `evals/`: Local eval entry points and report generation.
- `scripts/`: Local service startup, model checksum verification, health checks, and GGUF-oriented model scripts.
- `models/`: Local model manifest and model documentation.
- `tests/`: Pytest coverage for routing, schemas, dataset registry metadata, and no-cloud defaults.

## Run The Gradio Space Locally

The Space entrypoint is `app.py`.

```bash
python app.py
```

For a full laptop-local demo with staged model artifacts, start the local
backends first, then start the Gradio UI:

```bash
PYTHON_BIN=.venv/bin/python STARTUP_TIMEOUT_SECONDS=180 scripts/start_all_local.sh
PYTHON_BIN=.venv/bin/python scripts/start_gradio_app.sh
```

Then open:

- Gradio UI: `http://127.0.0.1:7860`
- Gateway health: `http://127.0.0.1:8000/health`

Verify required local services:

```bash
.venv/bin/python scripts/healthcheck.py --gateway http://127.0.0.1:8000 --require-running
```

### NVIDIA CUDA llama.cpp Preflight

Text and vision launchers pass `--n-gpu-layers -1` by default so a GPU-capable
llama.cpp build will offload model layers. For a demo where GPU use is required,
run the preflight and set `LLAMA_REQUIRE_CUDA=1` so startup fails fast instead of
silently running on CPU:

```bash
LLAMA_SERVER=/home/newthingnow/work-session/llama.cpp/build-cuda-sm120-3/bin/llama-server \
LLAMA_CUDA_LIBRARY_PATH=/home/newthingnow/.local/lib/python3.14/site-packages/nvidia/cu13/lib:/usr/lib/wsl/lib \
.venv/bin/python scripts/check_llamacpp_cuda.py --require

LLAMA_SERVER=/home/newthingnow/work-session/llama.cpp/build-cuda-sm120-3/bin/llama-server \
LLAMA_CUDA_LIBRARY_PATH=/home/newthingnow/.local/lib/python3.14/site-packages/nvidia/cu13/lib:/usr/lib/wsl/lib \
LLAMA_REQUIRE_CUDA=1 \
LLAMA_GPU_LAYERS=-1 \
PYTHON_BIN=.venv/bin/python \
STARTUP_TIMEOUT_SECONDS=180 \
scripts/start_all_local.sh
```

If the preflight reports no devices or a CUDA driver/runtime mismatch, fix the
host GPU visibility first. `nvidia-smi` must work in the same shell, and the
llama.cpp CUDA build must match the installed driver/toolkit.

Primary tabs:

- Ask: simple local assistant answers through the text endpoint, with a local sample question and a visible local-model-unavailable fallback.
- Read: summarize local article text or Wikipedia-style samples; optional bounded web fetch remains off by default and only runs after per-request opt-in.
- Documents: convert bills, invoices, receipts, notes, or statements to JSON/XLSX/TXT/PDF.
- Images: describe photos for accessibility or translate visible text through the local vision path.
- Speech: transcribe local audio through the local Parakeet ASR service when staged.
- Ask, Read, Images, and Speech results can be saved locally as TXT, XLSX, or PDF.
- PDF saves are generated locally with `reportlab`; no external document service is used.
- Settings / Privacy: show privacy status, demo readiness, model budget, and accessibility controls; raw runtime JSON stays in an advanced panel.

For a quick local smoke test that does not require model weights:

```bash
python3 scripts/smoke_test_local.py --mock-model-endpoints
```

## Fine-Tuning, Packaging, And Hosted Space Files

The Modal fine-tuning, GGUF packaging, Hugging Face model publish, and public
Gradio Space upload steps are documented in
`docs/hf_space_and_model_ops.md`.

The standalone hosted Space bundle lives in `spaces/local_lm_accessible/`.
Build the upload directory with:

```bash
.venv/bin/python scripts/prepare_hf_space_bundle.py
```

That smoke test starts loopback-only mock llama.cpp-compatible `/completion`
servers for text and vision, then confirms the gateway uses local HTTP model
clients for Ask, Read, and image tasks without cloud calls. It also writes a
sample invoice JSON, XLSX, TXT, and real local PDF under the smoke work
directory.

After starting verified local model services, require real text, vision, and ASR
service readiness before accepting the smoke result. This mode also sends a tiny
local WAV through `/tasks/speech_to_text` and requires the ASR response to report
`model_ready=true`:

```bash
python3 scripts/smoke_test_local.py --require-real-model-services
```

Use `--request-timeout 120` or a similar bounded value on CPU-only machines
where local text/vision requests can take longer than the default quick checks.

## Hosted Space Privacy Note

When this runs as a Hugging Face Space, user files are processed inside the Space
runtime rather than on the user's laptop. The app still does not call external
model APIs, cloud OCR, external telemetry, AWS, Azure, Google Cloud, or OpenAI.
Laptop-local mode uses the same Gradio and FastAPI gateway architecture after
models are downloaded locally.

## Local-Only Defaults

The repository defaults to local serving through `llama.cpp`/GGUF-compatible endpoints for text and vision, and a local ASR service for speech. Model sources are restricted to OpenBMB, NVIDIA, Cohere, and BFL. The v1 deployed model bundle must remain below 32B total parameters.

No runtime component should require AWS, Azure, Google Cloud, hosted OCR, hosted LLM APIs, remote telemetry, or external APIs.

## License

The application source code and repository documentation are licensed under
Apache-2.0; see `LICENSE`.

Model weights, datasets, and third-party artifacts are not relicensed by this
repository. They remain under their upstream licenses and must continue to pass
the project license, vendor, PII, and local-only gates before use.

## Attribution

This repository is maintained by Gravelaw and was implemented with assistance
from OpenAI Codex.

## Dataset Policy

Training data must focus on India, Southeast Asia, North America, and Europe. Every dataset entry must include license, region, country, language, modality, task, and PII metadata. Unknown, missing, ambiguous, or unverifiable licenses are rejected.

Real financial, medical, legal, or identity documents are blocked unless synthetic, redacted, or provided with explicit user opt-in.

### Dataset Discovery, Review, And Preparation

Dataset catalogs are used for discovery and metadata import only. They are not automatic approval to train, and the app runtime must not depend on Hugging Face Datasets, Kaggle, AWS, Google Cloud, Azure, EPO OPS, Wikimedia downloads, or other external dataset services.

The registry module supports offline seeded discovery plus local CSV/JSON/JSONL manifest imports for:

- Hugging Face Datasets
- Kaggle Datasets
- AWS Registry of Open Data
- Google BigQuery Public Datasets / Google Cloud public datasets
- Azure Open Datasets
- UCI Machine Learning Repository
- European Data Portal / data.europa.eu
- European Patent Office data / EPO Open Patent Services
- Awesome Public Datasets
- Google Dataset Search manual exports
- Wikimedia dumps
- Manual manifests
- Synthetic datasets generated by this repo
- User opt-in redacted datasets

Create the offline candidate registry without downloading dataset contents:

```bash
.venv/bin/python scripts/discover_datasets.py \
  --sources huggingface,kaggle,uci,aws,google,azure,europe,epo,wikimedia,manual \
  --query "invoice OCR" \
  --max-results 50 \
  --output data/registry/dataset_candidates.jsonl \
  --no-download
```

`--no-download` is the default. Large datasets are written to `data/registry/dataset_candidates.large_downloads.json` and require an explicit approval path before any content download or preparation script may fetch them.

Audit candidates and split them into approved, review/eval-only, and rejected queues:

```bash
.venv/bin/python scripts/audit_source_registry.py
```

The audit writes:

- `data/registry/approved_datasets.jsonl`
- `data/registry/research_eval_datasets.jsonl`
- `data/registry/rejected_datasets.jsonl`
- `reports/dataset_registry_audit.json`
- `reports/dataset_registry_audit.md`

Approval is explicit and fail-closed:

```bash
.venv/bin/python scripts/approve_dataset.py "dataset-id-or-name"
.venv/bin/python scripts/reject_dataset.py "dataset-id-or-name" --reason "documented reason"
```

Approved datasets must have a task mapping and generated card before training use:

```bash
.venv/bin/python scripts/map_datasets_to_tasks.py
.venv/bin/python scripts/create_dataset_card.py
.venv/bin/python scripts/build_training_mix.py
.venv/bin/python scripts/check_regional_balance.py
.venv/bin/python scripts/check_no_cloud_runtime_dependency.py
```

Generated outputs include `data/registry/task_mapped_datasets.jsonl`, `reports/dataset_task_coverage.*`, `data/dataset_cards/*.md`, and train/validation/test/regional-stress-test manifests under `data/splits/`. If regional targets are missed, `scripts/check_regional_balance.py` still writes `reports/regional_balance.*` and exits nonzero so the imbalance remains visible.

Prepare the first approved small-dataset slice:

```bash
.venv/bin/python scripts/download_prepare_small_datasets.py
```

This script uses `data/registry/small_dataset_ingestion_status.json` as a checkpoint. It writes raw artifacts under `data/raw/datasets/`, prepared artifacts under `data/processed/datasets/`, and reports to `reports/small_dataset_ingestion.*`. It refuses large registry records by default. In the current slice it prepares SROIE from Hugging Face, downloads/indexes CORD v2 from Hugging Face, downloads/prepares XFUND from GitHub release assets, downloads/prepares FATURA from Zenodo, generates local synthetic document data, downloads and samples UCI Online Retail, and writes an explicit manual collection manifest for FLEURS eval-subset selection.

Each dataset is capped at 10 GB by default:

```bash
.venv/bin/python scripts/download_prepare_small_datasets.py --max-dataset-size-gb 10
```

The cap is checked against registry `size_bytes` metadata before download and against local raw/processed files after preparation. Datasets above the cap must be split into smaller approved subsets before ingestion.

CORD parquet artifacts are normalized into receipt-extraction JSONL when
`pyarrow` is available. The file index is still written for traceability.

### Modal Data Prep And Fine-Tuning

The Build Small Hackathon provides Modal credits for data processing and model
fine-tuning. The Modal workflow is intentionally separate from the Gradio
runtime: `modal` is an optional dependency, and the shipped app does not import
Modal, cloud SDKs, or dataset loaders.

Install the optional Modal tooling locally:

```bash
.venv/bin/python -m pip install -e ".[modal]"
```

Create the Modal volumes and Hugging Face token secret once:

```bash
.venv/bin/modal volume create local-lm-data
.venv/bin/modal volume create local-lm-cache
.venv/bin/modal secret create huggingface-secret HF_TOKEN=hf_...
```

Run remote ingestion and preparation with the same 10 GB dataset cap:

```bash
.venv/bin/modal run modal_workflows/local_lm_pipeline.py \
  --action ingest_data \
  --max-dataset-size-gb 10
```

Run registry batch processing and prepared training-manifest generation:

```bash
.venv/bin/modal run modal_workflows/local_lm_pipeline.py --action batch_process
```

Run the full remote data pipeline:

```bash
.venv/bin/modal run modal_workflows/local_lm_pipeline.py \
  --action prepare_all_data \
  --max-dataset-size-gb 10
```

Run fine-tuning dry runs on Modal GPU compute:

```bash
.venv/bin/modal run modal_workflows/local_lm_pipeline.py --action finetune_text --dry-run
.venv/bin/modal run modal_workflows/local_lm_pipeline.py --action finetune_vision --dry-run
.venv/bin/modal run modal_workflows/local_lm_pipeline.py --action create_vision_readiness
.venv/bin/modal run modal_workflows/local_lm_pipeline.py --action check_asr_contingency
```

The default text fine-tuning backend is Hugging Face TRL `SFTTrainer` with PEFT
LoRA/QLoRA because it is already covered by local tests. The default Modal text
model is `nvidia/Llama-3.1-Nemotron-Nano-8B-v1`, which avoids the
`causal-conv1d`/`mamba-ssm` source-build blocker seen with the hybrid Mamba
Nemotron path. Unsloth remains a candidate acceleration backend after dependency
smoke tests pass. The optional native-extension step runs a CUDA toolchain
preflight first and pins the CUDA host compiler to `/usr/bin/g++` through
`CUDAHOSTCXX`, `CMAKE_ARGS`, `CUDAFLAGS`, and `NVCC_PREPEND_FLAGS` so CUDA 13
does not accidentally select an unsupported `clang++`.
If the capped job is memory- or throughput-bound, increase Modal GPU class in
this order: `A10G`, `A100-40GB`, `A100-80GB`, `H100`, then `H200`.

Run the first guarded text adapter job only after dry runs and preflight reports
pass:

```bash
.venv/bin/modal run modal_workflows/local_lm_pipeline.py --action check_training_toolchain
.venv/bin/modal run modal_workflows/local_lm_pipeline.py --action prepare_nemotron_dependencies
.venv/bin/modal run modal_workflows/local_lm_pipeline.py --action finetune_text --no-dry-run
.venv/bin/modal run modal_workflows/local_lm_pipeline.py --action evaluate_text_adapter
.venv/bin/modal run modal_workflows/local_lm_pipeline.py --action plan_text_adapter_packaging
.venv/bin/modal run modal_workflows/local_lm_pipeline.py --action run_text_adapter_packaging
.venv/bin/modal run modal_workflows/local_lm_pipeline.py --action smoke_test_packaged_gguf
.venv/bin/modal run modal_workflows/local_lm_pipeline.py --action check_finetuning_completion
```

The final text job writes the LoRA adapter under
`/vol/local-lm/training/text/llama_nemotron_nano_modal_lora`, emits
`adapter_manifest.json` with checksum metadata, and writes final reports under
`/vol/local-lm/reports`. `evaluate_text_adapter` now runs LoRA adapter
generation, writes `final_text_adapter_eval.*`, and writes
`final_text_adapter_readiness.*`. `plan_text_adapter_packaging` writes the
reviewable merge/export/quantize plan to `final_text_adapter_packaging_plan.*`;
`run_text_adapter_packaging` performs the reviewed merge, F16 GGUF export, and
Q4/Q5 quantization; `smoke_test_packaged_gguf` verifies the Q4 GGUF with
`llama.cpp`; and `check_finetuning_completion` fails closed unless text
training, adapter eval, packaging, GGUF smoke, vision readiness, and ASR
contingency reports are all present.

Vision readiness writes `vision_readiness.*` and validates the OpenBMB
MiniCPM-V LoRA/QLoRA command plan without downloading or training on images
unless `--require-images` is explicitly used by the underlying script. ASR
contingency writes `asr_contingency.*`; alternate ASR candidates are restricted
to NVIDIA families, must be verified locally, and stay eval-only until WER and
unsupported-language checks pass.

The Modal app copies source code into `/workspace/local-lm`, stores data and
reports on the `local-lm-data` volume under `/vol/local-lm`, and uses
`local-lm-cache` for package and Hugging Face caches. `.modalignore` excludes
local virtual environments, raw data, prepared data, model weights, and training
outputs from the uploaded source bundle.

Before fine-tuning, confirm these Modal outputs exist:

- `/vol/local-lm/data/processed/training/document_extraction_*.jsonl`
- `/vol/local-lm/data/processed/training/text_sft_*.jsonl`
- `/vol/local-lm/data/processed/training/asr_*.jsonl`
- `/vol/local-lm/data/processed/training/tabular_eval.jsonl`
- `/vol/local-lm/reports/prepared_dataset_gaps.*`
- `/vol/local-lm/reports/fine_tuning_*_preflight.json`

## Development

Install development dependencies in a local environment, then run:

```bash
pytest
ruff check .
```

Training and eval workspaces include local dry-run paths for text, vision, and
ASR. Modal text fine-tuning uses prepared manifests on `local-lm-data`; full
vision adapter training remains backend- and artifact-dependent. Runtime
installs keep training-heavy packages optional; install the training extra only
when running LoRA/QLoRA or eval work:

Vector or embedding databases are useful for local retrieval, example selection,
deduplication, and eval-set construction. They should not be treated as a
replacement for supervised fine-tuning. Any TurboVec-style backend must pass
license review and local-runtime checks before being added to the app path.

```bash
pip install -e ".[dev,training]"
```

Install the Nemotron native-extension extra only on CUDA-capable training
machines:

```bash
pip install -e ".[nemotron-training]"
```

Model packaging checks:

```bash
python3 scripts/verify_model_checksums.py --model text
python3 scripts/verify_model_checksums.py --model vision
PYTHON_BIN=.venv/bin/python STARTUP_TIMEOUT_SECONDS=180 scripts/start_all_local.sh
.venv/bin/python scripts/healthcheck.py --gateway http://127.0.0.1:8000 --require-running
python3 scripts/smoke_test_local.py --require-real-model-services
python3 scripts/release_gate.py
```

`scripts/release_gate.py` verifies required model checksums and metadata before
release. Its failure path prints structured JSON with `failures` and
`next_actions` instead of a traceback.

## Model Download Flow

Model downloads are explicit operator actions. Print the plan first:

```bash
python3 scripts/download_models.py --print-plan
python3 scripts/download_models.py --model asr --print-plan
```

Parakeet ASR is staged locally in this workspace. To restage it:

```bash
python3 scripts/download_models.py --model asr --download --allow-large-download
python3 scripts/verify_model_checksums.py --model asr --write-manifest-checksum
```

Then verify the staged artifact and release gate:

```bash
python3 scripts/verify_model_checksums.py --model asr
python3 scripts/release_gate.py
```

## Demo Checklist

See `docs/HACKATHON_DEMO.md` for the demo-video checklist, field-notes outline,
and judge-facing run script.
