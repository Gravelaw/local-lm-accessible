---
title: local-lm-accessible
sdk: gradio
sdk_version: 6.16.0
app_file: app.py
python_version: "3.11"
---

# local-lm-accessible

`local-lm-accessible` is a local-first small-model assistant for elderly users and low-vision users. v1 is designed to run without cloud inference, remote file upload, hosted OCR, external APIs, or default telemetry.

Hackathon track: Hugging Face Build Small Hackathon, Backyard AI. The app is built as a Gradio Space, keeps total deployed model parameters below 32B, and is designed to continue working offline after model download except for explicitly enabled optional web fetch.

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

## Dataset Policy

Training data must focus on India, Southeast Asia, North America, and Europe. Every dataset entry must include license, region, country, language, modality, task, and PII metadata. Unknown, missing, ambiguous, or unverifiable licenses are rejected.

Real financial, medical, legal, or identity documents are blocked unless synthetic, redacted, or provided with explicit user opt-in.

## Development

Install development dependencies in a local environment, then run:

```bash
pytest
ruff check .
```

Training and eval workspaces include local dry-run paths for text, vision, and
ASR. Full adapter training remains hardware- and artifact-dependent. Runtime
installs keep training-heavy packages optional; install the training extra only
when running LoRA/QLoRA or eval work:

```bash
pip install -e ".[dev,training]"
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
