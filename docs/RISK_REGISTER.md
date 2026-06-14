# Risk Register And Follow-Up Backlog

This file tracks repository risks found during the parallel review of the local-lm scaffold. It is organized by the project structure so future PRs can address the highest-risk areas first.

Status: generated from a read-only review of dataset, training, evaluation, and deployment code.
This register is now being maintained as implementation progresses.

## Current Implementation Status

### Resolved Or Mitigated

- Dataset registry now separates approved training datasets from `research_eval_only` datasets.
- Manual dataset approval refuses research/eval-only records for the training allowlist.
- Ambiguous and unknown license markers are rejected by the registry acceptance path.
- Synthetic document `metadata.jsonl` rows now include license, modality, task, source type, and `pii_status`, and repeated generation rewrites unique records by `document_id` instead of appending duplicates.
- Text SFT dry-run now validates assistant-only label masking so user/system tokens are ignored in loss labels.
- Text adapter eval records explicit prediction sources instead of silently scoring labels.
- MiniCPM-V vision training is no longer only a placeholder; it has a local-only LoRA/QLoRA scaffold, separate train/eval sample JSONL files, split-overlap checks, backend command planning for LLaMA-Factory or SWIFT, config validation, and dry-run tests.
- ASR manifests include `country`, `modality`, `task`, license, and PII metadata; unsupported Indian/SEA non-English ASR is flagged experimental.
- ASR service now rejects missing local audio files and reports whether the Parakeet artifact is checksum-ready.
- ASR service now has a lazy local-only Parakeet transcription path for checksum-verified artifacts and falls back visibly if the local runtime dependency is unavailable.
- ASR NeMo training launcher now defaults to dry-run, validates train/validation manifests before training, rejects remote audio upload, and requires audio files for real training.
- Modal text fine-tuning defaults to HF TRL/PEFT. Unsloth is tracked as an acceleration candidate for Nemotron 3, but must pass dependency, license, and manifest compatibility checks before becoming default.
- ASR eval now writes tested JSON/Markdown reports and lists unsupported-language detection failures explicitly, so unsupported-language hallucinations are auditable instead of only aggregated.
- Image/PDF document conversion now attempts local vision-model JSON extraction, validates outputs with Pydantic invoice/bank schemas, and falls back with warnings on malformed JSON.
- Gradio XLSX document exports now include a metadata sheet with task, status, confidence, warnings, source, model endpoint, and human-review flag.
- Offline Wikipedia/local article summarization now uses a local SQLite FTS5 index seeded from the demo article and never requires network access.
- Gateway `/health` and the Gradio status panel now expose required-model readiness, local artifact presence, checksum configuration, and pending ASR state without hashing large models on every health call.
- Unified evals now use explicit target adapters and a socket-level loopback network guard.
- Unified evals can now run the `llama_cpp_endpoint` target against separate loopback text and vision endpoints while keeping ASR on the ASR/deterministic path instead of incorrectly routing speech tasks to llama.cpp.
- Eval reports now include explicit metrics for route accuracy, JSON validity, summary source coverage, invoice/bank reconciliation, low-confidence human-review flags, unsafe advice, identity guessing, unsupported-language handling, invalid refusal rate, and readability.
- Critical-failure detection has direct tests for every registered failure type.
- Text adapter merge defaults to local-files-only, disables remote code trust by default, and validates adapter/output paths before importing training dependencies.
- Text GGUF export validates the merged HF model directory and `.gguf` output path; quantization writes checksum sidecars.
- `training/text/run_llama_server.sh` rejects non-loopback host binds.
- Text and vision startup scripts derive verified artifact paths from the manifest and reject mismatched path overrides.
- Healthcheck and smoke-test gateway URLs are loopback-only by default.
- Local text/vision model clients now support llama.cpp native `/completion` and fallback to loopback-only OpenAI-compatible `/v1/chat/completions` when the native path is not available.
- `start_all_local.sh` now waits for loopback HTTP readiness for text, vision, ASR, and gateway services instead of relying only on process liveness.
- `scripts/healthcheck.py --require-running` now requires loopback `/health` readiness for required services and treats optional omni readiness failure as non-blocking.
- `scripts/smoke_test_local.py --mock-model-endpoints` now starts loopback-only llama.cpp-compatible `/completion` mocks and proves gateway text/vision endpoint integration without model downloads.
- `scripts/smoke_test_local.py` now verifies sample invoice JSON, XLSX, TXT, and real local PDF outputs.
- Local smoke tests now exercise the Ask/general assistant route as well as Read, document, image, and speech paths.
- `scripts/smoke_test_local.py --require-real-model-services` now fails closed unless required loopback text, vision, and ASR service `/health` checks pass before smoke tasks run, and it requires the speech-to-text smoke call to return a ready ASR model transcript.
- Release readiness gate verifies required local model artifacts against manifest checksums and checks vendor policy, parameter budget, placeholder text, unresolved license/commercial review, and manifest/config consistency.
- Release readiness now requires enabled `configs/models.yaml` parameter metadata to match `models/manifest.json` base parameters within a narrow tolerance; ASR config metadata is aligned to the 0.6B Parakeet manifest entry.
- Required text, vision, and ASR manifest entries now carry concrete reviewed license/commercial metadata: Nemotron Open Model License, Apache-2.0, and CC-BY-4.0 respectively.
- `scripts/verify_model_checksums.py --write-manifest-checksum` now computes deterministic local artifact checksums, updates manifest entries after explicit staging, and refuses empty directory artifacts.
- Space runtime requirements exclude training-heavy dependencies and invalid non-runtime packages.
- Core `pyproject.toml` dependencies now stay runtime-oriented; training-heavy libraries are isolated in the optional `training` extra.
- Hugging Face Space metadata, `requirements.txt`, and `pyproject.toml` now pin the same Gradio SDK version; runtime requirements include `uvicorn` for local service startup while keeping training-heavy packages out.
- Editable dev installs now use explicit setuptools package discovery, and `ruff check .` passes locally after applying project-wide formatting/lint fixes.
- Release gate CLI failures now emit structured JSON with `failures` and `next_actions`, while preserving a nonzero exit code for automation.
- Gradio readiness now calls `run_release_gate(verify_checksums=False)` for a metadata-only release-gate summary, including ASR blocker details and next actions, without performing full checksum hashing during page load. The CLI `scripts/release_gate.py` remains the full checksum-verifying gate.
- Gradio first-screen disclosure now states the hosted-Space compute boundary, no external model APIs/cloud OCR/telemetry, and laptop-local architecture before users open Settings.
- Gradio Settings / Privacy now exposes a local demo-flow check that runs the sample Ask, Read, Document, Image, and Speech handlers and verifies structured local outputs, visible warnings, and document downloads.
- Gateway and Gradio ASR requests now preserve language, region, country, and explicit experimental opt-in; unsupported or Indian/SEA non-English Parakeet requests fail closed locally with warnings unless opted in.
- Image accessibility output now blocks identity-like claims from local vision responses, removes guessed names, records uncertainty in the Pydantic schema, and returns a human-review warning.
- Synthetic document generation now emits an idempotent `synthetic_dataset_candidates.jsonl` with dataset-registry-compatible metadata, and invoice/bank-statement candidates pass the default acceptance gate.
- Unified sample document evals now use real generated synthetic PNG, ground-truth JSON, and expected XLSX artifacts for invoice, receipt, bank-statement, and handwritten-note cases.
- Text training now enforces train/eval split separation with content fingerprints, uses a separate eval sample file, and overrides `WANDB_DISABLED=true` when W&B is disabled in config.
- Vision training now enforces train/eval split separation with content fingerprints, validates OpenBMB/local-only LoRA/QLoRA defaults, and writes a non-executing local backend command plan instead of requiring cloud training infrastructure.
- Gradio now exposes a first-tab Ask flow for the general local assistant route, includes a local sample question, and its fallback explicitly says the local text endpoint is unavailable without echoing the user's prompt as an answer.
- Gradio keeps the plain privacy/runtime status visible while moving raw runtime JSON into a closed advanced details panel.
- Gradio Settings / Privacy now includes a data-backed Demo Readiness panel that separates demo readiness from release readiness and lists the next local verification commands.
- Gradio demo readiness now depends on actual required artifact presence plus checksum metadata, and empty directory artifacts are treated as missing.
- Gradio primary text outputs now append warning text directly in the user-visible result box instead of relying only on JSON status panels.
- Ask, Read, Image, and Speech result panes can now save local TXT/XLSX/PDF files, so the demo covers the requested save-output flow beyond document conversion.
- Local PDF exports now generate bounded reportlab PDFs instead of text placeholder files with a `.pdf` suffix.
- Gradio high-contrast mode now emits an actual local stylesheet from the Settings / Privacy toggle instead of only reporting status text.
- Gradio base CSS now applies large button sizing to all buttons, including secondary sample loaders, not only primary action buttons.
- Gradio base accessibility CSS is now embedded as a local style component in the Space tree instead of relying only on a launch-time CSS argument.
- Optional web-page summarization now has a bounded per-request opt-in path for public text/HTML pages; default runtime health/config still reports `allow_web=false`.
- Optional web fetch now rejects localhost, private IP literals, unresolvable hostnames, and hostnames resolving to non-public addresses before any bounded public HTTP(S) text/HTML fetch.
- General assistant and summarization prompts with obvious financial, medical, or legal content now carry uncertainty and qualified-human-review warnings in both gateway responses and Gradio-visible text.
- Gradio now generates local demo image and speech samples on demand, so every primary task tab can be demonstrated without committed binary assets or external files.
- Image description and visible-text translation fallbacks now explicitly say the selected file stayed local, no cloud OCR or remote inference was used, and the local vision/OCR model is unavailable.
- README and `docs/HACKATHON_DEMO.md` now document Space/local launch, hosted Space privacy nuance, model download/checksum flow, smoke tests, demo-video steps, and field notes.
- Parakeet ASR is now staged locally and checksum-verified in `models/manifest.json`; `scripts/release_gate.py` passes required model checksum and metadata gates.
- Local text, vision, ASR, and gateway services now start with staged artifacts through `start_all_local.sh` when `PYTHON_BIN=.venv/bin/python` is used, and live loopback healthcheck passes with `--require-running`.
- A local-only Gradio launcher now binds to `127.0.0.1:7860` by default, rejects non-loopback binds, and disables Gradio analytics by default.
- llama.cpp launchers now pass `--n-gpu-layers -1` by default and expose `LLAMA_REQUIRE_CUDA=1` plus `scripts/check_llamacpp_cuda.py` for fail-fast CUDA-required startup.
- Real-service smoke tests now have a bounded configurable gateway request timeout for slow local inference, avoiding the previous 5-second document-conversion timeout.

### Still Open

- The app still falls back to visible local stubs when model services are unavailable; demo judging must start local text, vision, and ASR services before recording or live review.
- Vision/document extraction from images is wired through the local vision endpoint contract, but real MiniCPM-V execution still depends on starting a verified local vision service.
- `evals/run_all_evals.py` still defaults to deterministic sample predictions unless live local text/vision endpoints are explicitly enabled.
- Smoke tests can now require real local model-service readiness, but the exact real-service smoke path still needs to be rerun under the final demo hardware and dependency environment.
- ASR runtime readiness is still a demo risk. If Parakeet cannot produce non-stub local transcriptions reliably in the target runtime, evaluate an alternate local ASR model with a compatible license, offline artifact path, checksum manifest entry, and the same language/region safeguards.
- CUDA GPU execution is not verified in this shell. `nvidia-smi` reports GPU access blocked, no NVIDIA device nodes are visible, and the CUDA driver API reports zero devices; the current runtime can only validate CPU fallback until host GPU access is fixed.
- Current live real-service startup now fails closed at ASR readiness when the runtime cannot prove `model_ready=true`; the Parakeet artifact is checksum-ready, but non-stub transcription still requires a working local Transformers ASR runtime or an approved alternate ASR model.

## Project Structure Map

- `data/`: dataset source registry schemas, candidate intake, acceptance gates, and synthetic document metadata.
- `scripts/`: dataset import/audit tools, synthetic document generators, model download/startup, healthcheck, and local smoke tests.
- `services/`: local gateway, router, model clients, local tools, safety checks, and ASR service.
- `training/text/`: Nemotron LoRA SFT loop, adapter merge, GGUF export, quantization, and adapter eval.
- `training/vision/`: local MiniCPM-V LoRA/QLoRA scaffold, dataset validation, dry-run summaries, split checks, and backend command planning.
- `training/asr/`: Parakeet manifest prep, augmentation, NeMo training launcher, and WER eval.
- `evals/`: unified local evaluation harness, critical failure detection, regional breakdowns, and reports.
- `models/`: local deployment manifest and model notes.
- `tests/`: pytest coverage for routing, schemas, registry, synthetic data, ASR, evals, and packaging/runtime checks.

## Blocking Issues

### Data Registry And Synthetic Data

- `scripts/audit_source_registry.py`: `research_eval_only` datasets are written to `approved_datasets.jsonl`, which makes the approved file unsafe as a training allowlist.
- `scripts/approve_dataset.py`: manual approval can append `RESEARCH_EVAL_ONLY` records to the approved dataset registry.
- `data/schemas/source_registry.py`: license validation is too weak and only rejects a small unknown-license set. Ambiguous licenses can pass.
- Importers with ambiguous license examples need stricter handling: `scripts/import_awesome_public_datasets.py`, `scripts/import_google_cloud_marketplace_manifest.py`, and `scripts/import_wikimedia_manifest.py`.
- `scripts/synthetic_documents.py`: synthetic metadata is incomplete. `metadata.jsonl` is missing `license`, `modality`, `task`, and `pii_status`; ground truth uses `pii` instead of the required PII metadata naming.

### Training

- `training/text/eval_text_adapter.py`: evaluation scores ground-truth assistant messages instead of model or adapter outputs.
- `training/text/train_nemotron_lora.py`: training uses full rendered chat text without assistant-only label masking.

### Evaluation

- `evals/run_all_evals.py`, `evals/text_eval.py`, `evals/vision_eval.py`, and `evals/asr_eval.py`: unified model comparison is not real; sample evaluators return deterministic or hard-coded predictions.
- `evals/critical_failures.py`: cloud-call detection only works if a prediction voluntarily sets `attempted_cloud_call`.
- Evaluation metrics are too shallow for the project requirements. Missing metrics include WER, CER, JSON validity, tool argument accuracy, source coverage, readability, invalid refusal rate, extraction reconciliation, and grouped critical-failure breakdowns.
- Vision and document evals do not exercise generated artifacts. Current invoice Excel scoring checks totals only and does not evaluate image-to-document behavior.

### Deployment And Runtime

- Real-service smoke still needs a fresh pass in the target runtime with text, vision, ASR, and gateway services running.
- CUDA/GPU execution is not verified in this shell; GPU-required demos must run `scripts/check_llamacpp_cuda.py --require` in an environment where NVIDIA device nodes are visible.

## Non-Blocking Issues

### Data Registry And Synthetic Data

- `data/schemas/dataset_manifest.schema.json`: uses legacy singular fields while candidate records use plural fields and `candidate_tasks`.
- Candidate deduplication silently overwrites records by `(source_catalog, dataset_name)`.
- Synthetic generation appends metadata every run and can duplicate stale records.
- Font loading assumes `DejaVuSans.ttf` exists. This does not download fonts, but the fallback or error path should be clearer.

### Training

- Training dependencies use broad lower bounds only.
- QLoRA compute dtype is hardcoded to bfloat16 even when unavailable.
- `WANDB_DISABLED` is set with `setdefault`, so caller environment can still enable it.
- `training/text/merge_adapter.py` uses `trust_remote_code=True` and does not enforce `local_files_only`.
- Text README content is stale where it says training is not implemented.

### Evaluation

- `evals/run_all_evals.py` only supports `--sample`; non-sample mode raises.
- Markdown reports omit some breakdowns that exist in JSON.
- Sample coverage is very small.
- High-stakes advice detection is keyword-based and easy to bypass.

### Deployment And Runtime

- GGUF export and quantization scripts do not produce or update checksum metadata.

## Missing Tests

### Data Registry And Synthetic Data

- Approved registry must exclude `research_eval_only` datasets.
- `approve_dataset.py` must refuse `RESEARCH_EVAL_ONLY` unless a separate limited-use registry is introduced.
- Ambiguous license strings must be rejected or quarantined.
- Synthetic `metadata.jsonl` must include required metadata fields.
- Repeated synthetic generation should be idempotent or avoid duplicate metadata records.
- Regional labels and tax fields should be covered across India, Southeast Asia, North America, and Europe.

### Training

- Text eval must load model, adapter, merged HF model, or GGUF outputs instead of scoring labels.
- Assistant-only masking must ensure user and system tokens are ignored in loss labels.
- Mocked QLoRA trainer construction should validate config behavior without downloading models.
- Adapter merge, GGUF export args, and llama.cpp server validation need tests.

### Evaluation

- Evals should call a local model client or llama.cpp endpoint.
- Outbound cloud or network calls should fail during eval.
- Every critical failure type needs a direct test, including invalid JSON, unsafe advice, unsupported-language hallucination, identity guessing, and cloud calls.
- Region and language sample coverage thresholds should be tested.
- Document evals should run against synthetic PNG, ground-truth JSON, and XLSX fixtures.
- Report schema stability should be tested.

### Deployment And Runtime

- Real-service smoke should be covered by a repeatable target-hardware run record.

## Recommended Next PRs

1. Registry hardening: split approved training datasets from research/eval-only datasets, add strict license evidence fields, and normalize metadata schemas.
2. Training correctness: add assistant-only masking, real adapter eval, reproducible dependency pins, offline/local-files controls, and ASR manifest validation before training.
3. Vision training: add a proper local MiniCPM-V scaffold or explicitly defer it with acceptance criteria.
4. Eval harness v2: increase live-target coverage for HF base, PEFT adapter, merged HF, GGUF, and llama.cpp endpoint targets with local network guards.
5. ASR model contingency: if Parakeet remains unavailable or too slow, select and validate an alternate local ASR model that satisfies vendor/license/offline constraints.
6. Real-service demo validation: run text, vision, ASR, gateway, healthcheck, and `scripts/smoke_test_local.py --require-real-model-services` on the target hardware.
7. GPU validation: verify CUDA visibility and latency on the target laptop before recording or live review.

## Files That Need Changes

### Data

- `data/schemas/source_registry.py`
- `data/schemas/dataset_acceptance.py`
- `scripts/audit_source_registry.py`
- `scripts/approve_dataset.py`
- `scripts/import_awesome_public_datasets.py`
- `scripts/import_google_cloud_marketplace_manifest.py`
- `scripts/import_wikimedia_manifest.py`
- `scripts/synthetic_documents.py`

### Training

- `training/text/train_nemotron_lora.py`
- `training/text/eval_text_adapter.py`
- `training/text/merge_adapter.py`
- `training/text/configs/nemotron_router_summary_lora.yaml`
- `training/asr/train_parakeet_nemo.sh`
- `training/asr/prepare_manifest.py`
- `training/vision/README.md`

### Evaluation

- `evals/run_all_evals.py`
- `evals/text_eval.py`
- `evals/vision_eval.py`
- `evals/asr_eval.py`
- `evals/critical_failures.py`
- `evals/regional_breakdown.py`
- `evals/report.py`

### Deployment

- `scripts/start_text_llamacpp.sh`
- `scripts/start_vision_llamacpp.sh`
- `scripts/start_asr_service.sh`
- `scripts/start_all_local.sh`
- `scripts/healthcheck.py`
- `scripts/smoke_test_local.py`
- `training/text/run_llama_server.sh`
- `training/text/export_gguf.sh`
- `training/text/quantize_gguf.sh`
- `models/manifest.json`
- `configs/models.yaml`

### Tests

- `tests/test_dataset_acceptance.py`
- `tests/test_dataset_registry.py`
- `tests/test_text_dataset_format.py`
- `tests/test_asr_manifest.py`
- `tests/test_eval_sample.py`
- `tests/test_packaging_scripts.py`
- `tests/test_no_cloud_runtime.py`
