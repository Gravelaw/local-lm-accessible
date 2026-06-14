# local-lm Hackathon Implementation Plan

## Summary

Build `local-lm` as a Backyard AI Gradio Space for elderly, low-vision, and accessibility-constrained users. The immediate priority is Demo First: ship a believable end-to-end Hugging Face Space experience, then harden registry, eval, deployment, and training risks behind it.

Primary constraints:

- Hugging Face Build Small Hackathon: Gradio app, hosted as a Space, total model parameters <=32B.
- BRD: local-first, no cloud inference/OCR/telemetry, `privacy_mode=strict`, `allow_web=false` by default.
- v1 model stack: NVIDIA Nemotron GGUF for text, OpenBMB MiniCPM-V GGUF for vision, NVIDIA Parakeet for ASR.
- Risk register is part of this plan: blocking risks become explicit milestone gates, not optional cleanup.

Current repo state:

- Gradio `app.py` exists and wraps gateway stubs.
- Nemotron, MiniCPM-V, and Parakeet ASR artifacts are staged and checksum-verified.
- Optional web-page fetch is still disabled by default and now rejects localhost, private IP literals, unresolvable hostnames, and hostnames resolving to non-public addresses before bounded fetch.
- Local text/vision clients use loopback-only llama.cpp endpoints and can fall back from `/completion` to the OpenAI-compatible `/v1/chat/completions` API when the native path is not available.
- Gradio demo readiness now requires required model artifacts to be present and checksum metadata to be configured; empty directory artifacts are treated as missing.
- Gradio base accessibility styles are embedded in the Space component tree, so imported Space objects and launched apps share the same large-control styling.
- Ask, Read, Image, and Speech result panes now support local TXT/XLSX/PDF saves in addition to the document conversion exports.
- PDF saves are now real bounded local PDFs generated with `reportlab`, not text placeholder files with a `.pdf` suffix.
- Hugging Face Space metadata pins `sdk=gradio`, `sdk_version=6.16.0`, `app_file=app.py`, and Python 3.11; `requirements.txt` and `pyproject.toml` use the same Gradio version and include `uvicorn` while keeping training packages optional.
- Editable dev installs now work through explicit setuptools package discovery, and `ruff check .` passes locally.
- The release gate now passes with required text, vision, and ASR checksums recorded.
- The Gradio Settings / Privacy readiness panel calls `run_release_gate(verify_checksums=False)` for a metadata-only release-gate summary without hashing multi-GB model artifacts on page load; the `scripts/release_gate.py` CLI remains the full checksum-verifying gate.
- The first Gradio screen now includes the hosted-Space privacy disclosure: Space compute is used when hosted, no external model APIs/cloud OCR/telemetry are called, and laptop-local mode uses the same architecture.
- The Settings / Privacy tab now includes a Gradio-level "Run Local Demo Samples" check that exercises Ask, Read, Document, Image, and Speech sample handlers locally, verifies structured outputs, visible warnings, and document downloads.
- Milestone 2 ASR runtime metadata is now wired through the gateway and Gradio Speech tab: language, region, country, and explicit experimental opt-in are preserved, and Indian/SEA non-English or unsupported Parakeet languages fail closed with visible local warnings unless opted in.
- Milestone 3 image accessibility responses now guard against identity-like claims from the local vision model: guessed names are removed, uncertainty is recorded in the Pydantic schema, and a human-review warning is returned.
- Milestone 4 synthetic document generation now emits an idempotent `synthetic_dataset_candidates.jsonl` registry alongside `metadata.jsonl`; synthetic invoice and bank-statement candidates validate through the dataset registry schema and default acceptance gate.
- Milestone 5 sample document evals now use real generated synthetic PNG, ground-truth JSON, and expected XLSX fixtures for invoice, receipt, bank-statement, and handwritten-note tasks; expected totals and balances are derived from the generated ground truth.
- Milestone 6 text training now enforces train/eval split separation with content fingerprints, uses a separate tiny eval JSONL sample, and overrides W&B disabled state for local-only logging.
- Milestone 7 vision training scaffold now uses separate train/eval JSONL samples, rejects split overlap, validates OpenBMB/local-only LoRA/QLoRA config defaults, and emits non-executing LLaMA-Factory or SWIFT command plans for local MiniCPM-V adapter training.
- Milestone 8 ASR eval/runtime completion now has focused report-emission tests, explicit unsupported-language failure reporting, verified local manifest/runtime guards, and ASR-local staging/eval documentation.
- Milestone 9 deployment/package readiness now keeps ASR parameter metadata consistent at 0.6B and the release gate fails manifest/config parameter mismatches.
- ASR staging is complete: Parakeet was downloaded to `models/asr/parakeet-tdt-0.6b-v3`, its deterministic directory checksum was written to `models/manifest.json`, and `scripts/release_gate.py` passes.
- Local service startup is verified with staged artifacts: text (`8081`), vision (`8082`), ASR (`8090`), and gateway (`8000`) pass loopback health checks when started with `PYTHON_BIN=.venv/bin/python STARTUP_TIMEOUT_SECONDS=180 scripts/start_all_local.sh`.
- A loopback-only Gradio launcher exists at `scripts/start_gradio_app.sh` and disables Gradio analytics by default for local demo runs.
- llama.cpp text/vision launchers now default to full GPU layer offload (`--n-gpu-layers -1`) and support `LLAMA_REQUIRE_CUDA=1` through `scripts/check_llamacpp_cuda.py` so CUDA-required demos fail fast instead of silently falling back to CPU.
- Current shell evidence shows CUDA is not usable here: `nvidia-smi` reports GPU access blocked, no `/dev/nvidia*` devices are visible, and CUDA driver API device count is zero. Host GPU visibility/driver setup remains required before GPU demo validation.
- Real-service smoke uses a configurable bounded gateway request timeout for slow local inference. The exact real-service path still needs to be rerun in the current runtime after text, vision, ASR, and gateway services are started.
- ASR has a contingency path: if Parakeet cannot provide reliable non-stub local transcriptions in the target runtime, evaluate an alternate local ASR model that satisfies license, vendor/source, offline, checksum, and language/region safety constraints.
- Dataset registry, synthetic data, training, eval, gateway, and packaging scaffolds exist.
- Major gaps remain: eval realism with live targets, real-model smoke coverage under the exact demo environment, GPU visibility/latency validation on target laptop hardware, and final quality review.

Source requirements:

- BRD: `local-lm_BRD.md`
- Risk register: `docs/RISK_REGISTER.md`
- Hackathon page: `https://huggingface.co/build-small-hackathon`

## Implementation Milestones

### 1. Demo-Ready Gradio Space

Goal: make the app usable for a judge or real user before deep training work.

Implement:

- Upgrade `app.py` from stub wrapper to task-first accessible UI:
  - Read / Summarize
  - Convert Document
  - Describe Image
  - Translate Image Text
  - Speech to Text
  - Settings / Privacy
- Add high-contrast mode, large text, large buttons, short-answer-first output, and an advanced model/runtime panel hidden by default.
- Add explicit Space disclosure:
  - This demo runs inside Hugging Face Space compute.
  - No external model APIs or cloud OCR are called.
  - Laptop-local mode uses the same architecture.
- Add demo samples:
  - synthetic invoice text/document sample
  - synthetic bank-statement text/document sample
  - simple article text
  - generated local sample media where binary assets are not tracked
- Add Gradio manual task override matching router tasks.
- Preserve default web behavior:
  - URL fetch remains blocked unless `allow_web=true`.
  - Wikipedia/local text summarization works without network.

Acceptance:

- Gradio app starts locally with `python app.py`.
- App shows privacy/locality status on first screen.
- Each task tab returns structured output and visible warnings.
- Sample files can drive a complete demo without external APIs.

### 2. Local Model Runtime Integration

Goal: replace UI/gateway stubs with local model calls where possible.

Implement:

- Text model client:
  - call local llama.cpp-compatible endpoint at `127.0.0.1:8081`
  - support both llama.cpp `/completion` and `/v1/chat/completions` response formats
  - support summarization, JSON repair, simple explanation, and tool-plan prompts
  - timeout with clear user-facing fallback if service is not running
- Vision model client:
  - call local MiniCPM-V runtime at `127.0.0.1:8082`
  - support image description, OCR-style visible text extraction, document extraction prompts, and image translation prompts
  - include `mmproj` in startup path
- ASR client:
  - integrate local Parakeet service at `127.0.0.1:8090`
  - keep Indian/SEA non-English languages marked experimental
- Add model availability status in Gradio and `/health`.

Risk gates:

- Startup scripts must verify the same artifact path they launch.
- Non-loopback hosts must be rejected.
- Healthcheck must not accept arbitrary remote gateway URLs by default.

Acceptance:

- Text and vision tasks call local endpoints when running.
- If a model service is down, the app reports "local model unavailable" rather than silently using a stub.
- No runtime path can call AWS, Azure, Google Cloud, OpenAI APIs, cloud OCR, or remote telemetry.

### 3. Pydantic Schemas And Document Pipeline

Goal: make document outputs structured, exportable, and review-safe.

Implement:

- Define Pydantic schemas for:
  - invoice/bill/receipt extraction
  - bank-statement transactions
  - handwritten note transcription
  - image accessibility description
  - image translation
  - task response envelope with confidence, warnings, and `human_review_required`
- Wire document conversion flow:
  - image/PDF preprocessing
  - local vision extraction
  - schema validation
  - JSON/TXT/XLSX/PDF export
- Add numeric reconciliation:
  - invoice subtotal + tax = total
  - bank-statement running balance checks where fields exist
- Always mark bank statements as human-review-required.
- Add financial/medical/legal warning injection for sensitive outputs.

Risk gates:

- No silent financial total hallucination.
- Low-confidence document extraction must require human review.
- All document extraction outputs must use Pydantic schemas.

Acceptance:

- Invalid model JSON is repaired or rejected with a visible warning.
- XLSX exports open and include warnings/confidence fields.
- Bank statements always show a human-review warning.
- Image descriptions never guess identity and include uncertainty when ambiguous.

### 4. Dataset Registry And Synthetic Data Hardening

Goal: make training/eval data safe enough to trust.

Implement:

- Split dataset outputs into approved training, research/eval-only, and rejected datasets.
- Reject unknown, missing, ambiguous, or unverifiable licenses.
- Treat non-commercial licenses as research/eval-only unless explicitly configured.
- Block high-PII datasets unless synthetic, redacted, or explicit opt-in.
- Normalize required metadata across registry and synthetic generation.
- Fix synthetic metadata to include `license`, `modality`, `task`, and `pii_status`.
- Make repeated synthetic generation idempotent or clearly versioned.

Risk gates:

- `research_eval_only` cannot enter training allowlist.
- Ambiguous licenses cannot pass acceptance.
- Synthetic records must satisfy project metadata requirements.

Acceptance:

- 20+ sample candidate records audit cleanly.
- Audit report counts source, task, region, language, modality, license, and PII risk.
- Tests prove research/eval-only records are not approved for training.

### 5. Evaluation Harness V2

Goal: make evals reflect real app/model behavior instead of hard-coded predictions.

Implement:

- Replace deterministic fake predictions with target adapters for base HF, adapter, merged HF, quantized GGUF, and local llama.cpp endpoint targets.
- Add local network guard: outbound cloud calls fail tests/evals, loopback endpoints allowed.
- Expand metrics: router accuracy, JSON validity, tool-call argument accuracy, source coverage, readability, WER/CER, invoice total reconciliation, bank balance reconciliation, invalid refusal rate.
- Count critical failures: hallucinated financial totals, missing human-review flag, unsafe advice, identity guessing, invalid JSON, unsupported-language hallucination, cloud call attempted.
- Group reports by region, country, language, document type, and task.

Risk gates:

- Eval harness must call local model clients/endpoints.
- Cloud-call detection cannot depend on voluntary prediction fields.
- Document evals must use real synthetic PNG/JSON/XLSX fixtures.

Acceptance:

- `evals/run_all_evals.py --sample` completes locally.
- Reports emit summary JSON/Markdown, failure JSONL, and examples.
- Report compares base vs adapter vs merged vs GGUF vs llama.cpp endpoint.

### 6. Text Training Correctness

Goal: make Nemotron LoRA/QLoRA training reproducible and aligned with actual tasks.

Implement:

- Fix assistant-only label masking so user/system tokens are ignored in loss.
- Enforce train/eval split separation.
- Keep W&B disabled by default and logs local.
- Add offline/local-files controls for merge/eval paths.
- Make eval generate predictions from the model/adapter instead of scoring labels.
- Save best adapter.
- Keep GGUF export and quantization scripts aligned with llama.cpp.

Risk gates:

- `eval_text_adapter.py` must evaluate generated outputs.
- Training must not optimize on full rendered chat labels.
- Merge/export must avoid unsafe remote assumptions.

Acceptance:

- Dry-run works on 32 examples.
- Text eval writes JSON and Markdown reports.
- Tests cover label masking, dataset format, mocked trainer config, split separation, and eval output generation.

### 7. Vision Training Scaffold

Goal: add a real MiniCPM-V training path without overbuilding.

Implement:

- Add MiniCPM-V LoRA/QLoRA training scaffold using LLaMA-Factory preferred, SWIFT fallback.
- Define multimodal JSONL sample format with image path, prompt/task, expected structured output, region/country/language, license, and PII metadata.
- Add dry-run mode that validates dataset and emits config summary without downloading models.
- Explicitly document full training as optional/hardware-dependent for hackathon.

Risk gates:

- Vision training cannot remain only a placeholder.
- Dataset/config/dry-run tests must exist.

Acceptance:

- Vision dry-run validates a tiny synthetic sample.
- README documents supported tasks and local-only training assumptions.
- Tests cover schema validation and dry-run behavior.

### 8. ASR Eval And Runtime Completion

Goal: finish eval-first Parakeet support.

Contingency:

- If Parakeet is not viable for the target demo/runtime, select an alternate local ASR model.
- The replacement must remain local-only, have reviewed license/commercial metadata, be checksum-pinned in `models/manifest.json`, and preserve unsupported-language and experimental-region safeguards.

Implement:

- Download/stage Parakeet artifact or document exact operator download command.
- Fill manifest checksum once artifact is staged.
- Validate ASR manifests before training/eval.
- Add country/modality/task/PII metadata to ASR schema.
- Add unsupported-language detection and warning behavior.
- Keep Indian/SEA non-English ASR experimental until eval proves usable.

Risk gates:

- ASR manifest schema must include required metadata.
- NeMo training entrypoint must be tested or clearly marked experimental.
- Missing audio, unknown license, and unsupported language cases must fail safely.

Acceptance:

- Tiny local ASR manifest eval runs.
- WER/CER metrics emit JSON/Markdown.
- Unsupported-language examples are flagged, not hallucinated as supported.

### 9. Deployment, Packaging, And Space Readiness

Goal: make the repo shippable as a Space and runnable on a laptop.

Implement:

- Finalize `requirements.txt` for Space runtime only.
- Keep training extras out of Space default install unless needed.
- Add documented command for local Gradio launch.
- Harden startup scripts:
  - derive default paths from manifest
  - reject unchecked path overrides
  - reject non-loopback host binds
  - verify checksum before launch
- Add release manifest gate:
  - non-empty checksums for required models
  - no placeholders for required v1 models
  - allowed vendors only
  - total params <=32B
  - manifest/config consistency
- Update README with Space usage, laptop-local mode, model download/checksum flow, privacy limitations of hosted Space compute, and demo script.
- Add demo-video checklist and field-notes outline.

Risk gates:

- Placeholder manifest entries cannot ship as required models.
- Smoke test must be end-to-end or explicitly mock local endpoints.
- Gateway/smoke healthcheck cannot accept remote URLs by default.
- Optional web fetch must fail closed for localhost, private IPs, and private DNS resolutions.

Acceptance:

- Space entrypoint is `app.py`.
- Required models verify by checksum.
- Local smoke test runs without cloud calls.
- README makes hosted-Space vs laptop-local privacy nuance explicit.

## Public Interfaces And Behavior Changes

- Gradio UI becomes the primary app interface; FastAPI gateway remains internal/local orchestration.
- Task responses use a shared envelope with task, status, result, confidence when available, warnings, `human_review_required`, and local/privacy metadata.
- Document extraction outputs use Pydantic schemas only.
- Model clients use loopback endpoints only:
  - text `127.0.0.1:8081`
  - vision `127.0.0.1:8082`
  - optional omni `127.0.0.1:8083`
  - ASR `127.0.0.1:8090`
- Web fetch remains optional and disabled by default; no task may enable it implicitly, and enabled fetches are bounded to public HTTP(S) text/HTML endpoints.

## Test Plan

Run targeted tests after each milestone, then a full local test pass before packaging.

Required test groups:

- Gradio: strict privacy defaults, task handlers, local exports, manual route override, no web by default.
- Router/gateway: URL/Wikipedia routing, document/image/audio routing, loopback-only endpoint validation, no raw user-data logging.
- Schemas/document extraction: valid/invalid JSON, invoice total reconciliation, bank balance reconciliation, human-review warnings.
- Dataset registry: unknown license rejection, non-commercial research/eval-only handling, high-PII blocking, synthetic metadata completeness.
- Eval harness: local endpoint invocation, cloud-call guard, critical failure counting, regional/language grouping.
- Packaging: checksum verification, startup rejects unverified path override, manifest/config consistency, parameter budget under 32B.
- Smoke: sample invoice export must create JSON, XLSX, TXT, and a real local PDF without cloud calls.

Minimum final verification commands:

```bash
python3 scripts/verify_model_checksums.py --model text
python3 scripts/verify_model_checksums.py --model vision
python3 scripts/healthcheck.py
python3 scripts/smoke_test_local.py
.venv/bin/python -m pytest
```

## Assumptions And Defaults

- Priority is Demo First, then risk hardening, then training depth.
- Hosted Hugging Face Space is acceptable for hackathon submission if it uses no external model APIs, cloud OCR, or telemetry; laptop-local mode must be documented.
- Required v1 deployment models are Nemotron GGUF, MiniCPM-V GGUF, and Parakeet ASR. Optional omni remains non-blocking.
- Existing `docs/RISK_REGISTER.md` remains the issue register; this plan promotes blocking risks into milestone gates rather than duplicating every detail.
- Fine-tuned adapters are a target deliverable, but the hackathon MVP may run base/quantized local models if adapter training cannot complete safely in time.
- Real financial, medical, legal, or identity documents are not used for training unless synthetic, redacted, or explicit opt-in.
