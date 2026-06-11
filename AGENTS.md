# AGENTS.md

Repository guidance for `local-lm`.

`local-lm` is a local-first small-model assistant for elderly users and low-vision users. Optimize for privacy, accessibility, reliability, and verifiable offline behavior.

## Product Constraints

- The shipped product must not use cloud inference.
- User files must never be uploaded remotely.
- Runtime behavior must not depend on AWS, Azure, Google Cloud, hosted OCR, hosted LLM APIs, remote telemetry, or any external API.
- Remote telemetry is disabled by default.
- Prefer `llama.cpp` and GGUF for local model serving.
- Models must come only from OpenBMB, NVIDIA, Cohere, or BFL.
- The v1 deployed model bundle must stay below 32B total parameters.
- All evals must run locally.
- No cloud OCR APIs are allowed.

## User Safety And Accessibility

- The primary users include elderly users and low-vision users. Favor simple flows, large readable text, clear affordances, predictable navigation, and robust error messages.
- Financial, medical, and legal outputs must include uncertainty language and human-review warnings.
- Real financial, medical, legal, or identity documents are blocked unless they are synthetic, redacted, or provided with explicit user opt-in.
- Treat privacy and local-only operation as core requirements, not implementation details.

## Model And Training Rules

- Training may use PyTorch, Transformers, TRL, PEFT, LLaMA-Factory, SWIFT, or NeMo.
- Training data must focus on India, Southeast Asia, North America, and Europe.
- Every dataset record or dataset manifest must include:
  - license
  - region
  - country
  - language
  - modality
  - task
  - PII metadata
- Unknown, missing, ambiguous, or unverifiable dataset licenses mean reject the dataset.
- Dataset validation must fail closed. Do not silently accept incomplete metadata.

## Document Extraction

- All document extraction outputs must use Pydantic schemas.
- Validate extracted fields before downstream use or export.
- Excel export must use `openpyxl`.
- Offline search must use SQLite FTS5.

## Engineering Standards

- Use Python 3.11 or newer.
- Use FastAPI for the local gateway.
- Use Pydantic for schemas.
- Use `pytest` for tests.
- Use `ruff` for linting.
- Keep changes small, explicit, and easy to verify.
- Prefer simple control flow, bounded operations, timeouts, and explicit error handling.
- Check return values and failure paths.
- Avoid unnecessary abstraction, metaprogramming, or speculative refactors.

## Done Criteria

A task is not done until the relevant local checks pass or the gap is explicitly reported:

- Tests pass.
- Data validation passes.
- Training dry-run works on a tiny sample.
- Eval loop emits JSON and Markdown reports.
- GGUF export scripts are present.
- Local smoke test runs without cloud calls.

When reporting completed work, state what changed, what was verified, what was not verified, and any residual risk.
