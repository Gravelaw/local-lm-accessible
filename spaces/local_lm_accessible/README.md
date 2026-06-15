---
title: Local LM Accessible For Elders
sdk: gradio
sdk_version: 6.16.0
app_file: app.py
python_version: "3.11"
tags:
  - build-small-hackathon
  - backyard-ai
  - accessibility
  - local-first
  - best-minicpm-build
  - best-use-of-codex
  - nemotron
  - modal
  - off-brand
  - best-demo
  - best-agent
---

# Local LM Accessible For Elders

Hosted Gradio demo for `local-lm`, a local-first assistant for elderly and
low-vision users.

This Space is a public demonstration surface. It does not call external model
APIs, hosted OCR, AWS, Google Cloud, Azure, Kaggle, OpenAI, or remote telemetry.
Do not enter private documents, medical records, legal records, bank statements,
or identity data into the hosted Space.

## Competition Evidence

- Track: Backyard AI
- Official Space: `build-small-hackathon/Local-lm-accessible-for-elders`
- GitHub: https://github.com/Gravelaw/local-lm-accessible
- Model artifacts: pending org model repo creation for GGUF and LoRA uploads
- Demo video: pending
- Social post: pending

## What This Demo Shows

- Accessible Gradio UI for summarization, JSON repair, document-field drafting,
  and tool routing.
- Privacy boundary that avoids external model APIs and hosted OCR.
- Local-first deployment path using Modal-trained Nemotron LoRA, GGUF export,
  Q4/Q5 quantization, and llama.cpp smoke tests.
- MiniCPM-V and Parakeet paths documented for local vision and ASR workflows.

## Runtime Note

The Space uses Hugging Face ZeroGPU with a `/data` mount. GPU-backed callbacks
are decorated with `@spaces.GPU`, and the app keeps deterministic fallbacks so
the public demo still starts if optional model artifacts are unavailable.

The production target remains laptop-local inference with the published GGUF
model and llama.cpp.
