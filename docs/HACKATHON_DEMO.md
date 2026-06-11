# Hackathon Demo Checklist

Use this checklist for the Hugging Face Build Small Hackathon Backyard AI demo.

## Preflight

- Run `python3 scripts/smoke_test_local.py --mock-model-endpoints`.
- Run `.venv/bin/python -m pytest`.
- Start local backends with `PYTHON_BIN=.venv/bin/python STARTUP_TIMEOUT_SECONDS=180 scripts/start_all_local.sh`.
- Start the Gradio UI with `PYTHON_BIN=.venv/bin/python scripts/start_gradio_app.sh`.
- For GPU-required demos, run `.venv/bin/python scripts/check_llamacpp_cuda.py --require` with the intended `LLAMA_SERVER` and `LLAMA_CUDA_LIBRARY_PATH`, then launch with `LLAMA_REQUIRE_CUDA=1 LLAMA_GPU_LAYERS=-1`.
- Confirm `/health` shows `privacy_mode=strict`, `allow_web=false`, no telemetry, and loopback endpoints.
- Confirm text, vision, and ASR artifacts are ready in the Gradio status panel.
- Run `.venv/bin/python scripts/healthcheck.py --gateway http://127.0.0.1:8000 --require-running`.
- Run `python3 scripts/release_gate.py` and confirm it passes before release/demo submission.

## Demo Flow

1. Open the Gradio Space.
2. Show the privacy/runtime status panel.
3. Open Settings / Privacy and show the Demo Readiness panel.
4. Load the sample question, ask it, and show the local assistant answer or local-model-unavailable fallback.
5. Load the sample article and summarize it offline.
6. Save an Ask or Read result as TXT/PDF/XLSX and show the file is created locally.
7. Load the synthetic invoice sample and export JSON plus XLSX.
8. Open the XLSX and show the metadata sheet with confidence, warnings, and human-review flag.
9. Load the generated sample image and show that fallback/model behavior keeps the selected file local and avoids cloud OCR or remote inference.
10. Load the generated sample speech WAV and show local ASR behavior.
11. Show the 32B model-budget status and passing release-gate metadata/checksum status.

## Voiceover Points

- No cloud inference, cloud OCR, external telemetry, or user-file uploads to third-party model APIs.
- Hosted Space compute is not the same as laptop-local privacy; laptop-local mode is the target deployment.
- Models are constrained to OpenBMB, NVIDIA, Cohere, or BFL and stay under the 32B parameter cap.
- Financial outputs are drafts and require human review.
- Indian and Southeast Asian non-English ASR is experimental until local eval proves it usable.

## Field Notes Outline

- Device tested:
- CPU/GPU/RAM:
- Operating system:
- Text model artifact and quantization:
- llama.cpp binary:
- CUDA preflight result:
- Vision model artifact and projector:
- ASR artifact status:
- Startup command:
- Gradio URL:
- Gateway URL:
- Smoke-test result:
- Noted latency:
- Accessibility observations:
- Failure cases:
- Follow-up fixes:
