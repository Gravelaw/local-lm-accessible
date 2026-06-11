# Models

Model artifacts are not committed to this repository. The manifest at `models/manifest.json` is the source of truth for local paths, ports, runtimes, licenses, checksums, and supported tasks.

## Startup Rule

No model service should start unless its local artifact exists and its SHA-256 matches the manifest. After placing or changing an artifact locally, update `sha256`, then run:

```bash
python3 scripts/verify_model_checksums.py --model text
```

Empty `sha256` values are treated as not ready.

Currently staged locally:

- Text: `models/text/NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf`
- Vision: `models/vision/MiniCPM-V-4_6-Q4_K_M.gguf`
- Vision projector: `models/vision/mmproj-model-f16.gguf`

ASR and optional omni artifacts are still pending unless their manifest entries have non-empty checksums.

## ASR Staging

Parakeet is a required v1 model, but the artifact is not staged until an operator
downloads it locally and records the checksum. Use the guarded repository script.
It prints the exact `hf download` command by default and only downloads when the
large-download flag is explicit:

```bash
python3 scripts/download_models.py --model asr --print-plan
python3 scripts/download_models.py --model asr --download --allow-large-download
```

Equivalent direct `hf` CLI command:

```bash
hf download nvidia/parakeet-tdt-0.6b-v3 --local-dir models/asr/parakeet-tdt-0.6b-v3
```

After download, compute and write the local checksum into the manifest:

```bash
python3 scripts/verify_model_checksums.py --model asr --write-manifest-checksum
```

Then verify the checksum and release gate:

```bash
python3 scripts/verify_model_checksums.py --model asr
python3 scripts/release_gate.py
```

`scripts/start_asr_service.sh` refuses to start until the checksum passes.

## Ports

- Text `llama.cpp`: `127.0.0.1:8081`
- Vision `llama.cpp` or local Transformers server: `127.0.0.1:8082`
- Optional omni server: `127.0.0.1:8083`
- ASR service: `127.0.0.1:8090`
- Gateway default: `127.0.0.1:8000`

## Platform Notes

- CPU: use GGUF quantizations such as `Q4_K_M` for lower memory use.
- CUDA: build `llama.cpp` with CUDA and tune layer offload outside this manifest.
- Metal: build `llama.cpp` with Metal on macOS.
- Windows: use PowerShell or WSL2; keep all service hosts bound to `127.0.0.1`.
- Linux: the shell scripts assume Bash and local `llama-server`/`llama-quantize` on `PATH`.
- Android: use a mobile-compatible llama.cpp build and keep model paths local to the device.

## One-Command Smoke Test

```bash
python3 scripts/smoke_test_local.py
```

The default smoke test exercises local gateway behavior in-process and does not
require model weights. To verify the gateway-to-llama.cpp HTTP contract without
downloading or starting real models, run:

```bash
python3 scripts/smoke_test_local.py --mock-model-endpoints
```

That mode starts loopback-only mock `/completion` servers on the text and vision
ports and confirms the gateway uses them. Model service startup still requires
checksums to match.

After verified model services are actually running, require real service
readiness before accepting the smoke test:

```bash
python3 scripts/healthcheck.py --require-running
python3 scripts/smoke_test_local.py --require-real-model-services
```

This mode cannot be combined with mock endpoints. It checks loopback `/health`
readiness for required text, vision, and ASR services before running the smoke
tasks, and it sends a tiny local WAV through `/tasks/speech_to_text`. The real
mode only passes when ASR returns a non-empty ready-model transcript.
