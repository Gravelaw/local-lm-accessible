# Text Training

This folder contains the local-first NVIDIA Nemotron LoRA/QLoRA text SFT path.
The default backend is Hugging Face TRL `SFTTrainer` with PEFT because that path
has local tests and explicit assistant-only label masking. The tiny sample
config stays available for local validation, while Modal uses
`training/text/configs/nemotron_modal_prepared_lora.yaml` to read prepared
manifests from `/vol/local-lm/data/processed/training`.

Unsloth is a candidate acceleration backend for Nemotron 3, but it is not the
default until a Modal dependency smoke test and manifest compatibility test pass.
Use higher Modal GPU classes before changing training semantics when the issue is
capacity or throughput rather than framework compatibility.

Run the local sample dry run:

```bash
python training/text/train_nemotron_lora.py \
  --config training/text/configs/nemotron_router_summary_lora.yaml \
  --dry-run \
  --limit 32
```

Run the Modal prepared-manifest dry run:

```bash
.venv/bin/modal run modal_workflows/local_lm_pipeline.py --action finetune_text --dry-run
```

Real Modal text training remains guarded by `--no-dry-run`, runs the explicit
Mamba/Nemotron dependency install step, uses capped `max_steps`, disables remote
logging, and writes adapter outputs under `/vol/local-lm/training/text`.
Before installing Mamba, the Modal workflow runs `scripts/check_cuda_toolchain.py`
with the CUDA host compiler pinned to `/usr/bin/g++`; this prevents CUDA 13 from
falling through to an unsupported or partially installed `clang++`.
