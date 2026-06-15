# Text Training

This folder contains the local-first NVIDIA text LoRA/QLoRA SFT path. The
default Modal model is `nvidia/Llama-3.1-Nemotron-Nano-8B-v1`, a dense
Transformer model that runs through Hugging Face TRL `SFTTrainer` with PEFT and
does not require compiling `causal-conv1d` or `mamba-ssm`. The tiny sample
config stays available for local validation, while Modal uses
`training/text/configs/llama_nemotron_nano_modal_lora.yaml` to read prepared
manifests from `/vol/local-lm/data/processed/training`.

Unsloth is a candidate acceleration backend, but it is not the default until a
Modal dependency smoke test and manifest compatibility test pass. The hybrid
Mamba Nemotron config remains available at
`training/text/configs/nemotron_modal_prepared_lora.yaml`, but Mamba
dependencies are optional because source builds of `causal-conv1d` can be slow or
unreliable in the CUDA image. Use higher Modal GPU classes before changing
training semantics when the issue is capacity or throughput rather than
framework compatibility.

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

Real Modal text training remains guarded by `--no-dry-run`, uses capped
`max_steps`, disables remote logging, and writes adapter outputs under
`/vol/local-lm/training/text`. The Modal workflow runs
`scripts/check_cuda_toolchain.py` with the CUDA host compiler pinned to
`/usr/bin/g++`; this prevents CUDA 13 from falling through to an unsupported or
partially installed `clang++`. To explicitly test the optional Mamba dependency
path, run `prepare_nemotron_dependencies` with
`--install-mamba-dependencies`.
