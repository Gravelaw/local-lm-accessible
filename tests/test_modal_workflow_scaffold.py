from __future__ import annotations

import tomllib
from pathlib import Path

import yaml


def test_modal_dependency_is_optional() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    runtime_dependencies = " ".join(pyproject["project"]["dependencies"]).casefold()
    modal_extra = pyproject["project"]["optional-dependencies"]["modal"]

    assert "modal" not in runtime_dependencies
    assert any(str(dependency).startswith("modal") for dependency in modal_extra)


def test_modal_workflow_uses_image_volumes_and_secrets() -> None:
    source = Path("modal_workflows/local_lm_pipeline.py").read_text(encoding="utf-8")

    assert "modal.Image.debian_slim" in source
    assert "nvidia/cuda:13.0.2-devel-ubuntu24.04" in source
    assert "modal.Volume.from_name" in source
    assert "modal.Secret.from_name" in source
    assert "scripts/download_prepare_small_datasets.py" in source
    assert "--max-dataset-size-gb" in source
    assert "training/text/train_nemotron_lora.py" in source
    assert "training/vision/train_minicpm_v_lora.py" in source
    assert "check_training_toolchain" in source
    assert "scripts/preflight_finetuning_manifests.py" in source
    assert "nemotron_modal_prepared_lora.yaml" in source
    assert "minicpm_v_modal_document_lora.yaml" in source
    assert "mamba-ssm" in source
    assert "causal-conv1d" in source
    assert '"CUDAHOSTCXX": host_compiler' in source
    assert 'host_compiler = "/usr/bin/g++"' in source
    assert "scripts/check_cuda_toolchain.py" in source
    assert "NVCC_PREPEND_FLAGS" in source
    assert "CMAKE_CUDA_HOST_COMPILER" in source
    assert "torch==2.12.0" in source
    assert "LD_LIBRARY_PATH" in source


def test_modalignore_excludes_local_artifacts() -> None:
    ignore_text = Path(".modalignore").read_text(encoding="utf-8")

    assert ".venv/" in ignore_text
    assert "data/raw/" in ignore_text
    assert "models/text/" in ignore_text
    assert "training/**/outputs/" in ignore_text


def test_modal_pipeline_tracks_backend_and_gpu_escalation() -> None:
    config = yaml.safe_load(Path("configs/modal_pipeline.yaml").read_text(encoding="utf-8"))
    finetuning = config["finetuning"]

    assert finetuning["default_text_backend"] == "hf_trl_peft"
    assert finetuning["optional_text_backends"]["unsloth"]["status"] == "candidate"
    assert finetuning["gpu_upgrade_order"] == [
        "A10G",
        "A100-40GB",
        "A100-80GB",
        "H100",
        "H200",
    ]
    assert finetuning["retrieval_augmentation"]["vector_index_status"] == "backlog"
