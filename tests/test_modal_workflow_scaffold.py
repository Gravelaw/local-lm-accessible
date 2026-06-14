from __future__ import annotations

import tomllib
from pathlib import Path


def test_modal_dependency_is_optional() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    runtime_dependencies = " ".join(pyproject["project"]["dependencies"]).casefold()
    modal_extra = pyproject["project"]["optional-dependencies"]["modal"]

    assert "modal" not in runtime_dependencies
    assert any(str(dependency).startswith("modal") for dependency in modal_extra)


def test_modal_workflow_uses_image_volumes_and_secrets() -> None:
    source = Path("modal_workflows/local_lm_pipeline.py").read_text(encoding="utf-8")

    assert "modal.Image.debian_slim" in source
    assert "modal.Volume.from_name" in source
    assert "modal.Secret.from_name" in source
    assert "scripts/download_prepare_small_datasets.py" in source
    assert "--max-dataset-size-gb" in source
    assert "training/text/train_nemotron_lora.py" in source
    assert "training/vision/train_minicpm_v_lora.py" in source


def test_modalignore_excludes_local_artifacts() -> None:
    ignore_text = Path(".modalignore").read_text(encoding="utf-8")

    assert ".venv/" in ignore_text
    assert "data/raw/" in ignore_text
    assert "models/text/" in ignore_text
    assert "training/**/outputs/" in ignore_text
