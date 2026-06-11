from __future__ import annotations

import tomllib
from pathlib import Path

import yaml


def _readme_front_matter() -> dict[str, object]:
    readme = Path("README.md").read_text(encoding="utf-8")
    assert readme.startswith("---\n")
    _, front_matter, _body = readme.split("---", maxsplit=2)
    payload = yaml.safe_load(front_matter)
    assert isinstance(payload, dict)
    return payload


def test_readme_documents_space_launch_privacy_and_model_flow() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "python app.py" in readme
    assert "scripts/smoke_test_local.py --mock-model-endpoints" in readme
    assert "Ask, Read, and image tasks" in readme
    assert "Hosted Space Privacy Note" in readme
    assert "does not call external" in readme
    assert "python3 scripts/download_models.py --model asr --print-plan" in readme
    assert "python3 scripts/release_gate.py" in readme
    assert "docs/HACKATHON_DEMO.md" in readme
    assert "Ask: simple local assistant" in readme
    assert "local sample question" in readme
    assert "optional bounded web fetch remains off by default" in readme
    assert "Settings / Privacy" in readme
    assert "advanced panel" in readme
    assert 'pip install -e ".[dev,training]"' in readme


def test_readme_has_hugging_face_space_metadata() -> None:
    metadata = _readme_front_matter()

    assert metadata["title"] == "local-lm"
    assert metadata["sdk"] == "gradio"
    assert metadata["sdk_version"] == "6.16.0"
    assert metadata["app_file"] == "app.py"
    assert metadata["python_version"] == "3.11"


def test_requirements_cover_space_runtime_without_training_stack() -> None:
    requirements = Path("requirements.txt").read_text(encoding="utf-8").casefold()

    assert "gradio==6.16.0" in requirements
    assert "fastapi" in requirements
    assert "uvicorn" in requirements
    assert "reportlab" in requirements
    for training_package in (
        "transformers",
        "datasets",
        "accelerate",
        "peft",
        "trl",
        "bitsandbytes",
    ):
        assert training_package not in requirements


def test_gradio_version_is_consistent_across_space_metadata_and_packaging() -> None:
    metadata = _readme_front_matter()
    requirements = Path("requirements.txt").read_text(encoding="utf-8")
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    expected = f"gradio=={metadata['sdk_version']}"

    assert expected in requirements.splitlines()
    assert expected in pyproject["project"]["dependencies"]


def test_hackathon_demo_checklist_covers_required_demo_points() -> None:
    checklist = Path("docs/HACKATHON_DEMO.md").read_text(encoding="utf-8")

    assert "Demo Flow" in checklist
    assert "privacy/runtime status panel" in checklist
    assert "Load the sample question" in checklist
    assert "metadata sheet" in checklist
    assert "32B model-budget status" in checklist
    assert "No cloud inference" in checklist
    assert "Field Notes Outline" in checklist
