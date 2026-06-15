from __future__ import annotations

from pathlib import Path

from scripts.prepare_hf_space_bundle import prepare_space_bundle

SPACE_ROOT = Path("spaces/local_lm_accessible")


def test_standalone_hf_space_bundle_has_required_files() -> None:
    assert (SPACE_ROOT / "README.md").exists()
    assert (SPACE_ROOT / "app.py").exists()
    assert (SPACE_ROOT / "requirements.txt").exists()

    readme = (SPACE_ROOT / "README.md").read_text(encoding="utf-8")
    requirements = (SPACE_ROOT / "requirements.txt").read_text(encoding="utf-8")

    assert "sdk: gradio" in readme
    assert "app_file: app.py" in readme
    assert "gradio" in requirements


def test_standalone_hf_space_avoids_cloud_runtime_calls() -> None:
    source = (SPACE_ROOT / "app.py").read_text(encoding="utf-8").casefold()

    blocked_markers = (
        "openai",
        "requests.",
        "urllib",
        "boto3",
        "google.cloud",
        "azure.",
        "kaggle",
        "load_dataset(",
        "gr.file",
    )
    for marker in blocked_markers:
        assert marker not in source


def test_prepare_hf_space_bundle_copies_upload_files(tmp_path: Path) -> None:
    output = tmp_path / "space"

    result = prepare_space_bundle(output=output)

    assert (output / "README.md").exists()
    assert (output / "app.py").exists()
    assert (output / "requirements.txt").exists()
    assert result["output"] == str(output.resolve())
    assert result["publish_commands"][0][:4] == [
        "hf",
        "repos",
        "create",
        "build-small-hackathon/Local-lm-accessible-for-elders",
    ]
