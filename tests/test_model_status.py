from __future__ import annotations

import json
from pathlib import Path

from services.gateway.model_status import model_readiness_by_key


def test_model_readiness_reports_present_artifacts_and_missing_checksums(tmp_path: Path) -> None:
    model_file = tmp_path / "text.gguf"
    model_file.write_bytes(b"model")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "key": "text",
                        "model_id": "nvidia/text",
                        "runtime": "llama.cpp",
                        "local_path": str(model_file),
                        "sha256": "abc",
                    },
                    {
                        "key": "asr",
                        "model_id": "nvidia/asr",
                        "runtime": "parakeet_service",
                        "local_path": str(tmp_path / "missing-asr"),
                        "sha256": "",
                    },
                    {
                        "key": "omni",
                        "model_id": "nvidia/omni",
                        "runtime": "optional",
                        "local_path": str(tmp_path / "missing-omni"),
                        "sha256": "",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    statuses = model_readiness_by_key(manifest_path)

    assert statuses["text"]["ready"] is True
    assert statuses["text"]["artifact_present"] is True
    assert statuses["text"]["checksum_configured"] is True
    assert statuses["asr"]["ready"] is False
    assert statuses["asr"]["required"] is True
    assert statuses["asr"]["warnings"] == [
        "required artifact is not present locally",
        "required checksum is not configured",
    ]
    assert statuses["omni"]["required"] is False


def test_model_readiness_rejects_empty_directory_artifact(tmp_path: Path) -> None:
    empty_asr_dir = tmp_path / "empty-asr"
    empty_asr_dir.mkdir()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "key": "asr",
                        "model_id": "nvidia/asr",
                        "runtime": "parakeet_service",
                        "local_path": str(empty_asr_dir),
                        "sha256": "abc",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    statuses = model_readiness_by_key(manifest_path)

    assert statuses["asr"]["ready"] is False
    assert statuses["asr"]["artifact_present"] is False
    assert statuses["asr"]["checksum_configured"] is True
    assert statuses["asr"]["warnings"] == ["required artifact is not present locally"]
