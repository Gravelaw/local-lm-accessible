from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import yaml

from scripts.download_models import build_download_plan, build_hf_download_command, download_model
from scripts.model_launch_info import launch_info
from scripts.release_gate import run_release_gate
from scripts.smoke_test_local import run_smoke
from scripts.verify_model_checksums import (
    compute_model_artifact_checksums,
    load_manifest,
    sha256_path,
    update_manifest_checksum,
    verify_model,
)

TRAINING_PACKAGES = {"transformers", "datasets", "accelerate", "peft", "trl", "bitsandbytes"}


def test_manifest_only_validation_and_download_plan() -> None:
    manifest = load_manifest()
    plan = build_download_plan(manifest)

    assert len(plan) == len(manifest["models"])
    assert all(item["manual"] is True for item in plan)
    assert {item["port"] for item in plan} == {8081, 8082, 8083, 8090}
    assert all(item["command"][0:2] == ["hf", "download"] for item in plan)


def test_pyproject_keeps_training_dependencies_optional() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    dependencies = {
        str(dependency).split(">=", maxsplit=1)[0].split("==", maxsplit=1)[0]
        for dependency in pyproject["project"]["dependencies"]
    }
    training_extra = {
        str(dependency).split(">=", maxsplit=1)[0].split("==", maxsplit=1)[0]
        for dependency in pyproject["project"]["optional-dependencies"]["training"]
    }

    assert dependencies.isdisjoint(TRAINING_PACKAGES)
    assert training_extra >= TRAINING_PACKAGES


def test_pyproject_uses_explicit_package_discovery() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    discovery = pyproject["tool"]["setuptools"]["packages"]["find"]

    assert discovery["where"] == ["."]
    assert discovery["namespaces"] is True
    assert set(discovery["include"]) == {"data*", "evals*", "scripts*", "services*", "training*"}


def test_download_plan_can_filter_one_model() -> None:
    manifest = load_manifest()
    plan = build_download_plan(manifest, model_key="asr")

    assert len(plan) == 1
    assert plan[0]["key"] == "asr"
    assert plan[0]["command"] == [
        "hf",
        "download",
        "nvidia/parakeet-tdt-0.6b-v3",
        "--local-dir",
        str(Path("models/asr/parakeet-tdt-0.6b-v3").resolve()),
    ]


def test_hf_download_command_includes_gguf_artifacts() -> None:
    manifest = load_manifest()
    vision = next(model for model in manifest["models"] if model["key"] == "vision")

    command = build_hf_download_command(vision)

    assert command[:3] == ["hf", "download", "openbmb/MiniCPM-V-4.6-gguf"]
    assert "--local-dir" in command
    assert "MiniCPM-V-4_6-Q4_K_M.gguf" in command
    assert "mmproj-model-f16.gguf" in command


def test_hf_download_command_uses_directory_for_asr() -> None:
    manifest = load_manifest()
    asr = next(model for model in manifest["models"] if model["key"] == "asr")

    command = build_hf_download_command(asr)

    assert command == [
        "hf",
        "download",
        "nvidia/parakeet-tdt-0.6b-v3",
        "--local-dir",
        str(Path("models/asr/parakeet-tdt-0.6b-v3").resolve()),
    ]


def test_download_model_requires_explicit_large_download_flag() -> None:
    manifest = load_manifest()

    with pytest.raises(ValueError, match="allow-large-download"):
        download_model(manifest, "asr", allow_large_download=False)


def test_download_model_invokes_hf_download_when_explicitly_allowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = load_manifest()
    captured: dict[str, object] = {}

    def fake_run(command: list[str], cwd: Path, check: bool) -> None:
        captured["command"] = command
        captured["cwd"] = cwd
        captured["check"] = check

    monkeypatch.setattr("scripts.download_models.subprocess.run", fake_run)

    command = download_model(manifest, "asr", allow_large_download=True)

    assert command[0:3] == ["hf", "download", "nvidia/parakeet-tdt-0.6b-v3"]
    assert captured["command"] == command
    assert captured["check"] is True


def test_checksum_verification_fails_closed_when_sha_missing() -> None:
    with pytest.raises(ValueError, match="missing sha256"):
        verify_model(
            {
                "key": "synthetic",
                "local_path": "models/text/NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf",
                "sha256": "",
            }
        )


def test_checksum_update_writes_primary_and_additional_artifact_hashes(tmp_path: Path) -> None:
    model_file = tmp_path / "model.gguf"
    projector_file = tmp_path / "mmproj.gguf"
    model_file.write_bytes(b"model bytes")
    projector_file.write_bytes(b"projector bytes")
    manifest = {
        "version": "test",
        "local_only": True,
        "privacy_mode": "strict",
        "allowed_vendors": ["NVIDIA"],
        "models": [
            {
                "key": "demo",
                "model_name": "Demo",
                "model_id": "nvidia/demo",
                "vendor": "NVIDIA",
                "base_params": 1,
                "adapter_params": 0,
                "quantization": "Q4_K_M",
                "local_path": str(model_file),
                "sha256": "",
                "license": "Demo license",
                "runtime": "llama.cpp",
                "port": 8081,
                "additional_artifacts": [
                    {
                        "name": "mmproj",
                        "local_path": str(projector_file),
                        "sha256": "",
                    }
                ],
                "tasks": ["demo"],
                "supported_languages": ["en"],
                "unsupported_languages": [],
                "commercial_status": "allowed",
                "notes": "test manifest",
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = update_manifest_checksum("demo", manifest_path=manifest_path)
    updated = load_manifest(manifest_path)
    model = updated["models"][0]

    assert result["sha256"] == sha256_path(model_file)
    assert model["sha256"] == sha256_path(model_file)
    assert model["additional_artifacts"][0]["sha256"] == sha256_path(projector_file)
    assert verify_model(model)["verified"] is True


def test_directory_checksum_fails_closed_for_empty_artifact(tmp_path: Path) -> None:
    empty_model_dir = tmp_path / "empty-asr"
    empty_model_dir.mkdir()

    with pytest.raises(ValueError, match="directory artifact is empty"):
        compute_model_artifact_checksums(
            {
                "key": "asr",
                "local_path": str(empty_model_dir),
            }
        )


def test_launch_info_returns_verified_manifest_paths() -> None:
    text = launch_info("text")
    vision = launch_info("vision")

    assert Path(text["model_path"]).exists()
    assert Path(vision["model_path"]).exists()
    assert Path(vision["artifacts"]["mmproj"]).exists()
    assert text["port"] == 8081
    assert vision["port"] == 8082


def test_text_launcher_rejects_mismatched_model_override() -> None:
    environment = os.environ.copy()
    environment["TEXT_MODEL_PATH"] = "/tmp/not-the-verified-model.gguf"

    result = subprocess.run(
        ["bash", "scripts/start_text_llamacpp.sh"],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Refusing TEXT_MODEL_PATH override" in result.stderr


def test_vision_launcher_rejects_mismatched_mmproj_override() -> None:
    environment = os.environ.copy()
    environment["VISION_MMPROJ_PATH"] = "/tmp/not-the-verified-mmproj.gguf"

    result = subprocess.run(
        ["bash", "scripts/start_vision_llamacpp.sh"],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Refusing VISION_MMPROJ_PATH override" in result.stderr


def test_local_smoke_test_creates_invoice_exports(tmp_path: Path) -> None:
    result = run_smoke(tmp_path, gateway=None)

    assert result["status"] == "ok"
    assert Path(result["sample_invoice_image"]).exists()
    assert Path(result["sample_audio"]).exists()
    assert Path(result["outputs"]["json"]).exists()
    assert Path(result["outputs"]["xlsx"]).exists()
    assert Path(result["outputs"]["txt"]).exists()
    assert Path(result["outputs"]["pdf"]).read_bytes().startswith(b"%PDF")
    assert result["checks"]["gateway_health"] is True
    assert result["checks"]["general_assistant"] is True
    assert result["checks"]["invoice_txt"] is True
    assert result["checks"]["invoice_pdf"] is True
    assert result["checks"]["wikipedia_summary"] is True
    assert result["checks"]["image_description"] is True
    assert result["checks"]["speech_to_text"] is True


def test_local_smoke_test_can_exercise_mock_llama_endpoints(tmp_path: Path) -> None:
    if _tcp_port_is_open(8081) or _tcp_port_is_open(8082):
        pytest.skip("mock llama.cpp smoke test requires free fixed local ports 8081 and 8082")

    result = run_smoke(tmp_path, gateway=None, mock_model_endpoints=True)

    assert result["status"] == "ok"
    assert result["mock_model_endpoints"] is True
    assert result["checks"]["text_model_endpoint"] is True
    assert result["checks"]["assistant_text_model_endpoint"] is True
    assert result["checks"]["vision_model_endpoint"] is True


def test_local_smoke_test_rejects_mock_and_real_modes_together(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        run_smoke(
            tmp_path,
            gateway=None,
            mock_model_endpoints=True,
            require_real_model_services=True,
        )


def test_local_smoke_test_can_require_real_service_readiness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_check_ports(require_running: bool, opener: object) -> list[dict[str, object]]:
        assert require_running is True
        return [
            {"name": "text", "optional": False, "http_ready": True},
            {"name": "vision", "optional": False, "http_ready": True},
            {"name": "asr", "optional": False, "http_ready": True},
            {"name": "omni", "optional": True, "http_ready": False},
        ]

    def fake_client_get(
        path: str,
        gateway: str | None,
        *,
        request_timeout_seconds: float,
    ) -> dict[str, object]:
        assert path == "/health"
        assert gateway is None
        assert request_timeout_seconds > 0
        return {"local_only": True, "allow_web": False}

    def fake_client_post(
        path: str,
        payload: dict[str, object],
        gateway: str | None,
        *,
        request_timeout_seconds: float,
    ) -> dict[str, object]:
        assert gateway is None
        assert request_timeout_seconds > 0
        if path == "/tasks/document_to_excel":
            return {"task": "document_to_excel", "status": "ok", "result": {}}
        if path == "/tasks/general":
            return {
                "task": "general_local_assistant",
                "status": "ok",
                "local_only": True,
                "result": {
                    "model_endpoint": "http://127.0.0.1:8081/",
                    "text": "local assistant answer",
                },
            }
        if path == "/tasks/summarize_wikipedia":
            return {
                "task": "summarize_wikipedia",
                "status": "ok",
                "result": {"source": "local_text_model", "summary": "local summary"},
            }
        if path == "/tasks/describe_image":
            return {
                "task": "describe_image",
                "status": "ok",
                "result": {
                    "model_endpoint": "http://127.0.0.1:8082/",
                    "description": "local image description",
                },
            }
        if path == "/tasks/speech_to_text":
            return {
                "task": "speech_to_text",
                "status": "ok",
                "local_only": True,
                "result": {
                    "asr_endpoint": "http://127.0.0.1:8090",
                    "model_ready": True,
                    "text": "local asr transcript",
                },
            }
        raise AssertionError(f"unexpected path: {path}")

    monkeypatch.setattr("scripts.smoke_test_local.check_ports", fake_check_ports)
    monkeypatch.setattr("scripts.smoke_test_local._client_get", fake_client_get)
    monkeypatch.setattr("scripts.smoke_test_local._client_post", fake_client_post)

    result = run_smoke(tmp_path, gateway=None, require_real_model_services=True)

    assert result["status"] == "ok"
    assert result["require_real_model_services"] is True
    assert result["mock_model_endpoints"] is False
    assert result["checks"]["real_model_services_ready"] is True
    assert result["checks"]["text_model_endpoint"] is True
    assert result["checks"]["assistant_text_model_endpoint"] is True
    assert result["checks"]["vision_model_endpoint"] is True
    assert result["checks"]["speech_to_text"] is True
    assert result["checks"]["asr_model_endpoint"] is True
    assert {service["name"] for service in result["model_services"]} >= {"text", "vision", "asr"}


def test_release_gate_passes_current_required_model_metadata() -> None:
    summary = run_release_gate()

    assert summary["status"] == "ok"
    assert summary["total_required_parameters"] == 12_600_000_000


def test_release_gate_cli_reports_current_release_ready_state() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/release_gate.py"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    payload = json.loads(result.stdout)

    assert result.returncode == 0
    assert payload["status"] == "ok"
    assert payload["required_models"] == ["text", "vision", "asr"]
    assert payload["total_required_parameters"] == 12_600_000_000
    assert "Traceback" not in result.stderr


def test_release_gate_flags_required_unresolved_license_review(tmp_path: Path) -> None:
    manifest = _complete_manifest_with_local_artifacts(tmp_path)
    for model in manifest["models"]:
        if model["key"] == "text":
            model["license"] = "review_required"
            model["commercial_status"] = "review_required"
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="required model text has unresolved license"):
        run_release_gate(manifest_path=manifest_path)


def test_release_gate_flags_required_checksum_mismatch(tmp_path: Path) -> None:
    manifest = _complete_manifest_with_local_artifacts(tmp_path)
    for model in manifest["models"]:
        if model["key"] == "text":
            model["sha256"] = "0" * 64
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="required model text failed checksum verification"):
        run_release_gate(manifest_path=manifest_path)


def test_release_gate_metadata_mode_skips_checksum_verification(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest = _complete_manifest_with_local_artifacts(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    config_path = tmp_path / "models.yaml"
    config_path.write_text(
        yaml.safe_dump(_complete_models_config_for_release_gate()),
        encoding="utf-8",
    )

    def fail_if_called(_model: dict[str, object]) -> dict[str, object]:
        raise AssertionError("metadata-only release gate should not hash model artifacts")

    monkeypatch.setattr("scripts.release_gate.verify_model", fail_if_called)

    summary = run_release_gate(
        manifest_path=manifest_path,
        models_config_path=config_path,
        verify_checksums=False,
    )

    assert summary["status"] == "ok"
    assert summary["checksum_verification"] is False


def test_release_gate_passes_complete_manifest_and_matching_config(tmp_path: Path) -> None:
    manifest = _complete_manifest_with_local_artifacts(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    config = _complete_models_config_for_release_gate()
    config_path = tmp_path / "models.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    summary = run_release_gate(manifest_path=manifest_path, models_config_path=config_path)

    assert summary["status"] == "ok"
    assert summary["total_required_parameters"] == 12_600_000_000


def test_release_gate_flags_manifest_config_parameter_mismatch(tmp_path: Path) -> None:
    manifest = _complete_manifest_with_local_artifacts(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    config = _complete_models_config_for_release_gate()
    config["models"]["asr"]["parameters_b"] = 1
    config_path = tmp_path / "models.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="model asr config parameters_b does not match"):
        run_release_gate(manifest_path=manifest_path, models_config_path=config_path)


def _complete_manifest_with_local_artifacts(tmp_path: Path) -> dict[str, object]:
    manifest = load_manifest()
    artifact_paths = {
        "text": tmp_path / "text.gguf",
        "vision": tmp_path / "vision.gguf",
        "asr": tmp_path / "asr",
    }
    artifact_paths["text"].write_bytes(b"text model")
    artifact_paths["vision"].write_bytes(b"vision model")
    artifact_paths["asr"].mkdir()
    (artifact_paths["asr"] / "model.nemo").write_bytes(b"asr model")
    mmproj_path = tmp_path / "mmproj.gguf"
    mmproj_path.write_bytes(b"vision projector")

    for model in manifest["models"]:
        if model["key"] not in {"text", "vision", "asr"}:
            continue
        local_path = artifact_paths[str(model["key"])]
        model["local_path"] = str(local_path)
        model["sha256"] = sha256_path(local_path)
        model["license"] = "Release-reviewed license metadata"
        model["commercial_status"] = "allowed"
        if model["key"] == "vision":
            model["additional_artifacts"] = [
                {
                    "name": "mmproj",
                    "local_path": str(mmproj_path),
                    "sha256": sha256_path(mmproj_path),
                }
            ]
        if model["key"] == "asr":
            model["model_id"] = "nvidia/parakeet-tdt-0.6b-v3"
    return manifest


def _tcp_port_is_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.1)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _complete_models_config_for_release_gate() -> dict[str, object]:
    return {
        "runtime": {
            "local_only": True,
            "allow_remote_inference": False,
            "allow_remote_file_uploads": False,
            "allow_external_apis": False,
            "telemetry_enabled": False,
        },
        "model_policy": {
            "allowed_sources": ["OpenBMB", "NVIDIA", "Cohere", "BFL"],
            "max_v1_total_parameters_b": 32,
        },
        "models": {
            "text": {
                "id": "nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF",
                "source": "NVIDIA",
                "endpoint": "http://127.0.0.1:8081",
                "parameters_b": 4,
                "enabled": True,
            },
            "vision": {
                "id": "openbmb/MiniCPM-V-4.6-gguf",
                "source": "OpenBMB",
                "endpoint": "http://127.0.0.1:8082",
                "parameters_b": 8,
                "enabled": True,
            },
            "asr": {
                "id": "nvidia/parakeet-tdt-0.6b-v3",
                "source": "NVIDIA",
                "endpoint": "http://127.0.0.1:8090",
                "parameters_b": 0.6,
                "enabled": True,
            },
        },
    }
