from __future__ import annotations

import json
import os
import subprocess
import wave
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from services.stt import parakeet_service
from services.stt.parakeet_service import TranscriptionRequest, app
from training.asr.eval_wer import evaluate_manifest, write_reports
from training.asr.prepare_manifest import ASRManifestRecord, validate_manifest, write_jsonl
from training.asr.train_parakeet_nemo import build_nemo_command, dry_run_summary, run_training

MANIFEST = Path("training/asr/sample_data/tiny_manifest.jsonl")
PREDICTIONS = Path("training/asr/sample_data/tiny_predictions.json")
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def reset_parakeet_runtime_cache() -> object:
    parakeet_service._TRANSCRIBER_CACHE = None
    parakeet_service._TRANSCRIBER_LOADER = None
    yield
    parakeet_service._TRANSCRIBER_CACHE = None
    parakeet_service._TRANSCRIBER_LOADER = None


def _write_tiny_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16_000)
        wav_file.writeframes(b"\x00\x00" * 160)


def test_tiny_asr_manifest_validates_and_marks_experimental_languages() -> None:
    records = validate_manifest(MANIFEST)

    assert len(records) == 3
    assert records[0].language == "en"
    assert records[0].experimental is False
    assert records[2].language == "hi"
    assert records[2].experimental is True
    assert records[2].supported_by_parakeet_v3 is False
    for record in records:
        assert record.license
        assert record.pii_status in {"none", "redacted", "consented"}
        assert record.country
        assert record.modality == "audio"
        assert record.task == "speech_to_text"


def test_manifest_rejects_unknown_license() -> None:
    with pytest.raises(ValidationError, match="unknown license is rejected"):
        ASRManifestRecord.model_validate(
            {
                "audio_filepath": "local.wav",
                "duration": 1.0,
                "text": "hello",
                "language": "en",
                "region": "india",
                "country": "india",
                "modality": "audio",
                "task": "speech_to_text",
                "accent": "indian_english",
                "speaker_age_bucket": "adult",
                "license": "unknown",
                "pii_status": "none",
            }
        )


def test_manifest_requires_country_modality_and_task() -> None:
    with pytest.raises(ValidationError):
        ASRManifestRecord.model_validate(
            {
                "audio_filepath": "local.wav",
                "duration": 1.0,
                "text": "hello",
                "language": "en",
                "region": "india",
                "accent": "indian_english",
                "speaker_age_bucket": "adult",
                "license": "CC0-1.0",
                "pii_status": "none",
            }
        )


def test_manifest_missing_audio_validation_fails(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    record = ASRManifestRecord(
        audio_filepath=str(tmp_path / "missing.wav"),
        duration=1.0,
        text="hello",
        language="en",
        region="india",
        country="india",
        modality="audio",
        task="speech_to_text",
        accent="indian_english",
        speaker_age_bucket="adult",
        license="CC0-1.0",
        pii_status="none",
    )
    write_jsonl(manifest_path, [record])

    with pytest.raises(FileNotFoundError, match="missing audio files"):
        validate_manifest(manifest_path, require_audio_exists=True)


def test_write_manifest_outputs_jsonl(tmp_path: Path) -> None:
    records = validate_manifest(MANIFEST)
    output = tmp_path / "manifest.jsonl"

    write_jsonl(output, records)

    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(records)
    assert json.loads(lines[0])["audio_filepath"] == records[0].audio_filepath


def test_asr_eval_works_on_tiny_manifest() -> None:
    metrics = evaluate_manifest(MANIFEST, PREDICTIONS)

    assert metrics["count"] == 3
    assert metrics["wer"] == 0.0
    assert metrics["cer"] == 0.0
    assert metrics["language_detection_accuracy"] == 1.0
    assert metrics["unsupported_language_detection"] == 1.0
    assert metrics["noisy_room_wer"] == 0.0
    assert metrics["elderly_speaker_wer"] == 0.0
    assert metrics["unsupported_language_failures"] == []
    assert metrics["remote_uploads"] is False


def test_asr_eval_reports_unsupported_language_detection_failure(tmp_path: Path) -> None:
    predictions_path = tmp_path / "bad_predictions.json"
    predictions = json.loads(PREDICTIONS.read_text(encoding="utf-8"))
    predictions[-1]["unsupported_language"] = False
    predictions_path.write_text(json.dumps(predictions), encoding="utf-8")

    metrics = evaluate_manifest(MANIFEST, predictions_path)

    assert metrics["unsupported_language_detection"] == 0.0
    assert metrics["unsupported_language_failures"] == [
        "training/asr/sample_data/hi_elderly_eval.wav"
    ]


def test_asr_eval_writes_json_and_markdown_reports(tmp_path: Path) -> None:
    metrics = evaluate_manifest(MANIFEST, PREDICTIONS)
    report_json = tmp_path / "asr_eval.json"
    report_md = tmp_path / "asr_eval.md"

    write_reports(metrics, report_json, report_md)

    assert json.loads(report_json.read_text(encoding="utf-8"))["wer"] == 0.0
    markdown = report_md.read_text(encoding="utf-8")
    assert "# ASR Eval" in markdown
    assert "- wer: 0.0" in markdown
    assert "- unsupported_language_detection: 1.0" in markdown


def test_asr_service_rejects_experimental_without_flag() -> None:
    with pytest.raises(ValidationError, match="allow_experimental"):
        TranscriptionRequest(
            audio_filepath="training/asr/sample_data/hi_elderly_eval.wav",
            language="hi",
            region="india",
            allow_experimental=False,
        )


def test_asr_service_accepts_experimental_with_flag() -> None:
    request = TranscriptionRequest(
        audio_filepath="training/asr/sample_data/hi_elderly_eval.wav",
        language="hi",
        region="india",
        allow_experimental=True,
    )

    assert request.language == "hi"


def test_asr_service_response_flags_unsupported_language(tmp_path: Path) -> None:
    audio_path = tmp_path / "hi_elderly_eval.wav"
    _write_tiny_wav(audio_path)
    client = TestClient(app)

    response = client.post(
        "/transcribe",
        json={
            "audio_filepath": str(audio_path),
            "language": "hi",
            "region": "india",
            "country": "india",
            "allow_experimental": True,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["local_only"] is True
    assert payload["experimental"] is True
    assert payload["unsupported_language"] is True
    assert payload["warnings"]
    assert payload["model_ready"] is True


def test_asr_service_uses_verified_local_parakeet_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_path = tmp_path / "english.wav"
    model_path = tmp_path / "parakeet-model"
    model_path.mkdir()
    _write_tiny_wav(audio_path)
    loaded_paths: list[Path] = []

    def fake_verify_manifest(model_key: str) -> list[dict[str, object]]:
        assert model_key == "asr"
        return [{"path": str(model_path)}]

    def fake_loader(path: Path) -> parakeet_service.Transcriber:
        loaded_paths.append(path)

        def fake_transcriber(audio_filepath: str) -> dict[str, str]:
            assert audio_filepath == str(audio_path)
            return {"text": "hello from local parakeet"}

        return fake_transcriber

    monkeypatch.setattr(parakeet_service, "verify_manifest", fake_verify_manifest)
    monkeypatch.setattr(parakeet_service, "_TRANSCRIBER_LOADER", fake_loader)
    client = TestClient(app)

    response = client.post(
        "/transcribe",
        json={
            "audio_filepath": str(audio_path),
            "language": "en",
            "region": "north_america",
            "country": "us",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["text"] == "hello from local parakeet"
    assert payload["local_only"] is True
    assert payload["model_ready"] is True
    assert loaded_paths == [model_path]


def test_asr_service_falls_back_visibly_when_local_pipeline_cannot_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_path = tmp_path / "english.wav"
    model_path = tmp_path / "parakeet-model"
    model_path.mkdir()
    _write_tiny_wav(audio_path)

    def fake_verify_manifest(model_key: str) -> list[dict[str, object]]:
        assert model_key == "asr"
        return [{"path": str(model_path)}]

    def fake_loader(path: Path) -> parakeet_service.Transcriber:
        raise RuntimeError("mock Parakeet runtime is unavailable")

    monkeypatch.setattr(parakeet_service, "verify_manifest", fake_verify_manifest)
    monkeypatch.setattr(parakeet_service, "_TRANSCRIBER_LOADER", fake_loader)
    client = TestClient(app)

    response = client.post(
        "/transcribe",
        json={
            "audio_filepath": str(audio_path),
            "language": "en",
            "region": "north_america",
            "country": "us",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "stub"
    assert payload["text"] == f"LOCAL_STUB_TRANSCRIPTION_FOR:{audio_path.name}"
    assert payload["model_ready"] is True
    assert "mock Parakeet runtime is unavailable" in payload["warnings"]


def test_asr_service_health_reports_checksum_ready_model() -> None:
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["local_only"] is True
    assert payload["status"] == "ok"
    assert payload["model_ready"] is True
    assert payload["warnings"] == []


def test_asr_service_rejects_missing_audio_file() -> None:
    client = TestClient(app)

    response = client.post(
        "/transcribe",
        json={
            "audio_filepath": "training/asr/sample_data/missing.wav",
            "language": "en",
            "region": "india",
            "country": "india",
        },
    )

    assert response.status_code == 400
    assert "local file" in response.json()["detail"]


def test_parakeet_nemo_command_is_explicit_and_local_manifest_based(tmp_path: Path) -> None:
    command = build_nemo_command(
        model_name="nvidia/parakeet-tdt-0.6b-v3",
        train_manifest=MANIFEST,
        val_manifest=MANIFEST,
        output_dir=tmp_path / "out",
        max_steps=3,
    )

    assert command[:3] == ["python3", "-m", "nemo.collections.asr"]
    assert f"train_manifest={MANIFEST}" in command
    assert f"validation_manifest={MANIFEST}" in command
    assert "trainer.max_steps=3" in command


def test_parakeet_nemo_dry_run_validates_manifest_without_audio_files(tmp_path: Path) -> None:
    summary = dry_run_summary(
        model_name="nvidia/parakeet-tdt-0.6b-v3",
        train_manifest=MANIFEST,
        val_manifest=MANIFEST,
        output_dir=tmp_path / "out",
        max_steps=5,
    )

    assert summary["mode"] == "dry_run"
    assert summary["local_only"] is True
    assert summary["train_records"] == 3
    assert summary["val_records"] == 3
    assert "hi" in summary["experimental_languages"]
    assert summary["remote_audio_uploads"] is False


def test_parakeet_nemo_training_requires_audio_files_by_default(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="missing audio files"):
        run_training(
            model_name="nvidia/parakeet-tdt-0.6b-v3",
            train_manifest=MANIFEST,
            val_manifest=MANIFEST,
            output_dir=tmp_path / "out",
            max_steps=1,
        )


def test_parakeet_nemo_shell_defaults_to_dry_run(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["TRAIN_MANIFEST"] = str(MANIFEST)
    env["VAL_MANIFEST"] = str(MANIFEST)
    env["OUTPUT_DIR"] = str(tmp_path / "out")
    env["MAX_STEPS"] = "2"

    result = subprocess.run(
        ["bash", "training/asr/train_parakeet_nemo.sh"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["mode"] == "dry_run"
    assert payload["command"][-1] == "trainer.max_steps=2"


def test_parakeet_nemo_shell_rejects_remote_audio_upload(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["TRAIN_MANIFEST"] = str(MANIFEST)
    env["VAL_MANIFEST"] = str(MANIFEST)
    env["OUTPUT_DIR"] = str(tmp_path / "out")
    env["ALLOW_REMOTE_AUDIO_UPLOAD"] = "1"

    result = subprocess.run(
        ["bash", "training/asr/train_parakeet_nemo.sh"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Remote audio upload is not allowed" in result.stderr
