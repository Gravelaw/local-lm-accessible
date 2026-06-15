from __future__ import annotations

import json
from pathlib import Path

from scripts.check_asr_contingency import build_asr_contingency_report
from training.asr.eval_wer import evaluate_manifest, write_reports

MANIFEST = Path("models/manifest.json")
ASR_SAMPLE_MANIFEST = Path("training/asr/sample_data/tiny_manifest.jsonl")
ASR_PREDICTIONS = Path("training/asr/sample_data/tiny_predictions.json")


def test_asr_contingency_keeps_primary_when_eval_passes(tmp_path: Path) -> None:
    eval_json = tmp_path / "asr_eval.json"
    eval_md = tmp_path / "asr_eval.md"
    write_reports(evaluate_manifest(ASR_SAMPLE_MANIFEST, ASR_PREDICTIONS), eval_json, eval_md)

    report = build_asr_contingency_report(
        manifest_path=MANIFEST,
        eval_report_path=eval_json,
        report_json=tmp_path / "asr_contingency.json",
        report_md=tmp_path / "asr_contingency.md",
    )

    assert report["status"] == "keep_primary_with_runtime_validation"
    assert report["alternate_required"] is False
    assert report["primary"]["model_id"] == "nvidia/parakeet-tdt-0.6b-v3"
    assert report["primary"]["vendor"] == "NVIDIA"
    assert report["primary"]["checksum_configured"] is True
    assert report["remote_audio_uploads"] is False
    rule_text = "\n".join(report["candidate_rules"])
    assert "Allowed ASR vendor is NVIDIA only." in rule_text
    assert "no remote audio upload" in rule_text


def test_asr_contingency_requires_alternate_when_checksum_missing(tmp_path: Path) -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for model in manifest["models"]:
        if model["key"] == "asr":
            model["sha256"] = ""
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    eval_json = tmp_path / "asr_eval.json"
    eval_md = tmp_path / "asr_eval.md"
    write_reports(evaluate_manifest(ASR_SAMPLE_MANIFEST, ASR_PREDICTIONS), eval_json, eval_md)

    report = build_asr_contingency_report(
        manifest_path=manifest_path,
        eval_report_path=eval_json,
        report_json=tmp_path / "asr_contingency.json",
        report_md=tmp_path / "asr_contingency.md",
    )

    assert report["alternate_required"] is True
    assert report["status"] == "evaluate_alternate"
    assert report["primary"]["checksum_configured"] is False


def test_asr_contingency_requires_alternate_when_eval_fails(tmp_path: Path) -> None:
    eval_json = tmp_path / "bad_asr_eval.json"
    eval_json.write_text(
        json.dumps(
            {
                "wer": 0.4,
                "unsupported_language_detection": 0.0,
                "missing_predictions": ["missing.wav"],
                "remote_uploads": False,
            }
        ),
        encoding="utf-8",
    )
    report_md = tmp_path / "asr_contingency.md"

    report = build_asr_contingency_report(
        manifest_path=MANIFEST,
        eval_report_path=eval_json,
        report_json=tmp_path / "asr_contingency.json",
        report_md=report_md,
    )

    assert report["alternate_required"] is True
    assert report["status"] == "evaluate_alternate"
    assert "remote_audio_uploads: False" in report_md.read_text(encoding="utf-8")
