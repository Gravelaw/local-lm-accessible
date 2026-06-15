from __future__ import annotations

import json
from pathlib import Path

from scripts.check_finetuning_completion import check_finetuning_completion


def test_finetuning_completion_fails_closed_when_text_adapter_missing(tmp_path: Path) -> None:
    report = check_finetuning_completion(
        adapter_dir=tmp_path / "missing_adapter",
        finalization_report=tmp_path / "final.json",
        eval_report=tmp_path / "eval.json",
        readiness_report=tmp_path / "readiness.json",
        packaging_report=tmp_path / "packaging.json",
        smoke_report=tmp_path / "smoke.json",
        vision_report=tmp_path / "vision.json",
        asr_report=tmp_path / "asr.json",
        report_json=tmp_path / "completion.json",
        report_md=tmp_path / "completion.md",
    )

    assert report["complete"] is False
    assert report["text_finetuning"]["complete"] is False
    assert "adapter_config.json" in report["text_finetuning"]["missing"]
    assert report["next_modal_actions"][0].endswith("--action finetune_text")
    assert any("--action create_vision_readiness" in item for item in report["next_modal_actions"])
    assert any("--action evaluate_asr_tiny" in item for item in report["next_modal_actions"])
    assert any("--action check_asr_contingency" in item for item in report["next_modal_actions"])


def test_finetuning_completion_passes_with_text_packaging_smoke_and_reports(
    tmp_path: Path,
) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    for filename in ("adapter_config.json", "adapter_model.safetensors", "adapter_manifest.json"):
        (adapter_dir / filename).write_text("{}", encoding="utf-8")
    _write_json(tmp_path / "final.json", {"artifact_type": "lora_adapter"})
    _write_json(tmp_path / "eval.json", {"prediction_sources": ["lora_adapter_generation"]})
    _write_json(tmp_path / "readiness.json", {"ready_for_merge_and_gguf": True})
    _write_json(
        tmp_path / "packaging.json",
        {
            "f16_gguf": {"sha256": "a"},
            "q4_gguf": {"sha256": "b"},
            "q5_gguf": {"sha256": "c"},
        },
    )
    _write_json(tmp_path / "smoke.json", {"passed": True})
    _write_json(tmp_path / "vision.json", {"status": "ready_for_modal_dry_run"})
    _write_json(
        tmp_path / "asr.json",
        {
            "status": "keep_primary_with_runtime_validation",
            "alternate_required": False,
            "eval_metrics": {"wer": 0.0, "unsupported_language_detection": 1.0},
        },
    )

    report = check_finetuning_completion(
        adapter_dir=adapter_dir,
        finalization_report=tmp_path / "final.json",
        eval_report=tmp_path / "eval.json",
        readiness_report=tmp_path / "readiness.json",
        packaging_report=tmp_path / "packaging.json",
        smoke_report=tmp_path / "smoke.json",
        vision_report=tmp_path / "vision.json",
        asr_report=tmp_path / "asr.json",
        report_json=tmp_path / "completion.json",
        report_md=tmp_path / "completion.md",
    )

    assert report["complete"] is True
    assert report["text_finetuning"]["complete"] is True
    assert report["next_modal_actions"] == []
    assert "text_finetuning_complete: True" in (tmp_path / "completion.md").read_text(
        encoding="utf-8"
    )


def test_finetuning_completion_rejects_asr_report_without_passing_eval(
    tmp_path: Path,
) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    for filename in ("adapter_config.json", "adapter_model.safetensors", "adapter_manifest.json"):
        (adapter_dir / filename).write_text("{}", encoding="utf-8")
    _write_json(tmp_path / "final.json", {"artifact_type": "lora_adapter"})
    _write_json(tmp_path / "eval.json", {"prediction_sources": ["lora_adapter_generation"]})
    _write_json(tmp_path / "readiness.json", {"ready_for_merge_and_gguf": True})
    _write_json(
        tmp_path / "packaging.json",
        {
            "f16_gguf": {"sha256": "a"},
            "q4_gguf": {"sha256": "b"},
            "q5_gguf": {"sha256": "c"},
        },
    )
    _write_json(tmp_path / "smoke.json", {"passed": True})
    _write_json(tmp_path / "vision.json", {"status": "ready_for_modal_dry_run"})
    _write_json(
        tmp_path / "asr.json",
        {"status": "evaluate_alternate", "alternate_required": True, "eval_metrics": None},
    )

    report = check_finetuning_completion(
        adapter_dir=adapter_dir,
        finalization_report=tmp_path / "final.json",
        eval_report=tmp_path / "eval.json",
        readiness_report=tmp_path / "readiness.json",
        packaging_report=tmp_path / "packaging.json",
        smoke_report=tmp_path / "smoke.json",
        vision_report=tmp_path / "vision.json",
        asr_report=tmp_path / "asr.json",
        report_json=tmp_path / "completion.json",
        report_md=tmp_path / "completion.md",
    )

    assert report["complete"] is False
    assert report["asr_contingency"]["complete"] is False
    assert report["text_finetuning"]["complete"] is True
    assert any("--action evaluate_asr_tiny" in item for item in report["next_modal_actions"])


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")
