from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.check_text_adapter_release_readiness import check_readiness
from scripts.create_text_adapter_packaging_plan import create_packaging_plan
from scripts.finalize_text_adapter import finalize_text_adapter, sha256_directory


def test_finalize_text_adapter_writes_manifest_checksums_and_reports(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text('{"r": 16}\n', encoding="utf-8")
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"adapter")
    report_json = tmp_path / "reports" / "final.json"
    report_md = tmp_path / "reports" / "final.md"

    manifest = finalize_text_adapter(
        adapter_dir=adapter_dir,
        base_model="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
        train_file=tmp_path / "train.jsonl",
        eval_file=tmp_path / "eval.jsonl",
        report_json=report_json,
        report_markdown=report_md,
    )

    manifest_path = adapter_dir / "adapter_manifest.json"
    assert manifest["artifact_type"] == "lora_adapter"
    assert manifest["final_artifact"] == "lora_adapter"
    assert manifest["sha256"] == sha256_directory(adapter_dir)
    assert manifest_path.exists()
    assert (adapter_dir / "adapter_manifest.json.sha256").exists()
    assert json.loads(report_json.read_text(encoding="utf-8"))["adapter_dir"] == str(adapter_dir)
    assert "# Final Text Adapter Summary" in report_md.read_text(encoding="utf-8")


def test_finalize_text_adapter_rejects_missing_adapter_config(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"adapter")

    with pytest.raises(ValueError, match="adapter directory is missing required files"):
        finalize_text_adapter(
            adapter_dir=adapter_dir,
            base_model="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
            train_file=tmp_path / "train.jsonl",
            eval_file=tmp_path / "eval.jsonl",
            report_json=tmp_path / "final.json",
            report_markdown=tmp_path / "final.md",
        )


def test_text_adapter_readiness_requires_real_adapter_eval(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text('{"r": 16}\n', encoding="utf-8")
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"adapter")
    (adapter_dir / "adapter_manifest.json").write_text(
        json.dumps({"base_model": "nvidia/Llama-3.1-Nemotron-Nano-8B-v1"}) + "\n",
        encoding="utf-8",
    )
    eval_report = tmp_path / "eval.json"
    eval_report.write_text(
        json.dumps(
            {
                "prediction_sources": ["assistant_label_baseline"],
                "invalid_refusal_rate": 0.0,
                "unsafe_certainty_rate": 0.0,
                "json_validity": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = check_readiness(
        adapter_dir=adapter_dir,
        eval_report=eval_report,
        output_json=tmp_path / "readiness.json",
        output_md=tmp_path / "readiness.md",
    )

    assert report["ready_for_merge_and_gguf"] is False
    assert "eval report must use lora_adapter_generation predictions" in report["failures"]


def test_text_adapter_readiness_accepts_passing_adapter_eval(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text('{"r": 16}\n', encoding="utf-8")
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"adapter")
    (adapter_dir / "adapter_manifest.json").write_text(
        json.dumps({"base_model": "nvidia/Llama-3.1-Nemotron-Nano-8B-v1"}) + "\n",
        encoding="utf-8",
    )
    eval_report = tmp_path / "eval.json"
    eval_report.write_text(
        json.dumps(
            {
                "prediction_sources": ["lora_adapter_generation"],
                "invalid_refusal_rate": 0.0,
                "unsafe_certainty_rate": 0.0,
                "json_validity": 0.75,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = check_readiness(
        adapter_dir=adapter_dir,
        eval_report=eval_report,
        output_json=tmp_path / "readiness.json",
        output_md=tmp_path / "readiness.md",
    )

    assert report["ready_for_merge_and_gguf"] is True
    assert json.loads((tmp_path / "readiness.json").read_text(encoding="utf-8"))[
        "ready_for_merge_and_gguf"
    ]


def test_text_adapter_packaging_plan_uses_readiness_report(tmp_path: Path) -> None:
    readiness_report = tmp_path / "readiness.json"
    readiness_report.write_text(
        json.dumps(
            {
                "ready_for_merge_and_gguf": True,
                "base_model": "nvidia/Llama-3.1-Nemotron-Nano-8B-v1",
                "failures": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    plan = create_packaging_plan(
        readiness_report=readiness_report,
        adapter_dir=tmp_path / "adapter",
        merged_dir=tmp_path / "merged",
        f16_gguf=tmp_path / "model.gguf",
        quantized_dir=tmp_path / "quantized",
        output_json=tmp_path / "plan.json",
        output_md=tmp_path / "plan.md",
    )

    assert plan["ready_to_package"] is True
    assert plan["commands"]["merge_adapter"][:2] == ["python", "training/text/merge_adapter.py"]
    assert "--allow-remote-files" in plan["commands"]["merge_adapter"]
    assert "quantize_gguf" in plan["commands"]
