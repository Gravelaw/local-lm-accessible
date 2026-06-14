from __future__ import annotations

import json
from pathlib import Path

import pytest

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
