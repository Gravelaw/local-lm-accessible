from __future__ import annotations

import json
from pathlib import Path

from data.schemas.source_registry import read_jsonl, write_jsonl
from scripts.build_prepared_training_manifests import build_all_manifests


def _vision_row(task: str, split: str = "train") -> dict[str, object]:
    return {
        "image_path": "image.jpg",
        "prompt": "Extract fields.",
        "expected_output": {"total": "1.00"},
        "region": "Southeast Asia",
        "country": "Malaysia",
        "language": "en",
        "task": task,
        "source_type": "test",
        "source_dataset": "unit",
        "license": "CC-BY-4.0",
        "pii_status": "redacted",
        "modality": "image",
        "document_type": "receipt",
        "split_usage": split,
    }


def test_build_prepared_training_manifests_writes_expected_outputs(tmp_path: Path) -> None:
    processed = tmp_path / "processed" / "datasets"
    output = tmp_path / "processed" / "training"
    reports = tmp_path / "reports"
    write_jsonl(
        processed / "huggingface-ryanznie-sroie-2019-with-labels" / "receipt_extraction.jsonl",
        [_vision_row("receipt_extraction", "train")],
    )
    write_jsonl(
        processed / "manual-xfund" / "form_understanding.jsonl",
        [_vision_row("document_ocr", "validation")],
    )
    write_jsonl(
        processed / "uci-online-retail" / "tabular_reasoning_eval.jsonl",
        [{"dataset_id": "uci:online-retail", "task": "tabular_reasoning"}],
    )

    report = build_all_manifests(processed, output, reports)

    assert report["document_extraction"]["rows"] == 2
    assert (output / "document_extraction_train.jsonl").exists()
    assert (output / "document_extraction_validation.jsonl").exists()
    assert read_jsonl(output / "tabular_eval.jsonl")[0]["task"] == "tabular_reasoning"
    assert (output / "text_sft_train.jsonl").exists()
    assert (output / "asr_train.jsonl").exists()
    gaps = json.loads((reports / "prepared_dataset_gaps.json").read_text(encoding="utf-8"))
    assert any("asr" in gap for gap in gaps["gaps"])
