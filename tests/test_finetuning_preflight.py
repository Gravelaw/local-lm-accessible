from __future__ import annotations

from pathlib import Path

import pytest

from data.schemas.source_registry import write_jsonl
from scripts.preflight_finetuning_manifests import (
    preflight_text,
    preflight_vision,
)
from training.text.train_nemotron_lora import load_config as load_text_config
from training.text.train_nemotron_lora import load_jsonl as load_text_jsonl
from training.vision.prepare_dataset import load_jsonl as load_vision_jsonl
from training.vision.train_minicpm_v_lora import load_config as load_vision_config

TEXT_CONFIG = Path("training/text/configs/nemotron_modal_prepared_lora.yaml")
VISION_CONFIG = Path("training/vision/configs/minicpm_v_modal_document_lora.yaml")
TEXT_SAMPLE = Path("training/text/sample_data/router_summary_32.jsonl")
TEXT_EVAL_SAMPLE = Path("training/text/sample_data/router_summary_eval_8.jsonl")
VISION_SAMPLE = Path("training/vision/sample_data/tiny_multimodal.jsonl")
VISION_EVAL_SAMPLE = Path("training/vision/sample_data/tiny_multimodal_eval.jsonl")


def test_modal_text_config_uses_volume_prepared_manifests() -> None:
    config = load_text_config(TEXT_CONFIG)

    assert (
        config["data"]["train_file"]
        == "/vol/local-lm/data/processed/training/text_sft_train.jsonl"
    )
    assert (
        config["data"]["eval_file"]
        == "/vol/local-lm/data/processed/training/text_sft_validation.jsonl"
    )
    assert config["training"]["output_dir"].startswith("/vol/local-lm/training/text/")
    assert config["training"]["backend"] == "hf_trl_peft"
    assert "unsloth" in config["training"]["optional_backend_candidates"]
    assert config["training"]["max_steps"] == 20


def test_modal_vision_config_uses_volume_prepared_manifests() -> None:
    config = load_vision_config(VISION_CONFIG)

    assert (
        config["data"]["train_file"]
        == "/vol/local-lm/data/processed/training/document_extraction_train.jsonl"
    )
    assert (
        config["data"]["eval_file"]
        == "/vol/local-lm/data/processed/training/document_extraction_validation.jsonl"
    )
    assert config["training"]["output_dir"].startswith("/vol/local-lm/training/vision/")


def test_text_preflight_validates_non_empty_manifests_and_writes_report(tmp_path: Path) -> None:
    train_path = tmp_path / "text_sft_train.jsonl"
    eval_path = tmp_path / "text_sft_validation.jsonl"
    config_path = tmp_path / "text_config.yaml"
    report_path = tmp_path / "reports" / "text_preflight.json"
    write_jsonl(train_path, load_text_jsonl(TEXT_SAMPLE, limit=2))
    write_jsonl(eval_path, load_text_jsonl(TEXT_EVAL_SAMPLE, limit=2))
    config = TEXT_CONFIG.read_text(encoding="utf-8")
    config = config.replace(
        "/vol/local-lm/data/processed/training/text_sft_train.jsonl",
        str(train_path),
    )
    config = config.replace(
        "/vol/local-lm/data/processed/training/text_sft_validation.jsonl",
        str(eval_path),
    )
    config_path.write_text(config, encoding="utf-8")

    report = preflight_text(config_path, report_path=report_path)

    assert report["ready"] is True
    assert report["train_rows"] == 2
    assert report["eval_rows"] == 2
    assert report_path.exists()


def test_vision_preflight_validates_non_empty_manifests_and_writes_report(tmp_path: Path) -> None:
    train_path = tmp_path / "document_extraction_train.jsonl"
    eval_path = tmp_path / "document_extraction_validation.jsonl"
    config_path = tmp_path / "vision_config.yaml"
    report_path = tmp_path / "reports" / "vision_preflight.json"
    write_jsonl(train_path, load_vision_jsonl(VISION_SAMPLE, limit=2))
    write_jsonl(eval_path, load_vision_jsonl(VISION_EVAL_SAMPLE, limit=2))
    config = VISION_CONFIG.read_text(encoding="utf-8")
    config = config.replace(
        "/vol/local-lm/data/processed/training/document_extraction_train.jsonl",
        str(train_path),
    )
    config = config.replace(
        "/vol/local-lm/data/processed/training/document_extraction_validation.jsonl",
        str(eval_path),
    )
    config_path.write_text(config, encoding="utf-8")

    report = preflight_vision(config_path, report_path=report_path)

    assert report["ready"] is True
    assert report["train_rows"] == 2
    assert report["eval_rows"] == 2
    assert report_path.exists()


def test_preflight_rejects_empty_text_manifest(tmp_path: Path) -> None:
    train_path = tmp_path / "text_sft_train.jsonl"
    eval_path = tmp_path / "text_sft_validation.jsonl"
    config_path = tmp_path / "text_config.yaml"
    train_path.write_text("", encoding="utf-8")
    write_jsonl(eval_path, load_text_jsonl(TEXT_EVAL_SAMPLE, limit=1))
    config = TEXT_CONFIG.read_text(encoding="utf-8")
    config = config.replace(
        "/vol/local-lm/data/processed/training/text_sft_train.jsonl",
        str(train_path),
    )
    config = config.replace(
        "/vol/local-lm/data/processed/training/text_sft_validation.jsonl",
        str(eval_path),
    )
    config_path.write_text(config, encoding="utf-8")

    with pytest.raises(ValueError, match="text train manifest is empty"):
        preflight_text(config_path)
