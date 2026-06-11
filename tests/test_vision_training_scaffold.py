from __future__ import annotations

import os
from pathlib import Path

import pytest

from training.vision.prepare_dataset import (
    ALLOWED_TASKS,
    VisionTrainingRecord,
    load_jsonl,
    validate_records,
)
from training.vision.train_minicpm_v_lora import (
    build_backend_command,
    configure_local_training_environment,
    dry_run,
    load_config,
    record_fingerprint,
    train,
    validate_local_config,
    validate_train_eval_separation,
)

SAMPLE_PATH = Path("training/vision/sample_data/tiny_multimodal.jsonl")
EVAL_SAMPLE_PATH = Path("training/vision/sample_data/tiny_multimodal_eval.jsonl")
CONFIG_PATH = Path("training/vision/configs/minicpm_v_document_lora.yaml")


def test_tiny_vision_manifest_validates_metadata_and_tasks() -> None:
    records = validate_records(SAMPLE_PATH)

    assert len(records) == 6
    assert {record.task for record in records}.issubset(ALLOWED_TASKS)
    assert {record.region for record in records} == {
        "India",
        "Southeast Asia",
        "North America",
        "Europe",
    }
    for record in records:
        assert record.modality == "image"
        assert record.license
        assert record.country
        assert record.language


def test_vision_manifest_rejects_unknown_license() -> None:
    record = load_jsonl(SAMPLE_PATH, limit=1)[0]
    record["license"] = "unknown"

    with pytest.raises(ValueError, match="unknown license is rejected"):
        validate_records([record])


def test_financial_documents_require_synthetic_redacted_or_opt_in_pii() -> None:
    record = load_jsonl(SAMPLE_PATH, limit=1)[0]
    record["pii_status"] = "none"

    with pytest.raises(ValueError, match="financial document records require"):
        validate_records([record])


def test_bank_statement_is_forced_to_human_review() -> None:
    record = load_jsonl(SAMPLE_PATH, limit=3)[2]
    record["human_review_required"] = False

    validated = VisionTrainingRecord.model_validate(record)

    assert validated.task == "bank_statement_extraction"
    assert validated.human_review_required is True


def test_dry_run_writes_local_summary(tmp_path: Path) -> None:
    config = load_config(CONFIG_PATH)
    config["training"]["output_dir"] = str(tmp_path / "vision_adapter")

    summary = dry_run(config, limit=6)

    assert summary["examples"] == 6
    assert summary["eval_examples"] == 3
    assert summary["base_model"] == "openbmb/MiniCPM-V-4.6"
    assert summary["local_only"] is True
    assert summary["qlora_enabled"] is True
    assert summary["training_backend"] == "llama_factory"
    assert summary["train_eval_split_separate"] is True
    assert summary["command_plan"]["executes_training"] is False
    assert (tmp_path / "vision_adapter" / "dry_run_summary.json").exists()


def test_require_images_fails_for_missing_sample_images() -> None:
    with pytest.raises(FileNotFoundError, match="missing image files"):
        validate_records(SAMPLE_PATH, require_images=True)


def test_default_config_uses_separate_train_and_eval_files() -> None:
    config = load_config(CONFIG_PATH)

    assert config["data"]["train_file"] != config["data"]["eval_file"]
    train_records = validate_records(SAMPLE_PATH)
    eval_records = validate_records(EVAL_SAMPLE_PATH)

    validate_train_eval_separation(train_records, eval_records)
    assert not (
        {record_fingerprint(record) for record in train_records}
        & {record_fingerprint(record) for record in eval_records}
    )


def test_vision_train_eval_split_rejects_duplicate_records() -> None:
    record = validate_records(SAMPLE_PATH, limit=1)[0]

    with pytest.raises(ValueError, match="vision train/eval split overlap"):
        validate_train_eval_separation([record], [record])


def test_vision_config_requires_local_only_defaults() -> None:
    config = load_config(CONFIG_PATH)
    validate_local_config(config)

    config["training"]["wandb"] = "enabled"
    with pytest.raises(ValueError, match="disable W&B"):
        validate_local_config(config)


def test_local_training_environment_overrides_wandb(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WANDB_DISABLED", "false")

    configure_local_training_environment(load_config(CONFIG_PATH))

    assert os.environ["WANDB_DISABLED"] == "true"


def test_backend_command_plan_is_local_and_non_executing() -> None:
    config = load_config(CONFIG_PATH)

    command_plan = build_backend_command(config)

    assert command_plan["backend"] == "llama_factory"
    assert command_plan["executes_training"] is False
    assert command_plan["environment"]["WANDB_DISABLED"] == "true"
    assert command_plan["environment"]["HF_HUB_OFFLINE"] == "1"
    assert command_plan["command"][:2] == ["llamafactory-cli", "train"]
    assert "--report_to" in command_plan["command"]
    assert "none" in command_plan["command"]


def test_train_writes_backend_command_plan_without_executing(tmp_path: Path) -> None:
    config = load_config(CONFIG_PATH)
    config["training"]["output_dir"] = str(tmp_path / "vision_adapter")

    command_plan = train(config, require_images=False)

    assert command_plan["executes_training"] is False
    assert (tmp_path / "vision_adapter" / "local_backend_command_plan.json").exists()
