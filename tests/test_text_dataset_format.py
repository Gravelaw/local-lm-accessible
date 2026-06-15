from __future__ import annotations

import os
from pathlib import Path

import pytest

from training.text.eval_text_adapter import (
    adapter_predictions_from_records,
    baseline_predictions_from_records,
    evaluate_predictions,
    evaluate_records,
)
from training.text.train_nemotron_lora import (
    ALLOWED_TASKS,
    TextSFTExample,
    build_tokenized_dataset,
    configure_local_training_environment,
    dry_run,
    example_fingerprint,
    latest_checkpoint,
    load_config,
    load_jsonl,
    render_prompt_text,
    sft_training_args_kwargs,
    tokenize_assistant_only,
    validate_examples,
    validate_train_eval_separation,
)

SAMPLE_PATH = Path("training/text/sample_data/router_summary_32.jsonl")
CONFIG_PATH = Path("training/text/configs/nemotron_router_summary_lora.yaml")


class FakeTokenizer:
    eos_token_id = 99

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        del add_special_tokens
        return {"input_ids": [ord(character) % 50 + 1 for character in text]}


def test_text_sft_sample_has_32_valid_examples_and_all_tasks() -> None:
    records = load_jsonl(SAMPLE_PATH)
    examples = validate_examples(records)

    assert len(examples) == 32
    assert {example.metadata.task for example in examples} == ALLOWED_TASKS
    for example in examples:
        assert example.messages[-1].role == "assistant"
        assert example.metadata.license
        assert example.metadata.pii_status in {"synthetic", "redacted", "explicit_user_opt_in"}


def test_text_sft_rejects_unknown_license() -> None:
    record = load_jsonl(SAMPLE_PATH, limit=1)[0]
    record["metadata"]["license"] = "unknown"

    with pytest.raises(ValueError, match="unknown license is rejected"):
        validate_examples([record])


def test_text_sft_rejects_unsupported_task() -> None:
    record = load_jsonl(SAMPLE_PATH, limit=1)[0]
    record["metadata"]["task"] = "unsupported"

    with pytest.raises(ValueError, match="unsupported task"):
        validate_examples([record])


def test_dry_run_validates_32_examples_and_writes_summary(tmp_path: Path) -> None:
    config = load_config(CONFIG_PATH)
    config["training"]["output_dir"] = str(tmp_path / "adapter")

    summary = dry_run(config, limit=32)

    assert summary["examples"] == 32
    assert summary["qlora_enabled"] is True
    assert summary["assistant_only_labels"] is True
    assert summary["wandb_disabled"] is True
    assert summary["eval_examples"] == 8
    assert summary["train_eval_split_separate"] is True
    assert (tmp_path / "adapter" / "dry_run_summary.json").exists()


def test_default_config_uses_separate_train_and_eval_files() -> None:
    config = load_config(CONFIG_PATH)

    assert config["data"]["train_file"] != config["data"]["eval_file"]
    train_examples = validate_examples(load_jsonl(Path(config["data"]["train_file"])))
    eval_examples = validate_examples(load_jsonl(Path(config["data"]["eval_file"])))

    validate_train_eval_separation(train_examples, eval_examples)
    assert not (
        {example_fingerprint(example) for example in train_examples}
        & {example_fingerprint(example) for example in eval_examples}
    )


def test_train_eval_split_rejects_duplicate_examples() -> None:
    example = validate_examples(load_jsonl(SAMPLE_PATH, limit=1))[0]

    with pytest.raises(ValueError, match="train/eval split overlap"):
        validate_train_eval_separation([example], [example])


def test_local_training_environment_overrides_wandb(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WANDB_DISABLED", "false")

    configure_local_training_environment({"training": {"disable_wandb": True}})

    assert os.environ["WANDB_DISABLED"] == "true"


def test_text_sft_example_requires_assistant_target() -> None:
    record = load_jsonl(SAMPLE_PATH, limit=1)[0]
    record["messages"][-1] = {"role": "user", "content": "bad target"}

    with pytest.raises(ValueError, match="last message must be the assistant target"):
        TextSFTExample.model_validate(record)


def test_assistant_only_tokenization_masks_prompt_tokens() -> None:
    example = validate_examples(load_jsonl(SAMPLE_PATH, limit=1))[0]
    tokenizer = FakeTokenizer()

    tokenized = tokenize_assistant_only(example, tokenizer, max_seq_length=4096)
    prompt_length = len(tokenizer(render_prompt_text(example))["input_ids"])

    assert tokenized["labels"][:prompt_length] == [-100] * prompt_length
    assert any(label != -100 for label in tokenized["labels"][prompt_length:])
    assert tokenized["labels"][-1] == tokenizer.eos_token_id
    assert len(tokenized["input_ids"]) == len(tokenized["labels"])


def test_assistant_only_tokenization_fails_when_target_truncated() -> None:
    example = validate_examples(load_jsonl(SAMPLE_PATH, limit=1))[0]

    with pytest.raises(ValueError, match="assistant labels were truncated"):
        tokenize_assistant_only(example, FakeTokenizer(), max_seq_length=2)


def test_build_tokenized_dataset_outputs_labels() -> None:
    examples = validate_examples(load_jsonl(SAMPLE_PATH, limit=2))

    dataset = build_tokenized_dataset(examples, FakeTokenizer(), max_seq_length=4096)

    assert len(dataset) == 2
    assert all("labels" in item for item in dataset)
    assert all(any(label != -100 for label in item["labels"]) for item in dataset)


def test_latest_checkpoint_picks_highest_numeric_checkpoint(tmp_path: Path) -> None:
    (tmp_path / "checkpoint-2").mkdir()
    (tmp_path / "checkpoint-10").mkdir()
    (tmp_path / "checkpoint-final").mkdir()

    assert latest_checkpoint(tmp_path) == tmp_path / "checkpoint-10"


def test_sft_training_args_skip_prepare_for_pretokenized_dataset() -> None:
    config = load_config(CONFIG_PATH)
    args = sft_training_args_kwargs(config, bf16=True)

    assert args["max_length"] == config["data"]["max_seq_length"]
    assert args["dataset_kwargs"] == {"skip_prepare_dataset": True}
    assert args["remove_unused_columns"] is False
    assert args["report_to"] == []


def test_text_eval_scores_explicit_predictions() -> None:
    predictions = [
        {
            "task": "tool_call_json",
            "prediction": '{"tool":"excel_export","arguments":{"format":"xlsx"}}',
            "prediction_source": "unit-test",
        },
        {
            "task": "uncertainty_warning",
            "prediction": "Please verify this with a qualified human.",
            "prediction_source": "unit-test",
        },
    ]

    metrics = evaluate_predictions(predictions)

    assert metrics["total_examples"] == 2
    assert metrics["prediction_sources"] == ["unit-test"]
    assert metrics["json_validity"] == 1.0
    assert metrics["tool_call_argument_accuracy"] == 1.0
    assert metrics["unsafe_certainty_rate"] == 0.0


def test_text_eval_baseline_predictions_are_explicitly_marked() -> None:
    records = load_jsonl(SAMPLE_PATH, limit=2)

    predictions = baseline_predictions_from_records(records)
    metrics = evaluate_records(records)

    assert predictions[0]["prediction_source"] == "assistant_label_baseline"
    assert metrics["prediction_sources"] == ["assistant_label_baseline"]


def test_text_eval_adapter_predictions_reject_missing_adapter(tmp_path: Path) -> None:
    records = load_jsonl(SAMPLE_PATH, limit=1)

    with pytest.raises(FileNotFoundError, match="missing adapter directory"):
        adapter_predictions_from_records(
            records,
            base_model="nvidia/Llama-3.1-Nemotron-Nano-8B-v1",
            adapter_dir=tmp_path / "missing-adapter",
        )
