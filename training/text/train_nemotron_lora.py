from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

ALLOWED_TASKS = {
    "route_task",
    "summarize_article",
    "summarize_with_sources",
    "tool_call_json",
    "repair_json",
    "elderly_explanation",
    "uncertainty_warning",
}
ALLOWED_REGIONS = {"India", "Southeast Asia", "North America", "Europe"}
UNKNOWN_LICENSE_VALUES = {"", "unknown", "unk", "n/a", "na", "none", "tbd", "unlicensed"}
REQUIRED_METADATA = {
    "region",
    "country",
    "language",
    "task",
    "source_type",
    "license",
    "pii_status",
}


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str = Field(min_length=1)


class TextSFTMetadata(BaseModel):
    region: str = Field(min_length=1)
    country: str = Field(min_length=1)
    language: str = Field(min_length=1)
    task: str = Field(min_length=1)
    source_type: str = Field(min_length=1)
    license: str = Field(min_length=1)
    pii_status: str = Field(min_length=1)

    @field_validator("region")
    @classmethod
    def validate_region(cls, value: str) -> str:
        if value not in ALLOWED_REGIONS:
            raise ValueError(f"unsupported region: {value}")
        return value

    @field_validator("task")
    @classmethod
    def validate_task(cls, value: str) -> str:
        if value not in ALLOWED_TASKS:
            raise ValueError(f"unsupported task: {value}")
        return value

    @field_validator("license")
    @classmethod
    def reject_unknown_license(cls, value: str) -> str:
        if value.strip().casefold() in UNKNOWN_LICENSE_VALUES:
            raise ValueError("unknown license is rejected")
        return value


class TextSFTExample(BaseModel):
    messages: list[ChatMessage] = Field(min_length=2)
    metadata: TextSFTMetadata

    @model_validator(mode="after")
    def require_assistant_target(self) -> TextSFTExample:
        if self.messages[-1].role != "assistant":
            raise ValueError("last message must be the assistant target")
        if not any(message.role == "user" for message in self.messages):
            raise ValueError("at least one user message is required")
        return self


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError("training config must be a mapping")
    return config


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as jsonl_file:
        for line_number, line in enumerate(jsonl_file, start=1):
            if limit is not None and len(records) >= limit:
                break
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"record must be a JSON object at {path}:{line_number}")
            records.append(record)
    return records


def validate_examples(records: list[dict[str, Any]]) -> list[TextSFTExample]:
    examples: list[TextSFTExample] = []
    for index, record in enumerate(records, start=1):
        try:
            examples.append(TextSFTExample.model_validate(record))
        except ValidationError as exc:
            raise ValueError(f"invalid SFT example #{index}: {exc}") from exc
    return examples


def example_fingerprint(example: TextSFTExample) -> str:
    payload = [
        {"role": message.role, "content": message.content.strip()} for message in example.messages
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_train_eval_separation(
    train_examples: list[TextSFTExample],
    eval_examples: list[TextSFTExample],
) -> None:
    train_fingerprints = {example_fingerprint(example) for example in train_examples}
    eval_fingerprints = {example_fingerprint(example) for example in eval_examples}
    overlap = train_fingerprints & eval_fingerprints
    if overlap:
        raise ValueError(
            "train/eval split overlap detected; eval examples must not duplicate training examples"
        )


def configure_local_training_environment(config: dict[str, Any]) -> None:
    training_config = config.get("training", {})
    if bool(training_config.get("disable_wandb", True)):
        os.environ["WANDB_DISABLED"] = "true"
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")


def render_training_text(example: TextSFTExample, tokenizer: Any | None = None) -> str:
    messages = [message.model_dump() for message in example.messages]
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return str(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        )
    rendered = []
    for message in example.messages:
        rendered.append(f"<|{message.role}|>\n{message.content}")
    rendered.append("<|end|>")
    return "\n".join(rendered)


def render_prompt_text(example: TextSFTExample, tokenizer: Any | None = None) -> str:
    prompt_messages = [message for message in example.messages if message.role != "assistant"]
    messages = [message.model_dump() for message in prompt_messages]
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return str(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    rendered = []
    for message in prompt_messages:
        rendered.append(f"<|{message.role}|>\n{message.content}")
    rendered.append("<|assistant|>\n")
    return "\n".join(rendered)


def assistant_target_text(example: TextSFTExample) -> str:
    assistant_messages = [
        message.content for message in example.messages if message.role == "assistant"
    ]
    if not assistant_messages:
        raise ValueError("assistant target is required")
    return "\n".join(assistant_messages)


def tokenize_assistant_only(
    example: TextSFTExample,
    tokenizer: Any,
    max_seq_length: int,
) -> dict[str, list[int]]:
    prompt_text = render_prompt_text(example, tokenizer)
    target_text = assistant_target_text(example)
    prompt_ids = _tokenize_text(tokenizer, prompt_text)
    target_ids = _tokenize_text(tokenizer, target_text)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        target_ids = [*target_ids, int(eos_token_id)]
    input_ids = [*prompt_ids, *target_ids][:max_seq_length]
    labels = ([-100] * len(prompt_ids) + target_ids)[:max_seq_length]
    attention_mask = [1] * len(input_ids)
    if len(input_ids) != len(labels):
        raise ValueError("input_ids and labels must have the same length")
    if not any(label != -100 for label in labels):
        raise ValueError("assistant labels were truncated; increase max_seq_length")
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def build_tokenized_dataset(
    examples: list[TextSFTExample],
    tokenizer: Any,
    max_seq_length: int,
) -> list[dict[str, list[int]]]:
    return [tokenize_assistant_only(example, tokenizer, max_seq_length) for example in examples]


def sft_training_args_kwargs(config: dict[str, Any], *, bf16: bool) -> dict[str, Any]:
    return {
        "output_dir": config["training"]["output_dir"],
        "per_device_train_batch_size": config["training"]["per_device_train_batch_size"],
        "per_device_eval_batch_size": config["training"]["per_device_eval_batch_size"],
        "gradient_accumulation_steps": config["training"]["gradient_accumulation_steps"],
        "learning_rate": config["training"]["learning_rate"],
        "num_train_epochs": config["training"]["num_train_epochs"],
        "max_steps": config["training"]["max_steps"],
        "eval_strategy": "steps",
        "eval_steps": config["training"]["eval_steps"],
        "save_steps": config["training"]["save_steps"],
        "logging_steps": config["training"]["logging_steps"],
        "save_total_limit": config["training"]["save_total_limit"],
        "load_best_model_at_end": bool(config["training"]["save_best_adapter"]),
        "metric_for_best_model": config["training"]["metric_for_best_model"],
        "greater_is_better": config["training"]["greater_is_better"],
        "bf16": bf16,
        "fp16": not bf16 and bool(config["training"]["fp16"]),
        "report_to": [],
        "logging_dir": str(Path(config["training"]["output_dir"]) / "logs"),
        "seed": config["training"]["seed"],
        "remove_unused_columns": False,
        "do_train": True,
        "do_eval": True,
        "max_length": int(config["data"]["max_seq_length"]),
        "dataset_kwargs": {"skip_prepare_dataset": True},
    }


def build_sft_trainer(
    *,
    sft_trainer_cls: Any,
    sft_config_cls: Any,
    model: Any,
    tokenizer: Any,
    train_dataset: Any,
    eval_dataset: Any,
    peft_config: Any,
    training_args_kwargs: dict[str, Any],
    data_collator: Any,
) -> Any:
    trainer_signature = inspect.signature(sft_trainer_cls)
    if sft_config_cls is not None:
        args = sft_config_cls(**training_args_kwargs)
    else:
        from transformers import TrainingArguments

        fallback_kwargs = {
            key: value
            for key, value in training_args_kwargs.items()
            if key not in {"max_length", "dataset_kwargs"}
        }
        args = TrainingArguments(**fallback_kwargs)
    kwargs = {
        "model": model,
        "args": args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "peft_config": peft_config,
        "data_collator": data_collator,
    }
    if "processing_class" in trainer_signature.parameters:
        kwargs["processing_class"] = tokenizer
    else:
        kwargs["tokenizer"] = tokenizer
        kwargs["max_seq_length"] = int(training_args_kwargs["max_length"])
    return sft_trainer_cls(**kwargs)


def _tokenize_text(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    return [int(token_id) for token_id in input_ids]


def dry_run(config: dict[str, Any], limit: int) -> dict[str, Any]:
    configure_local_training_environment(config)
    train_file = Path(config["data"]["train_file"])
    eval_file = Path(config["data"]["eval_file"])
    examples = validate_examples(load_jsonl(train_file, limit=limit))
    eval_examples = validate_examples(load_jsonl(eval_file))
    validate_train_eval_separation(examples, eval_examples)
    tasks = sorted({example.metadata.task for example in examples})
    regions = sorted({example.metadata.region for example in examples})
    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "mode": "dry_run",
        "base_model": config["model"]["base_model"],
        "examples": len(examples),
        "tasks": tasks,
        "regions": regions,
        "max_seq_length": int(config["data"]["max_seq_length"]),
        "qlora_enabled": bool(config["qlora"]["enabled"]),
        "assistant_only_labels": True,
        "wandb_disabled": os.environ.get("WANDB_DISABLED") == "true",
        "eval_examples": len(eval_examples),
        "train_eval_split_separate": True,
    }
    (output_dir / "dry_run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def latest_checkpoint(output_dir: Path) -> Path | None:
    if not output_dir.exists():
        return None
    checkpoints = []
    for path in output_dir.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        suffix = path.name.removeprefix("checkpoint-")
        if suffix.isdigit():
            checkpoints.append((int(suffix), path))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda item: item[0])[1]


def train(config: dict[str, Any]) -> dict[str, Any]:
    configure_local_training_environment(config)

    import torch
    from datasets import Dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForSeq2Seq,
    )
    from trl import SFTTrainer

    try:
        from trl import SFTConfig
    except ImportError:
        SFTConfig = None

    try:
        import bitsandbytes as _bitsandbytes  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("bitsandbytes is required for QLoRA training") from exc

    train_examples = validate_examples(load_jsonl(Path(config["data"]["train_file"])))
    eval_examples = validate_examples(load_jsonl(Path(config["data"]["eval_file"])))
    validate_train_eval_separation(train_examples, eval_examples)

    base_model = config["model"]["base_model"]
    bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    quantization_config = None
    if config["qlora"]["enabled"]:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=bool(config["qlora"]["load_in_4bit"]),
            bnb_4bit_quant_type=str(config["qlora"]["bnb_4bit_quant_type"]),
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=bool(config["qlora"]["bnb_4bit_use_double_quant"]),
        )

    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=bool(config["model"].get("trust_remote_code", False)),
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=bool(config["model"].get("trust_remote_code", False)),
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16 if bf16 else torch.float16,
        device_map="auto",
    )
    if config["training"]["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
    if config["qlora"]["enabled"]:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=int(config["lora"]["r"]),
        lora_alpha=int(config["lora"]["alpha"]),
        lora_dropout=float(config["lora"]["dropout"]),
        bias=str(config["lora"]["bias"]),
        task_type=str(config["lora"]["task_type"]),
        target_modules=list(config["lora"]["target_modules"]),
    )
    train_dataset = Dataset.from_list(
        build_tokenized_dataset(
            train_examples,
            tokenizer,
            int(config["data"]["max_seq_length"]),
        )
    )
    eval_dataset = Dataset.from_list(
        build_tokenized_dataset(
            eval_examples,
            tokenizer,
            int(config["data"]["max_seq_length"]),
        )
    )

    training_args_kwargs = sft_training_args_kwargs(config, bf16=bf16)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )
    trainer = build_sft_trainer(
        sft_trainer_cls=SFTTrainer,
        sft_config_cls=SFTConfig,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
        training_args_kwargs=training_args_kwargs,
        data_collator=data_collator,
    )
    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    resume_checkpoint = latest_checkpoint(output_dir)
    train_result = trainer.train(
        resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint else None
    )
    trainer.save_model(config["training"]["output_dir"])
    tokenizer.save_pretrained(config["training"]["output_dir"])
    metrics = dict(getattr(train_result, "metrics", {}) or {})
    report = {
        "mode": "train",
        "base_model": base_model,
        "output_dir": str(output_dir),
        "train_examples": len(train_examples),
        "eval_examples": len(eval_examples),
        "resume_from_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
        "metrics": metrics,
        "adapter_saved": True,
    }
    report_path = output_dir / "final_training_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("training/text/configs/nemotron_router_summary_lora.yaml"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=32)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.dry_run:
        summary = dry_run(config, args.limit)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    report = train(config)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
