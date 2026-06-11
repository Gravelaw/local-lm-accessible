from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.vision.prepare_dataset import (  # noqa: E402
    VisionTrainingRecord,
    count_by,
    validate_records,
)

ALLOWED_BACKENDS = {"llama_factory", "swift"}
EXPECTED_BASE_MODEL = "openbmb/MiniCPM-V-4.6"
EXPECTED_VENDOR = "OpenBMB"


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError("vision training config must be a mapping")
    return config


def configure_local_training_environment(config: dict[str, Any]) -> None:
    training_config = config.get("training", {})
    if str(training_config.get("wandb", "disabled")).casefold() == "disabled":
        os.environ["WANDB_DISABLED"] = "true"
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")


def validate_local_config(config: dict[str, Any]) -> None:
    model_config = config.get("model", {})
    data_config = config.get("data", {})
    training_config = config.get("training", {})

    if model_config.get("base_model") != EXPECTED_BASE_MODEL:
        raise ValueError("vision training base model must be openbmb/MiniCPM-V-4.6")
    if model_config.get("allowed_vendor") != EXPECTED_VENDOR:
        raise ValueError("vision training model vendor must be OpenBMB")
    if not bool(config.get("lora", {}).get("enabled", False)):
        raise ValueError("vision training requires LoRA")
    if not bool(config.get("qlora", {}).get("enabled", False)):
        raise ValueError("vision training requires QLoRA")
    if not bool(training_config.get("local_logs_only", False)):
        raise ValueError("vision training logs must stay local")
    if str(training_config.get("wandb", "")).casefold() != "disabled":
        raise ValueError("vision training must disable W&B by default")

    backend = str(training_config.get("backend", ""))
    fallback_backend = str(training_config.get("fallback_backend", "swift"))
    if backend not in ALLOWED_BACKENDS:
        raise ValueError(f"unsupported vision training backend: {backend}")
    if fallback_backend not in ALLOWED_BACKENDS:
        raise ValueError(f"unsupported vision fallback backend: {fallback_backend}")

    train_file = str(data_config.get("train_file", ""))
    eval_file = str(data_config.get("eval_file", ""))
    output_dir = str(training_config.get("output_dir", ""))
    for field_name, value in {
        "train_file": train_file,
        "eval_file": eval_file,
        "output_dir": output_dir,
    }.items():
        if "://" in value:
            raise ValueError(f"{field_name} must be a local path")
    if train_file == eval_file:
        raise ValueError("vision train/eval files must be separate")


def record_fingerprint(record: VisionTrainingRecord) -> str:
    payload = {
        "image_path": record.image_path.strip(),
        "prompt": record.prompt.strip(),
        "expected_output": record.expected_output,
        "task": record.task,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_train_eval_separation(
    train_records: list[VisionTrainingRecord],
    eval_records: list[VisionTrainingRecord],
) -> None:
    train_fingerprints = {record_fingerprint(record) for record in train_records}
    eval_fingerprints = {record_fingerprint(record) for record in eval_records}
    overlap = train_fingerprints & eval_fingerprints
    if overlap:
        raise ValueError(
            "vision train/eval split overlap detected; "
            "eval records must not duplicate training records"
        )


def build_backend_command(config: dict[str, Any], backend: str | None = None) -> dict[str, Any]:
    validate_local_config(config)
    selected_backend = backend or str(config["training"]["backend"])
    if selected_backend not in ALLOWED_BACKENDS:
        raise ValueError(f"unsupported vision training backend: {selected_backend}")

    model_name = str(config["model"]["base_model"])
    train_file = str(config["data"]["train_file"])
    eval_file = str(config["data"]["eval_file"])
    output_dir = str(config["training"]["output_dir"])
    max_seq_length = str(config["data"]["max_seq_length"])

    if selected_backend == "llama_factory":
        command = [
            "llamafactory-cli",
            "train",
            "--stage",
            "sft",
            "--model_name_or_path",
            model_name,
            "--dataset",
            train_file,
            "--eval_dataset",
            eval_file,
            "--finetuning_type",
            "lora",
            "--output_dir",
            output_dir,
            "--cutoff_len",
            max_seq_length,
            "--per_device_train_batch_size",
            str(config["training"]["per_device_train_batch_size"]),
            "--gradient_accumulation_steps",
            str(config["training"]["gradient_accumulation_steps"]),
            "--learning_rate",
            str(config["training"]["learning_rate"]),
            "--num_train_epochs",
            str(config["training"]["num_train_epochs"]),
            "--template",
            "minicpm_v",
            "--report_to",
            "none",
        ]
    else:
        command = [
            "swift",
            "sft",
            "--model",
            model_name,
            "--train_type",
            "lora",
            "--dataset",
            train_file,
            "--val_dataset",
            eval_file,
            "--output_dir",
            output_dir,
            "--max_length",
            max_seq_length,
            "--gradient_checkpointing",
            "true",
            "--report_to",
            "none",
        ]

    return {
        "backend": selected_backend,
        "command": command,
        "environment": {
            "WANDB_DISABLED": "true",
            "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1",
            "HF_HUB_OFFLINE": "1",
        },
        "executes_training": False,
        "notes": (
            "Command plan only; install the backend and stage model files locally "
            "before execution."
        ),
    }


def dry_run(
    config: dict[str, Any], *, limit: int | None = None, require_images: bool = False
) -> dict[str, Any]:
    configure_local_training_environment(config)
    validate_local_config(config)
    train_file = Path(config["data"]["train_file"])
    eval_file = Path(config["data"]["eval_file"]) if config["data"].get("eval_file") else None
    train_records = validate_records(train_file, limit=limit, require_images=require_images)
    eval_records = (
        validate_records(eval_file, limit=limit, require_images=require_images) if eval_file else []
    )
    validate_train_eval_separation(train_records, eval_records)

    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    all_records = [*train_records, *eval_records]
    command_plan = build_backend_command(config)
    summary = {
        "mode": "dry_run",
        "base_model": config["model"]["base_model"],
        "examples": len(train_records),
        "eval_examples": len(eval_records),
        "tasks": sorted({record.task for record in all_records}),
        "regions": count_by(all_records, "region"),
        "languages": count_by(all_records, "language"),
        "max_seq_length": int(config["data"]["max_seq_length"]),
        "image_max_pixels": int(config["data"]["image_max_pixels"]),
        "lora_enabled": bool(config["lora"]["enabled"]),
        "qlora_enabled": bool(config["qlora"]["enabled"]),
        "gradient_checkpointing": bool(config["training"]["gradient_checkpointing"]),
        "local_logs_only": True,
        "wandb_disabled": True,
        "local_only": True,
        "training_backend": str(config["training"]["backend"]),
        "fallback_backend": str(config["training"].get("fallback_backend", "swift")),
        "train_eval_split_separate": True,
        "command_plan": command_plan,
    }
    (output_dir / "dry_run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def train(config: dict[str, Any], *, require_images: bool = True) -> dict[str, Any]:
    configure_local_training_environment(config)
    validate_local_config(config)
    train_records = validate_records(
        Path(config["data"]["train_file"]), require_images=require_images
    )
    eval_records = validate_records(
        Path(config["data"]["eval_file"]), require_images=require_images
    )
    validate_train_eval_separation(train_records, eval_records)

    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    command_plan = build_backend_command(config)
    plan_path = output_dir / "local_backend_command_plan.json"
    plan_path.write_text(
        json.dumps(command_plan, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return command_plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MiniCPM-V LoRA training scaffold")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--require-images", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.dry_run:
        summary = dry_run(config, limit=args.limit, require_images=args.require_images)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    command_plan = train(config)
    print(json.dumps(command_plan, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
