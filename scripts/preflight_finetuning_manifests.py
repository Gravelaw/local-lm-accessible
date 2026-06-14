from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from training.text.train_nemotron_lora import load_config as load_text_config
from training.text.train_nemotron_lora import load_jsonl as load_text_jsonl
from training.text.train_nemotron_lora import validate_examples
from training.vision.prepare_dataset import validate_records
from training.vision.train_minicpm_v_lora import load_config as load_vision_config
from training.vision.train_minicpm_v_lora import validate_local_config


def preflight_text(config_path: Path, *, report_path: Path | None = None) -> dict[str, Any]:
    config = load_text_config(config_path)
    train_path = Path(config["data"]["train_file"])
    eval_path = Path(config["data"]["eval_file"])
    train_records = _require_jsonl_rows(train_path, "text train manifest")
    eval_records = _require_jsonl_rows(eval_path, "text eval manifest")
    train_examples = validate_examples(train_records)
    eval_examples = validate_examples(eval_records)
    report = {
        "modality": "text",
        "config": str(config_path),
        "train_file": str(train_path),
        "eval_file": str(eval_path),
        "train_rows": len(train_examples),
        "eval_rows": len(eval_examples),
        "tasks": sorted({example.metadata.task for example in train_examples}),
        "regions": sorted({example.metadata.region for example in train_examples}),
        "base_model": config["model"]["base_model"],
        "output_dir": config["training"]["output_dir"],
        "ready": True,
    }
    _write_report(report_path, report)
    return report


def preflight_vision(
    config_path: Path,
    *,
    report_path: Path | None = None,
    require_images: bool = False,
) -> dict[str, Any]:
    config = load_vision_config(config_path)
    validate_local_config(config)
    train_path = Path(config["data"]["train_file"])
    eval_path = Path(config["data"]["eval_file"])
    _require_jsonl_rows(train_path, "vision train manifest")
    _require_jsonl_rows(eval_path, "vision eval manifest")
    train_records = validate_records(train_path, require_images=require_images)
    eval_records = validate_records(eval_path, require_images=require_images)
    report = {
        "modality": "vision",
        "config": str(config_path),
        "train_file": str(train_path),
        "eval_file": str(eval_path),
        "train_rows": len(train_records),
        "eval_rows": len(eval_records),
        "tasks": sorted({record.task for record in train_records}),
        "regions": sorted({record.region for record in train_records}),
        "base_model": config["model"]["base_model"],
        "output_dir": config["training"]["output_dir"],
        "require_images": require_images,
        "ready": True,
    }
    _write_report(report_path, report)
    return report


def _require_jsonl_rows(path: Path, label: str) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"missing {label}: {path}")
    records = load_text_jsonl(path)
    if not records:
        raise ValueError(f"{label} is empty: {path}")
    return records


def _write_report(report_path: Path | None, report: dict[str, Any]) -> None:
    if report_path is None:
        return
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", choices=("text", "vision"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--require-images", action="store_true")
    args = parser.parse_args()

    if args.modality == "text":
        report = preflight_text(args.config, report_path=args.report)
    else:
        report = preflight_vision(
            args.config,
            report_path=args.report,
            require_images=args.require_images,
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
