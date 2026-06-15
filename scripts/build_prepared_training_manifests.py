from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from data.schemas.source_registry import read_jsonl, write_jsonl
from training.asr.prepare_manifest import ASRManifestRecord
from training.text.train_nemotron_lora import validate_examples
from training.vision.prepare_dataset import validate_records

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATASETS = ROOT / "data" / "processed" / "datasets"
TRAINING_OUTPUT = ROOT / "data" / "processed" / "training"
REPORTS_DIR = ROOT / "reports"
TEXT_TRAIN_SAMPLE = ROOT / "training" / "text" / "sample_data" / "router_summary_32.jsonl"
TEXT_EVAL_SAMPLE = ROOT / "training" / "text" / "sample_data" / "router_summary_eval_8.jsonl"

SPLITS = ("train", "validation", "test", "regional_stress_test")
DOCUMENT_INPUTS = (
    "huggingface-ryanznie-sroie-2019-with-labels/receipt_extraction.jsonl",
    "manual-fatura/invoice_extraction.jsonl",
    "manual-xfund/form_understanding.jsonl",
    "manual-cord/receipt_extraction.jsonl",
    "synthetic-local-lm-regional-documents/vision_document_extraction.jsonl",
)


def build_all_manifests(
    processed_root: Path = PROCESSED_DATASETS,
    output_dir: Path = TRAINING_OUTPUT,
    reports_dir: Path = REPORTS_DIR,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "document_extraction": build_document_manifests(processed_root, output_dir),
        "text_sft": build_text_manifests(output_dir),
        "tabular_eval": build_tabular_eval(processed_root, output_dir),
        "asr": build_asr_manifests(processed_root, output_dir),
        "image_accessibility": build_empty_task_manifests(
            output_dir,
            prefix="image_accessibility",
            reason=(
                "No approved image accessibility dataset has been downloaded "
                "under the 10 GB cap."
            ),
        ),
    }
    report["gaps"] = collect_gaps(report)
    (reports_dir / "prepared_dataset_gaps.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (reports_dir / "prepared_dataset_gaps.md").write_text(
        render_report(report),
        encoding="utf-8",
    )
    return report


def build_document_manifests(processed_root: Path, output_dir: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    missing_inputs: list[str] = []
    for relative in DOCUMENT_INPUTS:
        path = processed_root / relative
        if not path.exists():
            missing_inputs.append(str(path))
            continue
        records.extend(read_jsonl(path))
    records = [_normalize_vision_record(record) for record in records]
    validate_records(records, require_images=False)
    split_records = split_by_usage(records)
    outputs = write_split_manifests(output_dir, "document_extraction", split_records)
    return {
        "rows": len(records),
        "outputs": outputs,
        "missing_inputs": missing_inputs,
        "counts": summarize_records(records),
    }


def build_text_manifests(output_dir: Path) -> dict[str, Any]:
    train_records = read_jsonl(TEXT_TRAIN_SAMPLE)
    eval_records = read_jsonl(TEXT_EVAL_SAMPLE)
    validate_examples(train_records)
    validate_examples(eval_records)
    outputs = {
        "train": output_dir / "text_sft_train.jsonl",
        "validation": output_dir / "text_sft_validation.jsonl",
        "test": output_dir / "text_sft_test.jsonl",
    }
    write_jsonl(outputs["train"], train_records)
    write_jsonl(outputs["validation"], eval_records)
    write_jsonl(outputs["test"], eval_records)
    return {
        "rows": len(train_records) + len(eval_records),
        "outputs": {key: str(path) for key, path in outputs.items()},
        "counts": summarize_text_records([*train_records, *eval_records]),
    }


def build_tabular_eval(processed_root: Path, output_dir: Path) -> dict[str, Any]:
    source = processed_root / "uci-online-retail" / "tabular_reasoning_eval.jsonl"
    output = output_dir / "tabular_eval.jsonl"
    if not source.exists():
        write_jsonl(output, [])
        return {"rows": 0, "outputs": [str(output)], "missing_inputs": [str(source)]}
    records = read_jsonl(source)
    write_jsonl(output, records)
    return {"rows": len(records), "outputs": [str(output)], "counts": summarize_records(records)}


def build_asr_manifests(processed_root: Path, output_dir: Path) -> dict[str, Any]:
    candidate_paths = [
        processed_root / "huggingface-google-fleurs" / "asr_manifest.jsonl",
        processed_root / "huggingface-google-fleurs" / "fleurs_asr_manifest.jsonl",
    ]
    records: list[dict[str, Any]] = []
    for path in candidate_paths:
        if path.exists():
            records.extend(read_jsonl(path))
    validated = [ASRManifestRecord.model_validate(record) for record in records]
    split_records = split_by_usage([record.model_dump() for record in validated])
    outputs = write_split_manifests(output_dir, "asr", split_records)
    return {
        "rows": len(records),
        "outputs": outputs,
        "missing_inputs": [] if records else [str(path) for path in candidate_paths],
        "notes": "FLEURS remains manual/eval-subset pending until audio manifests exist."
        if not records
        else "",
    }


def build_empty_task_manifests(output_dir: Path, *, prefix: str, reason: str) -> dict[str, Any]:
    outputs: dict[str, str] = {}
    for split in SPLITS:
        path = output_dir / f"{prefix}_{split}.jsonl"
        write_jsonl(path, [])
        outputs[split] = str(path)
    return {"rows": 0, "outputs": outputs, "notes": reason}


def split_by_usage(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    split_records = {split: [] for split in SPLITS}
    for index, record in enumerate(records):
        split = str(record.get("split_usage") or "").strip()
        if split not in split_records:
            split = ("train", "validation", "test")[index % 3]
        split_records[split].append(record)
        if split != "regional_stress_test" and _is_regional_stress_record(record):
            split_records["regional_stress_test"].append(record)
    if not split_records["regional_stress_test"]:
        split_records["regional_stress_test"] = records[: min(100, len(records))]
    return split_records


def write_split_manifests(
    output_dir: Path,
    prefix: str,
    split_records: dict[str, list[dict[str, Any]]],
) -> dict[str, str]:
    outputs: dict[str, str] = {}
    for split in SPLITS:
        path = output_dir / f"{prefix}_{split}.jsonl"
        write_jsonl(path, split_records.get(split, []))
        outputs[split] = str(path)
    return outputs


def _normalize_vision_record(record: dict[str, Any]) -> dict[str, Any]:
    item = dict(record)
    if item.get("task") == "image_translation":
        item["task"] = "image_text_translation"
    if item.get("pii_status") == "none" and item.get("task") in {
        "invoice_extraction",
        "receipt_extraction",
        "bill_extraction",
        "bank_statement_extraction",
    }:
        item["pii_status"] = "redacted"
    return item


def _is_regional_stress_record(record: dict[str, Any]) -> bool:
    return str(record.get("region", "")) in {"India", "Southeast Asia"}


def summarize_records(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    return {
        "region": dict(Counter(str(record.get("region", "")) for record in records)),
        "task": dict(Counter(str(record.get("task", "")) for record in records)),
        "source_dataset": dict(
            Counter(str(record.get("source_dataset", "")) for record in records)
        ),
    }


def summarize_text_records(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    return {
        "region": dict(
            Counter(str(record.get("metadata", {}).get("region", "")) for record in records)
        ),
        "task": dict(
            Counter(str(record.get("metadata", {}).get("task", "")) for record in records)
        ),
    }


def collect_gaps(report: dict[str, Any]) -> list[str]:
    gaps: list[str] = []
    for section, payload in report.items():
        if isinstance(payload, dict):
            if payload.get("rows") == 0:
                gaps.append(f"{section} has no prepared rows")
            for missing in payload.get("missing_inputs", []) or []:
                gaps.append(f"{section} missing input: {missing}")
            if payload.get("notes"):
                gaps.append(f"{section}: {payload['notes']}")
    return gaps


def render_report(report: dict[str, Any]) -> str:
    lines = ["# Prepared Dataset Gaps", ""]
    for section, payload in report.items():
        if section == "gaps":
            continue
        lines.append(f"## {section}")
        lines.append("")
        if isinstance(payload, dict):
            lines.append(f"- rows: {payload.get('rows', 0)}")
            if payload.get("notes"):
                lines.append(f"- notes: {payload['notes']}")
        lines.append("")
    lines.append("## Gaps")
    lines.append("")
    for gap in report.get("gaps", []):
        lines.append(f"- {gap}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", type=Path, default=PROCESSED_DATASETS)
    parser.add_argument("--output-dir", type=Path, default=TRAINING_OUTPUT)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    args = parser.parse_args()
    report = build_all_manifests(args.processed_root, args.output_dir, args.reports_dir)
    print(json.dumps({"gaps": report["gaps"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
