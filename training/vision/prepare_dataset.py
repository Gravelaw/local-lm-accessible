from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

ALLOWED_REGIONS = {"India", "Southeast Asia", "North America", "Europe"}
ALLOWED_TASKS = {
    "invoice_extraction",
    "receipt_extraction",
    "bill_extraction",
    "bank_statement_extraction",
    "handwritten_note_transcription",
    "image_accessibility_description",
    "image_translation",
    "ocr_text_extraction",
}
UNKNOWN_LICENSE_VALUES = {"", "unknown", "unk", "n/a", "na", "none", "tbd", "unlicensed"}
PROTECTED_DOCUMENT_TASKS = {
    "invoice_extraction",
    "receipt_extraction",
    "bill_extraction",
    "bank_statement_extraction",
}
ACCEPTED_PROTECTED_PII = {"synthetic", "redacted", "explicit_user_opt_in"}


class VisionTrainingRecord(BaseModel):
    image_path: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    expected_output: dict[str, Any] | str = Field(default_factory=dict)
    region: str = Field(min_length=1)
    country: str = Field(min_length=1)
    language: str = Field(min_length=1)
    task: str = Field(min_length=1)
    source_type: str = Field(min_length=1)
    license: str = Field(min_length=1)
    pii_status: Literal["none", "synthetic", "redacted", "explicit_user_opt_in"]
    modality: Literal["image"] = "image"
    document_type: str | None = None
    human_review_required: bool = False

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

    @model_validator(mode="after")
    def enforce_sensitive_document_policy(self) -> VisionTrainingRecord:
        if self.task in PROTECTED_DOCUMENT_TASKS and self.pii_status not in ACCEPTED_PROTECTED_PII:
            raise ValueError(
                "financial document records require synthetic, redacted, "
                "or explicit user opt-in PII"
            )
        if self.task == "bank_statement_extraction":
            self.human_review_required = True
        return self


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


def validate_records(
    records_or_path: list[dict[str, Any]] | Path,
    *,
    limit: int | None = None,
    require_images: bool = False,
    base_dir: Path | None = None,
) -> list[VisionTrainingRecord]:
    raw_records = (
        load_jsonl(records_or_path, limit=limit)
        if isinstance(records_or_path, Path)
        else records_or_path[:limit]
    )
    validated: list[VisionTrainingRecord] = []
    missing_images: list[str] = []
    image_base_dir = base_dir or (
        records_or_path.parent if isinstance(records_or_path, Path) else Path.cwd()
    )

    for index, record in enumerate(raw_records, start=1):
        try:
            item = VisionTrainingRecord.model_validate(record)
        except ValidationError as exc:
            raise ValueError(f"invalid vision training record #{index}: {exc}") from exc
        if require_images:
            image_path = Path(item.image_path)
            resolved = image_path if image_path.is_absolute() else image_base_dir / image_path
            if not resolved.exists():
                missing_images.append(item.image_path)
        validated.append(item)

    if missing_images:
        preview = ", ".join(missing_images[:5])
        raise FileNotFoundError(f"missing image files: {preview}")
    return validated


def count_by(records: list[VisionTrainingRecord], field_name: str) -> dict[str, int]:
    counts = Counter(str(getattr(record, field_name)) for record in records)
    return dict(sorted(counts.items()))
