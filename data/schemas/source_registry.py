from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

UNKNOWN_LICENSE_VALUES = {"", "unknown", "unk", "n/a", "na", "none", "tbd", "unlicensed"}
NON_COMMERCIAL_MARKERS = ("non-commercial", "noncommercial", "cc-by-nc", "by-nc")
AMBIGUOUS_LICENSE_MARKERS = (
    "mixed",
    "various",
    "multiple",
    "per-file",
    "terms",
    "terms of use",
    "marketplace terms",
    "research-friendly",
    "custom",
    "review required",
)


class SourceCatalog(StrEnum):
    HUGGINGFACE = "huggingface"
    KAGGLE = "kaggle"
    AWS_OPEN_DATA_REGISTRY = "aws_open_data_registry"
    GOOGLE_PUBLIC_DATASETS = "google_public_datasets"
    AZURE_OPEN_DATASETS = "azure_open_datasets"
    UCI_ML_REPOSITORY = "uci_ml_repository"
    EUROPEAN_DATA_PORTAL = "european_data_portal"
    EPO = "epo"
    AWESOME_PUBLIC_DATASETS = "awesome_public_datasets"
    GOOGLE_DATASET_SEARCH = "google_dataset_search"
    WIKIMEDIA = "wikimedia"
    MANUAL = "manual"
    SYNTHETIC = "synthetic"
    USER_OPT_IN_REDACTED = "user_opt_in_redacted"


class Modality(StrEnum):
    TEXT = "text"
    IMAGE = "image"
    DOCUMENT_IMAGE = "document_image"
    PDF = "pdf"
    AUDIO = "audio"
    TABULAR = "tabular"
    HTML = "html"
    MIXED = "mixed"


class CandidateTask(StrEnum):
    TEXT_SUMMARIZATION = "text_summarization"
    WIKIPEDIA_SUMMARIZATION = "wikipedia_summarization"
    WEB_PAGE_SUMMARIZATION = "web_page_summarization"
    DOCUMENT_OCR = "document_ocr"
    INVOICE_EXTRACTION = "invoice_extraction"
    RECEIPT_EXTRACTION = "receipt_extraction"
    BILL_EXTRACTION = "bill_extraction"
    BANK_STATEMENT_EXTRACTION = "bank_statement_extraction"
    HANDWRITTEN_NOTE_TRANSCRIPTION = "handwritten_note_transcription"
    IMAGE_ACCESSIBILITY_DESCRIPTION = "image_accessibility_description"
    IMAGE_TEXT_TRANSLATION = "image_text_translation"
    VISUAL_QUESTION_ANSWERING = "visual_question_answering"
    SPEECH_TO_TEXT = "speech_to_text"
    TABULAR_REASONING = "tabular_reasoning"
    TOOL_ROUTING = "tool_routing"
    JSON_REPAIR = "json_repair"
    EVAL_ONLY = "eval_only"
    BLOCKED = "blocked"


class PiiRisk(StrEnum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class IntendedUse(StrEnum):
    TRAINING = "training"
    VALIDATION = "validation"
    TEST = "test"
    EVAL = "eval"
    EVAL_ONLY = "eval_only"
    RESEARCH = "research"
    RESEARCH_EVAL = "research_eval"
    BLOCKED = "blocked"


class SourceQuality(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class DatasetCandidate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    dataset_id: str = Field(min_length=1)
    dataset_name: str = Field(min_length=1)
    source_catalog: SourceCatalog
    source_url: str = Field(min_length=1)
    mirror_url: str | None = None
    dataset_version: str = "unknown"
    access_date: str = Field(min_length=1)
    discovered_by: str = Field(min_length=1)
    modality: Modality
    candidate_tasks: list[CandidateTask] = Field(min_length=1)
    regions: list[str] = Field(min_length=1)
    countries: list[str] = Field(min_length=1)
    languages: list[str] = Field(min_length=1)
    license_name: str = Field(min_length=1)
    license_url: str | None = None
    commercial_use_allowed: bool
    redistribution_allowed: bool
    derivative_use_allowed: bool
    pii_risk: PiiRisk
    contains_sensitive_data: bool = False
    source_quality: SourceQuality
    intended_use: IntendedUse
    local_path: str | None = None
    checksum: str | None = None
    size_bytes: int | None = Field(default=None, ge=0)
    notes: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    redacted: bool = False
    explicit_user_opt_in: bool = False
    metadata_only: bool = True
    cloud_runtime_dependency: bool = False
    download_approved: bool = False

    @field_validator("license_name")
    @classmethod
    def reject_unknown_license_name(cls, value: str) -> str:
        if value.strip().casefold() in UNKNOWN_LICENSE_VALUES:
            raise ValueError("unknown license is rejected")
        return value

    @field_validator("candidate_tasks", "regions", "countries", "languages")
    @classmethod
    def reject_blank_list_items(cls, values: list[Any]) -> list[Any]:
        if any(not str(value).strip() for value in values):
            raise ValueError("list fields must not contain blank values")
        return values

    @model_validator(mode="after")
    def validate_discovery_record(self) -> DatasetCandidate:
        if self.source_catalog == SourceCatalog.KAGGLE and not str(
            self.metadata.get("kaggle_slug", "")
        ).strip():
            raise ValueError("Kaggle datasets must record kaggle_slug")
        if self.source_catalog == SourceCatalog.HUGGINGFACE and not str(
            self.metadata.get("hf_repo_id", "")
        ).strip():
            raise ValueError("Hugging Face datasets must record hf_repo_id")
        if self.source_catalog in {
            SourceCatalog.AWS_OPEN_DATA_REGISTRY,
            SourceCatalog.GOOGLE_PUBLIC_DATASETS,
            SourceCatalog.AZURE_OPEN_DATASETS,
        } and self.cloud_runtime_dependency:
            raise ValueError("cloud catalogs must not become runtime dependencies")
        if self.source_catalog == SourceCatalog.EUROPEAN_DATA_PORTAL and not str(
            self.metadata.get("publisher", "")
        ).strip():
            raise ValueError("European Data Portal records must preserve publisher metadata")
        return self

    @property
    def dataset_url(self) -> str:
        return self.source_url

    @property
    def is_non_commercial(self) -> bool:
        normalized = self.license_name.casefold()
        return not self.commercial_use_allowed or any(
            marker in normalized for marker in NON_COMMERCIAL_MARKERS
        )

    @property
    def has_ambiguous_license(self) -> bool:
        normalized = self.license_name.casefold()
        return any(marker in normalized for marker in AMBIGUOUS_LICENSE_MARKERS)

    @property
    def is_large(self) -> bool:
        return self.size_bytes is not None and self.size_bytes >= 1_000_000_000


def normalize_candidate_record(record: dict[str, Any]) -> dict[str, Any]:
    payload = dict(record)
    if "source_url" not in payload and "dataset_url" in payload:
        payload["source_url"] = payload.pop("dataset_url")
    if "dataset_id" not in payload:
        source = str(payload.get("source_catalog", "manual")).replace("-", "_")
        name = str(payload.get("dataset_name", "dataset")).lower().replace(" ", "_")
        payload["dataset_id"] = f"{source}:{name}"
    payload.setdefault("mirror_url", None)
    payload.setdefault("dataset_version", "unknown")
    payload.setdefault("access_date", "2026-06-13")
    payload.setdefault("discovered_by", "local-lm offline seed")
    payload.setdefault("license_url", None)
    payload.setdefault("contains_sensitive_data", False)
    payload.setdefault("local_path", None)
    payload.setdefault("checksum", None)
    payload.setdefault("size_bytes", None)
    payload.setdefault("notes", "")
    payload.setdefault("metadata", {})
    payload.setdefault("redacted", False)
    payload.setdefault("explicit_user_opt_in", False)
    payload.setdefault("metadata_only", True)
    payload.setdefault("cloud_runtime_dependency", False)
    payload.setdefault("download_approved", False)
    return payload


def validate_candidate_records(records: Iterable[dict[str, Any]]) -> list[DatasetCandidate]:
    return [
        DatasetCandidate.model_validate(normalize_candidate_record(record))
        for record in records
    ]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as jsonl_file:
        for line_number, line in enumerate(jsonl_file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"JSONL record must be an object at {path}:{line_number}")
            records.append(record)
    return records


def write_jsonl(path: Path, records: Iterable[BaseModel | dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as jsonl_file:
        for record in records:
            payload = record.model_dump(mode="json") if isinstance(record, BaseModel) else record
            jsonl_file.write(json.dumps(payload, sort_keys=True) + "\n")


def append_jsonl(path: Path, records: Iterable[BaseModel | dict[str, Any]]) -> None:
    existing = read_jsonl(path)
    existing.extend(
        record.model_dump(mode="json") if isinstance(record, BaseModel) else record
        for record in records
    )
    write_jsonl(path, existing)


def count_by(records: Iterable[DatasetCandidate], field_name: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for record in records:
        value = getattr(record, field_name)
        if isinstance(value, list):
            counter.update(str(item) for item in value)
        else:
            counter[str(value)] += 1
    return dict(sorted(counter.items()))


def dedupe_candidates(records: Iterable[DatasetCandidate]) -> list[DatasetCandidate]:
    by_key: dict[str, DatasetCandidate] = {}
    for record in records:
        by_key[record.dataset_id] = record
    return list(by_key.values())
