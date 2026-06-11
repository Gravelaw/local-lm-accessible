from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator

UNKNOWN_LICENSE_VALUES = {"", "unknown", "unk", "n/a", "na", "none", "tbd", "unlicensed"}
NON_COMMERCIAL_MARKERS = ("non-commercial", "noncommercial", "nc", "cc-by-nc")
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


class PiiRisk(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class IntendedUse(StrEnum):
    TRAINING = "training"
    EVAL = "eval"
    RESEARCH = "research"
    RESEARCH_EVAL = "research_eval"


class SourceQuality(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class DatasetCandidate(BaseModel):
    source_catalog: str = Field(min_length=1)
    dataset_name: str = Field(min_length=1)
    dataset_url: HttpUrl
    modality: str = Field(min_length=1)
    candidate_tasks: list[str] = Field(min_length=1)
    regions: list[str] = Field(min_length=1)
    countries: list[str] = Field(min_length=1)
    languages: list[str] = Field(min_length=1)
    license_name: str = Field(min_length=1)
    commercial_use_allowed: bool
    redistribution_allowed: bool
    derivative_use_allowed: bool
    pii_risk: PiiRisk
    intended_use: IntendedUse
    source_quality: SourceQuality
    notes: str = ""
    redacted: bool = False
    explicit_user_opt_in: bool = False
    metadata_only: bool = True
    cloud_hosted: bool = False

    @field_validator("license_name")
    @classmethod
    def reject_unknown_license_name(cls, value: str) -> str:
        if value.strip().lower() in UNKNOWN_LICENSE_VALUES:
            raise ValueError("unknown license is rejected")
        return value

    @field_validator("candidate_tasks", "regions", "countries", "languages")
    @classmethod
    def reject_blank_list_items(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("list fields must not contain blank values")
        return values

    @model_validator(mode="after")
    def require_metadata_only_discovery(self) -> DatasetCandidate:
        if not self.metadata_only:
            raise ValueError("dataset discovery records must be metadata-only")
        return self

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


def validate_candidate_records(records: Iterable[dict[str, Any]]) -> list[DatasetCandidate]:
    return [DatasetCandidate.model_validate(record) for record in records]


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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as jsonl_file:
        for record in records:
            payload = record.model_dump(mode="json") if isinstance(record, BaseModel) else record
            jsonl_file.write(json.dumps(payload, sort_keys=True) + "\n")


def count_by(records: Iterable[DatasetCandidate], field_name: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for record in records:
        value = getattr(record, field_name)
        if isinstance(value, list):
            counter.update(value)
        else:
            counter[str(value)] += 1
    return dict(sorted(counter.items()))


def dedupe_candidates(records: Iterable[DatasetCandidate]) -> list[DatasetCandidate]:
    by_key: dict[tuple[str, str], DatasetCandidate] = {}
    for record in records:
        key = (record.source_catalog, record.dataset_name)
        by_key[key] = record
    return list(by_key.values())
