from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class DatasetCard(BaseModel):
    dataset_name: str
    source_catalog: str
    source_url: str
    access_date: str
    license: str
    permitted_usage: str
    commercial_use_allowed: bool
    redistribution_allowed: bool
    derivative_use_allowed: bool
    regions: list[str]
    countries: list[str]
    languages: list[str]
    modality: str
    task_mapping: list[str] = Field(min_length=1)
    pii_sensitive_assessment: str
    preprocessing_required: list[str] = Field(default_factory=list)
    split_usage: list[str] = Field(default_factory=list)
    known_limitations: list[str] = Field(default_factory=list)
    approval_status: str
    reviewer_notes: str = ""


def render_dataset_card(card: DatasetCard) -> str:
    return "\n".join(
        [
            f"# {card.dataset_name}",
            "",
            f"- source_catalog: {card.source_catalog}",
            f"- source_url: {card.source_url}",
            f"- access_date: {card.access_date}",
            f"- license: {card.license}",
            f"- permitted_usage: {card.permitted_usage}",
            f"- commercial_use_allowed: {card.commercial_use_allowed}",
            f"- redistribution_allowed: {card.redistribution_allowed}",
            f"- derivative_use_allowed: {card.derivative_use_allowed}",
            "- region/country/language coverage: "
            f"{', '.join(card.regions)} | "
            f"{', '.join(card.countries)} | "
            f"{', '.join(card.languages)}",
            f"- modality: {card.modality}",
            f"- task mapping: {', '.join(card.task_mapping)}",
            f"- PII/sensitive-data assessment: {card.pii_sensitive_assessment}",
            f"- preprocessing required: {', '.join(card.preprocessing_required) or 'none'}",
            f"- split usage: {', '.join(card.split_usage) or 'not assigned'}",
            f"- known limitations: {', '.join(card.known_limitations) or 'none recorded'}",
            f"- approval status: {card.approval_status}",
            f"- reviewer notes: {card.reviewer_notes or 'none'}",
            "",
        ]
    )


def card_from_payload(
    payload: dict[str, Any],
    mapping: dict[str, Any] | None = None,
) -> DatasetCard:
    acceptance = payload.get("acceptance", {})
    if not isinstance(acceptance, dict):
        acceptance = {}
    mapped_tasks = []
    split_usage = []
    if mapping:
        mapped_tasks = [str(task) for task in mapping.get("mapped_tasks", [])]
        split_usage = [str(split) for split in mapping.get("split_usage", [])]
    if not mapped_tasks:
        mapped_tasks = [str(task) for task in payload.get("candidate_tasks", [])]
    return DatasetCard(
        dataset_name=str(payload["dataset_name"]),
        source_catalog=str(payload["source_catalog"]),
        source_url=str(payload["source_url"]),
        access_date=str(payload["access_date"]),
        license=str(payload["license_name"]),
        permitted_usage=str(acceptance.get("effective_use") or payload.get("intended_use")),
        commercial_use_allowed=bool(payload["commercial_use_allowed"]),
        redistribution_allowed=bool(payload["redistribution_allowed"]),
        derivative_use_allowed=bool(payload["derivative_use_allowed"]),
        regions=[str(item) for item in payload["regions"]],
        countries=[str(item) for item in payload["countries"]],
        languages=[str(item) for item in payload["languages"]],
        modality=str(payload["modality"]),
        task_mapping=mapped_tasks,
        pii_sensitive_assessment=(
            f"pii_risk={payload['pii_risk']}; "
            f"contains_sensitive_data={payload.get('contains_sensitive_data', False)}"
        ),
        preprocessing_required=_preprocessing(payload),
        split_usage=split_usage,
        known_limitations=_limitations(payload),
        approval_status=str(acceptance.get("status") or payload.get("approval_status", "unknown")),
        reviewer_notes=str(payload.get("reviewer_notes") or payload.get("notes") or ""),
    )


def write_dataset_card(path: Path, card: DatasetCard) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_dataset_card(card), encoding="utf-8")


def slugify(value: str) -> str:
    chars = [char.lower() if char.isalnum() else "-" for char in value]
    slug = "".join(chars).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "dataset"


def _preprocessing(payload: dict[str, Any]) -> list[str]:
    steps = ["verify license evidence", "preserve regional metadata"]
    modality = str(payload.get("modality", ""))
    if modality in {"document_image", "pdf", "image"}:
        steps.append("local OCR/image preprocessing")
    if modality == "audio":
        steps.append("local transcript validation")
    if payload.get("size_bytes") and int(payload["size_bytes"]) >= 1_000_000_000:
        steps.append("large download approval and resumable checkpoint")
    return steps


def _limitations(payload: dict[str, Any]) -> list[str]:
    limitations = []
    if payload.get("cloud_runtime_dependency"):
        limitations.append("must not be used at runtime")
    if payload.get("pii_risk") in {"medium", "high"}:
        limitations.append("requires PII review before training")
    if not payload.get("local_path"):
        limitations.append("metadata-only until explicitly downloaded")
    return limitations
