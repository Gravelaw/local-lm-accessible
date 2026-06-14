from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from data.schemas.dataset_acceptance import AcceptanceStatus
from data.schemas.source_registry import CandidateTask, DatasetCandidate, Modality, SourceCatalog


class ProjectTask(StrEnum):
    TEXT_SUMMARIZATION = "text_summarization"
    TECHNICAL_SUMMARIZATION = "technical_summarization"
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
    EXCEL_EXPORT_TESTS = "excel_export_tests"
    EVAL_ONLY = "eval_only"


class DatasetTaskMapping(BaseModel):
    dataset_id: str
    dataset_name: str
    source_catalog: str
    mapped_tasks: list[ProjectTask] = Field(min_length=1)
    split_usage: list[str] = Field(min_length=1)
    eval_only: bool
    reasons: list[str] = Field(default_factory=list)


def map_candidate_to_tasks(
    candidate: DatasetCandidate,
    acceptance_status: str | AcceptanceStatus | None = None,
) -> DatasetTaskMapping:
    mapped: set[ProjectTask] = set()
    reasons: list[str] = []
    status = str(acceptance_status or "")
    eval_only = status in {
        AcceptanceStatus.EVAL_ONLY,
        AcceptanceStatus.RESEARCH_EVAL_ONLY,
        "eval_only",
    }

    for task in candidate.candidate_tasks:
        if task == CandidateTask.EVAL_ONLY or task == CandidateTask.BLOCKED:
            mapped.add(ProjectTask.EVAL_ONLY)
            eval_only = True
        else:
            mapped.add(ProjectTask(str(task)))

    if (
        candidate.modality in {Modality.TEXT, Modality.HTML, Modality.PDF}
        and candidate.derivative_use_allowed
    ):
        mapped.add(ProjectTask.TEXT_SUMMARIZATION)
    if candidate.modality in {Modality.DOCUMENT_IMAGE, Modality.PDF}:
        mapped.add(ProjectTask.DOCUMENT_OCR)
    if candidate.modality == Modality.TABULAR:
        mapped.update(
            {
                ProjectTask.TABULAR_REASONING,
                ProjectTask.EXCEL_EXPORT_TESTS,
                ProjectTask.JSON_REPAIR,
            }
        )
    if candidate.modality == Modality.AUDIO:
        if (
            "transcript" in candidate.notes.casefold()
            or CandidateTask.SPEECH_TO_TEXT in candidate.candidate_tasks
        ):
            mapped.add(ProjectTask.SPEECH_TO_TEXT)
        else:
            reasons.append("audio dataset needs transcripts before ASR use")
            eval_only = True
    if candidate.modality == Modality.IMAGE:
        if _image_has_caption_or_vqa(candidate):
            mapped.add(ProjectTask.IMAGE_ACCESSIBILITY_DESCRIPTION)
        else:
            reasons.append("image dataset needs captions or VQA labels for accessibility use")
            eval_only = True
    if candidate.source_catalog == SourceCatalog.EPO:
        mapped = {
            ProjectTask.TECHNICAL_SUMMARIZATION,
            ProjectTask.TEXT_SUMMARIZATION,
            ProjectTask.EVAL_ONLY,
        }
        eval_only = True
        reasons.append("EPO/patent data is restricted to technical summarization and eval")
    if (
        "health" in candidate.dataset_name.casefold()
        or "covid" in candidate.dataset_name.casefold()
    ):
        eval_only = True
        mapped.add(ProjectTask.EVAL_ONLY)
        reasons.append("healthcare-like datasets default to eval-only")
    if "online retail" in candidate.dataset_name.casefold():
        eval_only = True
        mapped.add(ProjectTask.EVAL_ONLY)

    split_usage = (
        ["eval_only"] if eval_only else ["train", "validation", "test", "regional_stress_test"]
    )
    if eval_only:
        mapped.add(ProjectTask.EVAL_ONLY)

    return DatasetTaskMapping(
        dataset_id=candidate.dataset_id,
        dataset_name=candidate.dataset_name,
        source_catalog=str(candidate.source_catalog),
        mapped_tasks=sorted(mapped),
        split_usage=split_usage,
        eval_only=eval_only,
        reasons=reasons or ["mapped by modality, labels, and acceptance status"],
    )


def _image_has_caption_or_vqa(candidate: DatasetCandidate) -> bool:
    labels = " ".join(
        [
            candidate.dataset_name,
            candidate.notes,
            " ".join(str(task) for task in candidate.candidate_tasks),
        ]
    ).casefold()
    return any(marker in labels for marker in ("caption", "vqa", "textvqa", "vizwiz", "label"))
