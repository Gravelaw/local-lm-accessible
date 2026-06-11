from __future__ import annotations

from services.gateway.schemas import DocumentExtractionOutput


def extract_pdf_placeholder(document_type: str) -> DocumentExtractionOutput:
    return DocumentExtractionOutput(
        document_type=document_type,
        fields={},
        raw_text="",
        confidence=0.0,
        warnings=["PDF extraction implementation is pending."],
        human_review_required=True,
    )
