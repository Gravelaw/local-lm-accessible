from __future__ import annotations

import pytest
from pydantic import ValidationError

from services.gateway.schemas import (
    BankStatementExtractionOutput,
    DatasetMetadata,
    DocumentExtractionOutput,
    ImageAccessibilityOutput,
    ImageTranslationOutput,
    InvoiceExtractionOutput,
    RuntimeConfig,
)


def test_runtime_config_rejects_remote_flags() -> None:
    with pytest.raises(ValidationError, match="allow_remote_inference"):
        RuntimeConfig.model_validate(
            {
                "local_only": True,
                "allow_remote_inference": True,
                "allow_remote_file_uploads": False,
                "allow_external_apis": False,
                "telemetry_enabled": False,
            }
        )


def test_dataset_metadata_rejects_unknown_license() -> None:
    with pytest.raises(ValidationError, match="unknown or missing license"):
        DatasetMetadata.model_validate(
            {
                "name": "bad_dataset",
                "license": "unknown",
                "region": "Europe",
                "country": "France",
                "language": "fr",
                "modality": "text",
                "task": "assistant",
                "pii_metadata": {
                    "contains_pii": False,
                    "source": "synthetic",
                    "handling": "no PII",
                },
            }
        )


def test_document_extraction_uses_pydantic_schema() -> None:
    output = DocumentExtractionOutput(
        document_type="synthetic_invoice",
        fields={"total": "10.00"},
        raw_text="total 10.00",
        confidence=0.5,
        warnings=["Synthetic example only."],
        human_review_required=True,
    )

    assert output.model_dump()["fields"]["total"] == "10.00"
    assert output.human_review_required is True


def test_invoice_schema_reconciles_totals() -> None:
    output = InvoiceExtractionOutput(
        document_type="invoice",
        subtotal=100.0,
        tax_amount=18.0,
        total=118.0,
        currency="INR",
        confidence=0.9,
    )

    assert output.total == 118.0


def test_invoice_schema_rejects_mismatched_totals() -> None:
    with pytest.raises(ValidationError, match="invoice totals do not reconcile"):
        InvoiceExtractionOutput(
            document_type="invoice",
            subtotal=100.0,
            tax_amount=18.0,
            total=119.0,
            currency="INR",
            confidence=0.9,
        )


def test_bank_statement_schema_always_requires_review() -> None:
    with pytest.raises(ValidationError, match="bank statements must require human review"):
        BankStatementExtractionOutput(confidence=0.8, human_review_required=False)

    output = BankStatementExtractionOutput(confidence=0.8)
    assert output.human_review_required is True


def test_image_output_schemas_are_structured() -> None:
    accessibility = ImageAccessibilityOutput(
        short_description="A printed notice on a wall.",
        visible_text=["Clinic open 9 AM"],
        possible_hazards=[],
        uncertainties=["Small print may be incomplete."],
        spoken_response="A notice says the clinic opens at 9 AM.",
        confidence=0.7,
    )
    translation = ImageTranslationOutput(
        original_text=["Bonjour"],
        translated_text="Hello",
        target_language="English",
        confidence=0.8,
    )

    assert accessibility.visible_text == ["Clinic open 9 AM"]
    assert translation.translated_text == "Hello"
