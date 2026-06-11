from __future__ import annotations

from pathlib import Path

import pytest

from services.gateway.schemas import BankStatementExtractionOutput, InvoiceExtractionOutput
from services.tools.document_extract import (
    extract_bank_statement_text,
    extract_invoice_text,
    extract_local_document,
    extract_vision_document_json,
    extraction_rows,
)


def test_extract_invoice_text_reconciles_sample_totals() -> None:
    text = Path("samples/demo/invoice_sample.txt").read_text(encoding="utf-8")

    extraction = extract_invoice_text(text)

    assert isinstance(extraction, InvoiceExtractionOutput)
    assert extraction.fields["vendor"] == "Sunrise Home Services"
    assert extraction.fields["gstin"] == "29ABCDE1234F1Z5"
    assert extraction.currency == "INR"
    assert extraction.subtotal == 1500.0
    assert extraction.tax_amount == 270.0
    assert extraction.total == 1770.0
    assert extraction.human_review_required is True
    assert len(extraction.line_items) == 2


def test_extract_invoice_text_rejects_mismatched_total() -> None:
    text = Path("samples/demo/invoice_sample.txt").read_text(encoding="utf-8")
    bad_text = text.replace("Total: INR 1,770.00", "Total: INR 1,780.00")

    with pytest.raises(ValueError, match="invoice totals do not reconcile"):
        extract_invoice_text(bad_text)


def test_extract_bank_statement_text_requires_review_and_rows() -> None:
    text = Path("samples/demo/bank_statement_sample.txt").read_text(encoding="utf-8")

    extraction = extract_bank_statement_text(text)
    rows = extraction_rows(extraction)

    assert isinstance(extraction, BankStatementExtractionOutput)
    assert extraction.currency == "USD"
    assert extraction.human_review_required is True
    assert len(extraction.transactions) == 4
    assert rows[1]["description"] == "Grocery store"
    assert rows[1]["debit"] == 62.45
    assert rows[2]["credit"] == 900.0


def test_extract_local_document_dispatches_text_samples() -> None:
    invoice = extract_local_document(Path("samples/demo/invoice_sample.txt"))
    statement = extract_local_document(Path("samples/demo/bank_statement_sample.txt"))

    assert isinstance(invoice, InvoiceExtractionOutput)
    assert isinstance(statement, BankStatementExtractionOutput)


def test_extract_vision_document_json_validates_invoice_totals() -> None:
    model_text = """
    ```json
    {
      "document_type": "invoice",
      "fields": {"vendor": "Synthetic Vendor"},
      "line_items": [{"description": "service", "amount": 100.0}],
      "currency": "INR",
      "subtotal": 100.0,
      "tax_amount": 18.0,
      "total": 118.0,
      "raw_ocr_text": "Synthetic invoice",
      "confidence": 0.72,
      "warnings": []
    }
    ```
    """

    extraction = extract_vision_document_json(model_text)

    assert isinstance(extraction, InvoiceExtractionOutput)
    assert extraction.total == 118.0
    assert extraction.human_review_required is True
    assert "Local vision model extraction" in extraction.warnings[-1]


def test_extract_vision_document_json_rejects_invalid_json() -> None:
    with pytest.raises(ValueError, match="did not contain a JSON object"):
        extract_vision_document_json("not json")


def test_extract_vision_document_json_rejects_unreconciled_invoice() -> None:
    model_text = (
        '{"document_type":"invoice","currency":"INR","subtotal":100.0,'
        '"tax_amount":18.0,"total":119.0,"confidence":0.8}'
    )

    with pytest.raises(ValueError, match="invoice totals do not reconcile"):
        extract_vision_document_json(model_text)
