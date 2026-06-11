from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

from data.schemas.dataset_acceptance import AcceptanceStatus, evaluate_candidate
from data.schemas.source_registry import read_jsonl, validate_candidate_records
from scripts.synthetic_documents import generate_documents, money


def test_invoice_generation_outputs_and_reconciles_totals(tmp_path: Path) -> None:
    records = generate_documents(
        kind="invoice",
        output_dir=tmp_path,
        count=8,
        seed=123,
        augment=False,
    )

    assert {record["region"] for record in records} == {
        "India",
        "Southeast Asia",
        "North America",
        "Europe",
    }

    for record in records:
        document_dir = tmp_path / "invoice" / record["document_id"]
        assert (document_dir / "rendered.png").exists()
        assert (document_dir / "document.pdf").exists()
        assert (document_dir / "source.html").exists()
        assert (document_dir / "ground_truth.json").exists()
        assert (document_dir / "raw_ocr.txt").exists()
        assert (document_dir / "expected_excel_rows.xlsx").exists()

        subtotal = money(sum(Decimal(item["line_total"]) for item in record["items"]))
        tax = money(subtotal * Decimal(record["totals"]["tax_rate"]))
        secondary_tax = money(subtotal * Decimal(record["totals"]["secondary_tax_rate"]))
        total = money(subtotal + tax + secondary_tax)

        assert Decimal(record["totals"]["subtotal"]) == subtotal
        assert Decimal(record["totals"]["tax_amount"]) == tax
        assert Decimal(record["totals"]["secondary_tax_amount"]) == secondary_tax
        assert Decimal(record["totals"]["total"]) == total
        assert record["synthetic"] is True
        assert record["pii"] == "synthetic"


def test_india_invoice_contains_required_synthetic_tax_fields(tmp_path: Path) -> None:
    records = generate_documents(
        kind="invoice",
        output_dir=tmp_path,
        count=1,
        regions=("India",),
        seed=456,
        augment=False,
    )
    record = records[0]

    assert record["tax_id_label"] == "GSTIN"
    assert record["tax_id"].startswith("SYN")
    assert record["payment_reference"].startswith("UPI-SYN-")
    assert record["currency"] == "INR"
    assert record["totals"]["tax_label"] in {"CGST", "IGST"}
    assert "Hindi" in record["languages"]
    assert "Tamil" in record["languages"]
    assert all(item["tax_code"] for item in record["items"])


def test_synthetic_metadata_is_required_and_idempotent(tmp_path: Path) -> None:
    generate_documents(
        kind="invoice",
        output_dir=tmp_path,
        count=2,
        regions=("India",),
        seed=999,
        augment=False,
    )
    generate_documents(
        kind="invoice",
        output_dir=tmp_path,
        count=2,
        regions=("India",),
        seed=999,
        augment=False,
    )

    metadata_path = tmp_path / "metadata.jsonl"
    rows = [
        json.loads(line)
        for line in metadata_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(rows) == 2
    assert len({row["document_id"] for row in rows}) == 2
    for row in rows:
        assert row["license"] == "CC0-1.0"
        assert row["modality"] == "image"
        assert row["task"] == "invoice_extraction"
        assert row["pii_status"] == "synthetic"
        assert row["source_type"] == "synthetic"
        assert row["outputs"]["ground_truth"].endswith("ground_truth.json")


def test_synthetic_invoice_registry_records_validate_and_approve(tmp_path: Path) -> None:
    generate_documents(
        kind="invoice",
        output_dir=tmp_path,
        count=2,
        regions=("India",),
        seed=999,
        augment=False,
    )
    generate_documents(
        kind="invoice",
        output_dir=tmp_path,
        count=2,
        regions=("India",),
        seed=999,
        augment=False,
    )

    registry_path = tmp_path / "synthetic_dataset_candidates.jsonl"
    records = read_jsonl(registry_path)
    candidates = validate_candidate_records(records)

    assert len(candidates) == 2
    assert len({candidate.dataset_name for candidate in candidates}) == 2
    for candidate in candidates:
        assert candidate.source_catalog == "local-lm synthetic regional documents"
        assert candidate.license_name == "CC0-1.0"
        assert candidate.modality == "image"
        assert candidate.candidate_tasks == ["invoice_extraction"]
        assert candidate.regions == ["India"]
        assert candidate.countries == ["India"]
        assert "pii_status=synthetic" in candidate.notes
        assert evaluate_candidate(candidate).status == AcceptanceStatus.APPROVED
