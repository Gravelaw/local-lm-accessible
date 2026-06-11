from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from data.schemas.dataset_acceptance import AcceptanceStatus, evaluate_candidate
from data.schemas.source_registry import read_jsonl, validate_candidate_records
from scripts.synthetic_documents import generate_documents, money


def test_bank_statement_generation_outputs_and_reconciles_balances(tmp_path: Path) -> None:
    records = generate_documents(
        kind="bank_statement",
        output_dir=tmp_path,
        count=8,
        seed=789,
        augment=False,
    )

    assert {record["region"] for record in records} == {
        "India",
        "Southeast Asia",
        "North America",
        "Europe",
    }

    for record in records:
        document_dir = tmp_path / "bank_statement" / record["document_id"]
        assert (document_dir / "rendered.png").exists()
        assert (document_dir / "document.pdf").exists()
        assert (document_dir / "source.html").exists()
        assert (document_dir / "ground_truth.json").exists()
        assert (document_dir / "raw_ocr.txt").exists()
        assert (document_dir / "expected_excel_rows.xlsx").exists()

        balance = Decimal(record["starting_balance"])
        for transaction in record["transactions"]:
            balance = money(
                balance - Decimal(transaction["debit"]) + Decimal(transaction["credit"])
            )
            assert Decimal(transaction["balance"]) == balance

        assert Decimal(record["ending_balance"]) == balance
        assert record["account_number"].startswith("SYN-ACCT-****")
        assert record["synthetic"] is True
        assert record["pii"] == "synthetic"


def test_synthetic_bank_statement_registry_records_validate_and_approve(tmp_path: Path) -> None:
    generate_documents(
        kind="bank_statement",
        output_dir=tmp_path,
        count=2,
        regions=("North America",),
        seed=2026,
        augment=False,
    )

    registry_path = tmp_path / "synthetic_dataset_candidates.jsonl"
    candidates = validate_candidate_records(read_jsonl(registry_path))

    assert len(candidates) == 2
    for candidate in candidates:
        assert candidate.license_name == "CC0-1.0"
        assert candidate.modality == "image"
        assert candidate.candidate_tasks == ["bank_statement_extraction"]
        assert candidate.regions == ["North America"]
        assert candidate.pii_risk == "low"
        assert "pii_status=synthetic" in candidate.notes
        assert evaluate_candidate(candidate).status == AcceptanceStatus.APPROVED
