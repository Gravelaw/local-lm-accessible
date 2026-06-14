from __future__ import annotations

from pathlib import Path

from data.schemas.source_registry import read_jsonl, write_jsonl
from scripts.download_prepare_small_datasets import (
    approved_small_targets,
    assert_dataset_size_cap,
    ingest_dataset,
    load_checkpoint,
    prepare_fatura_invoice_extraction,
    prepare_sroie_receipt_extraction,
    prepare_xfund_form_understanding,
    write_report,
)


def _approved_record(dataset_id: str, **overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "dataset_id": dataset_id,
        "dataset_name": dataset_id,
        "source_catalog": "manual",
        "source_url": "https://example.com",
        "modality": "document_image",
        "candidate_tasks": ["document_ocr"],
        "regions": ["Europe"],
        "countries": ["France"],
        "languages": ["en"],
        "license_name": "CC-BY-4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "commercial_use_allowed": True,
        "redistribution_allowed": True,
        "derivative_use_allowed": True,
        "pii_risk": "low",
        "intended_use": "training",
        "source_quality": "high",
        "notes": "test",
        "metadata": {},
        "acceptance": {"status": "approved", "effective_use": "training"},
    }
    record.update(overrides)
    return record


def test_approved_small_targets_excludes_large_datasets(tmp_path: Path) -> None:
    approved = tmp_path / "approved.jsonl"
    write_jsonl(
        approved,
        [
            _approved_record("manual:CORD"),
            _approved_record("huggingface:facebook/textvqa", size_bytes=8_000_000_000),
        ],
    )

    targets = approved_small_targets(approved)

    assert [target["dataset_id"] for target in targets] == ["manual:CORD"]


def test_manual_target_writes_collection_manifest(tmp_path: Path) -> None:
    checkpoint = {"datasets": {}}
    result = ingest_dataset(
        _approved_record(
            "huggingface:google/fleurs",
            dataset_name="FLEURS",
            source_catalog="huggingface",
        ),
        checkpoint=checkpoint,
        raw_root=tmp_path / "raw",
        processed_root=tmp_path / "processed",
        checkpoint_path=tmp_path / "checkpoint.json",
        no_network=True,
        tabular_limit=5,
        synthetic_count_per_kind=1,
        hf_existing_file_threshold=200,
        max_dataset_size_bytes=10 * 1024 * 1024 * 1024,
    )

    assert result["status"] == "prepared"
    manifest = tmp_path / "raw" / "huggingface-google-fleurs" / "MANUAL_COLLECTION.json"
    assert manifest.exists()
    assert checkpoint["datasets"]["huggingface:google/fleurs"]["status"] == "prepared"


def test_synthetic_target_generates_training_jsonl(tmp_path: Path) -> None:
    checkpoint = {"datasets": {}}
    result = ingest_dataset(
        _approved_record(
            "synthetic:local-lm-regional-documents",
            dataset_name="local-lm synthetic regional documents",
            source_catalog="synthetic",
            source_url="file:data/synthetic",
            candidate_tasks=["invoice_extraction", "receipt_extraction"],
            regions=["India", "Southeast Asia", "North America", "Europe"],
            countries=["India", "Singapore", "United States", "France"],
            languages=["en", "hi", "fr"],
            license_name="CC0-1.0",
            license_url="https://creativecommons.org/publicdomain/zero/1.0/",
            pii_risk="none",
        ),
        checkpoint=checkpoint,
        raw_root=tmp_path / "raw",
        processed_root=tmp_path / "processed",
        checkpoint_path=tmp_path / "checkpoint.json",
        no_network=True,
        tabular_limit=5,
        synthetic_count_per_kind=1,
        hf_existing_file_threshold=200,
        max_dataset_size_bytes=10 * 1024 * 1024 * 1024,
    )

    output = (
        tmp_path
        / "processed"
        / "synthetic-local-lm-regional-documents"
        / "vision_document_extraction.jsonl"
    )
    rows = read_jsonl(output)
    assert result["status"] == "prepared"
    assert result["rows"] == 4
    assert len(rows) == 4
    assert {row["pii_status"] for row in rows} == {"synthetic"}


def test_network_target_fails_cleanly_when_network_disabled(tmp_path: Path) -> None:
    checkpoint = {"datasets": {}}
    result = ingest_dataset(
        _approved_record("uci:online-retail", dataset_name="UCI Online Retail"),
        checkpoint=checkpoint,
        raw_root=tmp_path / "raw",
        processed_root=tmp_path / "processed",
        checkpoint_path=tmp_path / "checkpoint.json",
        no_network=True,
        tabular_limit=5,
        synthetic_count_per_kind=1,
        hf_existing_file_threshold=200,
        max_dataset_size_bytes=10 * 1024 * 1024 * 1024,
    )

    assert result["status"] == "failed"
    assert "network disabled" in result["error"]
    assert checkpoint["datasets"]["uci:online-retail"]["status"] == "failed"


def test_sroie_preparation_pairs_images_with_entities(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    image_dir = raw / "train" / "img"
    entity_dir = raw / "train" / "entities"
    image_dir.mkdir(parents=True)
    entity_dir.mkdir(parents=True)
    (image_dir / "sample.jpg").write_bytes(b"fake image")
    (entity_dir / "sample.txt").write_text(
        '{"company": "Shop", "date": "2026-06-13", "address": "X", "total": "1.00"}',
        encoding="utf-8",
    )

    result = prepare_sroie_receipt_extraction(raw, processed)
    rows = read_jsonl(processed / "receipt_extraction.jsonl")

    assert result["rows"] == 1
    assert rows[0]["task"] == "receipt_extraction"
    assert rows[0]["country"] == "Malaysia"
    assert rows[0]["expected_output"]["total"] == "1.00"


def test_xfund_preparation_pairs_european_json_with_images(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    image_dir = raw / "extracted" / "de.train"
    image_dir.mkdir(parents=True)
    (image_dir / "de_train_0.jpg").write_bytes(b"fake image")
    (raw / "de.train.json").write_text(
        '{"documents":[{"id":"de_train_0","document":[{"text":"Name","label":"question"}]}]}',
        encoding="utf-8",
    )

    result = prepare_xfund_form_understanding(raw, processed)
    rows = read_jsonl(processed / "form_understanding.jsonl")

    assert result["rows"] == 1
    assert rows[0]["source_dataset"] == "XFUND"
    assert rows[0]["country"] == "Germany"


def test_fatura_preparation_pairs_csv_images_and_annotations(tmp_path: Path) -> None:
    base = tmp_path / "raw" / "extracted" / "FATURA" / "invoices_dataset_final"
    (base / "images").mkdir(parents=True)
    (base / "Annotations" / "Original_Format").mkdir(parents=True)
    (base / "images" / "invoice.jpg").write_bytes(b"fake image")
    (base / "Annotations" / "Original_Format" / "invoice.json").write_text(
        '{"TOTAL":{"text":"TOTAL : 1.00"}}',
        encoding="utf-8",
    )
    (base / "strat1_train.csv").write_text(
        "img_path,annot_path\ninvoice.jpg,invoice.json\n",
        encoding="utf-8",
    )

    result = prepare_fatura_invoice_extraction(tmp_path / "raw", tmp_path / "processed")
    rows = read_jsonl(tmp_path / "processed" / "invoice_extraction.jsonl")

    assert result["rows"] == 1
    assert rows[0]["source_dataset"] == "FATURA"
    assert rows[0]["expected_output"]["TOTAL"]["text"] == "TOTAL : 1.00"


def test_declared_dataset_size_cap_is_enforced(tmp_path: Path) -> None:
    dataset = _approved_record("manual:too-large", size_bytes=11 * 1024 * 1024 * 1024)

    try:
        assert_dataset_size_cap(
            dataset,
            tmp_path / "raw",
            tmp_path / "processed",
            max_dataset_size_bytes=10 * 1024 * 1024 * 1024,
        )
    except ValueError as exc:
        assert "above cap" in str(exc)
    else:
        raise AssertionError("expected dataset size cap failure")


def test_local_dataset_size_cap_is_enforced(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "oversized.bin").write_bytes(b"123456")

    try:
        assert_dataset_size_cap(
            _approved_record("manual:local-too-large"),
            raw,
            tmp_path / "processed",
            max_dataset_size_bytes=5,
        )
    except ValueError as exc:
        assert "local files" in str(exc)
    else:
        raise AssertionError("expected local dataset size cap failure")


def test_checkpoint_and_report_are_valid_json(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoint.json"
    assert load_checkpoint(checkpoint_path) == {"datasets": {}}
    report = write_report(
        [{"dataset_id": "manual:CORD", "status": "prepared", "rows": 0, "outputs": []}],
        tmp_path / "reports",
    )

    assert report["prepared"] == 1
    assert (tmp_path / "reports" / "small_dataset_ingestion.json").exists()
    assert (tmp_path / "reports" / "small_dataset_ingestion.md").exists()
