from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from data.schemas.source_registry import read_jsonl, validate_candidate_records
from scripts.approve_dataset import approve_dataset

REQUIRED_FIELDS = {
    "dataset_id",
    "dataset_name",
    "source_catalog",
    "source_url",
    "mirror_url",
    "dataset_version",
    "access_date",
    "discovered_by",
    "modality",
    "candidate_tasks",
    "regions",
    "countries",
    "languages",
    "license_name",
    "license_url",
    "commercial_use_allowed",
    "redistribution_allowed",
    "derivative_use_allowed",
    "pii_risk",
    "contains_sensitive_data",
    "source_quality",
    "intended_use",
    "local_path",
    "checksum",
    "size_bytes",
    "notes",
}


def test_discovery_creates_sample_registry_with_at_least_30_candidates(tmp_path: Path) -> None:
    output = tmp_path / "dataset_candidates.jsonl"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/discover_datasets.py",
            "--output",
            str(output),
            "--max-results",
            "100",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    records = read_jsonl(output)
    candidates = validate_candidate_records(records)
    assert "candidate records" in result.stdout
    assert len(candidates) >= 30
    assert (tmp_path / "dataset_candidates.large_downloads.json").exists()
    for record in records:
        assert set(record) >= REQUIRED_FIELDS


def test_audit_writes_approval_outputs_and_reports(tmp_path: Path) -> None:
    candidates_path = tmp_path / "dataset_candidates.jsonl"
    approved_path = tmp_path / "approved_datasets.jsonl"
    research_eval_path = tmp_path / "research_eval_datasets.jsonl"
    rejected_path = tmp_path / "rejected_datasets.jsonl"
    reports_dir = tmp_path / "reports"

    subprocess.run(
        [
            sys.executable,
            "scripts/discover_datasets.py",
            "--output",
            str(candidates_path),
            "--max-results",
            "100",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = subprocess.run(
        [
            sys.executable,
            "scripts/audit_source_registry.py",
            "--candidates",
            str(candidates_path),
            "--approved",
            str(approved_path),
            "--research-eval",
            str(research_eval_path),
            "--rejected",
            str(rejected_path),
            "--reports-dir",
            str(reports_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    for section in ["source", "task", "region", "language", "modality", "license", "pii_risk"]:
        assert f"counts by {section}:" in result.stdout

    report = json.loads((reports_dir / "dataset_registry_audit.json").read_text(encoding="utf-8"))
    assert report["total_candidates"] >= 30
    assert report["approved"] > 0
    assert report["research_eval_only"] > 0
    assert report["rejected"] > 0
    assert approved_path.exists()
    assert research_eval_path.exists()
    assert rejected_path.exists()
    assert (reports_dir / "dataset_registry_audit.md").exists()
    assert any(record["source_catalog"] == "epo" for record in read_jsonl(research_eval_path))
    assert any(record["acceptance"]["status"] == "rejected" for record in read_jsonl(rejected_path))


def test_manual_approval_refuses_eval_only_dataset(tmp_path: Path) -> None:
    candidates_path = tmp_path / "dataset_candidates.jsonl"
    approved_path = tmp_path / "approved_datasets.jsonl"
    record = {
        "dataset_id": "manual:noncommercial",
        "source_catalog": "manual",
        "dataset_name": "noncommercial sample",
        "source_url": "https://example.com/noncommercial",
        "modality": "text",
        "candidate_tasks": ["text_summarization"],
        "regions": ["Europe"],
        "countries": ["France"],
        "languages": ["fr"],
        "license_name": "CC-BY-NC-4.0",
        "license_url": "https://creativecommons.org/licenses/by-nc/4.0/",
        "commercial_use_allowed": False,
        "redistribution_allowed": True,
        "derivative_use_allowed": True,
        "pii_risk": "low",
        "intended_use": "training",
        "source_quality": "high",
        "notes": "unit test",
    }
    candidates_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not training-approvable"):
        approve_dataset("noncommercial sample", candidates_path, approved_path)

    assert not approved_path.exists()
