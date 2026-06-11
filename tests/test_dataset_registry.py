from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from data.schemas.source_registry import read_jsonl, validate_candidate_records
from scripts.approve_dataset import approve_dataset

REQUIRED_FIELDS = {
    "source_catalog",
    "dataset_name",
    "dataset_url",
    "modality",
    "candidate_tasks",
    "regions",
    "countries",
    "languages",
    "license_name",
    "commercial_use_allowed",
    "redistribution_allowed",
    "derivative_use_allowed",
    "pii_risk",
    "intended_use",
    "source_quality",
    "notes",
}


def test_discovery_creates_sample_registry_with_20_candidates(tmp_path: Path) -> None:
    output = tmp_path / "dataset_candidates.jsonl"

    result = subprocess.run(
        [sys.executable, "scripts/discover_datasets.py", "--output", str(output)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "wrote 20 candidate records" in result.stdout
    records = read_jsonl(output)
    candidates = validate_candidate_records(records)
    assert len(candidates) == 20
    for record in records:
        assert set(record) >= REQUIRED_FIELDS


def test_audit_prints_required_counts_and_writes_reports(tmp_path: Path) -> None:
    candidates_path = tmp_path / "dataset_candidates.jsonl"
    approved_path = tmp_path / "approved_datasets.jsonl"
    research_eval_path = tmp_path / "research_eval_datasets.jsonl"
    rejected_path = tmp_path / "rejected_datasets.jsonl"
    reports_dir = tmp_path / "reports"

    subprocess.run(
        [sys.executable, "scripts/discover_datasets.py", "--output", str(candidates_path)],
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
    assert report["total_candidates"] == 20
    assert "approved" in report
    assert "research_eval_only" in report
    assert approved_path.exists()
    assert research_eval_path.exists()
    assert rejected_path.exists()
    assert (reports_dir / "dataset_registry_audit.md").exists()

    approved = read_jsonl(approved_path)
    limited = read_jsonl(research_eval_path)
    rejected = read_jsonl(rejected_path)
    assert not any(record["acceptance"]["status"] == "research_eval_only" for record in approved)
    assert all(record["acceptance"]["status"] == "research_eval_only" for record in limited)
    assert any("mixed" in record["license_name"].casefold() for record in rejected)


def test_manual_approval_refuses_research_eval_only_dataset(tmp_path: Path) -> None:
    candidates_path = tmp_path / "dataset_candidates.jsonl"
    approved_path = tmp_path / "approved_datasets.jsonl"

    record = {
        "source_catalog": "unit-test",
        "dataset_name": "noncommercial sample",
        "dataset_url": "https://example.com/noncommercial",
        "modality": "text",
        "candidate_tasks": ["summarization"],
        "regions": ["Europe"],
        "countries": ["France"],
        "languages": ["fr"],
        "license_name": "CC-BY-NC-4.0",
        "commercial_use_allowed": False,
        "redistribution_allowed": True,
        "derivative_use_allowed": True,
        "pii_risk": "low",
        "intended_use": "training",
        "source_quality": "high",
        "notes": "unit test",
        "metadata_only": True,
    }
    candidates_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="research/eval-only"):
        approve_dataset("noncommercial sample", candidates_path, approved_path)

    assert not approved_path.exists()
