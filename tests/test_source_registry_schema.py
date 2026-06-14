from __future__ import annotations

import pytest
from pydantic import ValidationError

from data.schemas.source_registry import DatasetCandidate


def _candidate(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "dataset_id": "huggingface:example",
        "dataset_name": "Example dataset",
        "source_catalog": "huggingface",
        "source_url": "https://huggingface.co/datasets/example/example",
        "access_date": "2026-06-13",
        "discovered_by": "unit test",
        "modality": "text",
        "candidate_tasks": ["text_summarization"],
        "regions": ["Europe"],
        "countries": ["France"],
        "languages": ["fr"],
        "license_name": "CC-BY-4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "commercial_use_allowed": True,
        "redistribution_allowed": True,
        "derivative_use_allowed": True,
        "pii_risk": "low",
        "source_quality": "high",
        "intended_use": "training",
        "notes": "unit test",
        "metadata": {"hf_repo_id": "example/example"},
    }
    values.update(overrides)
    return values


def test_candidate_schema_accepts_required_public_dataset_metadata() -> None:
    candidate = DatasetCandidate.model_validate(_candidate())

    assert candidate.dataset_id == "huggingface:example"
    assert candidate.source_catalog == "huggingface"
    assert candidate.dataset_url == candidate.source_url
    assert candidate.cloud_runtime_dependency is False


def test_kaggle_candidate_requires_slug() -> None:
    with pytest.raises(ValidationError, match="kaggle_slug"):
        DatasetCandidate.model_validate(
            _candidate(
                dataset_id="kaggle:missing-slug",
                source_catalog="kaggle",
                source_url="https://www.kaggle.com/datasets/example/missing-slug",
                metadata={},
            )
        )


def test_cloud_catalog_cannot_be_runtime_dependency() -> None:
    with pytest.raises(ValidationError, match="runtime dependencies"):
        DatasetCandidate.model_validate(
            _candidate(
                dataset_id="aws:catalog",
                source_catalog="aws_open_data_registry",
                source_url="https://registry.opendata.aws/",
                metadata={},
                cloud_runtime_dependency=True,
            )
        )


def test_european_data_portal_requires_publisher_metadata() -> None:
    with pytest.raises(ValidationError, match="publisher metadata"):
        DatasetCandidate.model_validate(
            _candidate(
                dataset_id="europe:catalog",
                source_catalog="european_data_portal",
                source_url="https://data.europa.eu/",
                metadata={},
            )
        )
