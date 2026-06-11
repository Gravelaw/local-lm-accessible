from __future__ import annotations

import pytest
from pydantic import ValidationError

from data.schemas.dataset_acceptance import AcceptanceStatus, evaluate_candidate
from data.schemas.source_registry import DatasetCandidate


def _candidate(**overrides: object) -> DatasetCandidate:
    values = {
        "source_catalog": "unit-test",
        "dataset_name": "unit dataset",
        "dataset_url": "https://example.com/dataset",
        "modality": "text",
        "candidate_tasks": ["qa"],
        "regions": ["Europe"],
        "countries": ["France"],
        "languages": ["fr"],
        "license_name": "CC-BY-4.0",
        "commercial_use_allowed": True,
        "redistribution_allowed": True,
        "derivative_use_allowed": True,
        "pii_risk": "low",
        "intended_use": "training",
        "source_quality": "high",
        "notes": "unit test",
    }
    values.update(overrides)
    return DatasetCandidate.model_validate(values)


def test_unknown_license_is_rejected() -> None:
    with pytest.raises(ValidationError, match="unknown license is rejected"):
        _candidate(license_name="unknown")


def test_non_commercial_license_is_research_eval_only() -> None:
    decision = evaluate_candidate(
        _candidate(
            license_name="CC-BY-NC-4.0",
            commercial_use_allowed=False,
            intended_use="training",
        )
    )

    assert decision.status == AcceptanceStatus.RESEARCH_EVAL_ONLY
    assert decision.effective_use == "research_eval"


def test_ambiguous_license_is_rejected_by_acceptance_gate() -> None:
    decision = evaluate_candidate(_candidate(license_name="Research-friendly mixed licenses"))

    assert decision.status == AcceptanceStatus.REJECTED
    assert decision.effective_use is None
    assert "ambiguous license" in decision.reasons[0]


def test_high_pii_is_blocked_without_redaction_or_opt_in() -> None:
    decision = evaluate_candidate(_candidate(pii_risk="high"))

    assert decision.status == AcceptanceStatus.REJECTED
    assert decision.effective_use is None


def test_high_pii_is_allowed_when_redacted() -> None:
    decision = evaluate_candidate(_candidate(pii_risk="high", redacted=True))

    assert decision.status == AcceptanceStatus.APPROVED


def test_cloud_hosted_metadata_discovery_is_allowed() -> None:
    candidate = _candidate(cloud_hosted=True, metadata_only=True)
    decision = evaluate_candidate(candidate)

    assert candidate.cloud_hosted is True
    assert candidate.metadata_only is True
    assert decision.status == AcceptanceStatus.APPROVED
