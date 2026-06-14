from __future__ import annotations

import pytest
from pydantic import ValidationError

from data.schemas.dataset_acceptance import AcceptanceStatus, evaluate_candidate
from data.schemas.source_registry import DatasetCandidate


def _candidate(**overrides: object) -> DatasetCandidate:
    values = {
        "dataset_id": "manual:unit",
        "dataset_name": "unit dataset",
        "source_catalog": "manual",
        "source_url": "https://example.com/dataset",
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
        "intended_use": "training",
        "source_quality": "high",
        "notes": "unit test",
    }
    values.update(overrides)
    return DatasetCandidate.model_validate(values)


def test_unknown_license_is_rejected() -> None:
    with pytest.raises(ValidationError, match="unknown license is rejected"):
        _candidate(license_name="unknown")


def test_missing_license_url_needs_review() -> None:
    decision = evaluate_candidate(_candidate(license_url=None))

    assert decision.status == AcceptanceStatus.NEEDS_REVIEW
    assert decision.effective_use is None


def test_non_commercial_license_is_eval_only_by_default() -> None:
    decision = evaluate_candidate(
        _candidate(
            license_name="CC-BY-NC-4.0",
            license_url="https://creativecommons.org/licenses/by-nc/4.0/",
            commercial_use_allowed=False,
        )
    )

    assert decision.status == AcceptanceStatus.EVAL_ONLY
    assert decision.effective_use == "eval_only"


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


def test_user_opt_in_redacted_exception_allows_financial_training() -> None:
    decision = evaluate_candidate(
        _candidate(
            dataset_id="user_opt_in_redacted:sample",
            source_catalog="user_opt_in_redacted",
            modality="document_image",
            candidate_tasks=["bank_statement_extraction"],
            redacted=True,
            explicit_user_opt_in=True,
        )
    )

    assert decision.status == AcceptanceStatus.APPROVED


def test_sensitive_real_financial_documents_are_blocked_without_redaction() -> None:
    decision = evaluate_candidate(
        _candidate(
            modality="document_image",
            candidate_tasks=["invoice_extraction"],
            pii_risk="medium",
            contains_sensitive_data=True,
        )
    )

    assert decision.status == AcceptanceStatus.REJECTED
