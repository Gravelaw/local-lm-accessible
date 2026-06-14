from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from data.schemas.source_registry import (
    CandidateTask,
    DatasetCandidate,
    IntendedUse,
    PiiRisk,
    SourceCatalog,
)


class AcceptanceStatus(StrEnum):
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REVIEW = "needs_review"
    EVAL_ONLY = "eval_only"
    RESEARCH_EVAL_ONLY = "research_eval_only"


class AcceptanceConfig(BaseModel):
    allow_non_commercial_training: bool = False
    allow_high_pii_with_redaction_or_opt_in: bool = True
    allow_research_only_training: bool = False
    require_license_url: bool = True
    require_dataset_card_for_approval: bool = False


class AcceptanceDecision(BaseModel):
    dataset_id: str
    dataset_name: str
    source_catalog: str
    status: AcceptanceStatus
    effective_use: IntendedUse | None
    reasons: list[str] = Field(default_factory=list)


def evaluate_candidate(
    candidate: DatasetCandidate,
    config: AcceptanceConfig | None = None,
) -> AcceptanceDecision:
    acceptance_config = config or AcceptanceConfig()
    reasons: list[str] = []

    if (
        CandidateTask.BLOCKED in candidate.candidate_tasks
        or candidate.intended_use == IntendedUse.BLOCKED
    ):
        return _decision(candidate, AcceptanceStatus.REJECTED, None, ["candidate is blocked"])

    if candidate.source_catalog == SourceCatalog.EPO:
        if acceptance_config.require_license_url and not candidate.license_url:
            return _decision(
                candidate,
                AcceptanceStatus.NEEDS_REVIEW,
                None,
                ["missing license URL"],
            )
        reasons.append("EPO data is limited to patent/technical/legal summarization and eval")
        return _decision(candidate, AcceptanceStatus.EVAL_ONLY, IntendedUse.EVAL_ONLY, reasons)

    if candidate.has_ambiguous_license:
        return _decision(
            candidate,
            AcceptanceStatus.REJECTED,
            None,
            ["ambiguous license requires documented review before use"],
        )

    if acceptance_config.require_license_url and not candidate.license_url:
        return _decision(candidate, AcceptanceStatus.NEEDS_REVIEW, None, ["missing license URL"])

    if (
        candidate.source_catalog == SourceCatalog.KAGGLE
        and (not candidate.metadata.get("kaggle_slug") or not candidate.license_name)
    ):
        return _decision(
            candidate,
            AcceptanceStatus.NEEDS_REVIEW,
            None,
            ["Kaggle datasets must record Kaggle dataset slug and license"],
        )

    if candidate.source_catalog == SourceCatalog.HUGGINGFACE and not candidate.metadata.get(
        "hf_repo_id"
    ):
        return _decision(
            candidate,
            AcceptanceStatus.NEEDS_REVIEW,
            None,
            ["Hugging Face datasets must record repo ID"],
        )

    if candidate.source_catalog in {
        SourceCatalog.AWS_OPEN_DATA_REGISTRY,
        SourceCatalog.GOOGLE_PUBLIC_DATASETS,
        SourceCatalog.AZURE_OPEN_DATASETS,
    }:
        reasons.append("cloud catalog is discovery/training-time only, not a runtime dependency")

    if (
        candidate.source_catalog == SourceCatalog.EUROPEAN_DATA_PORTAL
        and (
            not candidate.metadata.get("publisher")
            or not candidate.metadata.get("publisher_license")
        )
    ):
        return _decision(
            candidate,
            AcceptanceStatus.NEEDS_REVIEW,
            None,
            ["European Data Portal records must preserve publisher and licence metadata"],
        )

    if candidate.pii_risk == PiiRisk.HIGH or candidate.contains_sensitive_data:
        allowed_sensitive = (
            acceptance_config.allow_high_pii_with_redaction_or_opt_in
            and (candidate.redacted or candidate.explicit_user_opt_in)
        )
        if not allowed_sensitive:
            return _decision(
                candidate,
                AcceptanceStatus.REJECTED,
                None,
                ["high PII or sensitive data requires redaction or explicit user opt-in"],
            )
        reasons.append(
            "sensitive data allowed only because redaction or explicit opt-in is present"
        )

    if _real_financial_document(candidate) and candidate.pii_risk not in {
        PiiRisk.NONE,
        PiiRisk.LOW,
    }:
        return _decision(
            candidate,
            AcceptanceStatus.REJECTED,
            None,
            ["real invoices/receipts/statements require none/low PII or redaction/explicit opt-in"],
        )

    if candidate.is_non_commercial and not acceptance_config.allow_non_commercial_training:
        reasons.append("non-commercial license restricted to research/eval-only use")
        return _decision(candidate, AcceptanceStatus.EVAL_ONLY, IntendedUse.EVAL_ONLY, reasons)

    if not candidate.derivative_use_allowed:
        reasons.append("derivative use is not allowed; do not train derivative models")
        return _decision(candidate, AcceptanceStatus.EVAL_ONLY, IntendedUse.EVAL_ONLY, reasons)

    if not candidate.redistribution_allowed:
        reasons.append("redistribution is not allowed; keep source-referenced metadata only")

    return _decision(
        candidate,
        AcceptanceStatus.APPROVED,
        candidate.intended_use,
        reasons or ["candidate satisfies default acceptance rules"],
    )


def _real_financial_document(candidate: DatasetCandidate) -> bool:
    financial_tasks = {
        CandidateTask.INVOICE_EXTRACTION,
        CandidateTask.RECEIPT_EXTRACTION,
        CandidateTask.BILL_EXTRACTION,
        CandidateTask.BANK_STATEMENT_EXTRACTION,
    }
    if not any(task in financial_tasks for task in candidate.candidate_tasks):
        return False
    return candidate.source_catalog not in {
        SourceCatalog.SYNTHETIC,
        SourceCatalog.USER_OPT_IN_REDACTED,
    }


def _decision(
    candidate: DatasetCandidate,
    status: AcceptanceStatus,
    effective_use: IntendedUse | None,
    reasons: list[str],
) -> AcceptanceDecision:
    return AcceptanceDecision(
        dataset_id=candidate.dataset_id,
        dataset_name=candidate.dataset_name,
        source_catalog=str(candidate.source_catalog),
        status=status,
        effective_use=effective_use,
        reasons=reasons,
    )
