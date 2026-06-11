from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from data.schemas.source_registry import DatasetCandidate, IntendedUse, PiiRisk


class AcceptanceStatus(StrEnum):
    APPROVED = "approved"
    REJECTED = "rejected"
    RESEARCH_EVAL_ONLY = "research_eval_only"


class AcceptanceConfig(BaseModel):
    allow_non_commercial_training: bool = False
    allow_high_pii_with_redaction_or_opt_in: bool = True


class AcceptanceDecision(BaseModel):
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

    if candidate.has_ambiguous_license:
        return AcceptanceDecision(
            dataset_name=candidate.dataset_name,
            source_catalog=candidate.source_catalog,
            status=AcceptanceStatus.REJECTED,
            effective_use=None,
            reasons=["ambiguous license requires documented review before use"],
        )

    if candidate.pii_risk == PiiRisk.HIGH:
        allowed_high_pii = acceptance_config.allow_high_pii_with_redaction_or_opt_in and (
            candidate.redacted or candidate.explicit_user_opt_in
        )
        if not allowed_high_pii:
            return AcceptanceDecision(
                dataset_name=candidate.dataset_name,
                source_catalog=candidate.source_catalog,
                status=AcceptanceStatus.REJECTED,
                effective_use=None,
                reasons=["high PII risk requires redaction or explicit user opt-in"],
            )
        reasons.append("high PII allowed only because redaction or explicit opt-in is present")

    if candidate.is_non_commercial and not acceptance_config.allow_non_commercial_training:
        reasons.append("non-commercial license restricted to research/eval-only use")
        return AcceptanceDecision(
            dataset_name=candidate.dataset_name,
            source_catalog=candidate.source_catalog,
            status=AcceptanceStatus.RESEARCH_EVAL_ONLY,
            effective_use=IntendedUse.RESEARCH_EVAL,
            reasons=reasons,
        )

    if not candidate.redistribution_allowed:
        reasons.append("redistribution is not allowed; keep source-referenced metadata only")

    if not candidate.derivative_use_allowed:
        reasons.append("derivative use is not allowed; do not train derivative models")
        return AcceptanceDecision(
            dataset_name=candidate.dataset_name,
            source_catalog=candidate.source_catalog,
            status=AcceptanceStatus.RESEARCH_EVAL_ONLY,
            effective_use=IntendedUse.RESEARCH_EVAL,
            reasons=reasons,
        )

    return AcceptanceDecision(
        dataset_name=candidate.dataset_name,
        source_catalog=candidate.source_catalog,
        status=AcceptanceStatus.APPROVED,
        effective_use=candidate.intended_use,
        reasons=reasons or ["candidate satisfies default acceptance rules"],
    )
