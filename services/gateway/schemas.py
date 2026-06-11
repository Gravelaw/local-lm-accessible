from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator

UNKNOWN_LICENSE_VALUES = {"", "unknown", "unk", "n/a", "na", "none", "tbd", "unlicensed"}
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


class RuntimeConfig(BaseModel):
    local_only: bool = True
    privacy_mode: Literal["strict"] = "strict"
    allow_web: bool = False
    allow_remote_inference: bool = False
    allow_remote_file_uploads: bool = False
    allow_external_apis: bool = False
    telemetry_enabled: bool = False

    @model_validator(mode="after")
    def require_local_only(self) -> RuntimeConfig:
        if not self.local_only:
            raise ValueError("runtime must be local_only")
        if self.privacy_mode != "strict":
            raise ValueError("privacy_mode must be strict")
        blocked_flags = {
            "allow_web": self.allow_web,
            "allow_remote_inference": self.allow_remote_inference,
            "allow_remote_file_uploads": self.allow_remote_file_uploads,
            "allow_external_apis": self.allow_external_apis,
            "telemetry_enabled": self.telemetry_enabled,
        }
        enabled = [name for name, value in blocked_flags.items() if value]
        if enabled:
            raise ValueError(f"cloud or remote runtime flags enabled: {', '.join(enabled)}")
        return self


class PiiMetadata(BaseModel):
    contains_pii: bool
    source: str = Field(min_length=1)
    handling: str = Field(min_length=1)


class DatasetMetadata(BaseModel):
    name: str = Field(min_length=1)
    license: str = Field(min_length=1)
    region: str = Field(min_length=1)
    country: str = Field(min_length=1)
    language: str = Field(min_length=1)
    modality: str = Field(min_length=1)
    task: str = Field(min_length=1)
    pii_metadata: PiiMetadata

    @field_validator("license")
    @classmethod
    def reject_unknown_license(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized in UNKNOWN_LICENSE_VALUES:
            raise ValueError("unknown or missing license is rejected")
        return value


class RouteTarget(BaseModel):
    provider: str = Field(pattern="^local$")
    model_key: str = Field(min_length=1)
    endpoint: HttpUrl

    @field_validator("endpoint")
    @classmethod
    def require_loopback_endpoint(cls, value: HttpUrl) -> HttpUrl:
        host = value.host or ""
        if host not in LOCAL_HOSTS:
            raise ValueError("route endpoints must use loopback hosts only")
        return value


class TaskName(StrEnum):
    GENERAL_LOCAL_ASSISTANT = "general_local_assistant"
    SUMMARIZE_URL = "summarize_url"
    SUMMARIZE_WIKIPEDIA = "summarize_wikipedia"
    DOCUMENT_TO_EXCEL = "document_to_excel"
    DESCRIBE_IMAGE = "describe_image"
    TRANSLATE_IMAGE_TEXT = "translate_image_text"
    SPEECH_TO_TEXT = "speech_to_text"


class UserRequest(BaseModel):
    task: str = Field(min_length=1)
    text: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RouteRequest(BaseModel):
    intent: str = Field(default="", max_length=2000)
    file_path: str | None = Field(default=None, max_length=4096)
    mime_type: str | None = Field(default=None, max_length=255)
    url: str | None = Field(default=None, max_length=4096)
    allow_web: bool = False


class RouteDecision(BaseModel):
    task: TaskName
    provider: Literal["local"] = "local"
    model_key: str
    endpoint: HttpUrl
    privacy_mode: Literal["strict"] = "strict"
    allow_web: bool = False
    reason: str


class TaskRequest(BaseModel):
    text: str | None = Field(default=None, max_length=100_000)
    url: str | None = Field(default=None, max_length=4096)
    file_path: str | None = Field(default=None, max_length=4096)
    mime_type: str | None = Field(default=None, max_length=255)
    target_language: str | None = Field(default=None, max_length=64)
    language: str | None = Field(default=None, max_length=16)
    region: str | None = Field(default=None, max_length=64)
    country: str | None = Field(default=None, max_length=64)
    allow_experimental_asr: bool = False
    allow_web: bool = False


class TaskResponse(BaseModel):
    task: TaskName
    status: Literal["ok", "blocked", "stub"]
    local_only: bool = True
    privacy_mode: Literal["strict"] = "strict"
    result: dict[str, Any] = Field(default_factory=dict)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    human_review_required: bool = False
    warnings: list[str] = Field(default_factory=list)


class HealthService(BaseModel):
    name: str
    endpoint: HttpUrl
    local_only: bool
    optional: bool = False
    configured: bool
    model_id: str | None = None
    required: bool = False
    artifact_present: bool = False
    checksum_configured: bool = False
    ready: bool = False
    warnings: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: Literal["ok"]
    local_only: bool
    privacy_mode: Literal["strict"]
    allow_web: bool
    telemetry_enabled: bool
    services: list[HealthService]


class DocumentExtractionOutput(BaseModel):
    document_type: str = Field(min_length=1)
    fields: dict[str, Any] = Field(default_factory=dict)
    raw_text: str = ""
    confidence: float = Field(ge=0.0, le=1.0)
    warnings: list[str] = Field(default_factory=list)
    human_review_required: bool = False


class InvoiceExtractionOutput(BaseModel):
    document_type: Literal["invoice", "bill", "receipt"]
    fields: dict[str, Any] = Field(default_factory=dict)
    line_items: list[dict[str, Any]] = Field(default_factory=list)
    currency: str = Field(default="", max_length=16)
    subtotal: float | None = None
    tax_amount: float | None = None
    total: float | None = None
    raw_ocr_text: str = ""
    confidence: float = Field(ge=0.0, le=1.0)
    warnings: list[str] = Field(default_factory=list)
    human_review_required: bool = False

    @model_validator(mode="after")
    def require_reconciled_totals(self) -> InvoiceExtractionOutput:
        if self.subtotal is None or self.tax_amount is None or self.total is None:
            return self
        expected_total = round(self.subtotal + self.tax_amount, 2)
        actual_total = round(self.total, 2)
        if expected_total != actual_total:
            raise ValueError("invoice totals do not reconcile")
        return self


class BankTransaction(BaseModel):
    date: str = Field(min_length=1)
    description: str = Field(min_length=1)
    debit: float = Field(default=0.0, ge=0.0)
    credit: float = Field(default=0.0, ge=0.0)
    balance: float | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    needs_review: bool = True


class BankStatementExtractionOutput(BaseModel):
    document_type: Literal["bank_statement"] = "bank_statement"
    transactions: list[BankTransaction] = Field(default_factory=list)
    currency: str = Field(default="", max_length=16)
    raw_ocr_text: str = ""
    confidence: float = Field(ge=0.0, le=1.0)
    warnings: list[str] = Field(default_factory=list)
    human_review_required: bool = True

    @model_validator(mode="after")
    def bank_statements_always_require_review(self) -> BankStatementExtractionOutput:
        if not self.human_review_required:
            raise ValueError("bank statements must require human review")
        return self


class ImageAccessibilityOutput(BaseModel):
    short_description: str = ""
    visible_text: list[str] = Field(default_factory=list)
    possible_hazards: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    spoken_response: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    warnings: list[str] = Field(default_factory=list)


class ImageTranslationOutput(BaseModel):
    original_text: list[str] = Field(default_factory=list)
    translated_text: str = ""
    target_language: str = Field(min_length=1)
    uncertain_text: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    warnings: list[str] = Field(default_factory=list)


class SafetyWarning(BaseModel):
    category: str = Field(min_length=1)
    message: str = Field(min_length=1)
    requires_human_review: bool = False
