from __future__ import annotations

from collections.abc import Iterable

from services.gateway.schemas import DatasetMetadata, SafetyWarning

SENSITIVE_OUTPUTS = {"financial", "medical", "legal"}
SENSITIVE_KEYWORDS = {
    "financial": {
        "bank",
        "bill",
        "budget",
        "credit",
        "debt",
        "finance",
        "financial",
        "invoice",
        "loan",
        "money",
        "payment",
        "tax",
    },
    "medical": {
        "clinic",
        "doctor",
        "dose",
        "health",
        "hospital",
        "medical",
        "medicine",
        "prescription",
        "symptom",
        "treatment",
    },
    "legal": {
        "attorney",
        "contract",
        "court",
        "law",
        "lawyer",
        "legal",
        "notice",
        "rights",
        "sue",
    },
}


def validate_dataset_registry(entries: Iterable[dict[str, object]]) -> list[DatasetMetadata]:
    return [DatasetMetadata.model_validate(entry) for entry in entries]


def warnings_for_output(category: str) -> list[SafetyWarning]:
    if category not in SENSITIVE_OUTPUTS:
        return []
    return [
        SafetyWarning(
            category=category,
            message=("This output may be uncertain and should be reviewed by a qualified human."),
            requires_human_review=True,
        )
    ]


def sensitive_categories_for_text(text: str) -> list[str]:
    normalized = text.casefold()
    matches = [
        category
        for category, keywords in SENSITIVE_KEYWORDS.items()
        if any(keyword in normalized for keyword in keywords)
    ]
    return sorted(matches)


def assert_no_cloud_runtime(settings: dict[str, object]) -> None:
    blocked = {
        "allow_remote_inference": True,
        "allow_remote_file_uploads": True,
        "allow_external_apis": True,
        "telemetry_enabled": True,
        "allow_web": True,
    }
    enabled = [name for name, disallowed in blocked.items() if settings.get(name) == disallowed]
    if enabled:
        raise ValueError(f"cloud or remote settings are enabled: {', '.join(enabled)}")
    if settings.get("privacy_mode", "strict") != "strict":
        raise ValueError("privacy_mode must be strict")
