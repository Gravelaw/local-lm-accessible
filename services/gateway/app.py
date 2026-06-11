from __future__ import annotations

import re
from pathlib import Path

from fastapi import FastAPI
from pydantic import ValidationError

from services.gateway.model_clients import LocalModelError, default_clients
from services.gateway.model_status import model_readiness_by_key
from services.gateway.router import load_routes
from services.gateway.schemas import (
    HealthResponse,
    ImageAccessibilityOutput,
    ImageTranslationOutput,
    RouteRequest,
    TaskName,
    TaskRequest,
    TaskResponse,
)
from services.gateway.tool_registry import assert_all_tools_local, assert_no_raw_logging, list_tools
from services.tools.document_extract import (
    extract_local_document,
    extract_vision_document_json,
    extraction_rows,
)
from services.tools.excel_export import export_rows
from services.tools.safety_checks import sensitive_categories_for_text, warnings_for_output
from services.tools.web_fetch import fetch_url, summarize_url_blocked_by_default
from services.tools.wiki_index import summarize_wikipedia_offline_stub

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
ROUTES_PATH = CONFIG_DIR / "routes.yaml"
VISION_DOCUMENT_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".pdf"}
PARAKEET_SUPPORTED_LANGUAGES = {
    "en",
    "bg",
    "cs",
    "da",
    "de",
    "el",
    "es",
    "et",
    "fi",
    "fr",
    "hr",
    "hu",
    "it",
    "lt",
    "lv",
    "mt",
    "nl",
    "pl",
    "pt",
    "ro",
    "sk",
    "sl",
    "sv",
    "uk",
    "ga",
    "is",
}
PARAKEET_EXPERIMENTAL_REGIONS = {"india", "southeast_asia"}
IDENTITY_GUESS_RE = re.compile(
    r"\b(?:this is|that is|person is|man is|woman is|he is|she is|they are)\s+"
    r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b",
    flags=re.IGNORECASE,
)

app = FastAPI(title="local-lm gateway")


@app.get("/health")
def health() -> HealthResponse:
    router = load_routes(ROUTES_PATH)
    assert_all_tools_local()
    assert_no_raw_logging()
    return HealthResponse(
        status="ok",
        local_only=True,
        privacy_mode=router.runtime.privacy_mode,
        allow_web=router.runtime.allow_web,
        telemetry_enabled=router.runtime.telemetry_enabled,
        services=_health_services_with_model_status(router.health_services()),
    )


@app.get("/tools")
def tools() -> dict[str, object]:
    return {"tools": [tool.__dict__ for tool in list_tools()]}


@app.post("/tasks/route")
def route_request(request: RouteRequest) -> dict[str, object]:
    decision = load_routes(ROUTES_PATH).decide(request)
    return decision.model_dump(mode="json")


@app.post("/tasks/summarize_url")
def summarize_url(request: TaskRequest) -> TaskResponse:
    if not request.allow_web:
        return TaskResponse(
            task=TaskName.SUMMARIZE_URL,
            status="blocked",
            result=summarize_url_blocked_by_default(request.url or ""),
            warnings=["Web access is disabled by default in strict privacy mode."],
        )
    try:
        fetched = fetch_url(request.url or "")
    except (RuntimeError, ValueError) as exc:
        return TaskResponse(
            task=TaskName.SUMMARIZE_URL,
            status="blocked",
            result={
                "url": request.url or "",
                "summary": "",
                "blocked": True,
                "reason": str(exc),
                "remote_uploads": False,
            },
            warnings=[f"Optional web fetch was not completed: {exc}"],
        )
    source_text = str(fetched["text"])
    safety_warnings = _safety_warning_messages(source_text)
    prompt = (
        "Summarize this user-enabled fetched web page for an elderly or low-vision user. "
        "Use simple language, keep uncertainty visible, and do not add unsupported claims.\n\n"
        f"URL: {fetched['url']}\n\n{source_text}"
    )
    generated = _try_local_generation("text", prompt, max_tokens=320)
    if generated is not None:
        return TaskResponse(
            task=TaskName.SUMMARIZE_URL,
            status="ok",
            result={
                "url": request.url,
                "summary": generated["text"],
                "source": "local_text_model_with_user_enabled_web_fetch",
                "model_endpoint": generated["endpoint"],
                "bytes_read": fetched["bytes_read"],
                "remote_uploads": False,
            },
            confidence=0.65,
            human_review_required=bool(safety_warnings),
            warnings=[
                "Optional web fetch was explicitly enabled for this request.",
                *safety_warnings,
            ],
        )
    fallback = summarize_wikipedia_offline_stub(source_text)
    fallback.update(
        {
            "url": request.url,
            "source": "optional_web_fetch_local_fallback",
            "bytes_read": fetched["bytes_read"],
            "remote_uploads": False,
        }
    )
    return TaskResponse(
        task=TaskName.SUMMARIZE_URL,
        status="stub",
        result=fallback,
        confidence=0.4,
        human_review_required=bool(safety_warnings),
        warnings=[
            "Optional web fetch was explicitly enabled for this request.",
            "Local text model is unavailable; returned offline summary fallback.",
            *safety_warnings,
        ],
    )


@app.post("/tasks/summarize_wikipedia")
def summarize_wikipedia(request: TaskRequest) -> TaskResponse:
    source = request.text or request.url or ""
    safety_warnings = _safety_warning_messages(source)
    prompt = (
        "Summarize the following article for an elderly or low-vision user. "
        "Use simple language, keep uncertainty visible, and do not add unsupported claims.\n\n"
        f"{source}"
    )
    generated = _try_local_generation("text", prompt, max_tokens=320)
    if generated is not None:
        return TaskResponse(
            task=TaskName.SUMMARIZE_WIKIPEDIA,
            status="ok",
            result={
                "query": source,
                "summary": generated["text"],
                "model_endpoint": generated["endpoint"],
                "source": "local_text_model",
            },
            confidence=0.7,
            human_review_required=bool(safety_warnings),
            warnings=safety_warnings,
        )
    return TaskResponse(
        task=TaskName.SUMMARIZE_WIKIPEDIA,
        status="stub",
        result=summarize_wikipedia_offline_stub(source),
        human_review_required=bool(safety_warnings),
        warnings=[
            "Local text model is unavailable; returned offline summary stub.",
            *safety_warnings,
        ],
    )


@app.post("/tasks/document_to_excel")
def document_to_excel(request: TaskRequest) -> TaskResponse:
    file_path = Path(request.file_path) if request.file_path else Path("")
    vision_warning = ""
    vision_endpoint = ""
    if file_path.exists() and file_path.suffix.casefold() in VISION_DOCUMENT_SUFFIXES:
        generated = _try_local_generation(
            "vision",
            _document_extraction_prompt(file_path),
            max_tokens=1024,
        )
        if generated is not None:
            try:
                extraction = extract_vision_document_json(str(generated["text"]))
            except (ValidationError, ValueError) as exc:
                vision_warning = f"Local vision model returned invalid document JSON: {exc}"
            else:
                vision_endpoint = str(generated["endpoint"])
                rows = extraction_rows(extraction)
                return TaskResponse(
                    task=TaskName.DOCUMENT_TO_EXCEL,
                    status="ok",
                    result={
                        "file_path": request.file_path,
                        "rows": rows,
                        "exporter": export_rows.__name__,
                        "schema": extraction.model_dump(),
                        "source": "local_vision_model",
                        "model_endpoint": vision_endpoint,
                    },
                    confidence=extraction.confidence,
                    human_review_required=extraction.human_review_required,
                    warnings=extraction.warnings,
                )

    extraction = (
        extract_local_document(file_path)
        if file_path.exists()
        else extract_local_document(Path(""))
    )
    rows = extraction_rows(extraction)
    status = "ok" if file_path.exists() and file_path.suffix.casefold() == ".txt" else "stub"
    warnings = list(extraction.warnings)
    if vision_warning:
        warnings.insert(0, vision_warning)
    return TaskResponse(
        task=TaskName.DOCUMENT_TO_EXCEL,
        status=status,
        result={
            "file_path": request.file_path,
            "rows": rows,
            "exporter": export_rows.__name__,
            "schema": extraction.model_dump(),
            "source": "local_fallback",
            "model_endpoint": vision_endpoint or "http://127.0.0.1:8082",
        },
        confidence=extraction.confidence,
        human_review_required=extraction.human_review_required,
        warnings=warnings,
    )


@app.post("/tasks/describe_image")
def describe_image(request: TaskRequest) -> TaskResponse:
    prompt = (
        "Describe this local image for a low-vision user. Mention hazards first if visible, "
        "separate visible text from the general description, and do not guess identity.\n\n"
        f"Local image path: {request.file_path or '<not provided>'}"
    )
    generated = _try_local_generation("vision", prompt, max_tokens=320)
    if generated is not None:
        safe_description, uncertainties, warnings = _safe_image_description(generated["text"])
        output = ImageAccessibilityOutput(
            short_description=safe_description,
            visible_text=[],
            possible_hazards=[],
            uncertainties=uncertainties,
            spoken_response=safe_description,
            confidence=0.6,
            warnings=warnings,
        )
        return TaskResponse(
            task=TaskName.DESCRIBE_IMAGE,
            status="ok",
            result={
                "file_path": request.file_path,
                "description": output.spoken_response,
                "model_endpoint": generated["endpoint"],
                "schema": output.model_dump(),
            },
            confidence=output.confidence,
            human_review_required=True,
            warnings=output.warnings,
        )

    selected_path = request.file_path or "the selected local image"
    description = (
        "Local vision model is unavailable. The image stayed local and was not sent to "
        f"cloud OCR or remote inference. Selected file: {selected_path}"
    )
    output = ImageAccessibilityOutput(
        short_description=description,
        visible_text=[],
        possible_hazards=[],
        uncertainties=["Local vision model is unavailable, so no visual content was analyzed."],
        spoken_response=description,
        confidence=0.0,
        warnings=[
            "Local vision model is unavailable; no remote image service was used.",
            "Upload remains local to this runtime.",
        ],
    )
    return TaskResponse(
        task=TaskName.DESCRIBE_IMAGE,
        status="stub",
        result={
            "file_path": request.file_path,
            "description": output.spoken_response,
            "schema": output.model_dump(),
        },
        confidence=output.confidence,
        human_review_required=True,
        warnings=output.warnings,
    )


@app.post("/tasks/translate_image_text")
def translate_image_text(request: TaskRequest) -> TaskResponse:
    target_language = request.target_language or "English"
    prompt = (
        "Read visible text from this local image and translate it. Keep original text and "
        "translation separate when possible, and mark uncertain text.\n\n"
        f"Target language: {target_language}\n"
        f"Local image path: {request.file_path or '<not provided>'}"
    )
    generated = _try_local_generation("vision", prompt, max_tokens=320)
    if generated is not None:
        output = ImageTranslationOutput(
            original_text=[],
            translated_text=generated["text"],
            target_language=target_language,
            uncertain_text=["Structured OCR parsing is not yet enabled."],
            confidence=0.6,
            warnings=["Review translated visible text for OCR uncertainty."],
        )
        return TaskResponse(
            task=TaskName.TRANSLATE_IMAGE_TEXT,
            status="ok",
            result={
                "file_path": request.file_path,
                "target_language": target_language,
                "text": output.translated_text,
                "model_endpoint": generated["endpoint"],
                "schema": output.model_dump(),
            },
            confidence=output.confidence,
            human_review_required=True,
            warnings=output.warnings,
        )

    selected_path = request.file_path or "the selected local image"
    translated_text = (
        "Local OCR/translation model is unavailable. The image stayed local and was not sent "
        f"to cloud OCR or remote inference. Selected file: {selected_path}"
    )
    output = ImageTranslationOutput(
        original_text=[],
        translated_text=translated_text,
        target_language=target_language,
        uncertain_text=["Local OCR model is unavailable, so visible text was not read."],
        confidence=0.0,
        warnings=[
            "Local OCR/translation model is unavailable; no remote image service was used.",
            "Upload remains local to this runtime.",
        ],
    )
    return TaskResponse(
        task=TaskName.TRANSLATE_IMAGE_TEXT,
        status="stub",
        result={
            "file_path": request.file_path,
            "target_language": request.target_language,
            "text": output.translated_text,
            "schema": output.model_dump(),
        },
        confidence=output.confidence,
        human_review_required=True,
        warnings=output.warnings,
    )


@app.post("/tasks/speech_to_text")
def speech_to_text(request: TaskRequest) -> TaskResponse:
    if request.file_path:
        language_status = _asr_language_status(request)
        if language_status["experimental"] and not request.allow_experimental_asr:
            return TaskResponse(
                task=TaskName.SPEECH_TO_TEXT,
                status="stub",
                result={
                    "file_path": request.file_path,
                    "asr_endpoint": "http://127.0.0.1:8090",
                    "text": "",
                    "language": language_status["language"],
                    "region": language_status["region"],
                    "country": language_status["country"],
                    "experimental": True,
                    "unsupported_language": language_status["unsupported_language"],
                    "model_ready": False,
                },
                confidence=0.0,
                human_review_required=True,
                warnings=language_status["warnings"],
            )
        generated = _try_local_asr(request)
        if generated is not None:
            return TaskResponse(
                task=TaskName.SPEECH_TO_TEXT,
                status="ok" if generated.get("status") == "ok" else "stub",
                result={
                    "file_path": request.file_path,
                    "asr_endpoint": "http://127.0.0.1:8090",
                    "text": generated.get("text", ""),
                    "language": generated.get("language", language_status["language"]),
                    "region": language_status["region"],
                    "country": language_status["country"],
                    "experimental": bool(generated.get("experimental", False)),
                    "unsupported_language": bool(generated.get("unsupported_language", False)),
                    "model_ready": bool(generated.get("model_ready", False)),
                },
                confidence=0.4 if generated.get("status") == "stub" else 0.8,
                human_review_required=bool(generated.get("experimental", False)),
                warnings=list(generated.get("warnings", [])),
            )
    return TaskResponse(
        task=TaskName.SPEECH_TO_TEXT,
        status="stub",
        result={
            "file_path": request.file_path,
            "asr_endpoint": "http://127.0.0.1:8090",
            "text": "Local ASR service is unavailable or no audio was uploaded.",
        },
        confidence=0.0,
        human_review_required=True,
        warnings=["Local ASR endpoint is unavailable; no remote speech service was used."],
    )


@app.post("/tasks/general")
def general_local_assistant(request: TaskRequest) -> TaskResponse:
    safety_warnings = _safety_warning_messages(request.text or "")
    if request.text:
        generated = _try_local_generation(
            "text",
            (
                "Answer in simple language for an elderly or accessibility-constrained user. "
                "Include uncertainty when needed.\n\n"
                f"{request.text}"
            ),
            max_tokens=256,
        )
        if generated is not None:
            return TaskResponse(
                task=TaskName.GENERAL_LOCAL_ASSISTANT,
                status="ok",
                result={"text": generated["text"], "model_endpoint": generated["endpoint"]},
                confidence=0.7,
                human_review_required=bool(safety_warnings),
                warnings=safety_warnings,
            )
    return TaskResponse(
        task=TaskName.GENERAL_LOCAL_ASSISTANT,
        status="stub",
        result={
            "text": (
                "Local text model is unavailable. No remote model or external API was used. "
                "Start the verified local text service on 127.0.0.1:8081 to answer this request."
            ),
            "model_endpoint": "http://127.0.0.1:8081",
        },
        confidence=0.0,
        human_review_required=True,
        warnings=[
            "Local text endpoint is unavailable; no remote assistant service was used.",
            *safety_warnings,
        ],
    )


def _safety_warning_messages(text: str) -> list[str]:
    messages: list[str] = []
    for category in sensitive_categories_for_text(text):
        messages.extend(warning.message for warning in warnings_for_output(category))
    return messages


def _try_local_generation(
    model_key: str,
    prompt: str,
    *,
    max_tokens: int,
) -> dict[str, object] | None:
    try:
        return default_clients()[model_key].generate(prompt, max_tokens=max_tokens)
    except (KeyError, LocalModelError):
        return None


def _document_extraction_prompt(file_path: Path) -> str:
    return (
        "Extract this local document into one strict JSON object only. "
        "Use document_type invoice, bill, receipt, or bank_statement when applicable. "
        "For invoices include fields, line_items, currency, subtotal, tax_amount, total, "
        "raw_ocr_text, confidence, warnings, and human_review_required. "
        "For bank statements include transactions, currency, raw_ocr_text, confidence, "
        "warnings, and human_review_required=true. Do not invent totals; if uncertain, "
        "lower confidence and require human review.\n\n"
        f"Local document path: {file_path}"
    )


def _safe_image_description(model_text: object) -> tuple[str, list[str], list[str]]:
    description = str(model_text).strip()
    uncertainties = ["Structured vision parsing is not yet enabled."]
    warnings = ["Review image descriptions for uncertainty before relying on them."]
    if not description:
        return (
            "Local vision model returned no usable description.",
            [*uncertainties, "No usable local vision description was returned."],
            [*warnings, "Local vision output was empty."],
        )
    if IDENTITY_GUESS_RE.search(description):
        return (
            "I can describe visible scene details, but I cannot identify or name people in "
            "images. The local vision output included an identity-like claim, so this result "
            "requires human review.",
            [*uncertainties, "Identity-like claim removed from the local vision output."],
            [*warnings, "Identity guessing from images is blocked."],
        )
    return description, uncertainties, warnings


def _try_local_asr(request: TaskRequest) -> dict[str, object] | None:
    language = (request.language or "en").strip().lower()
    region = (request.region or "unknown").strip().lower()
    country = (request.country or "unknown").strip().lower()
    try:
        return default_clients()["asr"].post_json(
            "/transcribe",
            {
                "audio_filepath": request.file_path,
                "language": language,
                "region": region,
                "country": country,
                "allow_experimental": request.allow_experimental_asr,
            },
        )
    except (KeyError, LocalModelError):
        return None


def _asr_language_status(request: TaskRequest) -> dict[str, object]:
    language = (request.language or "en").strip().lower()
    region = (request.region or "unknown").strip().lower()
    country = (request.country or "unknown").strip().lower()
    unsupported_language = language not in PARAKEET_SUPPORTED_LANGUAGES
    regional_experiment = region in PARAKEET_EXPERIMENTAL_REGIONS and language != "en"
    experimental = unsupported_language or regional_experiment
    warnings: list[str] = []
    if unsupported_language:
        warnings.append(
            "Parakeet v3 does not list this language as supported; ASR is experimental."
        )
    elif regional_experiment:
        warnings.append(
            "Indian and Southeast Asian non-English ASR is experimental until local eval "
            "proves usable."
        )
    if experimental and not request.allow_experimental_asr:
        warnings.append("Enable experimental ASR explicitly to process this language/region.")
    return {
        "language": language,
        "region": region,
        "country": country,
        "unsupported_language": unsupported_language,
        "experimental": experimental,
        "warnings": warnings,
    }


def _health_services_with_model_status(
    services: list[dict[str, object]],
) -> list[dict[str, object]]:
    statuses = model_readiness_by_key()
    enriched = []
    for service in services:
        payload = dict(service)
        status = statuses.get(str(service["name"]))
        if status is not None:
            payload.update(
                {
                    "model_id": status["model_id"],
                    "required": status["required"],
                    "artifact_present": status["artifact_present"],
                    "checksum_configured": status["checksum_configured"],
                    "ready": status["ready"],
                    "warnings": status["warnings"],
                }
            )
        enriched.append(payload)
    return enriched
