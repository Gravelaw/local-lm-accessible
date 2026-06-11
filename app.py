from __future__ import annotations

import json
import tempfile
import wave
from pathlib import Path
from typing import Any

import gradio as gr

from scripts.release_gate import ReleaseGateError, run_release_gate
from scripts.synthetic_documents import generate_documents
from services.gateway.app import (
    ROUTES_PATH,
)
from services.gateway.app import (
    describe_image as gateway_describe_image,
)
from services.gateway.app import (
    document_to_excel as gateway_document_to_excel,
)
from services.gateway.app import (
    general_local_assistant as gateway_general_local_assistant,
)
from services.gateway.app import (
    health as gateway_health,
)
from services.gateway.app import (
    route_request as gateway_route_request,
)
from services.gateway.app import (
    speech_to_text as gateway_speech_to_text,
)
from services.gateway.app import (
    summarize_url as gateway_summarize_url,
)
from services.gateway.app import (
    summarize_wikipedia as gateway_summarize_wikipedia,
)
from services.gateway.app import (
    translate_image_text as gateway_translate_image_text,
)
from services.gateway.model_status import model_readiness_by_key
from services.gateway.router import load_routes
from services.gateway.schemas import RouteRequest, TaskRequest
from services.tools.excel_export import export_rows
from services.tools.pdf_export import export_text_placeholder

ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = ROOT / "samples" / "demo"
EXPORT_DIR = Path(tempfile.gettempdir()) / "local-lm-gradio-exports"
DEMO_MEDIA_DIR = Path(tempfile.gettempdir()) / "local-lm-gradio-demo-media"
MODEL_MANIFEST_PATH = ROOT / "models" / "manifest.json"
AUTO_ROUTE = "Auto"
ADVANCED_RUNTIME_LABEL = "Advanced runtime details"
TASK_CHOICES = [
    AUTO_ROUTE,
    "summarize_wikipedia",
    "summarize_url",
    "document_to_excel",
    "describe_image",
    "translate_image_text",
    "speech_to_text",
    "general_local_assistant",
]
APP_CSS = """
.gradio-container {
  max-width: 1180px !important;
  font-size: 18px;
  line-height: 1.5;
}
.gradio-container button {
  min-height: 52px;
  font-size: 18px !important;
  padding: 10px 16px !important;
}
textarea, input {
  font-size: 18px !important;
  line-height: 1.45 !important;
}
.local-lm-status {
  border: 2px solid #111827;
  padding: 12px;
  font-size: 18px;
}
"""
HIGH_CONTRAST_CSS = """
<style id="local-lm-accessibility-style">
.gradio-container {
  background: #000 !important;
  color: #fff !important;
}
.gradio-container button,
.gradio-container input,
.gradio-container textarea,
.gradio-container select,
.gradio-container .wrap {
  background: #000 !important;
  color: #fff !important;
  border-color: #fff !important;
}
.gradio-container button.primary {
  background: #ffd400 !important;
  color: #000 !important;
  border-color: #ffd400 !important;
}
.gradio-container a {
  color: #7dd3fc !important;
}
</style>
"""
DEFAULT_ACCESSIBILITY_CSS = '<style id="local-lm-accessibility-style"></style>'


def base_style_html() -> str:
    return f'<style id="local-lm-base-style">{APP_CSS}</style>'


def _to_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, default=str)


def _file_path(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        return str(value)
    path = getattr(value, "path", None)
    if path:
        return str(path)
    name = getattr(value, "name", None)
    if name:
        return str(name)
    return str(value)


def _write_exports(base_name: str, payload: dict[str, Any], formats: list[str]) -> list[str]:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    requested = {item.lower() for item in formats}
    files: list[str] = []
    json_text = _to_json(payload)

    if "json" in requested:
        path = EXPORT_DIR / f"{base_name}.json"
        path.write_text(json_text, encoding="utf-8")
        files.append(str(path))
    if "txt" in requested:
        path = EXPORT_DIR / f"{base_name}.txt"
        path.write_text(json_text, encoding="utf-8")
        files.append(str(path))
    if "xlsx" in requested:
        rows = payload.get("result", {}).get("rows", [])
        path = EXPORT_DIR / f"{base_name}.xlsx"
        export_rows(
            path, rows if isinstance(rows, list) else [], metadata=_export_metadata(payload)
        )
        files.append(str(path))
    if "pdf" in requested:
        path = EXPORT_DIR / f"{base_name}.pdf"
        export_text_placeholder(path, json_text)
        files.append(str(path))
    return files


def save_text_result_exports(
    base_name: str,
    result_text: str,
    payload: dict[str, Any] | None,
    formats: list[str],
) -> tuple[list[str], str]:
    if not result_text.strip():
        return [], "Nothing to save yet."
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    requested = {item.lower() for item in formats}
    if not requested:
        requested = {"txt"}
    payload = payload or {}
    safe_name = _safe_export_name(base_name)
    metadata = _export_metadata(payload)
    files: list[str] = []

    if "json" in requested:
        path = EXPORT_DIR / f"{safe_name}.json"
        path.write_text(_to_json({"text": result_text, "status": payload}), encoding="utf-8")
        files.append(str(path))
    if "txt" in requested:
        path = EXPORT_DIR / f"{safe_name}.txt"
        path.write_text(result_text, encoding="utf-8")
        files.append(str(path))
    if "xlsx" in requested:
        path = EXPORT_DIR / f"{safe_name}.xlsx"
        export_rows(path, _text_result_rows(result_text, payload), metadata=metadata)
        files.append(str(path))
    if "pdf" in requested:
        path = EXPORT_DIR / f"{safe_name}.pdf"
        export_text_placeholder(path, result_text)
        files.append(str(path))
    return files, f"Saved {len(files)} local file(s)."


def _text_result_rows(result_text: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        {"field": "result", "value": result_text},
        {"field": "task", "value": payload.get("task", "")},
        {"field": "status", "value": payload.get("status", "")},
        {"field": "human_review_required", "value": payload.get("human_review_required", False)},
    ]
    warnings = payload.get("warnings", [])
    if warnings:
        rows.append({"field": "warnings", "value": warnings})
    return rows


def _safe_export_name(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)
    return cleaned.strip("_") or "local_lm_result"


def _export_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    result = payload.get("result", {})
    return {
        "task": payload.get("task", ""),
        "status": payload.get("status", ""),
        "confidence": payload.get("confidence"),
        "human_review_required": payload.get("human_review_required", False),
        "warnings": payload.get("warnings", []),
        "source": result.get("source", ""),
        "model_endpoint": result.get("model_endpoint", ""),
    }


def _with_visible_warnings(text: str, payload: dict[str, Any]) -> str:
    warnings = "\n".join(str(warning) for warning in payload.get("warnings", []))
    if not warnings:
        return text
    return f"{text}\n\nWarnings:\n{warnings}"


def demo_sample_paths() -> dict[str, str]:
    samples = {
        "question": SAMPLE_DIR / "ask_question.txt",
        "article": SAMPLE_DIR / "article_wikipedia_style.txt",
        "invoice": SAMPLE_DIR / "invoice_sample.txt",
        "bank_statement": SAMPLE_DIR / "bank_statement_sample.txt",
    }
    return {name: str(path) for name, path in samples.items() if path.exists()}


def load_demo_article() -> str:
    path = Path(demo_sample_paths().get("article", ""))
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def load_demo_question() -> str:
    path = Path(demo_sample_paths().get("question", ""))
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def load_demo_document(sample_name: str) -> str | None:
    key = (sample_name or "").strip().casefold().replace(" ", "_")
    return demo_sample_paths().get(key)


def load_demo_image() -> str:
    generated = generate_documents(
        kind="invoice",
        output_dir=DEMO_MEDIA_DIR,
        count=1,
        regions=("India",),
        seed=20260607,
        augment=False,
    )
    document_id = str(generated[0]["document_id"])
    image_path = DEMO_MEDIA_DIR / "invoice" / document_id / "rendered.png"
    return str(image_path)


def load_demo_audio() -> str:
    DEMO_MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    audio_path = DEMO_MEDIA_DIR / "sample_speech.wav"
    if not audio_path.exists():
        with wave.open(str(audio_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16_000)
            wav_file.writeframes(b"\x00\x00" * 160)
    return str(audio_path)


def privacy_disclosure() -> str:
    return (
        "### Privacy and runtime\n\n"
        "- This hackathon demo runs inside Hugging Face Space compute when hosted as a Space.\n"
        "- No external model APIs, cloud OCR, or remote telemetry are called.\n"
        "- User files are handled by the app runtime and are not sent to third-party "
        "inference APIs.\n"
        "- Web fetching is off by default and only allowed when explicitly enabled.\n"
        "- Laptop-local mode uses the same Gradio, gateway, and local model architecture."
    )


def first_screen_disclosure() -> str:
    return (
        "**Hosted Space note:** this demo runs on Hugging Face Space compute when hosted. "
        "It does not call external model APIs, cloud OCR, or remote telemetry. "
        "Laptop-local mode uses the same Gradio, gateway, and local model architecture."
    )


def model_budget_status() -> dict[str, Any]:
    manifest = json.loads(MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))
    models = manifest.get("models", [])
    readiness = model_readiness_by_key(MODEL_MANIFEST_PATH)
    total_params = sum(
        int(model.get("base_params", 0)) + int(model.get("adapter_params", 0)) for model in models
    )
    required_ready = [
        key for key, status in readiness.items() if status["required"] and status["ready"]
    ]
    pending = [
        key for key, status in readiness.items() if status["required"] and not status["ready"]
    ]
    checksum_configured = [
        key
        for key, status in readiness.items()
        if status["required"] and status["checksum_configured"]
    ]
    artifact_present = [
        key
        for key, status in readiness.items()
        if status["required"] and status["artifact_present"]
    ]
    return {
        "parameter_cap_b": 32,
        "total_params_b": round(total_params / 1_000_000_000, 2),
        "within_cap": total_params <= 32_000_000_000,
        "allowed_vendors": manifest.get("allowed_vendors", []),
        "ready_required_models": required_ready,
        "pending_required_models": pending,
        "checksum_configured_required_models": checksum_configured,
        "artifact_present_required_models": artifact_present,
        "required_model_readiness": readiness,
        "local_only": bool(manifest.get("local_only")),
        "privacy_mode": manifest.get("privacy_mode"),
    }


def demo_readiness_status() -> tuple[str, dict[str, Any]]:
    health_summary, health_payload = space_health()
    del health_summary
    budget = model_budget_status()
    samples = demo_sample_paths()
    release_gate = release_gate_metadata_status()
    ready_required = set(budget["ready_required_models"])
    checksum_configured = set(budget["checksum_configured_required_models"])
    artifact_present = set(budget["artifact_present_required_models"])
    pending_required = set(budget["pending_required_models"])
    checks = {
        "strict_privacy": health_payload["privacy_mode"] == "strict"
        and health_payload["allow_web"] is False
        and health_payload["telemetry_enabled"] is False,
        "model_budget_under_32b": budget["within_cap"] is True,
        "text_model_checksum_ready": "text" in checksum_configured,
        "vision_model_checksum_ready": "vision" in checksum_configured,
        "asr_model_checksum_ready": "asr" in checksum_configured,
        "text_model_artifact_present": "text" in artifact_present,
        "vision_model_artifact_present": "vision" in artifact_present,
        "asr_model_artifact_present": "asr" in artifact_present,
        "text_model_ready": "text" in ready_required,
        "vision_model_ready": "vision" in ready_required,
        "asr_model_ready": "asr" in ready_required,
        "release_gate_metadata_ready": release_gate["status"] == "ok",
        "demo_samples_available": {"question", "article", "invoice", "bank_statement"}
        <= set(samples),
        "generated_demo_media_available": Path(load_demo_image()).exists()
        and Path(load_demo_audio()).exists(),
    }
    blocking_release_gaps = []
    if "asr" in pending_required:
        blocking_release_gaps.append("Parakeet ASR artifact and checksum are not staged.")
    for key in ("text", "vision"):
        if key in pending_required:
            blocking_release_gaps.append(f"Required {key} model artifact/checksum is not ready.")
    if not checks["demo_samples_available"]:
        blocking_release_gaps.append("One or more demo samples are missing.")
    for failure in release_gate.get("failures", []):
        release_failure = f"Release gate: {failure}"
        if release_failure not in blocking_release_gaps:
            blocking_release_gaps.append(release_failure)
    demo_ready = (
        checks["strict_privacy"]
        and checks["model_budget_under_32b"]
        and checks["text_model_ready"]
        and checks["vision_model_ready"]
        and checks["demo_samples_available"]
        and checks["generated_demo_media_available"]
    )
    next_commands = ["python3 scripts/smoke_test_local.py --mock-model-endpoints"]
    if not checks["asr_model_ready"]:
        next_commands.append(
            "python3 scripts/verify_model_checksums.py --model asr --write-manifest-checksum"
        )
    next_commands.extend(
        [
            "python3 scripts/smoke_test_local.py --require-real-model-services",
            "python3 scripts/release_gate.py",
        ]
    )
    payload = {
        "demo_ready": demo_ready,
        "release_ready": demo_ready
        and checks["asr_model_ready"]
        and checks["release_gate_metadata_ready"],
        "checks": checks,
        "release_gate": release_gate,
        "blocking_release_gaps": blocking_release_gaps,
        "sample_paths": samples,
        "next_commands": next_commands,
    }
    markdown_lines = [
        "### Demo Readiness",
        "",
        f"- Demo path: {'ready' if demo_ready else 'needs attention'}",
        f"- Release gate: {'ready' if payload['release_ready'] else 'not ready'}",
        f"- Strict local/privacy defaults: {_yes_no(checks['strict_privacy'])}",
        f"- Parameter budget under 32B: {_yes_no(checks['model_budget_under_32b'])}",
        f"- Text model artifact/checksum: {_yes_no(checks['text_model_ready'])}",
        f"- Vision model artifact/checksum: {_yes_no(checks['vision_model_ready'])}",
        f"- ASR model artifact/checksum: {_yes_no(checks['asr_model_ready'])}",
        f"- Release gate metadata: {_yes_no(checks['release_gate_metadata_ready'])}",
        f"- Demo samples: {_yes_no(checks['demo_samples_available'])}",
        f"- Generated demo image/audio: {_yes_no(checks['generated_demo_media_available'])}",
    ]
    if blocking_release_gaps:
        markdown_lines.extend(["", "Release blockers:"])
        markdown_lines.extend(f"- {gap}" for gap in blocking_release_gaps)
    markdown_lines.extend(["", "Next local checks:"])
    markdown_lines.extend(f"- `{command}`" for command in payload["next_commands"])
    return "\n".join(markdown_lines), payload


def release_gate_metadata_status() -> dict[str, Any]:
    try:
        return run_release_gate(verify_checksums=False)
    except ReleaseGateError as exc:
        return exc.summary
    except (FileNotFoundError, KeyError, ValueError) as exc:
        return {
            "status": "failed",
            "checksum_verification": False,
            "failures": [str(exc)],
            "next_actions": [
                "Inspect models/manifest.json and configs/models.yaml against release requirements."
            ],
        }


def run_local_demo_flow() -> tuple[str, dict[str, Any]]:
    question_text = load_demo_question()
    article_text = load_demo_article()
    document_path = load_demo_document("invoice")
    image_path = load_demo_image()
    audio_path = load_demo_audio()

    ask_text, ask_payload = ask_assistant(question_text)
    article_summary, article_payload = summarize_article(article_text, "", False)
    document_json, document_files, document_warnings = convert_document(
        document_path,
        ["JSON", "XLSX"],
    )
    image_result, image_payload = describe_photo(image_path)
    speech_result, speech_payload = transcribe_speech(audio_path)

    document_payload = _parse_json_payload(document_json)
    steps = {
        "ask": _demo_step(ask_text, ask_payload, expects_visible_warning=True),
        "read": _demo_step(article_summary, article_payload, expects_visible_warning=True),
        "document": {
            **_demo_step(document_json, document_payload, expects_visible_warning=False),
            "downloads": document_files,
            "warnings_visible": bool(document_warnings.strip()),
        },
        "image": _demo_step(image_result, image_payload, expects_visible_warning=True),
        "speech": _demo_step(speech_result, speech_payload, expects_visible_warning=True),
    }
    checks = {
        "samples_loaded": bool(question_text and article_text and document_path),
        "generated_media_loaded": Path(image_path).exists() and Path(audio_path).exists(),
        "document_exports_created": all(Path(path).exists() for path in document_files),
        "all_steps_local": all(step["local_only"] for step in steps.values()),
        "structured_outputs": all(bool(step["task"]) for step in steps.values()),
        "visible_warnings": all(step["warnings_visible"] for step in steps.values()),
    }
    payload = {
        "status": "ok" if all(checks.values()) else "needs_attention",
        "local_only": True,
        "external_model_apis": False,
        "cloud_ocr": False,
        "remote_telemetry": False,
        "checks": checks,
        "steps": steps,
        "sample_paths": {
            "document": document_path,
            "image": image_path,
            "audio": audio_path,
        },
    }
    summary_lines = [
        "### Local Demo Flow",
        "",
        f"- Status: {payload['status']}",
        f"- Samples loaded: {_yes_no(checks['samples_loaded'])}",
        f"- Generated image/audio loaded: {_yes_no(checks['generated_media_loaded'])}",
        f"- Document exports created: {_yes_no(checks['document_exports_created'])}",
        f"- All steps local: {_yes_no(checks['all_steps_local'])}",
        f"- Structured outputs: {_yes_no(checks['structured_outputs'])}",
        f"- Visible warnings: {_yes_no(checks['visible_warnings'])}",
    ]
    return "\n".join(summary_lines), payload


def _parse_json_payload(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _demo_step(
    visible_text: str,
    payload: dict[str, Any],
    *,
    expects_visible_warning: bool,
) -> dict[str, Any]:
    warnings = payload.get("warnings", []) if isinstance(payload, dict) else []
    return {
        "task": payload.get("task", "") if isinstance(payload, dict) else "",
        "status": payload.get("status", "") if isinstance(payload, dict) else "",
        "local_only": payload.get("local_only") is True if isinstance(payload, dict) else False,
        "human_review_required": bool(payload.get("human_review_required", False))
        if isinstance(payload, dict)
        else False,
        "warnings_count": len(warnings) if isinstance(warnings, list) else 0,
        "warnings_visible": ("warnings:" in visible_text.casefold())
        if expects_visible_warning
        else True,
    }


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def accessibility_status(high_contrast: bool) -> str:
    return "High contrast mode is on." if high_contrast else "High contrast mode is off."


def accessibility_style(high_contrast: bool) -> str:
    return HIGH_CONTRAST_CSS if high_contrast else DEFAULT_ACCESSIBILITY_CSS


def accessibility_settings(high_contrast: bool) -> tuple[str, str]:
    return accessibility_status(high_contrast), accessibility_style(high_contrast)


def space_health() -> tuple[str, dict[str, Any]]:
    payload = gateway_health().model_dump(mode="json")
    services = ", ".join(service["name"] for service in payload["services"])
    ready_models = [
        service["name"]
        for service in payload["services"]
        if service.get("required") and service.get("ready")
    ]
    pending_models = [
        service["name"]
        for service in payload["services"]
        if service.get("required") and not service.get("ready")
    ]
    summary = (
        f"Status: {payload['status']}\n\n"
        f"Privacy: {payload['privacy_mode']}\n\n"
        f"Web: {'enabled' if payload['allow_web'] else 'off by default'}\n\n"
        f"Services: {services}\n\n"
        f"Ready required models: {', '.join(ready_models) if ready_models else 'none'}\n\n"
        f"Pending required models: {', '.join(pending_models) if pending_models else 'none'}\n\n"
        "No external model APIs or cloud OCR are configured."
    )
    return summary, payload


def route_intent(
    intent: str,
    file_path: Any,
    url: str,
    allow_web: bool,
    manual_task: str = AUTO_ROUTE,
) -> dict[str, Any]:
    if manual_task and manual_task != AUTO_ROUTE:
        router = load_routes(ROUTES_PATH)
        target = router.route(manual_task)
        return {
            "task": manual_task,
            "provider": target.provider,
            "model_key": target.model_key,
            "endpoint": str(target.endpoint),
            "privacy_mode": router.runtime.privacy_mode,
            "allow_web": router.runtime.allow_web and bool(allow_web),
            "manual_override": True,
            "reason": "Manual route override selected in the Gradio Space.",
        }
    decision = gateway_route_request(
        RouteRequest(
            intent=intent or "",
            file_path=_file_path(file_path),
            url=url or None,
            allow_web=bool(allow_web),
        )
    )
    payload = dict(decision)
    payload["manual_override"] = False
    return payload


def summarize_article(text: str, url: str, allow_web: bool) -> tuple[str, dict[str, Any]]:
    request = TaskRequest(text=text or None, url=url or None, allow_web=bool(allow_web))
    if url and "wikipedia.org" not in url.casefold():
        response = gateway_summarize_url(request)
    else:
        response = gateway_summarize_wikipedia(request)
    payload = response.model_dump(mode="json")
    summary = payload.get("result", {}).get("summary", "")
    if not summary:
        summary = _to_json(payload.get("result", {}))
    return _with_visible_warnings(summary, payload), payload


def ask_assistant(question: str) -> tuple[str, dict[str, Any]]:
    if not (question or "").strip():
        return "Type a question first.", {}
    response = gateway_general_local_assistant(TaskRequest(text=question))
    payload = response.model_dump(mode="json")
    answer = str(payload.get("result", {}).get("text", ""))
    return _with_visible_warnings(answer, payload), payload


def convert_document(document_file: Any, output_formats: list[str]) -> tuple[str, list[str], str]:
    file_path = _file_path(document_file)
    if not file_path:
        return "No local document selected.", [], "Select a local file first."
    response = gateway_document_to_excel(TaskRequest(file_path=file_path))
    payload = response.model_dump(mode="json")
    base_name = Path(file_path).stem or "document"
    files = _write_exports(base_name, payload, output_formats or ["JSON"])
    return _to_json(payload), files, "\n".join(payload.get("warnings", []))


def describe_photo(image_file: Any) -> tuple[str, dict[str, Any]]:
    file_path = _file_path(image_file)
    if not file_path:
        return "No local image selected.", {}
    response = gateway_describe_image(TaskRequest(file_path=file_path, mime_type="image/png"))
    payload = response.model_dump(mode="json")
    return _with_visible_warnings(
        str(payload.get("result", {}).get("description", "")), payload
    ), payload


def translate_visible_text(image_file: Any, target_language: str) -> tuple[str, dict[str, Any]]:
    file_path = _file_path(image_file)
    if not file_path:
        return "No local image selected.", {}
    response = gateway_translate_image_text(
        TaskRequest(
            file_path=file_path,
            mime_type="image/png",
            target_language=target_language or "English",
        )
    )
    payload = response.model_dump(mode="json")
    return _with_visible_warnings(str(payload.get("result", {}).get("text", "")), payload), payload


def transcribe_speech(
    audio_file: Any,
    language: str = "en",
    region: str = "unknown",
    allow_experimental_asr: bool = False,
) -> tuple[str, dict[str, Any]]:
    file_path = _file_path(audio_file)
    if not file_path:
        return "No local audio selected.", {}
    response = gateway_speech_to_text(
        TaskRequest(
            file_path=file_path,
            mime_type="audio/wav",
            language=language or "en",
            region=region or "unknown",
            country="unknown",
            allow_experimental_asr=bool(allow_experimental_asr),
        )
    )
    payload = response.model_dump(mode="json")
    return _with_visible_warnings(str(payload.get("result", {}).get("text", "")), payload), payload


def save_ask_result(text: str, status: Any, formats: list[str]) -> tuple[list[str], str]:
    return _save_named_text_result("ask_answer", text, status, formats)


def save_article_result(text: str, status: Any, formats: list[str]) -> tuple[list[str], str]:
    return _save_named_text_result("article_summary", text, status, formats)


def save_image_result(text: str, status: Any, formats: list[str]) -> tuple[list[str], str]:
    return _save_named_text_result("image_result", text, status, formats)


def save_speech_result(text: str, status: Any, formats: list[str]) -> tuple[list[str], str]:
    return _save_named_text_result("speech_transcript", text, status, formats)


def _save_named_text_result(
    base_name: str,
    text: str,
    status: Any,
    formats: list[str],
) -> tuple[list[str], str]:
    payload = status if isinstance(status, dict) else {}
    return save_text_result_exports(base_name, text or "", payload, formats or [])


def build_demo() -> gr.Blocks:
    with gr.Blocks(
        title="local-lm",
        analytics_enabled=False,
    ) as demo:
        gr.HTML(value=base_style_html(), show_label=False)
        gr.Markdown("# local-lm")
        gr.Markdown(first_screen_disclosure(), elem_classes=["local-lm-status"])
        with gr.Row():
            health_text = gr.Markdown(elem_classes=["local-lm-status"])
            with gr.Accordion(ADVANCED_RUNTIME_LABEL, open=False):
                health_json = gr.JSON(label="Runtime")
        demo.load(space_health, outputs=[health_text, health_json])

        with gr.Tabs():
            with gr.Tab("Ask"):
                load_question_button = gr.Button("Load Sample Question")
                ask_text = gr.Textbox(label="Question", lines=4)
                ask_button = gr.Button("Ask", variant="primary")
                ask_output = gr.Textbox(label="Answer", lines=8)
                ask_status = gr.JSON(label="Status")
                ask_save_formats = gr.CheckboxGroup(
                    ["TXT", "XLSX", "PDF"],
                    label="Save result as",
                    value=["TXT", "PDF"],
                )
                ask_save_button = gr.Button("Save Answer")
                ask_files = gr.File(label="Saved answer", file_count="multiple")
                ask_save_status = gr.Textbox(label="Save status", lines=1)
                ask_button.click(
                    ask_assistant,
                    inputs=[ask_text],
                    outputs=[ask_output, ask_status],
                )
                load_question_button.click(load_demo_question, outputs=[ask_text])
                ask_save_button.click(
                    save_ask_result,
                    inputs=[ask_output, ask_status, ask_save_formats],
                    outputs=[ask_files, ask_save_status],
                )

            with gr.Tab("Read"):
                load_article_button = gr.Button("Load Sample Article")
                article_text = gr.Textbox(label="Article or document text", lines=8)
                article_url = gr.Textbox(label="URL")
                article_allow_web = gr.Checkbox(label="Allow optional web fetch", value=False)
                article_button = gr.Button("Summarize", variant="primary")
                article_output = gr.Textbox(label="Summary", lines=8)
                article_status = gr.JSON(label="Status")
                article_save_formats = gr.CheckboxGroup(
                    ["TXT", "XLSX", "PDF"],
                    label="Save result as",
                    value=["TXT", "PDF"],
                )
                article_save_button = gr.Button("Save Summary")
                article_files = gr.File(label="Saved summary", file_count="multiple")
                article_save_status = gr.Textbox(label="Save status", lines=1)
                article_button.click(
                    summarize_article,
                    inputs=[article_text, article_url, article_allow_web],
                    outputs=[article_output, article_status],
                )
                load_article_button.click(load_demo_article, outputs=[article_text])
                article_save_button.click(
                    save_article_result,
                    inputs=[article_output, article_status, article_save_formats],
                    outputs=[article_files, article_save_status],
                )

            with gr.Tab("Documents"):
                sample_document = gr.Dropdown(
                    ["invoice", "bank_statement"],
                    label="Demo sample",
                    value="invoice",
                )
                load_document_button = gr.Button("Load Sample Document")
                document_file = gr.File(
                    label="Bill, invoice, receipt, note, or statement",
                    file_types=[".pdf", ".png", ".jpg", ".jpeg", ".webp", ".txt"],
                    type="filepath",
                )
                export_formats = gr.CheckboxGroup(
                    ["JSON", "XLSX", "TXT", "PDF"],
                    label="Save as",
                    value=["JSON", "XLSX"],
                )
                document_button = gr.Button("Convert", variant="primary")
                document_json = gr.Textbox(label="Extracted JSON", lines=12)
                document_files = gr.File(label="Downloads", file_count="multiple")
                document_warnings = gr.Textbox(label="Warnings", lines=3)
                document_button.click(
                    convert_document,
                    inputs=[document_file, export_formats],
                    outputs=[document_json, document_files, document_warnings],
                )
                load_document_button.click(
                    load_demo_document,
                    inputs=[sample_document],
                    outputs=[document_file],
                )

            with gr.Tab("Images"):
                load_image_button = gr.Button("Load Sample Image")
                image_file = gr.Image(
                    label="Photo or image",
                    sources=["upload", "webcam", "clipboard"],
                    type="filepath",
                )
                with gr.Row():
                    describe_button = gr.Button("Describe", variant="primary")
                    translate_button = gr.Button("Translate Text", variant="primary")
                target_language = gr.Textbox(label="Target language", value="English")
                image_text = gr.Textbox(label="Result", lines=8)
                image_status = gr.JSON(label="Status")
                image_save_formats = gr.CheckboxGroup(
                    ["TXT", "XLSX", "PDF"],
                    label="Save result as",
                    value=["TXT", "PDF"],
                )
                image_save_button = gr.Button("Save Image Result")
                image_files = gr.File(label="Saved image result", file_count="multiple")
                image_save_status = gr.Textbox(label="Save status", lines=1)
                describe_button.click(
                    describe_photo,
                    inputs=[image_file],
                    outputs=[image_text, image_status],
                )
                translate_button.click(
                    translate_visible_text,
                    inputs=[image_file, target_language],
                    outputs=[image_text, image_status],
                )
                load_image_button.click(load_demo_image, outputs=[image_file])
                image_save_button.click(
                    save_image_result,
                    inputs=[image_text, image_status, image_save_formats],
                    outputs=[image_files, image_save_status],
                )

            with gr.Tab("Speech"):
                load_audio_button = gr.Button("Load Sample Speech")
                audio_file = gr.Audio(
                    label="Speech",
                    sources=["upload", "microphone"],
                    type="filepath",
                )
                speech_language = gr.Textbox(label="Language code", value="en")
                speech_region = gr.Dropdown(
                    ["unknown", "india", "southeast_asia", "north_america", "europe"],
                    label="Region",
                    value="unknown",
                )
                speech_allow_experimental = gr.Checkbox(
                    label="Allow experimental ASR",
                    value=False,
                )
                speech_button = gr.Button("Transcribe", variant="primary")
                speech_text = gr.Textbox(label="Transcript", lines=6)
                speech_status = gr.JSON(label="Status")
                speech_save_formats = gr.CheckboxGroup(
                    ["TXT", "XLSX", "PDF"],
                    label="Save result as",
                    value=["TXT", "PDF"],
                )
                speech_save_button = gr.Button("Save Transcript")
                speech_files = gr.File(label="Saved transcript", file_count="multiple")
                speech_save_status = gr.Textbox(label="Save status", lines=1)
                speech_button.click(
                    transcribe_speech,
                    inputs=[
                        audio_file,
                        speech_language,
                        speech_region,
                        speech_allow_experimental,
                    ],
                    outputs=[speech_text, speech_status],
                )
                load_audio_button.click(load_demo_audio, outputs=[audio_file])
                speech_save_button.click(
                    save_speech_result,
                    inputs=[speech_text, speech_status, speech_save_formats],
                    outputs=[speech_files, speech_save_status],
                )

            with gr.Tab("Route"):
                route_intent_box = gr.Textbox(label="Intent", lines=3)
                route_file = gr.File(label="Optional local file", type="filepath")
                route_url = gr.Textbox(label="Optional URL")
                route_allow_web = gr.Checkbox(label="Allow optional web fetch", value=False)
                route_override = gr.Dropdown(
                    TASK_CHOICES,
                    label="Manual task override",
                    value=AUTO_ROUTE,
                )
                route_button = gr.Button("Route", variant="primary")
                route_output = gr.JSON(label="Route")
                route_button.click(
                    route_intent,
                    inputs=[
                        route_intent_box,
                        route_file,
                        route_url,
                        route_allow_web,
                        route_override,
                    ],
                    outputs=[route_output],
                )

            with gr.Tab("Settings / Privacy"):
                gr.Markdown(privacy_disclosure())
                readiness_markdown = gr.Markdown()
                readiness_json = gr.JSON(label="Demo readiness")
                demo.load(
                    demo_readiness_status,
                    outputs=[readiness_markdown, readiness_json],
                )
                demo_flow_button = gr.Button("Run Local Demo Samples", variant="primary")
                demo_flow_summary = gr.Markdown()
                demo_flow_json = gr.JSON(label="Local demo flow")
                demo_flow_button.click(
                    run_local_demo_flow,
                    outputs=[demo_flow_summary, demo_flow_json],
                )
                high_contrast = gr.Checkbox(label="High contrast mode", value=True)
                accessibility_css = gr.HTML(value=accessibility_style(True), show_label=False)
                accessibility_output = gr.Markdown()
                model_budget = gr.JSON(label="Model budget")
                demo.load(
                    accessibility_settings,
                    inputs=[high_contrast],
                    outputs=[accessibility_output, accessibility_css],
                )
                high_contrast.change(
                    accessibility_settings,
                    inputs=[high_contrast],
                    outputs=[accessibility_output, accessibility_css],
                )
                demo.load(model_budget_status, outputs=[model_budget])
    return demo


demo = build_demo()


def launch() -> None:
    demo.launch()


if __name__ == "__main__":
    launch()
