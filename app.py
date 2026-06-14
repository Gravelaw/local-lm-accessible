# ruff: noqa: E501, SIM117

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
  --background-fill-primary: #0b0e17;
  --background-fill-secondary: #111623;
  --block-background-fill: #171b2a;
  --block-border-color: #2a3043;
  --body-text-color: #f7f8fb;
  --body-text-color-subdued: #93a0bd;
  --border-color-primary: #30374d;
  --border-color-accent: #ee7445;
  --input-background-fill: #1b2032;
  --input-border-color: #30374d;
  --input-placeholder-color: #a2abc3;
  --button-primary-background-fill: #ee7445;
  --button-primary-background-fill-hover: #f08257;
  --button-primary-text-color: #ffffff;
  max-width: none !important;
  min-height: 100vh;
  background: #0b0e17 !important;
  color: #f7f8fb !important;
  font-size: 18px;
  line-height: 1.5;
  padding: 0 !important;
}
.gradio-container .main {
  background: #0b0e17 !important;
}
.gradio-container .contain,
.gradio-container .wrap,
.gradio-container .form,
.gradio-container .block,
.gradio-container .block.padded,
.gradio-container .panel,
.gradio-container .tabs,
.gradio-container .tabitem,
.gradio-container .accordion,
.gradio-container .json-holder,
.gradio-container .file-preview {
  background: transparent !important;
  color: #f7f8fb !important;
}
.gradio-container button {
  background: #202638 !important;
  border: 1px solid #30374d !important;
  color: #f7f8fb !important;
  min-height: 56px;
  border-radius: 8px !important;
  font-size: 18px !important;
  font-weight: 700 !important;
  padding: 12px 18px !important;
}
.gradio-container button.primary {
  background: #ee7445 !important;
  border-color: #ee7445 !important;
  box-shadow: 0 12px 28px rgba(238, 116, 69, 0.22);
  color: #ffffff !important;
}
textarea, input, select {
  background: #1b2032 !important;
  border-color: #30374d !important;
  color: #f7f8fb !important;
  font-size: 18px !important;
  line-height: 1.45 !important;
}
label, .block-title, .wrap .label-wrap span, .gradio-container .prose {
  color: #eef2ff !important;
  font-weight: 700 !important;
}
.gradio-container .prose p,
.gradio-container .prose li,
.gradio-container .prose span {
  color: #d8def0 !important;
}
.form, .block, .panel, .tabs, .tabitem, .accordion {
  background: transparent !important;
}
.local-lm-shell {
  min-height: 100vh;
  background: #0b0e17;
}
.local-lm-topbar {
  align-items: center;
  border-bottom: 1px solid #202638;
  display: flex;
  gap: 18px;
  justify-content: space-between;
  padding: 18px 8vw;
}
.local-lm-brand {
  align-items: center;
  display: flex;
  gap: 14px;
  min-width: 0;
}
.local-lm-mark {
  align-items: center;
  background: #ee7445;
  border-radius: 8px;
  color: #ffffff;
  display: inline-flex;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 24px;
  font-weight: 800;
  height: 52px;
  justify-content: center;
  width: 52px;
}
.local-lm-name {
  color: #f7f8fb;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 28px;
  font-weight: 800;
  letter-spacing: 0;
  white-space: nowrap;
}
.local-lm-subtitle,
.local-lm-muted {
  color: #93a0bd;
}
.local-lm-badges {
  align-items: center;
  display: flex;
  flex-wrap: wrap;
  gap: 14px;
  justify-content: flex-end;
}
.local-lm-badge {
  background: #1b2032;
  border: 1px solid #30374d;
  border-radius: 999px;
  color: #aab4cd;
  display: inline-flex;
  font-weight: 700;
  padding: 10px 18px;
}
.local-lm-badge.local {
  background: #132820;
  border-color: #255143;
  color: #6bd6a2;
}
.local-lm-layout {
  display: grid;
  gap: 36px;
  grid-template-columns: minmax(240px, 300px) minmax(0, 1fr);
  margin: 0 auto;
  max-width: 1500px;
  padding: 28px 8vw 48px;
}
.local-lm-rail {
  align-self: start;
  background: #0f1320;
  border: 1px solid #1a2030;
  border-radius: 8px;
  padding: 16px;
  position: sticky;
  top: 24px;
}
.local-lm-rail-title {
  color: #93a0bd;
  font-size: 15px;
  font-weight: 800;
  letter-spacing: 0.08em;
  padding: 8px 12px 16px;
  text-transform: uppercase;
}
.local-lm-rail-item {
  align-items: center;
  border-radius: 8px;
  color: #94a2c2;
  display: flex;
  font-weight: 700;
  gap: 12px;
  margin: 8px 0;
  min-height: 52px;
  padding: 12px 14px;
}
.local-lm-rail-item.active {
  background: #ee7445;
  box-shadow: 0 12px 28px rgba(238, 116, 69, 0.22);
  color: #ffffff;
}
.local-lm-rail-icon {
  align-items: center;
  border: 1px solid currentColor;
  border-radius: 6px;
  display: inline-flex;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 14px;
  height: 28px;
  justify-content: center;
  width: 28px;
}
.local-lm-content {
  min-width: 0;
}
.local-lm-section-title h2,
.local-lm-section-title h3 {
  color: #f7f8fb;
  font-size: 32px;
  line-height: 1.15;
  margin: 0;
}
.local-lm-section-title p {
  color: #9aa7c5;
  font-size: 22px;
  margin: 8px 0 20px;
}
.local-lm-card {
  background: #171b2a;
  border: 1px solid #2a3043;
  border-radius: 8px;
  margin-bottom: 18px;
  padding: 24px;
}
.local-lm-card > .block,
.local-lm-output-card > .block {
  background: transparent !important;
}
.local-lm-card.compact {
  padding: 20px;
}
.local-lm-output-card {
  background: #111623;
  border: 1px solid #2a3043;
  border-radius: 8px;
  padding: 24px;
}
.local-lm-result-drawer {
  background: #111623 !important;
  border: 1px solid #2a3043 !important;
  border-radius: 8px !important;
  margin-top: 18px;
}
.local-lm-result-drawer button {
  background: #171b2a !important;
  color: #f7f8fb !important;
  min-height: 46px !important;
}
.local-lm-status {
  background: #121827;
  border: 1px solid #2a3043;
  border-radius: 8px;
  color: #d8def0;
  padding: 16px 18px;
  font-size: 17px;
}
.local-lm-status strong {
  color: #6bd6a2;
}
.local-lm-disclosure {
  border-left: 4px solid #ee7445;
}
.local-lm-footer {
  align-items: center;
  border-top: 1px solid #202638;
  color: #93a0bd;
  display: flex;
  flex-wrap: wrap;
  gap: 18px;
  justify-content: space-between;
  padding: 24px 8vw;
}
.local-lm-cap {
  color: #6bd6a2;
  font-weight: 800;
}
.local-lm-tabs .tab-nav {
  background: #0f1320 !important;
  border: 1px solid #1f2536 !important;
  border-radius: 8px !important;
  display: flex !important;
  flex-wrap: wrap !important;
  gap: 8px !important;
  margin-bottom: 28px !important;
  overflow: visible !important;
  padding: 8px !important;
}
.local-lm-tabs .tab-container {
  background: #0f1320 !important;
  border: 1px solid #1f2536 !important;
  border-radius: 8px !important;
  display: flex !important;
  flex-wrap: wrap !important;
  gap: 8px !important;
  overflow: visible !important;
  padding: 8px !important;
}
.local-lm-tabs .tab-nav button {
  border-radius: 8px !important;
  color: #96a2bf !important;
  flex: 0 1 auto !important;
  min-height: 48px !important;
}
.local-lm-tabs .tab-container button {
  border-radius: 8px !important;
  color: #96a2bf !important;
  flex: 0 1 auto !important;
  min-height: 48px !important;
}
.local-lm-tabs .tab-nav button.selected {
  background: #ee7445 !important;
  color: #ffffff !important;
}
.local-lm-tabs .tab-container button.selected {
  background: #ee7445 !important;
  color: #ffffff !important;
}
.local-lm-uploader .wrap,
.local-lm-uploader .upload-container,
.local-lm-uploader .file-preview,
.local-lm-uploader .image-container,
.local-lm-uploader .audio-container {
  background: #171b2a !important;
  border: 2px dashed #3a4158 !important;
  border-radius: 8px !important;
  min-height: 190px;
}
.local-lm-settings-panel {
  background: #171b2a;
  border: 1px solid #2a3043;
  border-radius: 8px;
  overflow: hidden;
}
.local-lm-panel-head {
  background: #222842;
  color: #6bd6a2;
  font-size: 18px;
  font-weight: 800;
  letter-spacing: 0.12em;
  padding: 18px 22px;
  text-transform: uppercase;
}
.local-lm-panel-body {
  padding: 22px;
}
.local-lm-chip-row {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}
.local-lm-chip {
  background: #152820;
  border: 1px solid #2b5749;
  border-radius: 6px;
  color: #6bd6a2;
  display: inline-flex;
  font-weight: 700;
  padding: 8px 12px;
}
.local-lm-chip.warn {
  background: #2b2419;
  border-color: #6c5430;
  color: #f1cc68;
}
.local-lm-sr-only {
  clip: rect(0 0 0 0);
  clip-path: inset(50%);
  height: 1px;
  overflow: hidden;
  position: absolute;
  white-space: nowrap;
  width: 1px;
}
@media (max-width: 980px) {
  .local-lm-topbar,
  .local-lm-footer {
    align-items: flex-start;
    padding-left: 22px;
    padding-right: 22px;
  }
  .local-lm-topbar {
    display: grid;
    grid-template-columns: minmax(0, 1fr);
    row-gap: 12px;
  }
  .local-lm-badges {
    align-content: start;
    flex-direction: row;
    gap: 8px;
    justify-content: flex-start;
  }
  .local-lm-badge {
    padding: 8px 12px;
  }
  .local-lm-name {
    font-size: 24px;
  }
  .local-lm-layout {
    grid-template-columns: 1fr;
    padding: 22px 18px 42px;
  }
  .local-lm-rail {
    display: none !important;
  }
  .local-lm-section-title h2,
  .local-lm-section-title h3 {
    font-size: 28px;
  }
  .local-lm-section-title p {
    font-size: 19px;
  }
  .local-lm-card {
    padding: 18px;
  }
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
.local-lm-card,
.local-lm-output-card,
.local-lm-rail,
.local-lm-settings-panel {
  background: #000 !important;
  border-color: #fff !important;
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


def build_demo_legacy() -> gr.Blocks:
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
                high_contrast = gr.Checkbox(label="High contrast mode", value=False)
                accessibility_css = gr.HTML(value=accessibility_style(False), show_label=False)
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


def build_demo() -> gr.Blocks:
    with gr.Blocks(
        title="local-lm",
        analytics_enabled=False,
    ) as demo:
        gr.HTML(value=base_style_html(), show_label=False)
        gr.Markdown("# local-lm", elem_classes=["local-lm-sr-only"])
        gr.HTML(
            """
            <div class="local-lm-topbar">
              <div class="local-lm-brand">
                <div class="local-lm-mark">LM</div>
                <div>
                  <div class="local-lm-name">local-lm</div>
                  <div class="local-lm-subtitle">Backyard AI - HF Build Small Hackathon</div>
                </div>
              </div>
              <div class="local-lm-badges">
                <span class="local-lm-badge local">Local only</span>
                <span class="local-lm-badge">~8.6B active params</span>
              </div>
            </div>
            """,
            show_label=False,
        )
        gr.Markdown(
            first_screen_disclosure(),
            elem_classes=["local-lm-status", "local-lm-disclosure"],
        )
        with gr.Row(elem_classes=["local-lm-layout"]):
            with gr.Column(scale=1, min_width=240, elem_classes=["local-lm-rail"]):
                gr.HTML(
                    """
                    <div class="local-lm-rail-title">Tasks</div>
                    <div class="local-lm-rail-item"><span class="local-lm-rail-icon">RD</span>Read & Summarise</div>
                    <div class="local-lm-rail-item"><span class="local-lm-rail-icon">DC</span>Convert Document</div>
                    <div class="local-lm-rail-item"><span class="local-lm-rail-icon">IM</span>Describe Image</div>
                    <div class="local-lm-rail-item"><span class="local-lm-rail-icon">TR</span>Translate Image Text</div>
                    <div class="local-lm-rail-item"><span class="local-lm-rail-icon">ST</span>Speech to Text</div>
                    <div class="local-lm-rail-item"><span class="local-lm-rail-icon">PR</span>Privacy & Settings</div>
                    """,
                    show_label=False,
                )
                health_text = gr.Markdown(elem_classes=["local-lm-status"])
                with gr.Accordion(ADVANCED_RUNTIME_LABEL, open=False):
                    health_json = gr.JSON(label="Runtime")

            with (
                gr.Column(scale=4, min_width=520, elem_classes=["local-lm-content"]),
                gr.Tabs(elem_classes=["local-lm-tabs"]),
            ):
                    with gr.Tab("Read"):
                        gr.HTML(
                            """
                            <div class="local-lm-section-title">
                              <h2>Read & Summarise</h2>
                              <p>Paste text, load a local document sample, or enter a URL to get a simple summary.</p>
                            </div>
                            """,
                            show_label=False,
                        )
                        with gr.Group(elem_classes=["local-lm-card"]):
                            load_article_button = gr.Button("Load sample file")
                            article_text = gr.Textbox(
                                label="Paste your article or document text below",
                                lines=6,
                                placeholder="Paste text here - article, Wikipedia page, or any document...",
                            )
                            article_url = gr.Textbox(
                                label="Optional web URL",
                                placeholder="URL summarisation stays blocked unless web fetch is enabled.",
                            )
                            article_allow_web = gr.Checkbox(
                                label="Allow optional web fetch",
                                value=False,
                            )
                            article_button = gr.Button("Summarise", variant="primary")
                        with gr.Accordion(
                            "Summary and exports",
                            open=False,
                            elem_classes=["local-lm-result-drawer"],
                        ):
                            with gr.Group(elem_classes=["local-lm-output-card"]):
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

                    with gr.Tab("Docs"):
                        gr.HTML(
                            """
                            <div class="local-lm-section-title">
                              <h2>Convert Document</h2>
                              <p>Upload a bill, invoice, receipt, handwritten note, or bank statement and save structured local exports.</p>
                            </div>
                            """,
                            show_label=False,
                        )
                        with gr.Group(elem_classes=["local-lm-card"]):
                            with gr.Row():
                                sample_document = gr.Dropdown(
                                    ["invoice", "bank_statement"],
                                    label="Demo sample",
                                    value="invoice",
                                )
                                load_document_button = gr.Button("Load sample file")
                            document_file = gr.File(
                                label="Click to upload or drag and drop a document",
                                file_types=[".pdf", ".png", ".jpg", ".jpeg", ".webp", ".txt"],
                                type="filepath",
                                elem_classes=["local-lm-uploader"],
                            )
                            export_formats = gr.CheckboxGroup(
                                ["JSON", "XLSX", "TXT", "PDF"],
                                label="Save as",
                                value=["JSON", "XLSX"],
                            )
                            document_button = gr.Button("Convert to Excel + JSON", variant="primary")
                        with gr.Accordion(
                            "Extracted data and downloads",
                            open=False,
                            elem_classes=["local-lm-result-drawer"],
                        ):
                            with gr.Group(elem_classes=["local-lm-output-card"]):
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

                    with gr.Tab("Image"):
                        gr.HTML(
                            """
                            <div class="local-lm-section-title">
                              <h2>Describe Image</h2>
                              <p>Upload a photo to get a simple accessible description with visible text and hazard warnings.</p>
                            </div>
                            """,
                            show_label=False,
                        )
                        with gr.Group(elem_classes=["local-lm-card"]):
                            load_image_button = gr.Button("Load sample image")
                            image_file = gr.Image(
                                label="Click to upload or drag and drop an image",
                                sources=["upload", "webcam", "clipboard"],
                                type="filepath",
                                elem_classes=["local-lm-uploader"],
                            )
                            describe_button = gr.Button("Describe Image", variant="primary")
                        with gr.Accordion(
                            "Description and exports",
                            open=False,
                            elem_classes=["local-lm-result-drawer"],
                        ):
                            with gr.Group(elem_classes=["local-lm-output-card"]):
                                image_text = gr.Textbox(label="Accessible result", lines=8)
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
                        load_image_button.click(load_demo_image, outputs=[image_file])
                        image_save_button.click(
                            save_image_result,
                            inputs=[image_text, image_status, image_save_formats],
                            outputs=[image_files, image_save_status],
                        )

                    with gr.Tab("Text"):
                        gr.HTML(
                            """
                            <div class="local-lm-section-title">
                              <h2>Translate Image Text</h2>
                              <p>Upload a sign, label, menu, notice, or document and translate the visible text.</p>
                            </div>
                            """,
                            show_label=False,
                        )
                        with gr.Group(elem_classes=["local-lm-card"]):
                            translate_image_file = gr.Image(
                                label="Upload image with text to translate",
                                sources=["upload", "webcam", "clipboard"],
                                type="filepath",
                                elem_classes=["local-lm-uploader"],
                            )
                            target_language = gr.Textbox(label="Translate to", value="English")
                            translate_button = gr.Button("Extract & Translate", variant="primary")
                            load_translate_image_button = gr.Button("Load sample image")
                        with gr.Accordion(
                            "Translation result",
                            open=False,
                            elem_classes=["local-lm-result-drawer"],
                        ):
                            with gr.Group(elem_classes=["local-lm-output-card"]):
                                translate_text = gr.Textbox(label="Translation result", lines=8)
                                translate_status = gr.JSON(label="Status")
                        translate_button.click(
                            translate_visible_text,
                            inputs=[translate_image_file, target_language],
                            outputs=[translate_text, translate_status],
                        )
                        load_translate_image_button.click(
                            load_demo_image,
                            outputs=[translate_image_file],
                        )

                    with gr.Tab("Speech"):
                        gr.HTML(
                            """
                            <div class="local-lm-section-title">
                              <h2>Speech to Text</h2>
                              <p>Record or upload audio to get a local transcript.</p>
                            </div>
                            """,
                            show_label=False,
                        )
                        with gr.Group(elem_classes=["local-lm-card"]):
                            load_audio_button = gr.Button("Load sample speech")
                            audio_file = gr.Audio(
                                label="Record from microphone or upload audio",
                                sources=["upload", "microphone"],
                                type="filepath",
                                elem_classes=["local-lm-uploader"],
                            )
                            with gr.Row():
                                speech_language = gr.Textbox(label="Language code", value="en")
                                speech_region = gr.Dropdown(
                                    ["unknown", "india", "southeast_asia", "north_america", "europe"],
                                    label="Region",
                                    value="unknown",
                                )
                            speech_allow_experimental = gr.Checkbox(
                                label="Allow experimental ASR",
                                info="Use only for unevaluated regional languages; results require review.",
                                value=False,
                            )
                            gr.HTML(
                                """
                                <div class="local-lm-chip-row">
                                  <span class="local-lm-chip">English</span>
                                  <span class="local-lm-chip">Spanish</span>
                                  <span class="local-lm-chip">French</span>
                                  <span class="local-lm-chip">German</span>
                                  <span class="local-lm-chip">Italian</span>
                                  <span class="local-lm-chip warn">Hindi experimental</span>
                                  <span class="local-lm-chip warn">Tamil experimental</span>
                                  <span class="local-lm-chip warn">Bahasa Melayu experimental</span>
                                </div>
                                """,
                                show_label=False,
                            )
                            speech_button = gr.Button("Transcribe", variant="primary")
                        with gr.Accordion(
                            "Transcript and exports",
                            open=False,
                            elem_classes=["local-lm-result-drawer"],
                        ):
                            with gr.Group(elem_classes=["local-lm-output-card"]):
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

                    with gr.Tab("Ask"):
                        gr.HTML(
                            """
                            <div class="local-lm-section-title">
                              <h2>Ask Local Assistant</h2>
                              <p>Ask a short question and get a speech-ready answer with warnings when review is needed.</p>
                            </div>
                            """,
                            show_label=False,
                        )
                        with gr.Group(elem_classes=["local-lm-card"]):
                            load_question_button = gr.Button("Load sample question")
                            ask_text = gr.Textbox(label="Question", lines=4)
                            ask_button = gr.Button("Ask", variant="primary")
                        with gr.Accordion(
                            "Answer and exports",
                            open=False,
                            elem_classes=["local-lm-result-drawer"],
                        ):
                            with gr.Group(elem_classes=["local-lm-output-card"]):
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

                    with gr.Tab("Route"):
                        gr.HTML(
                            """
                            <div class="local-lm-section-title">
                              <h2>Route a Task</h2>
                              <p>Inspect deterministic routing for a typed intent, local file, or optional URL.</p>
                            </div>
                            """,
                            show_label=False,
                        )
                        with gr.Group(elem_classes=["local-lm-card"]):
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
                        with gr.Accordion(
                            "Route result",
                            open=False,
                            elem_classes=["local-lm-result-drawer"],
                        ):
                            route_output = gr.JSON(label="Route", elem_classes=["local-lm-output-card"])
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

                    with gr.Tab("Prefs"):
                        gr.HTML(
                            """
                            <div class="local-lm-section-title">
                              <h2>Privacy & Settings</h2>
                              <p>Manage privacy mode, model details, demo readiness, and accessibility options.</p>
                            </div>
                            <div class="local-lm-settings-panel">
                              <div class="local-lm-panel-head">Privacy & Locality</div>
                              <div class="local-lm-panel-body">
                                <div class="local-lm-chip-row">
                                  <span class="local-lm-chip">No cloud inference</span>
                                  <span class="local-lm-chip">No cloud OCR</span>
                                  <span class="local-lm-chip">No remote telemetry</span>
                                  <span class="local-lm-chip warn">Web fetch disabled by default</span>
                                </div>
                              </div>
                            </div>
                            """,
                            show_label=False,
                        )
                        gr.Markdown(privacy_disclosure(), elem_classes=["local-lm-status"])
                        with gr.Group(elem_classes=["local-lm-card"]):
                            high_contrast = gr.Checkbox(label="High contrast mode", value=False)
                            accessibility_css = gr.HTML(value=accessibility_style(False), show_label=False)
                            accessibility_output = gr.Markdown()
                            model_budget = gr.JSON(label="Model budget")
                        with gr.Accordion(
                            "Readiness and demo flow",
                            open=False,
                            elem_classes=["local-lm-result-drawer"],
                        ):
                            with gr.Group(elem_classes=["local-lm-output-card"]):
                                readiness_markdown = gr.Markdown()
                                readiness_json = gr.JSON(label="Demo readiness")
                                demo_flow_button = gr.Button("Run Local Demo Samples", variant="primary")
                                demo_flow_summary = gr.Markdown()
                                demo_flow_json = gr.JSON(label="Local demo flow")
                        demo.load(
                            demo_readiness_status,
                            outputs=[readiness_markdown, readiness_json],
                        )
                        demo_flow_button.click(
                            run_local_demo_flow,
                            outputs=[demo_flow_summary, demo_flow_json],
                        )
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
        gr.HTML(
            """
            <div class="local-lm-footer">
              <span>local-lm - Hugging Face Build Small Hackathon - Backyard AI track</span>
              <span>Models: Nemotron 3.97B - MiniCPM-V ~4B - Parakeet 0.6B <span class="local-lm-cap">&lt;=32B cap</span></span>
            </div>
            """,
            show_label=False,
        )
        demo.load(space_health, outputs=[health_text, health_json])
    return demo


demo = build_demo()


def launch() -> None:
    demo.launch()


if __name__ == "__main__":
    launch()
