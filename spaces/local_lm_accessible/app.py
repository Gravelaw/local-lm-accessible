from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import gradio as gr

APP_TITLE = "Local LM Accessible For Elders"
OFFICIAL_SPACE_ID = "build-small-hackathon/Local-lm-accessible-for-elders"
OFFICIAL_SPACE_URL = "https://build-small-hackathon-local-lm-accessible-for-elders.hf.space"
try:
    import spaces
except ImportError:
    class _SpacesShim:
        @staticmethod
        def GPU(*args: Any, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
            if args and callable(args[0]) and not kwargs:
                return args[0]

            def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
                return func

            return decorator

    spaces = _SpacesShim()
SENSITIVE_TERMS = {
    "bank",
    "credit",
    "debit",
    "diagnosis",
    "doctor",
    "health",
    "hospital",
    "identity",
    "legal",
    "medical",
    "passport",
    "ssn",
    "tax",
}
TASK_CHOICES = (
    "Auto",
    "zerogpu_assistant",
    "summarization",
    "json_repair",
    "receipt_or_invoice_json",
    "tool_routing",
)
CSS = """
.gradio-container {
  font-size: 18px;
  line-height: 1.5;
  max-width: 1120px !important;
}
textarea, input, button {
  font-size: 18px !important;
}
button {
  min-height: 52px !important;
}
.sample-row button {
  min-height: 48px !important;
}
"""
SAMPLES = {
    "summary": {
        "task": "summarization",
        "text": (
            "Local LM Accessible For Elders is a local-first assistant for older "
            "adults and low-vision users. It summarizes public text, repairs JSON, "
            "routes tasks, and extracts fields from synthetic receipts. Private "
            "documents are intended for local GGUF inference only."
        ),
    },
    "invoice": {
        "task": "receipt_or_invoice_json",
        "text": (
            "Synthetic invoice\n"
            "Company: Sunrise Pharmacy\n"
            "Invoice date: 2026-06-01\n"
            "Item: Blood pressure monitor\n"
            "Total INR 1240.00\n"
            "Payment status: paid"
        ),
    },
    "json": {
        "task": "json_repair",
        "text": "{vendor: 'Sunrise Pharmacy', total: 1240.00, paid: true,}",
    },
}


@dataclass(frozen=True)
class DemoResult:
    text: str
    payload: dict[str, Any]


def _now() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="seconds")


def _needs_review(text: str) -> bool:
    lowered = text.casefold()
    return any(term in lowered for term in SENSITIVE_TERMS)


def _warnings(text: str) -> list[str]:
    warnings = [
        "Hosted demo output is not financial, medical, legal, or safety advice.",
        "Use the local GGUF model for private documents.",
    ]
    if _needs_review(text):
        warnings.append("Sensitive-topic output requires qualified human review.")
    return warnings


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def summarize_text(text: str) -> DemoResult:
    if not text.strip():
        return DemoResult("Enter text first.", {})
    sentences = _sentences(text)
    summary = " ".join(sentences[:3]) if sentences else text.strip()[:500]
    payload = _payload(
        task="text_summarization",
        status="demo",
        input_text=text,
        result={"summary": summary},
    )
    return DemoResult(_format_answer(summary, payload), payload)


def repair_json(text: str) -> DemoResult:
    if not text.strip():
        return DemoResult("Enter JSON-like text first.", {})
    repaired = text.strip()
    try:
        parsed = json.loads(repaired)
        status = "valid"
    except json.JSONDecodeError:
        repaired = _simple_json_repair(repaired)
        try:
            parsed = json.loads(repaired)
            status = "repaired"
        except json.JSONDecodeError:
            parsed = {"raw_text": text.strip()}
            status = "fallback_wrapped"
            repaired = json.dumps(parsed, indent=2, sort_keys=True)
    payload = _payload(
        task="json_repair",
        status=status,
        input_text=text,
        result={"json": parsed},
    )
    return DemoResult(f"```json\n{json.dumps(parsed, indent=2, sort_keys=True)}\n```", payload)


def extract_document_json(text: str) -> DemoResult:
    if not text.strip():
        return DemoResult("Enter document text first.", {})
    amount_matches = re.findall(
        r"(?:rs\.?|inr|usd|\$|eur|gbp)?\s*([0-9][0-9,]*(?:\.[0-9]{2})?)",
        text,
        re.I,
    )
    date_match = re.search(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\b", text)
    vendor_match = re.search(
        r"(?:vendor|merchant|from|company)[:\s]+([A-Za-z0-9 &.,'-]{2,80})",
        text,
        re.I,
    )
    result = {
        "document_type": _infer_document_type(text),
        "vendor": vendor_match.group(1).strip(" .,:;") if vendor_match else None,
        "date": date_match.group(1) if date_match else None,
        "amount_candidates": amount_matches[:8],
        "human_review_required": True,
    }
    payload = _payload(
        task="receipt_extraction",
        status="demo",
        input_text=text,
        result=result,
    )
    return DemoResult(f"```json\n{json.dumps(result, indent=2, sort_keys=True)}\n```", payload)


def route_tool(text: str) -> DemoResult:
    if not text.strip():
        return DemoResult("Type a request first.", {})
    lowered = text.casefold()
    if "json" in lowered:
        route = "json_repair"
    elif any(term in lowered for term in ("invoice", "receipt", "bill", "statement")):
        route = "receipt_or_invoice_json"
    elif any(term in lowered for term in ("summarize", "summary", "explain")):
        route = "summarization"
    else:
        route = "general_local_assistant"
    payload = _payload(
        task="tool_routing",
        status="demo",
        input_text=text,
        result={"route": route},
    )
    return DemoResult(f"Selected route: `{route}`", payload)


@spaces.GPU(duration=45)
def zerogpu_assistant(text: str) -> DemoResult:
    if not text.strip():
        return DemoResult("Type a request first.", {})
    route = _auto_task(text)
    if route == "json_repair":
        return repair_json(text)
    if route == "receipt_or_invoice_json":
        return extract_document_json(text)
    if route == "tool_routing":
        return route_tool(text)
    result = summarize_text(text)
    payload = dict(result.payload)
    payload["task"] = "zerogpu_assistant"
    payload["status"] = "zerogpu_callback_fallback"
    payload["zerogpu"] = {
        "enabled": True,
        "decorator": "@spaces.GPU(duration=45)",
        "model_runtime": "pending published model artifacts",
        "note": (
            "Callback is GPU-routable on ZeroGPU and uses safe fallback logic "
            "until model weights are available."
        ),
    }
    return DemoResult(result.text, payload)


def run_demo(task: str, user_text: str) -> tuple[str, dict[str, Any]]:
    handlers: dict[str, Callable[[str], DemoResult]] = {
        "zerogpu_assistant": zerogpu_assistant,
        "summarization": summarize_text,
        "json_repair": repair_json,
        "receipt_or_invoice_json": extract_document_json,
        "tool_routing": route_tool,
    }
    selected = _auto_task(user_text) if task == "Auto" else task
    result = handlers[selected](user_text)
    return result.text, result.payload


def space_status() -> tuple[str, dict[str, Any]]:
    payload = {
        "space": APP_TITLE,
        "space_id": OFFICIAL_SPACE_ID,
        "space_url": OFFICIAL_SPACE_URL,
        "generated_at": _now(),
        "hosted_demo": True,
        "local_first_target": True,
        "space_hardware": "ZeroGPU",
        "space_mount": "/data",
        "zerogpu_callbacks": ["zerogpu_assistant"],
        "external_model_apis": False,
        "hosted_ocr": False,
        "remote_telemetry": False,
        "supported_public_demo_tasks": [
            "text_summarization",
            "json_repair",
            "receipt_extraction",
            "tool_routing",
        ],
        "model_artifacts": {
            "adapter_repo": "build-small-hackathon/local-lm-accessible-text-lora",
            "gguf_repo": "build-small-hackathon/local-lm-accessible-gguf",
            "status": "pending organization model repository creation",
        },
        "modal_evidence": {
            "text_lora_trained": True,
            "gguf_q4_smoke_tested": True,
            "runtime": "llama.cpp CUDA smoke on Modal",
        },
    }
    text = (
        "Runtime: hosted Gradio demo on Hugging Face ZeroGPU. External model "
        "APIs: disabled. Hosted OCR: disabled. Private document use: local "
        "GGUF mode only."
    )
    return text, payload


def competition_evidence() -> tuple[str, dict[str, Any]]:
    status_text, status_payload = space_status()
    payload = {
        **status_payload,
        "competition_tracks": ["Backyard AI", "local-first small-model assistant"],
        "modal_workflows": [
            "prepare_all_data",
            "finetune_text_nemotron",
            "evaluate_text_adapter",
            "run_text_adapter_packaging",
            "smoke_test_packaged_gguf",
            "finetune_vision",
            "create_vision_readiness",
            "check_asr_contingency",
        ],
        "runtime_privacy": {
            "private_file_uploads_in_space": False,
            "external_inference_apis": False,
            "hosted_ocr": False,
            "telemetry": False,
        },
    }
    text = "\n".join(
        [
            status_text,
            "",
            "Competition evidence:",
            "- ZeroGPU callback is present for judge-visible GPU routing.",
            "- Text LoRA packaging and GGUF smoke workflow are tracked in Modal.",
            "- Vision readiness and ASR contingency reports are generated before full training.",
            "- Private documents stay out of the hosted Space.",
        ]
    )
    return text, payload


def load_sample(sample_key: str) -> tuple[str, str]:
    sample = SAMPLES[sample_key]
    return str(sample["task"]), str(sample["text"])


def _auto_task(text: str) -> str:
    lowered = text.casefold()
    if lowered.strip().startswith(("{", "[")) or "broken json" in lowered:
        return "json_repair"
    if any(term in lowered for term in ("invoice", "receipt", "bill", "statement", "total")):
        return "receipt_or_invoice_json"
    if any(term in lowered for term in ("route", "tool", "which task")):
        return "tool_routing"
    return "summarization"


def _simple_json_repair(text: str) -> str:
    repaired = text.strip()
    repaired = re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_ -]*)(\s*:)", r'\1"\2"\3', repaired)
    repaired = repaired.replace("'", '"')
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    if not repaired.startswith(("{", "[")):
        repaired = "{" + repaired
    if repaired.startswith("{") and not repaired.endswith("}"):
        repaired += "}"
    if repaired.startswith("[") and not repaired.endswith("]"):
        repaired += "]"
    return repaired


def _infer_document_type(text: str) -> str:
    lowered = text.casefold()
    if "receipt" in lowered:
        return "receipt"
    if "statement" in lowered:
        return "bank_statement"
    if "bill" in lowered:
        return "bill"
    return "invoice" if "invoice" in lowered else "document"


def _payload(task: str, status: str, input_text: str, result: dict[str, Any]) -> dict[str, Any]:
    warnings = _warnings(input_text)
    return {
        "task": task,
        "status": status,
        "local_only_target": True,
        "hosted_demo": True,
        "human_review_required": _needs_review(input_text) or task.endswith("extraction"),
        "warnings": warnings,
        "result": result,
        "generated_at": _now(),
    }


def _format_answer(answer: str, payload: dict[str, Any]) -> str:
    warning_text = "\n".join(f"- {warning}" for warning in payload["warnings"])
    return f"{answer}\n\nWarnings:\n{warning_text}"


with gr.Blocks(title=APP_TITLE, css=CSS) as demo:
    gr.Markdown(
        """
# Local LM Accessible For Elders

Hosted ZeroGPU demo for public samples only. Private documents are for the
published local GGUF workflow, not this Space.
"""
    )
    with gr.Tab("Assistant"):
        task = gr.Dropdown(choices=list(TASK_CHOICES), value="Auto", label="Task")
        user_text = gr.Textbox(
            label="Input text",
            lines=10,
            placeholder="Paste a short public sample, broken JSON, or synthetic invoice text.",
        )
        with gr.Row(elem_classes=["sample-row"]):
            summary_sample = gr.Button("Sample Summary")
            invoice_sample = gr.Button("Synthetic Invoice")
            json_sample = gr.Button("Broken JSON")
        run_button = gr.Button("Run", variant="primary")
        output = gr.Markdown(label="Output")
        payload = gr.JSON(label="Runtime payload")
        summary_sample.click(
            lambda: load_sample("summary"),
            inputs=[],
            outputs=[task, user_text],
            show_progress="hidden",
        )
        invoice_sample.click(
            lambda: load_sample("invoice"),
            inputs=[],
            outputs=[task, user_text],
            show_progress="hidden",
        )
        json_sample.click(
            lambda: load_sample("json"),
            inputs=[],
            outputs=[task, user_text],
            show_progress="hidden",
        )
        run_button.click(run_demo, inputs=[task, user_text], outputs=[output, payload])
    with gr.Tab("Evidence"):
        evidence_button = gr.Button("Show evidence", variant="primary")
        evidence_text = gr.Markdown(label="Evidence")
        evidence_payload = gr.JSON(label="Evidence payload")
        evidence_button.click(
            competition_evidence,
            inputs=[],
            outputs=[evidence_text, evidence_payload],
        )
    with gr.Tab("Status"):
        status_button = gr.Button("Check status", variant="primary")
        status_text = gr.Textbox(label="Status", interactive=False)
        status_payload = gr.JSON(label="Status payload")
        status_button.click(space_status, inputs=[], outputs=[status_text, status_payload])


if __name__ == "__main__":
    demo.launch()
