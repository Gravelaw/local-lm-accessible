from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import gradio as gr

APP_TITLE = "local-lm accessible"
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
    "summarization",
    "json_repair",
    "receipt_or_invoice_json",
    "tool_routing",
)
CSS = """
.gradio-container {
  font-size: 18px;
  line-height: 1.5;
}
textarea, input, button {
  font-size: 18px !important;
}
button {
  min-height: 52px !important;
}
"""


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


def run_demo(task: str, user_text: str) -> tuple[str, dict[str, Any]]:
    handlers: dict[str, Callable[[str], DemoResult]] = {
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
        "generated_at": _now(),
        "hosted_demo": True,
        "local_first_target": True,
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
        },
    }
    text = (
        "Runtime: hosted Gradio demo. External model APIs: disabled. "
        "Hosted OCR: disabled. Private document use: local GGUF mode only."
    )
    return text, payload


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
# local-lm accessible

Hosted demo. Do not enter private documents. Use the published GGUF locally for
private work.
"""
    )
    with gr.Tab("Assistant"):
        task = gr.Dropdown(choices=list(TASK_CHOICES), value="Auto", label="Task")
        user_text = gr.Textbox(
            label="Input text",
            lines=10,
            placeholder="Paste a short public sample, broken JSON, or synthetic invoice text.",
        )
        run_button = gr.Button("Run", variant="primary")
        output = gr.Markdown(label="Output")
        payload = gr.JSON(label="Runtime payload")
        run_button.click(run_demo, inputs=[task, user_text], outputs=[output, payload])
    with gr.Tab("Status"):
        status_button = gr.Button("Check status", variant="primary")
        status_text = gr.Textbox(label="Status", interactive=False)
        status_payload = gr.JSON(label="Status payload")
        status_button.click(space_status, inputs=[], outputs=[status_text, status_payload])


if __name__ == "__main__":
    demo.launch()
