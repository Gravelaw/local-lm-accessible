from __future__ import annotations

import argparse
import json
import sys
import threading
import wave
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from openpyxl import Workbook

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.healthcheck import HTTP_OPENER, check_ports  # noqa: E402
from scripts.local_runtime import require_loopback_url  # noqa: E402
from scripts.synthetic_documents import generate_documents  # noqa: E402
from services.tools.pdf_export import export_text_placeholder  # noqa: E402

MOCK_TEXT_RESPONSE = "Mock local llama.cpp text summary for the smoke test."
MOCK_VISION_RESPONSE = "Mock local llama.cpp vision description for the smoke test."
DEFAULT_GATEWAY_REQUEST_TIMEOUT_SECONDS = 120.0
MAX_GATEWAY_REQUEST_TIMEOUT_SECONDS = 300.0


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


def _client_post(
    path: str,
    payload: dict[str, Any],
    gateway: str | None,
    *,
    request_timeout_seconds: float = DEFAULT_GATEWAY_REQUEST_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    timeout = _validate_request_timeout(request_timeout_seconds)
    if gateway is not None:
        require_loopback_url(gateway, label="gateway")
        request = Request(
            f"{gateway.rstrip('/')}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    from fastapi.testclient import TestClient

    from services.gateway.app import app

    response = TestClient(app).post(path, json=payload)
    response.raise_for_status()
    return response.json()


def _client_get(
    path: str,
    gateway: str | None,
    *,
    request_timeout_seconds: float = DEFAULT_GATEWAY_REQUEST_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    timeout = _validate_request_timeout(request_timeout_seconds)
    if gateway is not None:
        require_loopback_url(gateway, label="gateway")
        with urlopen(f"{gateway.rstrip('/')}{path}", timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    from fastapi.testclient import TestClient

    from services.gateway.app import app

    response = TestClient(app).get(path)
    response.raise_for_status()
    return response.json()


def _write_invoice_outputs(work_dir: Path, invoice: dict[str, Any]) -> dict[str, str]:
    json_path = work_dir / "sample_invoice_conversion.json"
    xlsx_path = work_dir / "sample_invoice_conversion.xlsx"
    txt_path = work_dir / "sample_invoice_conversion.txt"
    pdf_path = work_dir / "sample_invoice_conversion.pdf"
    invoice_text = json.dumps(invoice, indent=2, sort_keys=True)
    json_path.write_text(invoice_text, encoding="utf-8")
    txt_path.write_text(invoice_text, encoding="utf-8")

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "invoice"
    sheet.append(["field", "value"])
    sheet.append(["document_id", invoice["document_id"]])
    sheet.append(["region", invoice["region"]])
    sheet.append(["country", invoice["country"]])
    sheet.append(["total", invoice["totals"]["total"]])
    sheet.append(["tax", invoice["totals"]["tax_amount"]])
    workbook.save(xlsx_path)
    export_text_placeholder(pdf_path, invoice_text)
    return {
        "json": str(json_path),
        "xlsx": str(xlsx_path),
        "txt": str(txt_path),
        "pdf": str(pdf_path),
    }


def _write_tiny_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16_000)
        wav_file.writeframes(b"\x00\x00" * 160)


def run_smoke(
    work_dir: Path,
    gateway: str | None,
    *,
    mock_model_endpoints: bool = False,
    require_real_model_services: bool = False,
    request_timeout_seconds: float = DEFAULT_GATEWAY_REQUEST_TIMEOUT_SECONDS,
    health_opener: Callable[..., Any] = HTTP_OPENER,
) -> dict[str, Any]:
    if mock_model_endpoints and require_real_model_services:
        raise ValueError("mock_model_endpoints cannot be combined with require_real_model_services")
    request_timeout = _validate_request_timeout(request_timeout_seconds)
    work_dir.mkdir(parents=True, exist_ok=True)
    generated = generate_documents(
        kind="invoice",
        output_dir=work_dir,
        count=1,
        regions=("India",),
        seed=20260607,
        augment=False,
    )
    invoice = generated[0]
    document_dir = work_dir / "invoice" / invoice["document_id"]
    image_path = document_dir / "rendered.png"
    converted = _write_invoice_outputs(work_dir, invoice)
    audio_path = work_dir / "sample_speech.wav"
    _write_tiny_wav(audio_path)
    service_checks: list[dict[str, Any]] = []
    if require_real_model_services:
        service_checks = check_ports(require_running=True, opener=health_opener)

    with _mock_llama_servers(mock_model_endpoints):
        health = _client_get(
            "/health",
            gateway,
            request_timeout_seconds=request_timeout,
        )
        assistant_response = _client_post(
            "/tasks/general",
            {"text": "Explain local privacy in simple words."},
            gateway,
            request_timeout_seconds=request_timeout,
        )
        invoice_response = _client_post(
            "/tasks/document_to_excel",
            {"file_path": str(image_path), "mime_type": "image/png"},
            gateway,
            request_timeout_seconds=request_timeout,
        )
        wiki_response = _client_post(
            "/tasks/summarize_wikipedia",
            {"text": "Synthetic local Wikipedia text about GST invoices."},
            gateway,
            request_timeout_seconds=request_timeout,
        )
        image_response = _client_post(
            "/tasks/describe_image",
            {"file_path": str(image_path), "mime_type": "image/png"},
            gateway,
            request_timeout_seconds=request_timeout,
        )
        speech_response = _client_post(
            "/tasks/speech_to_text",
            {"file_path": str(audio_path), "mime_type": "audio/wav"},
            gateway,
            request_timeout_seconds=request_timeout,
        )

    checks = {
        "gateway_health": health["local_only"] is True and health["allow_web"] is False,
        "general_assistant": (
            assistant_response["task"] == "general_local_assistant"
            and assistant_response["local_only"] is True
        ),
        "invoice_json": Path(converted["json"]).exists(),
        "invoice_xlsx": Path(converted["xlsx"]).exists(),
        "invoice_txt": Path(converted["txt"]).exists(),
        "invoice_pdf": Path(converted["pdf"]).read_bytes().startswith(b"%PDF"),
        "invoice_endpoint": invoice_response["task"] == "document_to_excel",
        "wikipedia_summary": wiki_response["task"] == "summarize_wikipedia",
        "image_description": image_response["task"] == "describe_image",
        "speech_to_text": (
            speech_response["task"] == "speech_to_text"
            and speech_response["local_only"] is True
            and speech_response["result"].get("asr_endpoint") == "http://127.0.0.1:8090"
        ),
    }
    if mock_model_endpoints:
        checks["text_model_endpoint"] = (
            wiki_response["status"] == "ok"
            and wiki_response["result"].get("source") == "local_text_model"
            and MOCK_TEXT_RESPONSE in str(wiki_response["result"].get("summary", ""))
        )
        checks["assistant_text_model_endpoint"] = (
            assistant_response["status"] == "ok"
            and assistant_response["result"].get("model_endpoint") == "http://127.0.0.1:8081/"
            and MOCK_TEXT_RESPONSE in str(assistant_response["result"].get("text", ""))
        )
        checks["vision_model_endpoint"] = (
            image_response["status"] == "ok"
            and image_response["result"].get("model_endpoint") == "http://127.0.0.1:8082/"
            and MOCK_VISION_RESPONSE in str(image_response["result"].get("description", ""))
        )
    if require_real_model_services:
        required_services = [service for service in service_checks if not service["optional"]]
        checks["real_model_services_ready"] = all(
            service["http_ready"] is True for service in required_services
        )
        checks["text_model_endpoint"] = (
            wiki_response["status"] == "ok"
            and wiki_response["result"].get("source") == "local_text_model"
        )
        checks["assistant_text_model_endpoint"] = (
            assistant_response["status"] == "ok"
            and assistant_response["result"].get("model_endpoint") == "http://127.0.0.1:8081/"
            and bool(str(assistant_response["result"].get("text", "")).strip())
        )
        checks["vision_model_endpoint"] = (
            image_response["status"] == "ok"
            and image_response["result"].get("model_endpoint") == "http://127.0.0.1:8082/"
        )
        checks["asr_model_endpoint"] = (
            speech_response["status"] == "ok"
            and speech_response["result"].get("model_ready") is True
            and bool(str(speech_response["result"].get("text", "")).strip())
        )
    return {
        "status": "ok" if all(checks.values()) else "failed",
        "local_only": True,
        "mock_model_endpoints": mock_model_endpoints,
        "require_real_model_services": require_real_model_services,
        "request_timeout_seconds": request_timeout,
        "model_services": service_checks,
        "work_dir": str(work_dir),
        "sample_invoice_image": str(image_path),
        "sample_audio": str(audio_path),
        "outputs": converted,
        "checks": checks,
    }


@contextmanager
def _mock_llama_servers(enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return

    servers = [
        _start_mock_llama_server(8081, MOCK_TEXT_RESPONSE),
        _start_mock_llama_server(8082, MOCK_VISION_RESPONSE),
    ]
    try:
        yield
    finally:
        for server, thread in servers:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


def _start_mock_llama_server(
    port: int, content: str
) -> tuple[ThreadingHTTPServer, threading.Thread]:
    handler = _make_mock_llama_handler(content)
    server = ReusableThreadingHTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _make_mock_llama_handler(content: str) -> type[BaseHTTPRequestHandler]:
    class MockLlamaHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/health":
                self.send_error(404)
                return
            self._write_json({"status": "ok", "local_only": True})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/completion":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length).decode("utf-8")
            payload = json.loads(raw_body) if raw_body else {}
            if not str(payload.get("prompt", "")).strip():
                self.send_error(400, "prompt is required")
                return
            self._write_json({"content": content, "stop": True})

        def log_message(self, format: str, *args: object) -> None:
            del format, args

        def _write_json(self, payload: dict[str, object]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return MockLlamaHandler


def _validate_request_timeout(value: float) -> float:
    if value <= 0:
        raise ValueError("request_timeout_seconds must be positive")
    if value > MAX_GATEWAY_REQUEST_TIMEOUT_SECONDS:
        raise ValueError(
            "request_timeout_seconds must be less than or equal to "
            f"{MAX_GATEWAY_REQUEST_TIMEOUT_SECONDS:g}"
        )
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/local-lm-smoke"))
    parser.add_argument("--gateway", help="Optional running gateway URL.")
    parser.add_argument(
        "--mock-model-endpoints",
        action="store_true",
        help="Start loopback-only mock llama.cpp /completion servers on 8081 and 8082.",
    )
    parser.add_argument(
        "--require-real-model-services",
        action="store_true",
        help="Require loopback /health readiness for required text, vision, and ASR services.",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=DEFAULT_GATEWAY_REQUEST_TIMEOUT_SECONDS,
        help=(
            "Per-request gateway timeout in seconds. Increase for CPU-only local model "
            f"inference; maximum {MAX_GATEWAY_REQUEST_TIMEOUT_SECONDS:g}."
        ),
    )
    args = parser.parse_args()
    result = run_smoke(
        args.work_dir,
        args.gateway,
        mock_model_endpoints=args.mock_model_endpoints,
        require_real_model_services=args.require_real_model_services,
        request_timeout_seconds=args.request_timeout,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
