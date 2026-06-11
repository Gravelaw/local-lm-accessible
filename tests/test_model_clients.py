from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.error import HTTPError

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from services.gateway import app as gateway_app
from services.gateway.app import app
from services.gateway.model_clients import LocalModelClient, LocalModelError


class _FakeHTTPResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def __enter__(self) -> _FakeHTTPResponse:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class _FakeModelClient:
    def __init__(self, text: str, endpoint: str = "http://127.0.0.1:8081") -> None:
        self.text = text
        self.endpoint = endpoint
        self.prompts: list[str] = []

    def generate(self, prompt: str, *, max_tokens: int = 256) -> dict[str, object]:
        self.prompts.append(prompt)
        return {
            "text": self.text,
            "endpoint": self.endpoint,
            "model_key": "fake",
            "local_only": True,
            "raw": {"content": self.text},
        }


class _FakeAsrClient:
    def __init__(self) -> None:
        self.payloads: list[dict[str, object]] = []

    def post_json(self, path: str, payload: dict[str, object]) -> dict[str, object]:
        self.payloads.append({"path": path, **payload})
        return {
            "text": "local transcript",
            "language": "en",
            "status": "stub",
            "experimental": False,
            "unsupported_language": False,
            "model_ready": False,
            "warnings": ["Parakeet artifact is not checksum-verified locally."],
            "local_only": True,
        }


def test_local_model_client_rejects_remote_endpoint() -> None:
    with pytest.raises(ValidationError, match="loopback"):
        LocalModelClient(model_key="bad", endpoint="https://example.com")


def test_local_model_client_posts_to_loopback_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request: object, timeout: float) -> _FakeHTTPResponse:
        captured["url"] = request.full_url  # type: ignore[attr-defined]
        captured["body"] = json.loads(request.data.decode("utf-8"))  # type: ignore[attr-defined]
        captured["timeout"] = timeout
        return _FakeHTTPResponse({"content": "local summary"})

    monkeypatch.setattr("services.gateway.model_clients.urlopen", fake_urlopen)
    client = LocalModelClient(model_key="text", endpoint="http://127.0.0.1:8081")

    result = client.generate("Summarize this.", max_tokens=128)

    assert captured["url"] == "http://127.0.0.1:8081/completion"
    assert captured["body"]["prompt"] == "Summarize this."  # type: ignore[index]
    assert captured["body"]["n_predict"] == 128  # type: ignore[index]
    assert result["text"] == "local summary"
    assert result["api_path"] == "/completion"
    assert result["local_only"] is True


def test_local_model_client_falls_back_to_openai_chat_completions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, object]] = []

    def fake_urlopen(request: object, timeout: float) -> _FakeHTTPResponse:
        payload = json.loads(request.data.decode("utf-8"))  # type: ignore[attr-defined]
        captured.append(
            {
                "url": request.full_url,  # type: ignore[attr-defined]
                "body": payload,
                "timeout": timeout,
            }
        )
        if request.full_url.endswith("/completion"):  # type: ignore[attr-defined]
            raise HTTPError(
                request.full_url,  # type: ignore[attr-defined]
                404,
                "not found",
                hdrs=None,
                fp=None,
            )
        return _FakeHTTPResponse({"choices": [{"message": {"content": "local chat summary"}}]})

    monkeypatch.setattr("services.gateway.model_clients.urlopen", fake_urlopen)
    client = LocalModelClient(model_key="text", endpoint="http://127.0.0.1:8081")

    result = client.generate("Summarize this.", max_tokens=64)

    assert [item["url"] for item in captured] == [
        "http://127.0.0.1:8081/completion",
        "http://127.0.0.1:8081/v1/chat/completions",
    ]
    assert captured[0]["body"]["n_predict"] == 64  # type: ignore[index]
    assert captured[1]["body"]["max_tokens"] == 64  # type: ignore[index]
    assert captured[1]["body"]["messages"] == [  # type: ignore[index]
        {"role": "user", "content": "Summarize this."}
    ]
    assert result["text"] == "local chat summary"
    assert result["api_path"] == "/v1/chat/completions"
    assert result["local_only"] is True


def test_local_model_client_falls_back_to_chat_when_completion_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[str] = []

    def fake_urlopen(request: object, timeout: float) -> _FakeHTTPResponse:
        captured.append(request.full_url)  # type: ignore[attr-defined]
        if request.full_url.endswith("/completion"):  # type: ignore[attr-defined]
            return _FakeHTTPResponse({"content": "", "stop_type": "eos"})
        return _FakeHTTPResponse({"choices": [{"message": {"content": "usable chat text"}}]})

    monkeypatch.setattr("services.gateway.model_clients.urlopen", fake_urlopen)
    client = LocalModelClient(model_key="text", endpoint="http://127.0.0.1:8081")

    result = client.generate("Explain local privacy.", max_tokens=64)

    assert captured == [
        "http://127.0.0.1:8081/completion",
        "http://127.0.0.1:8081/v1/chat/completions",
    ]
    assert result["text"] == "usable chat text"
    assert result["api_path"] == "/v1/chat/completions"


def test_local_model_client_does_not_retry_chat_for_completion_server_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fake_urlopen(request: object, timeout: float) -> _FakeHTTPResponse:
        nonlocal calls
        calls += 1
        raise HTTPError(
            request.full_url,  # type: ignore[attr-defined]
            500,
            "server error",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr("services.gateway.model_clients.urlopen", fake_urlopen)
    client = LocalModelClient(model_key="text", endpoint="http://127.0.0.1:8081")

    with pytest.raises(LocalModelError, match="HTTP 500"):
        client.generate("Summarize this.")
    assert calls == 1


def test_local_model_client_posts_json_to_loopback_path(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request: object, timeout: float) -> _FakeHTTPResponse:
        captured["url"] = request.full_url  # type: ignore[attr-defined]
        captured["body"] = json.loads(request.data.decode("utf-8"))  # type: ignore[attr-defined]
        captured["timeout"] = timeout
        return _FakeHTTPResponse({"text": "local transcript"})

    monkeypatch.setattr("services.gateway.model_clients.urlopen", fake_urlopen)
    client = LocalModelClient(model_key="asr", endpoint="http://127.0.0.1:8090")

    result = client.post_json("/transcribe", {"audio_filepath": "/tmp/local.wav"})

    assert captured["url"] == "http://127.0.0.1:8090/transcribe"
    assert captured["body"]["audio_filepath"] == "/tmp/local.wav"  # type: ignore[index]
    assert result["text"] == "local transcript"
    assert result["local_only"] is True


def test_local_model_client_raises_on_empty_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_urlopen(request: object, timeout: float) -> _FakeHTTPResponse:
        return _FakeHTTPResponse({"content": ""})

    monkeypatch.setattr("services.gateway.model_clients.urlopen", fake_urlopen)
    client = LocalModelClient(model_key="text", endpoint="http://127.0.0.1:8081")

    with pytest.raises(LocalModelError, match="returned no text"):
        client.generate("Summarize this.")


def test_gateway_uses_local_text_model_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeModelClient("A simple local summary.")
    monkeypatch.setattr(gateway_app, "default_clients", lambda: {"text": fake})
    client = TestClient(app)

    response = client.post(
        "/tasks/summarize_wikipedia",
        json={"text": "Long local article about accessibility."},
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["status"] == "ok"
    assert payload["result"]["summary"] == "A simple local summary."
    assert payload["result"]["source"] == "local_text_model"
    assert fake.prompts


def test_gateway_summarization_adds_sensitive_output_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeModelClient("A simple local summary about the legal notice.")
    monkeypatch.setattr(gateway_app, "default_clients", lambda: {"text": fake})
    client = TestClient(app)

    response = client.post(
        "/tasks/summarize_wikipedia",
        json={"text": "This article explains a legal notice and court deadline."},
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["status"] == "ok"
    assert payload["human_review_required"] is True
    assert any("qualified human" in warning for warning in payload["warnings"])


def test_gateway_general_assistant_uses_local_text_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeModelClient("Here is a simple local answer.")
    monkeypatch.setattr(gateway_app, "default_clients", lambda: {"text": fake})
    client = TestClient(app)

    response = client.post(
        "/tasks/general",
        json={"text": "How do I save this as a PDF?"},
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["status"] == "ok"
    assert payload["task"] == "general_local_assistant"
    assert payload["result"]["text"] == "Here is a simple local answer."
    assert payload["result"]["model_endpoint"] == "http://127.0.0.1:8081"
    assert fake.prompts


def test_gateway_general_assistant_adds_sensitive_output_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeModelClient("This is a local explanation of the bill.")
    monkeypatch.setattr(gateway_app, "default_clients", lambda: {"text": fake})
    client = TestClient(app)

    response = client.post(
        "/tasks/general",
        json={"text": "Please explain this medical bill."},
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["status"] == "ok"
    assert payload["human_review_required"] is True
    assert payload["warnings"]
    assert any("qualified human" in warning for warning in payload["warnings"])


def test_gateway_general_assistant_fallback_is_not_prompt_echo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class UnavailableModelClient:
        def generate(self, prompt: str, *, max_tokens: int = 256) -> dict[str, object]:
            raise LocalModelError("local model service unavailable for text")

    monkeypatch.setattr(gateway_app, "default_clients", lambda: {"text": UnavailableModelClient()})
    client = TestClient(app)

    response = client.post(
        "/tasks/general",
        json={"text": "Please answer this exact prompt."},
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["status"] == "stub"
    assert payload["task"] == "general_local_assistant"
    assert "local text model is unavailable" in payload["result"]["text"].casefold()
    assert "Please answer this exact prompt." not in payload["result"]["text"]
    assert payload["human_review_required"] is True
    assert payload["warnings"]


def test_gateway_uses_local_vision_model_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeModelClient("A bright sign says Exit.")
    monkeypatch.setattr(gateway_app, "default_clients", lambda: {"vision": fake})
    client = TestClient(app)

    response = client.post(
        "/tasks/describe_image",
        json={"file_path": "/tmp/local-photo.png", "mime_type": "image/png"},
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["status"] == "ok"
    assert payload["result"]["description"] == "A bright sign says Exit."
    assert payload["human_review_required"] is True
    assert payload["result"]["schema"]["short_description"] == "A bright sign says Exit."
    assert fake.prompts


def test_gateway_blocks_identity_guess_from_local_vision_description(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeModelClient("This is John Smith standing near a bus stop.")
    monkeypatch.setattr(gateway_app, "default_clients", lambda: {"vision": fake})
    client = TestClient(app)

    response = client.post(
        "/tasks/describe_image",
        json={"file_path": "/tmp/local-photo.png", "mime_type": "image/png"},
    )

    payload = response.json()
    description = payload["result"]["description"]
    schema = payload["result"]["schema"]

    assert response.status_code == 200
    assert payload["status"] == "ok"
    assert "John Smith" not in description
    assert "cannot identify or name people" in description
    assert schema["spoken_response"] == description
    assert any("Identity-like claim removed" in item for item in schema["uncertainties"])
    assert any("Identity guessing" in warning for warning in payload["warnings"])
    assert payload["human_review_required"] is True


def test_gateway_uses_local_vision_json_for_image_document_extraction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "invoice.png"
    image_path.write_bytes(b"synthetic image bytes")
    fake = _FakeModelClient(
        json.dumps(
            {
                "document_type": "invoice",
                "fields": {"vendor": "Synthetic Vendor"},
                "line_items": [{"description": "service", "amount": 100.0}],
                "currency": "INR",
                "subtotal": 100.0,
                "tax_amount": 18.0,
                "total": 118.0,
                "raw_ocr_text": "Synthetic invoice",
                "confidence": 0.74,
                "warnings": [],
            }
        ),
        endpoint="http://127.0.0.1:8082/",
    )
    monkeypatch.setattr(gateway_app, "default_clients", lambda: {"vision": fake})
    client = TestClient(app)

    response = client.post(
        "/tasks/document_to_excel",
        json={"file_path": str(image_path), "mime_type": "image/png"},
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["status"] == "ok"
    assert payload["result"]["source"] == "local_vision_model"
    assert payload["result"]["model_endpoint"] == "http://127.0.0.1:8082/"
    assert payload["result"]["schema"]["total"] == 118.0
    assert payload["human_review_required"] is True
    assert fake.prompts


def test_gateway_falls_back_when_local_vision_document_json_is_invalid(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "invoice.png"
    image_path.write_bytes(b"synthetic image bytes")
    fake = _FakeModelClient("not json", endpoint="http://127.0.0.1:8082/")
    monkeypatch.setattr(gateway_app, "default_clients", lambda: {"vision": fake})
    client = TestClient(app)

    response = client.post(
        "/tasks/document_to_excel",
        json={"file_path": str(image_path), "mime_type": "image/png"},
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["status"] == "stub"
    assert payload["result"]["source"] == "local_fallback"
    assert "invalid document JSON" in payload["warnings"][0]
    assert payload["human_review_required"] is True


def test_gateway_uses_local_asr_service_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeAsrClient()
    monkeypatch.setattr(gateway_app, "default_clients", lambda: {"asr": fake})
    client = TestClient(app)

    response = client.post(
        "/tasks/speech_to_text",
        json={"file_path": "/tmp/local-voice.wav", "mime_type": "audio/wav"},
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["status"] == "stub"
    assert payload["result"]["text"] == "local transcript"
    assert payload["result"]["model_ready"] is False
    assert payload["warnings"]
    assert fake.payloads[0]["path"] == "/transcribe"


def test_gateway_passes_asr_language_region_metadata_to_local_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeAsrClient()
    monkeypatch.setattr(gateway_app, "default_clients", lambda: {"asr": fake})
    client = TestClient(app)

    response = client.post(
        "/tasks/speech_to_text",
        json={
            "file_path": "/tmp/local-voice.wav",
            "mime_type": "audio/wav",
            "language": "en",
            "region": "india",
            "country": "india",
            "allow_experimental_asr": False,
        },
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["result"]["language"] == "en"
    assert payload["result"]["region"] == "india"
    assert payload["result"]["country"] == "india"
    assert fake.payloads[0]["language"] == "en"
    assert fake.payloads[0]["region"] == "india"
    assert fake.payloads[0]["country"] == "india"
    assert fake.payloads[0]["allow_experimental"] is False


def test_gateway_blocks_experimental_asr_without_explicit_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeAsrClient()
    monkeypatch.setattr(gateway_app, "default_clients", lambda: {"asr": fake})
    client = TestClient(app)

    response = client.post(
        "/tasks/speech_to_text",
        json={
            "file_path": "/tmp/local-voice.wav",
            "mime_type": "audio/wav",
            "language": "hi",
            "region": "india",
            "country": "india",
            "allow_experimental_asr": False,
        },
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["status"] == "stub"
    assert payload["result"]["language"] == "hi"
    assert payload["result"]["experimental"] is True
    assert payload["result"]["unsupported_language"] is True
    assert payload["human_review_required"] is True
    assert any("Enable experimental ASR" in warning for warning in payload["warnings"])
    assert fake.payloads == []


def test_gateway_allows_experimental_asr_after_explicit_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeAsrClient()
    monkeypatch.setattr(gateway_app, "default_clients", lambda: {"asr": fake})
    client = TestClient(app)

    response = client.post(
        "/tasks/speech_to_text",
        json={
            "file_path": "/tmp/local-voice.wav",
            "mime_type": "audio/wav",
            "language": "hi",
            "region": "india",
            "country": "india",
            "allow_experimental_asr": True,
        },
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["result"]["language"] == "en"
    assert payload["result"]["region"] == "india"
    assert fake.payloads[0]["language"] == "hi"
    assert fake.payloads[0]["region"] == "india"
    assert fake.payloads[0]["allow_experimental"] is True
