from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from scripts.healthcheck import check_gateway, check_ports
from scripts.local_runtime import require_loopback_url
from scripts.smoke_test_local import run_smoke
from services.gateway import app as gateway_app
from services.gateway.app import app
from services.gateway.model_clients import default_clients
from services.gateway.tool_registry import list_tools


class _HealthResponse:
    status = 200

    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def __enter__(self) -> _HealthResponse:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None

    def getcode(self) -> int:
        return self.status

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_gateway_health_confirms_local_services() -> None:
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["local_only"] is True
    assert payload["privacy_mode"] == "strict"
    assert payload["allow_web"] is False
    assert payload["telemetry_enabled"] is False
    services = {service["name"]: service for service in payload["services"]}
    assert services["text"]["endpoint"] == "http://127.0.0.1:8081/"
    assert services["vision"]["endpoint"] == "http://127.0.0.1:8082/"
    assert services["omni"]["endpoint"] == "http://127.0.0.1:8083/"
    assert services["asr"]["endpoint"] == "http://127.0.0.1:8090/"
    assert services["text"]["ready"] is True
    assert services["vision"]["ready"] is True
    assert services["asr"]["ready"] is True
    assert services["asr"]["required"] is True
    assert services["asr"]["checksum_configured"] is True
    assert services["asr"]["warnings"] == []


def test_task_endpoints_do_not_call_cloud_when_web_disabled() -> None:
    client = TestClient(app)

    response = client.post(
        "/tasks/summarize_url",
        json={"url": "https://example.com/article", "allow_web": False},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "blocked"
    assert payload["local_only"] is True
    assert payload["result"]["blocked"] is True
    assert payload["result"]["remote_uploads"] is False


def test_optional_web_fetch_uses_local_text_model_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeTextClient:
        def generate(self, prompt: str, *, max_tokens: int = 256) -> dict[str, object]:
            assert "Fetched local test page" in prompt
            assert max_tokens == 320
            return {
                "text": "Fetched page summary from local text model.",
                "endpoint": "http://127.0.0.1:8081/",
            }

    def fake_fetch(url: str) -> dict[str, object]:
        assert url == "https://example.com/article"
        return {
            "url": url,
            "text": "Fetched local test page about accessibility.",
            "content_type": "text/html",
            "bytes_read": 128,
            "remote_uploads": False,
        }

    monkeypatch.setattr(gateway_app, "fetch_url", fake_fetch)
    monkeypatch.setattr(gateway_app, "default_clients", lambda: {"text": FakeTextClient()})
    client = TestClient(app)

    response = client.post(
        "/tasks/summarize_url",
        json={"url": "https://example.com/article", "allow_web": True},
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["status"] == "ok"
    assert payload["local_only"] is True
    assert payload["result"]["summary"] == "Fetched page summary from local text model."
    assert payload["result"]["source"] == "local_text_model_with_user_enabled_web_fetch"
    assert payload["result"]["remote_uploads"] is False
    assert payload["warnings"]


def test_optional_web_fetch_adds_sensitive_output_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeTextClient:
        def generate(self, prompt: str, *, max_tokens: int = 256) -> dict[str, object]:
            return {
                "text": "Fetched medical page summary.",
                "endpoint": "http://127.0.0.1:8081/",
            }

    def fake_fetch(url: str) -> dict[str, object]:
        return {
            "url": url,
            "text": "This page discusses medical symptoms and treatment options.",
            "content_type": "text/html",
            "bytes_read": 128,
            "remote_uploads": False,
        }

    monkeypatch.setattr(gateway_app, "fetch_url", fake_fetch)
    monkeypatch.setattr(gateway_app, "default_clients", lambda: {"text": FakeTextClient()})
    client = TestClient(app)

    response = client.post(
        "/tasks/summarize_url",
        json={"url": "https://example.com/health", "allow_web": True},
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["status"] == "ok"
    assert payload["human_review_required"] is True
    assert any("qualified human" in warning for warning in payload["warnings"])


def test_optional_web_fetch_blocks_unsafe_fetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_fetch(url: str) -> dict[str, object]:
        raise ValueError("optional web fetch rejects local hostnames")

    monkeypatch.setattr(gateway_app, "fetch_url", fake_fetch)
    client = TestClient(app)

    response = client.post(
        "/tasks/summarize_url",
        json={"url": "http://localhost:8000/private", "allow_web": True},
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["status"] == "blocked"
    assert payload["result"]["blocked"] is True
    assert payload["result"]["remote_uploads"] is False
    assert "Optional web fetch was not completed" in payload["warnings"][0]


def test_model_clients_are_loopback_only() -> None:
    clients = default_clients()

    assert {client.endpoint.host for client in clients.values()} == {"127.0.0.1"}
    assert clients["text"].endpoint.port == 8081
    assert clients["vision"].endpoint.port == 8082
    assert clients["omni"].endpoint.port == 8083
    assert clients["asr"].endpoint.port == 8090


def test_tools_are_local_and_do_not_log_raw_user_data() -> None:
    tools = list_tools()

    assert tools
    assert all(tool.local_only for tool in tools)
    assert not any(tool.logs_raw_user_data for tool in tools)


def test_gateway_url_validation_allows_only_loopback() -> None:
    assert require_loopback_url("http://127.0.0.1:8000") == "http://127.0.0.1:8000"
    assert require_loopback_url("http://localhost:8000") == "http://localhost:8000"
    with pytest.raises(ValueError, match="loopback"):
        require_loopback_url("https://example.com")


def test_healthcheck_rejects_remote_gateway_before_request() -> None:
    with pytest.raises(ValueError, match="loopback"):
        check_gateway("https://example.com")


def test_healthcheck_require_running_uses_http_readiness() -> None:
    urls: list[str] = []

    def opener(url: str, timeout: float) -> _HealthResponse:
        urls.append(url)
        return _HealthResponse({"status": "ok", "local_only": True})

    checks = check_ports(require_running=True, opener=opener)

    assert {check["name"] for check in checks} >= {"text", "vision", "asr"}
    assert all(check["http_ready"] for check in checks if not check["optional"])
    assert "http://127.0.0.1:8081/health" in urls
    assert "http://127.0.0.1:8090/health" in urls


def test_healthcheck_require_running_rejects_bad_http_readiness() -> None:
    def opener(url: str, timeout: float) -> _HealthResponse:
        if url == "http://127.0.0.1:8081/health":
            raise TimeoutError("not a local model health endpoint")
        return _HealthResponse({"status": "ok", "local_only": True})

    with pytest.raises(ConnectionError, match="HTTP readiness"):
        check_ports(require_running=True, opener=opener)


def test_healthcheck_require_running_allows_optional_http_readiness_failure() -> None:
    def opener(url: str, timeout: float) -> _HealthResponse:
        if url == "http://127.0.0.1:8083/health":
            raise TimeoutError("optional omni service is offline")
        return _HealthResponse({"status": "ok", "local_only": True})

    checks = check_ports(require_running=True, opener=opener)
    by_name = {check["name"]: check for check in checks}

    assert by_name["text"]["http_ready"] is True
    assert by_name["vision"]["http_ready"] is True
    assert by_name["asr"]["http_ready"] is True
    assert by_name["omni"]["optional"] is True
    assert by_name["omni"]["http_ready"] is False
    assert "optional omni service is offline" in by_name["omni"]["health_error"]


def test_smoke_test_rejects_remote_gateway(tmp_path) -> None:
    with pytest.raises(ValueError, match="loopback"):
        run_smoke(tmp_path, gateway="https://example.com")
