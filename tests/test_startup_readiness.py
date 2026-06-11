from __future__ import annotations

import json
from pathlib import Path
from urllib.error import URLError

import pytest

from scripts.wait_for_service import wait_for_http_service


class _ReadyResponse:
    def __init__(self, payload: dict[str, object], status: int = 200) -> None:
        self.payload = payload
        self.status = status

    def __enter__(self) -> _ReadyResponse:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None

    def getcode(self) -> int:
        return self.status

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_wait_for_http_service_accepts_loopback_local_ready_response() -> None:
    def opener(url: str, timeout: float) -> _ReadyResponse:
        assert url == "http://127.0.0.1:8000/health"
        assert timeout > 0
        return _ReadyResponse({"local_only": True, "status": "ok"})

    result = wait_for_http_service(
        "http://127.0.0.1:8000/health",
        name="gateway",
        require_local_only=True,
        opener=opener,
    )

    assert result["ready"] is True
    assert result["attempts"] == 1
    assert result["payload"]["local_only"] is True


def test_wait_for_http_service_rejects_remote_url_before_request() -> None:
    def opener(url: str, timeout: float) -> _ReadyResponse:
        raise AssertionError("remote URL should be rejected before opener is called")

    with pytest.raises(ValueError, match="loopback"):
        wait_for_http_service("https://example.com/health", name="gateway", opener=opener)


def test_wait_for_http_service_times_out_with_last_error() -> None:
    ticks = iter([0.0, 0.05, 0.2])

    def opener(url: str, timeout: float) -> _ReadyResponse:
        raise URLError("connection refused")

    with pytest.raises(TimeoutError, match="connection refused"):
        wait_for_http_service(
            "http://127.0.0.1:8000/health",
            name="gateway",
            timeout_seconds=0.1,
            interval_seconds=0.01,
            opener=opener,
            sleep=lambda seconds: None,
            monotonic=lambda: next(ticks),
        )


def test_wait_for_http_service_requires_local_only_when_requested() -> None:
    def opener(url: str, timeout: float) -> _ReadyResponse:
        return _ReadyResponse({"local_only": False, "status": "ok"})

    with pytest.raises(TimeoutError, match="local_only"):
        wait_for_http_service(
            "http://127.0.0.1:8000/health",
            name="gateway",
            timeout_seconds=0.1,
            interval_seconds=0.01,
            require_local_only=True,
            opener=opener,
            sleep=lambda seconds: None,
            monotonic=_three_ticks(),
        )


def test_start_all_local_uses_readiness_probe_for_requested_services() -> None:
    script = Path("scripts/start_all_local.sh").read_text(encoding="utf-8")

    assert "scripts/wait_for_service.py" in script
    assert 'wait_ready text "http://127.0.0.1:8081/health"' in script
    assert 'wait_ready vision "http://127.0.0.1:8082/health"' in script
    assert 'wait_ready asr "http://127.0.0.1:8090/health" --require-local-only' in script
    assert (
        'wait_ready gateway "http://${GATEWAY_HOST}:${GATEWAY_PORT}/health" --require-local-only'
        in script
    )
    assert "export PYTHON_BIN" in script
    assert "scripts/start_gradio_app.sh" in script


def test_llamacpp_launchers_support_cuda_preflight_and_gpu_layers() -> None:
    text_script = Path("scripts/start_text_llamacpp.sh").read_text(encoding="utf-8")
    vision_script = Path("scripts/start_vision_llamacpp.sh").read_text(encoding="utf-8")

    for script in (text_script, vision_script):
        assert 'LLAMA_GPU_LAYERS="${LLAMA_GPU_LAYERS:--1}"' in script
        assert 'LLAMA_REQUIRE_CUDA="${LLAMA_REQUIRE_CUDA:-0}"' in script
        assert 'LLAMA_CUDA_LIBRARY_PATH="${LLAMA_CUDA_LIBRARY_PATH:-}"' in script
        assert 'export LD_LIBRARY_PATH="${LLAMA_CUDA_LIBRARY_PATH}:${LD_LIBRARY_PATH:-}"' in script
        assert "scripts/check_llamacpp_cuda.py" in script
        assert '--n-gpu-layers "${LLAMA_GPU_LAYERS}"' in script


def test_start_gradio_app_launcher_is_loopback_only_and_disables_analytics() -> None:
    script = Path("scripts/start_gradio_app.sh").read_text(encoding="utf-8")

    assert 'HOST="${GRADIO_HOST:-127.0.0.1}"' in script
    assert 'PORT="${GRADIO_PORT:-7860}"' in script
    assert "Refusing to bind Gradio app to non-local host" in script
    assert 'export GRADIO_SERVER_NAME="${HOST}"' in script
    assert 'export GRADIO_SERVER_PORT="${PORT}"' in script
    assert 'export GRADIO_ANALYTICS_ENABLED="${GRADIO_ANALYTICS_ENABLED:-False}"' in script
    assert 'exec "${PYTHON_BIN}" app.py' in script


def _three_ticks() -> object:
    ticks = iter([0.0, 0.05, 0.2])
    return lambda: next(ticks)
