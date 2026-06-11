from __future__ import annotations

from pathlib import Path

import pytest

from services.gateway.router import LocalRouter, load_routes
from services.gateway.schemas import RouteRequest


def test_routes_general_assistant_to_local_llamacpp() -> None:
    router = load_routes(Path("configs/routes.yaml"))

    target = router.route("general_local_assistant")

    assert target.provider == "local"
    assert target.model_key == "text"
    assert target.endpoint.host == "127.0.0.1"


def test_unknown_task_is_rejected() -> None:
    router = load_routes(Path("configs/routes.yaml"))

    with pytest.raises(KeyError):
        router.route("cloud_task")


def test_remote_route_is_rejected() -> None:
    config = {
        "runtime": {"local_only": True, "privacy_mode": "strict"},
        "routes": {
            "general_local_assistant": {
                "provider": "local",
                "model_key": "text",
                "endpoint": "https://api.example.com/model",
            }
        },
    }

    with pytest.raises(ValueError, match="loopback"):
        LocalRouter(config)


@pytest.mark.parametrize(
    ("route_request", "expected_task"),
    [
        (RouteRequest(file_path="voice.wav", intent="transcribe"), "speech_to_text"),
        (RouteRequest(file_path="invoice.pdf", intent="export to Excel"), "document_to_excel"),
        (
            RouteRequest(file_path="photo.png", intent="describe for accessibility"),
            "describe_image",
        ),
        (RouteRequest(file_path="sign.jpg", intent="translate image text"), "translate_image_text"),
        (RouteRequest(url="https://en.wikipedia.org/wiki/Localhost"), "summarize_wikipedia"),
        (RouteRequest(url="https://example.com/article"), "summarize_url"),
        (RouteRequest(intent="hello"), "general_local_assistant"),
    ],
)
def test_router_classifies_task_requests(route_request: RouteRequest, expected_task: str) -> None:
    router = load_routes(Path("configs/routes.yaml"))

    decision = router.decide(route_request)

    assert decision.task == expected_task
    assert decision.provider == "local"
    assert decision.endpoint.host == "127.0.0.1"
    assert decision.privacy_mode == "strict"
    assert decision.allow_web is False


def test_router_reflects_explicit_web_opt_in_for_url_task() -> None:
    router = load_routes(Path("configs/routes.yaml"))

    decision = router.decide(RouteRequest(url="https://example.com/article", allow_web=True))

    assert decision.task == "summarize_url"
    assert decision.allow_web is True
