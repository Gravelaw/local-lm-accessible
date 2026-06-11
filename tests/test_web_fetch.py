from __future__ import annotations

import pytest

from services.tools.web_fetch import fetch_url


class _FakeResponse:
    headers = {"Content-Type": "text/html; charset=utf-8"}

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None

    def read(self, size: int) -> bytes:
        assert size > 0
        return (
            b"<html><body><h1>Title</h1><script>skip()</script><p>Hello reader.</p></body></html>"
        )


def _public_resolver(
    host: str, port: int, type: object
) -> list[tuple[object, object, object, str, tuple[str, int]]]:
    assert host == "example.com"
    assert port == 443
    assert type is not None
    return [(object(), object(), object(), "", ("93.184.216.34", port))]


def test_optional_web_fetch_extracts_text_without_remote_upload() -> None:
    def opener(request: object, timeout: int) -> _FakeResponse:
        assert request.full_url == "https://example.com/article"  # type: ignore[attr-defined]
        assert timeout == 10
        return _FakeResponse()

    payload = fetch_url("https://example.com/article", opener=opener, resolver=_public_resolver)

    assert payload["url"] == "https://example.com/article"
    assert "Title" in str(payload["text"])
    assert "Hello reader." in str(payload["text"])
    assert "skip" not in str(payload["text"])
    assert payload["remote_uploads"] is False


def test_optional_web_fetch_rejects_localhost() -> None:
    with pytest.raises(ValueError, match="local hostnames"):
        fetch_url("http://localhost:8000/private")


def test_optional_web_fetch_rejects_private_ip() -> None:
    with pytest.raises(ValueError, match="private IP"):
        fetch_url("http://127.0.0.1:8000/private")


def test_optional_web_fetch_rejects_hostname_resolving_to_private_ip() -> None:
    def resolver(
        host: str, port: int, type: object
    ) -> list[tuple[object, object, object, str, tuple[str, int]]]:
        assert host == "metadata.example"
        assert port == 80
        assert type is not None
        return [(object(), object(), object(), "", ("169.254.169.254", port))]

    with pytest.raises(ValueError, match="resolving to local or private IP"):
        fetch_url("http://metadata.example/private", resolver=resolver)


def test_optional_web_fetch_rejects_unresolvable_hostname() -> None:
    def resolver(
        host: str, port: int, type: object
    ) -> list[tuple[object, object, object, str, tuple[str, int]]]:
        assert host == "missing.example"
        assert port == 443
        assert type is not None
        return []

    with pytest.raises(ValueError, match="could not resolve hostname"):
        fetch_url("https://missing.example/article", resolver=resolver)
