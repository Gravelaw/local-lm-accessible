from __future__ import annotations

import ipaddress
import re
import socket
from collections.abc import Callable
from html.parser import HTMLParser
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import ParseResult, urlparse
from urllib.request import Request, urlopen

MAX_FETCH_BYTES = 200_000
DEFAULT_TIMEOUT_SECONDS = 10
LOCAL_HOSTS = {"localhost"}
TEXT_CONTENT_TYPES = {"text/html", "text/plain", "application/xhtml+xml"}


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1
        if tag in {"p", "br", "div", "section", "article", "li", "h1", "h2", "h3"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._skip_depth > 0:
            self._skip_depth -= 1
        if tag in {"p", "div", "section", "article", "li"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0 and data.strip():
            self.parts.append(data)


def fetch_url(
    url: str,
    *,
    opener: Callable[..., Any] = urlopen,
    resolver: Callable[..., Any] = socket.getaddrinfo,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    max_bytes: int = MAX_FETCH_BYTES,
) -> dict[str, object]:
    parsed = urlparse(url)
    _validate_public_http_url(parsed, resolver=resolver)
    request = Request(
        url,
        headers={"User-Agent": "local-lm/0.1 optional-user-enabled-fetch"},
        method="GET",
    )
    try:
        with opener(request, timeout=timeout_seconds) as response:
            content_type = str(response.headers.get("Content-Type", "text/plain")).split(";")[0]
            if content_type.lower() not in TEXT_CONTENT_TYPES:
                raise ValueError(f"unsupported content type for optional web fetch: {content_type}")
            raw = response.read(max_bytes + 1)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise RuntimeError("optional web fetch failed") from exc
    if len(raw) > max_bytes:
        raise ValueError(f"optional web fetch exceeded {max_bytes} bytes")
    text = _html_to_text(raw.decode("utf-8", errors="replace"))
    if not text:
        raise ValueError("optional web fetch returned no readable text")
    return {
        "url": url,
        "text": text,
        "content_type": content_type.lower(),
        "bytes_read": len(raw),
        "remote_uploads": False,
    }


def summarize_url_blocked_by_default(url: str) -> dict[str, object]:
    return {
        "url": url,
        "summary": "",
        "blocked": True,
        "reason": "allow_web=false in strict privacy mode",
        "remote_uploads": False,
    }


def _validate_public_http_url(
    parsed: ParseResult,
    *,
    resolver: Callable[..., Any],
) -> None:
    scheme = parsed.scheme
    host = parsed.hostname or ""
    if scheme not in {"http", "https"}:
        raise ValueError("optional web fetch requires http or https")
    if not host:
        raise ValueError("optional web fetch requires a hostname")
    if host.casefold() in LOCAL_HOSTS:
        raise ValueError("optional web fetch rejects local hostnames")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        _validate_resolved_host_is_public(host, _resolve_port(parsed), resolver)
        return
    if not _is_public_address(address):
        raise ValueError("optional web fetch rejects local or private IP addresses")


def _validate_resolved_host_is_public(
    host: str,
    port: int,
    resolver: Callable[..., Any],
) -> None:
    try:
        resolved = resolver(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValueError("optional web fetch could not resolve hostname") from exc

    addresses: set[ipaddress.IPv4Address | ipaddress.IPv6Address] = set()
    for item in resolved:
        if len(item) < 5:
            continue
        sockaddr = item[4]
        if not sockaddr:
            continue
        try:
            addresses.add(ipaddress.ip_address(sockaddr[0]))
        except (IndexError, ValueError):
            continue
    if not addresses:
        raise ValueError("optional web fetch could not resolve hostname")
    if any(not _is_public_address(address) for address in addresses):
        raise ValueError(
            "optional web fetch rejects hostnames resolving to local or private IP addresses"
        )


def _is_public_address(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return address.is_global


def _resolve_port(parsed: ParseResult) -> int:
    if parsed.port is not None:
        return parsed.port
    if parsed.scheme == "http":
        return 80
    if parsed.scheme == "https":
        return 443
    raise ValueError("optional web fetch requires http or https")


def _html_to_text(value: str) -> str:
    extractor = _TextExtractor()
    extractor.feed(value)
    text = " ".join(" ".join(extractor.parts).split())
    if not text:
        text = " ".join(value.split())
    return re.sub(r"\s+", " ", text).strip()
