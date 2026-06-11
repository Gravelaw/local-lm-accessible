from __future__ import annotations

from urllib.parse import urlparse

LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def require_loopback_url(url: str, *, label: str = "url") -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"{label} must use http or https")
    host = parsed.hostname or ""
    if host not in LOOPBACK_HOSTS:
        raise ValueError(f"{label} must use a loopback host")
    return url
