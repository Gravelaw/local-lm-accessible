from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.local_runtime import require_loopback_url  # noqa: E402


def wait_for_http_service(
    url: str,
    *,
    name: str,
    timeout_seconds: float = 45.0,
    interval_seconds: float = 0.5,
    request_timeout_seconds: float = 5.0,
    require_local_only: bool = False,
    opener: Callable[..., Any] = urlopen,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    require_loopback_url(url, label=f"{name} readiness url")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive")
    if request_timeout_seconds <= 0:
        raise ValueError("request_timeout_seconds must be positive")

    deadline = monotonic() + timeout_seconds
    attempts = 0
    last_error = "not attempted"
    while True:
        attempts += 1
        try:
            remaining_for_attempt = max(0.1, deadline - monotonic())
            request_timeout = min(request_timeout_seconds, remaining_for_attempt)
            with opener(url, timeout=request_timeout) as response:
                status = int(getattr(response, "status", response.getcode()))
                body = response.read().decode("utf-8")
            if 200 <= status < 300:
                payload = _decode_json_object(body)
                if require_local_only and payload.get("local_only") is not True:
                    raise ValueError("service did not report local_only=true")
                return {
                    "name": name,
                    "url": url,
                    "ready": True,
                    "attempts": attempts,
                    "status": status,
                    "payload": payload,
                }
            last_error = f"HTTP {status}"
        except (
            HTTPError,
            URLError,
            TimeoutError,
            OSError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            last_error = str(exc)

        remaining = deadline - monotonic()
        if remaining <= 0:
            raise TimeoutError(f"{name} did not become ready at {url}: {last_error}")
        sleep(min(interval_seconds, remaining))


def _decode_json_object(body: str) -> dict[str, Any]:
    if not body.strip():
        return {}
    payload = json.loads(body)
    if not isinstance(payload, dict):
        raise ValueError("service readiness response must be a JSON object")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Wait for a local HTTP service to become ready.")
    parser.add_argument("--url", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--request-timeout", type=float, default=5.0)
    parser.add_argument("--require-local-only", action="store_true")
    args = parser.parse_args()

    result = wait_for_http_service(
        args.url,
        name=args.name,
        timeout_seconds=args.timeout,
        interval_seconds=args.interval,
        request_timeout_seconds=args.request_timeout,
        require_local_only=args.require_local_only,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
