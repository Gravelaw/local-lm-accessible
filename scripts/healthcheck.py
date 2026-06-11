from __future__ import annotations

import argparse
import json
import socket
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.local_runtime import require_loopback_url  # noqa: E402
from scripts.verify_model_checksums import load_manifest  # noqa: E402
from services.gateway.router import load_routes  # noqa: E402
from services.gateway.schemas import RuntimeConfig  # noqa: E402

HTTP_OPENER = urlopen
SERVICE_HEALTH_TIMEOUT_SECONDS = 5


def check_configs() -> list[str]:
    checked = []
    for config_path in (ROOT / "configs").glob("*.yaml"):
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        runtime = config.get("runtime")
        if runtime is not None:
            RuntimeConfig.model_validate(runtime)
        checked.append(str(config_path.relative_to(ROOT)))
    return checked


def check_manifest() -> dict[str, Any]:
    manifest = load_manifest()
    return {
        "version": manifest["version"],
        "models": [
            {
                "key": model["key"],
                "runtime": model["runtime"],
                "port": model["port"],
                "local_path": model["local_path"],
                "sha256_configured": bool(str(model["sha256"]).strip()),
            }
            for model in manifest["models"]
        ],
    }


def check_gateway(
    gateway_url: str | None,
    *,
    opener: Callable[..., Any] = HTTP_OPENER,
) -> dict[str, Any] | None:
    if gateway_url is None:
        return None
    require_loopback_url(gateway_url, label="gateway")
    with opener(f"{gateway_url.rstrip('/')}/health", timeout=5) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if payload.get("local_only") is not True:
        raise ValueError("gateway did not report local_only=true")
    if payload.get("privacy_mode") != "strict":
        raise ValueError("gateway did not report privacy_mode=strict")
    if payload.get("allow_web") is not False:
        raise ValueError("gateway did not report allow_web=false")
    return payload


def check_ports(
    require_running: bool,
    *,
    opener: Callable[..., Any] = HTTP_OPENER,
) -> list[dict[str, Any]]:
    router = load_routes(ROOT / "configs" / "routes.yaml")
    checks = []
    for service in router.health_services():
        endpoint = service["endpoint"]
        host = endpoint.host
        port = endpoint.port
        running = _tcp_connects(str(host), int(port)) if port is not None else False
        health_url = f"{str(endpoint).rstrip('/')}/health"
        http_ready = False
        http_error = "not running"
        payload: dict[str, Any] = {}
        if running or require_running:
            http_ready, http_error, payload = _http_health_ready(health_url, opener=opener)
        if require_running and not service["optional"] and not http_ready:
            message = (
                f"required service did not pass HTTP readiness: "
                f"{service['name']} {health_url}: {http_error}"
            )
            raise ConnectionError(message)
        checks.append(
            {
                "name": service["name"],
                "host": host,
                "port": port,
                "optional": service["optional"],
                "running": running,
                "health_url": health_url,
                "http_ready": http_ready,
                "health_error": http_error,
                "health": payload,
            }
        )
    return checks


def _tcp_connects(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.25)
        return sock.connect_ex((host, port)) == 0


def _http_health_ready(
    url: str,
    *,
    opener: Callable[..., Any] = HTTP_OPENER,
) -> tuple[bool, str, dict[str, Any]]:
    require_loopback_url(url, label="service health")
    try:
        with opener(url, timeout=SERVICE_HEALTH_TIMEOUT_SECONDS) as response:
            status = int(getattr(response, "status", response.getcode()))
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError, ValueError) as exc:
        return False, str(exc), {}
    if not isinstance(payload, dict):
        return False, "health response must be a JSON object", {}
    if not 200 <= status < 300:
        return False, f"HTTP {status}", payload
    return True, "", payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gateway", help="Optional gateway URL, e.g. http://127.0.0.1:8000")
    parser.add_argument("--require-running", action="store_true")
    args = parser.parse_args()

    result = {
        "status": "ok",
        "local_only": True,
        "configs": check_configs(),
        "manifest": check_manifest(),
        "gateway": check_gateway(args.gateway),
        "services": check_ports(args.require_running),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
