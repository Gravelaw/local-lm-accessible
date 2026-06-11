from __future__ import annotations

import ipaddress
import socket
from types import TracebackType
from typing import Any


class CloudCallBlocked(RuntimeError):
    """Raised when an eval attempts network access outside loopback."""


class LocalOnlyNetworkGuard:
    def __init__(self) -> None:
        self._original_connect: Any = None

    def __enter__(self) -> LocalOnlyNetworkGuard:
        self._original_connect = socket.socket.connect

        def guarded_connect(sock: socket.socket, address: Any) -> Any:
            host = _host_from_address(address)
            if not is_loopback_host(host):
                raise CloudCallBlocked(
                    f"eval network access is limited to loopback hosts, got {host}"
                )
            return self._original_connect(sock, address)

        socket.socket.connect = guarded_connect
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        if self._original_connect is not None:
            socket.socket.connect = self._original_connect


def is_loopback_host(host: str) -> bool:
    if host in {"localhost", "::1"}:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _host_from_address(address: Any) -> str:
    if isinstance(address, tuple) and address:
        return str(address[0])
    return str(address)
