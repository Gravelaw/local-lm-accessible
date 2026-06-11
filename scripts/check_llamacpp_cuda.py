from __future__ import annotations

import argparse
import json
import os
import subprocess
from typing import Any

CUDA_INIT_FAILURE_MARKERS = (
    "ggml_cuda_init: failed",
    "cuda driver version is insufficient",
    "error while loading shared libraries",
    "libcudart.so",
    "libcublas.so",
)


def check_llamacpp_cuda(
    llama_server: str,
    *,
    timeout_seconds: float = 10.0,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    command = [llama_server, "--list-devices"]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=_cuda_env(env or os.environ.copy()),
        )
    except FileNotFoundError as exc:
        return {
            "status": "failed",
            "cuda_ready": False,
            "llama_server": llama_server,
            "returncode": None,
            "devices": [],
            "error": str(exc),
            "next_actions": _next_actions(),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "failed",
            "cuda_ready": False,
            "llama_server": llama_server,
            "returncode": None,
            "devices": [],
            "error": f"CUDA device check timed out after {timeout_seconds:g}s: {exc}",
            "next_actions": _next_actions(),
        }

    result = parse_llamacpp_devices(
        completed.stdout,
        completed.stderr,
        completed.returncode,
        llama_server=llama_server,
    )
    result["next_actions"] = [] if result["cuda_ready"] else _next_actions()
    return result


def parse_llamacpp_devices(
    stdout: str,
    stderr: str,
    returncode: int,
    *,
    llama_server: str,
) -> dict[str, Any]:
    combined = f"{stdout}\n{stderr}".strip()
    lower_output = combined.casefold()
    devices = _device_lines(stdout)
    init_failure = any(marker in lower_output for marker in CUDA_INIT_FAILURE_MARKERS)
    cuda_ready = returncode == 0 and bool(devices) and not init_failure
    error = ""
    if returncode != 0:
        error = f"llama-server --list-devices exited with code {returncode}"
    elif init_failure:
        error = "llama.cpp reported CUDA initialization or runtime-library failure"
    elif not devices:
        error = "llama.cpp listed no GPU devices"

    return {
        "status": "ok" if cuda_ready else "failed",
        "cuda_ready": cuda_ready,
        "llama_server": llama_server,
        "returncode": returncode,
        "devices": devices,
        "error": error,
        "stdout": stdout.strip(),
        "stderr": stderr.strip(),
    }


def _device_lines(output: str) -> list[str]:
    lines = [line.strip() for line in output.splitlines()]
    devices: list[str] = []
    in_devices = False
    for line in lines:
        if not line:
            continue
        if line.casefold().startswith("available devices"):
            in_devices = True
            continue
        if not in_devices:
            continue
        if line.startswith("-") or "cuda" in line.casefold() or "gpu" in line.casefold():
            devices.append(line)
    return devices


def _cuda_env(env: dict[str, str]) -> dict[str, str]:
    cuda_library_path = env.get("LLAMA_CUDA_LIBRARY_PATH", "").strip()
    if not cuda_library_path:
        return env
    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = (
        cuda_library_path if not existing else f"{cuda_library_path}:{existing}"
    )
    return env


def _next_actions() -> list[str]:
    return [
        "Confirm the host exposes an NVIDIA GPU to this shell: nvidia-smi must work.",
        (
            "On WSL, install/update the Windows NVIDIA driver and ensure "
            "/usr/lib/wsl/lib/libcuda.so.1 can see a device."
        ),
        (
            "Use a llama.cpp CUDA build compatible with the host driver, or rebuild "
            "with the installed CUDA toolkit."
        ),
        (
            "If using Python-packaged CUDA libraries, set LLAMA_CUDA_LIBRARY_PATH "
            "before launching llama.cpp."
        ),
        "Set LLAMA_REQUIRE_CUDA=1 during demo startup to fail fast instead of silently using CPU.",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether llama.cpp can see CUDA devices.")
    parser.add_argument("--llama-server", default=os.environ.get("LLAMA_SERVER", "llama-server"))
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--require", action="store_true")
    args = parser.parse_args()

    result = check_llamacpp_cuda(args.llama_server, timeout_seconds=args.timeout)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.require and not result["cuda_ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
