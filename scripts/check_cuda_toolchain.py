from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


def check_cuda_toolchain(*, require_cuda: bool = True) -> dict[str, Any]:
    env = os.environ
    report: dict[str, Any] = {
        "cc": _tool_report(env.get("CC", "gcc")),
        "cxx": _tool_report(env.get("CXX", "g++")),
        "cuda_host_cxx": _tool_report(env.get("CUDAHOSTCXX", "g++")),
        "cuda_cxx": _tool_report(env.get("CUDACXX", "nvcc")),
        "cuda_home": env.get("CUDA_HOME", ""),
        "cmake_args": env.get("CMAKE_ARGS", ""),
        "cudaflags": env.get("CUDAFLAGS", ""),
        "nvcc_prepend_flags": env.get("NVCC_PREPEND_FLAGS", ""),
        "clang": _tool_report("clang++", required=False),
    }
    errors: list[str] = []
    for key in ("cc", "cxx", "cuda_host_cxx"):
        if not report[key]["path"]:
            errors.append(f"missing {key}: {report[key]['requested']}")
    if require_cuda and not report["cuda_cxx"]["path"]:
        errors.append(f"missing cuda compiler: {report['cuda_cxx']['requested']}")
    cuda_home = report["cuda_home"]
    if require_cuda and (not cuda_home or not Path(cuda_home).exists()):
        errors.append(f"missing CUDA_HOME: {cuda_home}")
    if "clang++" in str(report["cuda_host_cxx"]["requested"]):
        clang_version = report["cuda_host_cxx"].get("major_version")
        if clang_version is None or not 7 <= clang_version < 21:
            errors.append("CUDAHOSTCXX selects an unsupported clang++ for CUDA 13")
    if errors:
        raise RuntimeError(json.dumps({**report, "errors": errors}, indent=2, sort_keys=True))
    report["ready"] = True
    return report


def _tool_report(requested: str, *, required: bool = True) -> dict[str, Any]:
    path = shutil.which(requested) if "/" not in requested else requested
    exists = bool(path and Path(path).exists())
    output = _version_output(path) if exists and path else ""
    return {
        "requested": requested,
        "path": path if exists else "",
        "required": required,
        "version": output,
        "major_version": _major_version(output),
    }


def _version_output(path: str) -> str:
    for args in ([path, "--version"], [path, "-v"]):
        completed = subprocess.run(
            args,
            check=False,
            text=True,
            capture_output=True,
            timeout=15,
        )
        output = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
        if output.strip():
            return output.strip().splitlines()[0]
    return ""


def _major_version(output: str) -> int | None:
    match = re.search(r"(\d+)(?:\.\d+){0,2}", output)
    return int(match.group(1)) if match else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-require-cuda", action="store_true")
    args = parser.parse_args()
    report = check_cuda_toolchain(require_cuda=not args.no_require_cuda)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
