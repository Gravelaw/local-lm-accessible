from __future__ import annotations

from pathlib import Path

import pytest

from scripts import check_cuda_toolchain


def test_cuda_toolchain_check_reports_non_cuda_local_toolchain() -> None:
    report = check_cuda_toolchain.check_cuda_toolchain(require_cuda=False)

    assert report["ready"] is True
    assert report["cc"]["path"]
    assert report["cxx"]["path"]
    assert report["cuda_host_cxx"]["path"]


def test_cuda_toolchain_check_rejects_bad_clang_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(check_cuda_toolchain.shutil, "which", lambda name: f"/tools/{name}")
    monkeypatch.setattr(Path, "exists", lambda self: True)

    def fake_version(path: str) -> str:
        if path.endswith("clang++"):
            return "clang version 0.0.0"
        if path.endswith("nvcc"):
            return "Cuda compilation tools, release 13.0, V13.0.88"
        return "gcc version 13.3.0"

    monkeypatch.setattr(check_cuda_toolchain, "_version_output", fake_version)
    monkeypatch.setenv("CC", "gcc")
    monkeypatch.setenv("CXX", "clang++")
    monkeypatch.setenv("CUDAHOSTCXX", "clang++")
    monkeypatch.setenv("CUDACXX", "nvcc")
    monkeypatch.setenv("CUDA_HOME", "/usr/local/cuda")

    with pytest.raises(RuntimeError, match="unsupported clang"):
        check_cuda_toolchain.check_cuda_toolchain(require_cuda=True)
