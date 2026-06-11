from __future__ import annotations

from scripts.check_llamacpp_cuda import check_llamacpp_cuda, parse_llamacpp_devices


def test_parse_llamacpp_cuda_devices_accepts_visible_gpu() -> None:
    result = parse_llamacpp_devices(
        "Available devices:\n  - CUDA0: NVIDIA GPU, 12000 MiB free\n",
        "",
        0,
        llama_server="llama-server",
    )

    assert result["cuda_ready"] is True
    assert result["status"] == "ok"
    assert result["devices"] == ["- CUDA0: NVIDIA GPU, 12000 MiB free"]


def test_parse_llamacpp_cuda_devices_rejects_no_devices() -> None:
    result = parse_llamacpp_devices(
        "Available devices:\n",
        "",
        0,
        llama_server="llama-server",
    )

    assert result["cuda_ready"] is False
    assert result["status"] == "failed"
    assert "listed no GPU devices" in result["error"]


def test_parse_llamacpp_cuda_devices_rejects_driver_runtime_mismatch() -> None:
    result = parse_llamacpp_devices(
        "Available devices:\n",
        "ggml_cuda_init: failed to initialize CUDA: CUDA driver version is insufficient",
        0,
        llama_server="llama-server",
    )

    assert result["cuda_ready"] is False
    assert result["status"] == "failed"
    assert "CUDA initialization" in result["error"]


def test_check_llamacpp_cuda_injects_requested_library_path() -> None:
    captured: dict[str, object] = {}

    def fake_run(
        command: list[str],
        check: bool,
        capture_output: bool,
        text: bool,
        timeout: float,
        env: dict[str, str],
    ) -> object:
        captured["command"] = command
        captured["check"] = check
        captured["capture_output"] = capture_output
        captured["text"] = text
        captured["timeout"] = timeout
        captured["ld_library_path"] = env["LD_LIBRARY_PATH"]

        class Completed:
            stdout = "Available devices:\n  - CUDA0: NVIDIA GPU\n"
            stderr = ""
            returncode = 0

        return Completed()

    import scripts.check_llamacpp_cuda as cuda_check

    original = cuda_check.subprocess.run
    try:
        cuda_check.subprocess.run = fake_run  # type: ignore[assignment]
        result = check_llamacpp_cuda(
            "/tmp/llama-server",
            env={"LLAMA_CUDA_LIBRARY_PATH": "/tmp/cuda/lib", "LD_LIBRARY_PATH": "/tmp/base"},
        )
    finally:
        cuda_check.subprocess.run = original

    assert result["cuda_ready"] is True
    assert captured["command"] == ["/tmp/llama-server", "--list-devices"]
    assert captured["ld_library_path"] == "/tmp/cuda/lib:/tmp/base"
