from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from training.text.merge_adapter import merge_adapter, model_load_kwargs, tokenizer_load_kwargs

ROOT = Path(__file__).resolve().parents[1]


def test_merge_adapter_defaults_to_local_files_only_and_no_remote_code() -> None:
    assert model_load_kwargs() == {
        "device_map": "auto",
        "trust_remote_code": False,
        "local_files_only": True,
    }
    assert tokenizer_load_kwargs() == {
        "trust_remote_code": False,
        "use_fast": True,
        "local_files_only": True,
    }


def test_merge_adapter_fails_before_import_when_adapter_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="missing adapter directory"):
        merge_adapter(
            "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16", tmp_path / "missing", tmp_path / "out"
        )


def test_training_llama_server_rejects_non_loopback_host(tmp_path: Path) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"not a real model")
    env = os.environ.copy()
    env["LLAMA_HOST"] = "0.0.0.0"

    result = subprocess.run(
        ["bash", "training/text/run_llama_server.sh", str(model)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "non-local host" in result.stderr


def test_export_gguf_requires_merged_model_directory_and_gguf_output(tmp_path: Path) -> None:
    convert_script = tmp_path / "convert_hf_to_gguf.py"
    convert_script.write_text("print('unused')\n", encoding="utf-8")
    env = os.environ.copy()
    env["CONVERT_SCRIPT"] = str(convert_script)

    missing_dir = tmp_path / "missing_model"
    result = subprocess.run(
        ["bash", "training/text/export_gguf.sh", str(missing_dir), str(tmp_path / "out.gguf")],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Missing merged HF model directory" in result.stderr

    merged_dir = tmp_path / "merged"
    merged_dir.mkdir()
    result = subprocess.run(
        ["bash", "training/text/export_gguf.sh", str(merged_dir), str(tmp_path / "out.bin")],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "must end with .gguf" in result.stderr


def test_quantize_gguf_writes_checksum_sidecars(tmp_path: Path) -> None:
    input_gguf = tmp_path / "input.gguf"
    input_gguf.write_bytes(b"f16 gguf")
    fake_quantize = tmp_path / "llama-quantize"
    fake_quantize.write_text(
        '#!/usr/bin/env bash\nset -euo pipefail\ncp "$1" "$2"\n',
        encoding="utf-8",
    )
    fake_quantize.chmod(0o755)
    output_dir = tmp_path / "quantized"
    env = os.environ.copy()
    env["LLAMA_QUANTIZE"] = str(fake_quantize)

    result = subprocess.run(
        ["bash", "training/text/quantize_gguf.sh", str(input_gguf), str(output_dir)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert (output_dir / "nemotron-router-summary-Q4_K_M.gguf").exists()
    assert (output_dir / "nemotron-router-summary-Q5_K_M.gguf").exists()
    assert (output_dir / "nemotron-router-summary-Q4_K_M.gguf.sha256").exists()
    assert (output_dir / "nemotron-router-summary-Q5_K_M.gguf.sha256").exists()


def test_quantize_gguf_supports_model_basename_override(tmp_path: Path) -> None:
    input_gguf = tmp_path / "input.gguf"
    input_gguf.write_bytes(b"f16 gguf")
    fake_quantize = tmp_path / "llama-quantize"
    fake_quantize.write_text(
        '#!/usr/bin/env bash\nset -euo pipefail\ncp "$1" "$2"\n',
        encoding="utf-8",
    )
    fake_quantize.chmod(0o755)
    output_dir = tmp_path / "quantized"
    env = os.environ.copy()
    env["LLAMA_QUANTIZE"] = str(fake_quantize)
    env["MODEL_BASENAME"] = "local-lm-accessible-text"

    result = subprocess.run(
        ["bash", "training/text/quantize_gguf.sh", str(input_gguf), str(output_dir)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert (output_dir / "local-lm-accessible-text-Q4_K_M.gguf").exists()
    assert (output_dir / "local-lm-accessible-text-Q5_K_M.gguf").exists()
