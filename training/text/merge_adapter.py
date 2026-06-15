from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any


def model_load_kwargs(
    *,
    local_files_only: bool = True,
    trust_remote_code: bool = False,
    torch_dtype: object | None = None,
) -> dict[str, Any]:
    kwargs = {
        "device_map": "auto",
        "trust_remote_code": trust_remote_code,
        "local_files_only": local_files_only,
    }
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype
    return kwargs


def tokenizer_load_kwargs(
    *,
    local_files_only: bool = True,
    trust_remote_code: bool = False,
) -> dict[str, Any]:
    return {
        "trust_remote_code": trust_remote_code,
        "use_fast": True,
        "local_files_only": local_files_only,
    }


def merge_adapter(
    base_model: str,
    adapter_dir: Path,
    output_dir: Path,
    *,
    local_files_only: bool = True,
    trust_remote_code: bool = False,
    dtype: str = "auto",
) -> None:
    if not adapter_dir.exists():
        raise FileNotFoundError(f"missing adapter directory: {adapter_dir}")
    if output_dir.resolve() == adapter_dir.resolve():
        raise ValueError("output_dir must differ from adapter_dir")
    if local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = _torch_dtype(dtype)

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        **model_load_kwargs(
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        ),
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        **tokenizer_load_kwargs(
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        ),
    )
    merged = PeftModel.from_pretrained(
        model,
        adapter_dir,
        is_trainable=False,
        local_files_only=local_files_only,
    ).merge_and_unload()
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16")
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--allow-remote-files",
        action="store_true",
        help="Allow Hugging Face downloads during merge. Disabled by default.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow model repository code execution. Disabled by default.",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
        help="Torch dtype for loading the base model before merging.",
    )
    args = parser.parse_args()
    merge_adapter(
        args.base_model,
        args.adapter_dir,
        args.output_dir,
        local_files_only=not args.allow_remote_files,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
    )


def _torch_dtype(dtype: str) -> object | None:
    if dtype == "auto":
        return None
    import torch

    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float16":
        return torch.float16
    if dtype == "float32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {dtype}")


if __name__ == "__main__":
    main()
