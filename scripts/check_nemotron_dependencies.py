from __future__ import annotations

import argparse
import importlib
import json
from importlib import metadata
from typing import Any

CORE_PACKAGES = {
    "torch": "torch",
    "transformers": "transformers",
    "peft": "peft",
    "trl": "trl",
    "bitsandbytes": "bitsandbytes",
}

MAMBA_PACKAGES = {
    "mamba-ssm": "mamba_ssm",
    "causal-conv1d": "causal_conv1d",
}


def check_dependencies(*, include_mamba: bool = False) -> dict[str, Any]:
    required_packages = dict(CORE_PACKAGES)
    if include_mamba:
        required_packages.update(MAMBA_PACKAGES)
    packages: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for distribution_name, import_name in required_packages.items():
        try:
            module = importlib.import_module(import_name)
            version = metadata.version(distribution_name)
        except Exception as exc:  # pragma: no cover - exact import errors vary by CUDA image
            packages[distribution_name] = {
                "import": import_name,
                "available": False,
                "error": str(exc),
            }
            missing.append(distribution_name)
            continue
        packages[distribution_name] = {
            "import": import_name,
            "available": True,
            "version": version,
            "module": getattr(module, "__name__", import_name),
        }

    torch_report = _torch_report()
    report = {
        "ready": not missing and bool(torch_report["cuda_available"]),
        "include_mamba": include_mamba,
        "missing": missing,
        "packages": packages,
        "torch": torch_report,
    }
    if not torch_report["cuda_available"]:
        missing.append("cuda")
        report["missing"] = missing
        report["ready"] = False
    if missing:
        raise RuntimeError(json.dumps(report, indent=2, sort_keys=True))
    return report


def _torch_report() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        return {"cuda_available": False, "error": str(exc)}
    return {
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "cuda_version": getattr(torch.version, "cuda", None),
        "bf16_supported": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--include-mamba",
        action="store_true",
        help="Require mamba-ssm and causal-conv1d native extensions.",
    )
    args = parser.parse_args()
    report = check_dependencies(include_mamba=args.include_mamba)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
