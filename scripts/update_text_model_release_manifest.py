from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from scripts.verify_model_checksums import sha256_path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "models" / "manifest.json"
MODELS_CONFIG_PATH = ROOT / "configs" / "models.yaml"


def update_text_model_release_manifest(
    *,
    gguf_path: Path,
    model_id: str,
    model_name: str,
    quantization: str,
    output_manifest: Path = MANIFEST_PATH,
    models_config: Path = MODELS_CONFIG_PATH,
    root: Path = ROOT,
) -> dict[str, Any]:
    if not gguf_path.exists():
        raise FileNotFoundError(f"missing GGUF artifact: {gguf_path}")
    relative_path = _relative_to_root(gguf_path, root=root)
    checksum = sha256_path(gguf_path)

    manifest = json.loads(output_manifest.read_text(encoding="utf-8"))
    for model in manifest["models"]:
        if model["key"] != "text":
            continue
        model.update(
            {
                "model_name": model_name,
                "model_id": model_id,
                "vendor": "NVIDIA",
                "base_params": 8_000_000_000,
                "adapter_params": 167_832_240,
                "quantization": quantization,
                "local_path": relative_path,
                "sha256": checksum,
                "license": "NVIDIA Open Model License Agreement",
                "runtime": "llama.cpp",
                "port": 8081,
                "commercial_status": "allowed",
                "notes": (
                    "Fine-tuned local-lm text adapter merged into "
                    "nvidia/Llama-3.1-Nemotron-Nano-8B-v1 and exported to GGUF. "
                    "Deployment target is local llama.cpp only."
                ),
            }
        )
        break
    else:
        raise KeyError("text model missing from manifest")
    output_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    config = yaml.safe_load(models_config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"{models_config} must contain a YAML mapping")
    text_config = config.setdefault("models", {}).setdefault("text", {})
    text_config.update(
        {
            "id": model_id,
            "source": "NVIDIA",
            "format": "GGUF",
            "serving": "llama.cpp",
            "endpoint": "http://127.0.0.1:8081",
            "parameters_b": 8,
            "enabled": True,
        }
    )
    models_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return {
        "model_key": "text",
        "model_id": model_id,
        "local_path": relative_path,
        "sha256": checksum,
        "manifest": str(output_manifest),
        "models_config": str(models_config),
    }


def _relative_to_root(path: Path, *, root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(root.resolve()))
    except ValueError:
        return str(resolved)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf-path", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument(
        "--model-name",
        default="local-lm accessible text Llama Nemotron Nano Q4_K_M",
    )
    parser.add_argument("--quantization", default="Q4_K_M")
    args = parser.parse_args()
    result = update_text_model_release_manifest(
        gguf_path=args.gguf_path,
        model_id=args.model_id,
        model_name=args.model_name,
        quantization=args.quantization,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
