from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.verify_model_checksums import MANIFEST_PATH, load_manifest  # noqa: E402

MODEL_FILE_SUFFIXES = {".gguf", ".bin", ".safetensors", ".pt", ".pth", ".onnx"}


def build_download_plan(
    manifest: dict[str, Any], model_key: str | None = None
) -> list[dict[str, Any]]:
    plan = []
    for model in manifest["models"]:
        if model_key is not None and model["key"] != model_key:
            continue
        local_path = ROOT / model["local_path"]
        command = build_hf_download_command(model)
        plan.append(
            {
                "key": model["key"],
                "model_id": model["model_id"],
                "vendor": model["vendor"],
                "local_path": str(local_path),
                "runtime": model["runtime"],
                "port": model["port"],
                "manual": True,
                "reason": "Large model downloads require an explicit operator action.",
                "command": command,
            }
        )
    if model_key is not None and not plan:
        raise KeyError(f"model not found in manifest: {model_key}")
    return plan


def build_hf_download_command(model: dict[str, Any]) -> list[str]:
    local_path = ROOT / str(model["local_path"])
    command = ["hf", "download", str(model["model_id"])]
    if local_path.suffix in MODEL_FILE_SUFFIXES:
        command.extend(["--local-dir", str(local_path.parent)])
        command.extend(["--include", local_path.name])
    else:
        command.extend(["--local-dir", str(local_path)])
    for artifact in model.get("additional_artifacts", []):
        artifact_path = ROOT / str(artifact["local_path"])
        if artifact_path.parent != local_path.parent:
            raise ValueError(
                f"additional artifact for {model['key']} must share the model local directory"
            )
        command.extend(["--include", artifact_path.name])
    return command


def download_model(
    manifest: dict[str, Any],
    model_key: str,
    *,
    allow_large_download: bool,
) -> list[str]:
    if not allow_large_download:
        raise ValueError("refusing large model download without --allow-large-download")
    models = {model["key"]: model for model in manifest["models"]}
    if model_key not in models:
        raise KeyError(f"model not found in manifest: {model_key}")
    model = models[model_key]
    command = build_hf_download_command(model)
    local_path = ROOT / str(model["local_path"])
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if local_path.suffix not in MODEL_FILE_SUFFIXES:
        local_path.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, cwd=ROOT, check=True)
    return command


def ensure_model_dirs(manifest: dict[str, Any]) -> None:
    for model in manifest["models"]:
        path = ROOT / model["local_path"]
        path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--model", help="Manifest key to download, e.g. text, vision, asr.")
    parser.add_argument("--print-plan", action="store_true")
    parser.add_argument("--create-dirs", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--allow-large-download", action="store_true")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    if args.create_dirs:
        ensure_model_dirs(manifest)
    if args.download:
        if not args.model:
            raise ValueError("--download requires --model")
        command = download_model(
            manifest,
            args.model,
            allow_large_download=args.allow_large_download,
        )
        print(json.dumps({"status": "ok", "command": command}, indent=2, sort_keys=True))
        return
    plan = build_download_plan(manifest, model_key=args.model)
    if args.print_plan or not args.create_dirs:
        print(json.dumps({"local_only": True, "download_plan": plan}, indent=2, sort_keys=True))
    else:
        print(json.dumps({"status": "ok", "created_dirs": True}, sort_keys=True))


if __name__ == "__main__":
    main()
