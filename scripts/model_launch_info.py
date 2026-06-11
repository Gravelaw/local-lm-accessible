from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.verify_model_checksums import load_manifest, verify_manifest  # noqa: E402


def launch_info(model_key: str) -> dict[str, Any]:
    manifest = load_manifest()
    model = next((item for item in manifest["models"] if item["key"] == model_key), None)
    if model is None:
        raise KeyError(f"model not found in manifest: {model_key}")
    verified = verify_manifest(model_key)[0]
    artifacts = {
        artifact["name"]: artifact["path"] for artifact in verified.get("additional_artifacts", [])
    }
    return {
        "key": model_key,
        "model_path": verified["path"],
        "port": int(model["port"]),
        "runtime": model["runtime"],
        "artifacts": artifacts,
    }


def shell_assignments(info: dict[str, Any]) -> str:
    lines = [
        f"MANIFEST_MODEL_PATH={shlex.quote(str(info['model_path']))}",
        f"MANIFEST_MODEL_PORT={shlex.quote(str(info['port']))}",
    ]
    if "mmproj" in info["artifacts"]:
        lines.append(f"MANIFEST_MMPROJ_PATH={shlex.quote(str(info['artifacts']['mmproj']))}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_key", choices=["text", "vision", "asr", "omni"])
    parser.add_argument("--shell", action="store_true")
    args = parser.parse_args()

    info = launch_info(args.model_key)
    if args.shell:
        print(shell_assignments(info))
    else:
        print(json.dumps(info, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
