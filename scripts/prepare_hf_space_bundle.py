from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SPACE_SOURCE = ROOT / "spaces" / "local_lm_accessible"
DEFAULT_OUTPUT = ROOT / "dist" / "hf_space_local_lm_accessible"
REQUIRED_FILES = ("README.md", "app.py", "requirements.txt")


def prepare_space_bundle(
    *,
    source: Path = SPACE_SOURCE,
    output: Path = DEFAULT_OUTPUT,
    clean: bool = True,
) -> dict[str, Any]:
    if not source.exists():
        raise FileNotFoundError(f"missing Space source directory: {source}")
    missing = [name for name in REQUIRED_FILES if not (source / name).exists()]
    if missing:
        raise FileNotFoundError(f"missing required Space file(s): {', '.join(missing)}")
    if clean and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for name in REQUIRED_FILES:
        source_path = source / name
        target_path = output / name
        shutil.copy2(source_path, target_path)
        copied.append(_display_path(target_path))
    return {
        "source": _display_path(source),
        "output": _display_path(output),
        "files": copied,
        "publish_commands": [
            [
                "hf",
                "repos",
                "create",
                "build-small-hackathon/local-lm-accessible",
                "--type",
                "space",
                "--space-sdk",
                "gradio",
                "--exist-ok",
            ],
            [
                "hf",
                "upload",
                "build-small-hackathon/local-lm-accessible",
                _display_path(output),
                "--type",
                "space",
            ],
        ],
    }


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SPACE_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-clean", action="store_true")
    args = parser.parse_args()
    result = prepare_space_bundle(source=args.source, output=args.output, clean=not args.no_clean)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
