from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.asr.prepare_manifest import ASRManifestRecord, validate_manifest  # noqa: E402

DEFAULT_MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v3"


def build_nemo_command(
    *,
    model_name: str,
    train_manifest: Path,
    val_manifest: Path,
    output_dir: Path,
    max_steps: int,
) -> list[str]:
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    return [
        "python3",
        "-m",
        "nemo.collections.asr",
        f"model={model_name}",
        f"train_manifest={train_manifest}",
        f"validation_manifest={val_manifest}",
        f"exp_manager.exp_dir={output_dir}",
        f"trainer.max_steps={max_steps}",
    ]


def validate_training_inputs(
    *,
    train_manifest: Path,
    val_manifest: Path,
    require_audio_exists: bool,
) -> dict[str, Any]:
    train_records = validate_manifest(train_manifest, require_audio_exists=require_audio_exists)
    val_records = validate_manifest(val_manifest, require_audio_exists=require_audio_exists)
    return {
        "train_records": len(train_records),
        "val_records": len(val_records),
        "train_languages": _languages(train_records),
        "val_languages": _languages(val_records),
        "experimental_languages": _experimental_languages([*train_records, *val_records]),
        "require_audio_exists": require_audio_exists,
        "remote_audio_uploads": False,
    }


def dry_run_summary(
    *,
    model_name: str,
    train_manifest: Path,
    val_manifest: Path,
    output_dir: Path,
    max_steps: int,
    require_audio_exists: bool = False,
) -> dict[str, Any]:
    validation = validate_training_inputs(
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        require_audio_exists=require_audio_exists,
    )
    command = build_nemo_command(
        model_name=model_name,
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        output_dir=output_dir,
        max_steps=max_steps,
    )
    return {
        "mode": "dry_run",
        "model": model_name,
        "training_backend": "nemo_experimental",
        "local_only": True,
        "command": command,
        **validation,
    }


def run_training(
    *,
    model_name: str,
    train_manifest: Path,
    val_manifest: Path,
    output_dir: Path,
    max_steps: int,
    require_audio_exists: bool = True,
) -> list[str]:
    validate_training_inputs(
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        require_audio_exists=require_audio_exists,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    command = build_nemo_command(
        model_name=model_name,
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        output_dir=output_dir,
        max_steps=max_steps,
    )
    subprocess.run(command, cwd=ROOT, check=True)
    return command


def _languages(records: list[ASRManifestRecord]) -> list[str]:
    return sorted({record.language for record in records})


def _experimental_languages(records: list[ASRManifestRecord]) -> list[str]:
    return sorted({record.language for record in records if record.experimental})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and launch experimental Parakeet NeMo training."
    )
    parser.add_argument("--model-name", default=os.environ.get("MODEL_NAME", DEFAULT_MODEL_NAME))
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--require-audio-exists", action="store_true")
    args = parser.parse_args()

    if os.environ.get("ALLOW_REMOTE_AUDIO_UPLOAD", "0") != "0":
        raise SystemExit("Remote audio upload is not allowed for local-lm ASR training.")

    if args.dry_run:
        summary = dry_run_summary(
            model_name=args.model_name,
            train_manifest=args.train_manifest,
            val_manifest=args.val_manifest,
            output_dir=args.output_dir,
            max_steps=args.max_steps,
            require_audio_exists=args.require_audio_exists,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    command = run_training(
        model_name=args.model_name,
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        require_audio_exists=True,
    )
    print(json.dumps({"status": "ok", "command": command}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
