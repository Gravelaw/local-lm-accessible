from __future__ import annotations

import argparse
import json
import random
import struct
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.asr.prepare_manifest import (  # noqa: E402
    ASRManifestRecord,
    validate_manifest,
    write_jsonl,
)


def _clamp_int16(value: int) -> int:
    return max(-32768, min(32767, value))


def add_noisy_room_wav(
    input_path: Path,
    output_path: Path,
    seed: int,
    noise_level: int = 350,
) -> None:
    rng = random.Random(seed)
    with wave.open(str(input_path), "rb") as source:
        params = source.getparams()
        if params.sampwidth != 2:
            raise ValueError("noise augmentation expects 16-bit PCM WAV input")
        frames = source.readframes(params.nframes)

    samples = struct.unpack(f"<{len(frames) // 2}h", frames)
    augmented = [
        _clamp_int16(sample + rng.randint(-noise_level, noise_level)) for sample in samples
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as target:
        target.setparams(params)
        target.writeframes(struct.pack(f"<{len(augmented)}h", *augmented))


def augment_manifest(
    input_manifest: Path,
    output_manifest: Path,
    output_audio_dir: Path,
) -> list[ASRManifestRecord]:
    records = validate_manifest(input_manifest, require_audio_exists=True)
    augmented_records: list[ASRManifestRecord] = []
    for index, record in enumerate(records, start=1):
        input_path = Path(record.audio_filepath)
        output_path = output_audio_dir / f"{input_path.stem}_noisy_room.wav"
        add_noisy_room_wav(input_path, output_path, seed=index)
        payload = record.model_dump()
        payload["audio_filepath"] = str(output_path)
        payload["accent"] = f"{record.accent}_noisy_room"
        payload["source_dataset"] = "synthetic_noisy_room"
        augmented_records.append(ASRManifestRecord.model_validate(payload))
    write_jsonl(output_manifest, augmented_records)
    return augmented_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-audio-dir", type=Path, required=True)
    args = parser.parse_args()
    records = augment_manifest(args.input_manifest, args.output_manifest, args.output_audio_dir)
    print(json.dumps({"augmented": len(records), "remote_uploads": False}, sort_keys=True))


if __name__ == "__main__":
    main()
