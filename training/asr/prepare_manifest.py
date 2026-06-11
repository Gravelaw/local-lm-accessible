from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

SUPPORTED_PARAKEET_V3_LANGUAGES = {
    "en",
    "bg",
    "cs",
    "da",
    "de",
    "el",
    "es",
    "et",
    "fi",
    "fr",
    "hr",
    "hu",
    "it",
    "lt",
    "lv",
    "mt",
    "nl",
    "pl",
    "pt",
    "ro",
    "sk",
    "sl",
    "sv",
    "uk",
    "ga",
    "is",
}
EXPERIMENTAL_REGIONS = {"india", "southeast_asia"}
NON_EXPERIMENTAL_LANGUAGES = SUPPORTED_PARAKEET_V3_LANGUAGES
UNKNOWN_LICENSE_VALUES = {"", "unknown", "unk", "n/a", "na", "none", "tbd", "unlicensed"}


class ASRManifestRecord(BaseModel):
    audio_filepath: str = Field(min_length=1)
    duration: float = Field(gt=0.0, le=24 * 60 * 60)
    text: str = Field(min_length=1)
    language: str = Field(min_length=2)
    region: str = Field(min_length=1)
    country: str = Field(min_length=1)
    modality: Literal["audio"] = "audio"
    task: Literal["speech_to_text", "asr"] = "speech_to_text"
    accent: str = Field(min_length=1)
    speaker_age_bucket: Literal["elderly", "adult", "unknown"]
    license: str = Field(min_length=1)
    pii_status: Literal["none", "redacted", "consented"]
    experimental: bool = False
    source_dataset: str = Field(default="unknown", min_length=1)

    @field_validator("language", "region", "accent", "country")
    @classmethod
    def normalize_lower(cls, value: str) -> str:
        return value.strip().lower()

    @field_validator("license")
    @classmethod
    def reject_unknown_license(cls, value: str) -> str:
        if value.strip().casefold() in UNKNOWN_LICENSE_VALUES:
            raise ValueError("unknown license is rejected")
        return value

    @model_validator(mode="after")
    def mark_experimental_non_english_india_sea(self) -> ASRManifestRecord:
        if self.region in EXPERIMENTAL_REGIONS and self.language != "en":
            self.experimental = True
        if self.language not in NON_EXPERIMENTAL_LANGUAGES:
            self.experimental = True
        return self

    @property
    def supported_by_parakeet_v3(self) -> bool:
        return self.language in SUPPORTED_PARAKEET_V3_LANGUAGES


def read_jsonl(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as manifest_file:
        for line_number, line in enumerate(manifest_file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"manifest row must be an object at {path}:{line_number}")
            records.append(payload)
    return records


def write_jsonl(path: Path, records: list[ASRManifestRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as manifest_file:
        for record in records:
            manifest_file.write(json.dumps(record.model_dump(), sort_keys=True) + "\n")


def validate_manifest(path: Path, require_audio_exists: bool = False) -> list[ASRManifestRecord]:
    records = [ASRManifestRecord.model_validate(record) for record in read_jsonl(path)]
    if require_audio_exists:
        missing = [
            record.audio_filepath for record in records if not Path(record.audio_filepath).exists()
        ]
        if missing:
            raise FileNotFoundError(f"missing audio files: {missing[:5]}")
    return records


def filter_manifest(
    input_path: Path,
    output_path: Path,
    allowed_datasets: set[str] | None = None,
) -> list[ASRManifestRecord]:
    records = validate_manifest(input_path)
    if allowed_datasets is not None:
        records = [record for record in records if record.source_dataset in allowed_datasets]
    write_jsonl(output_path, records)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-audio-exists", action="store_true")
    parser.add_argument(
        "--allowed-datasets",
        nargs="*",
        default=None,
        help=(
            "Optional dataset filter, e.g. common_voice fleurs indicvoices "
            "opt_in_local synthetic_noisy_room."
        ),
    )
    args = parser.parse_args()

    records = validate_manifest(args.input, require_audio_exists=args.require_audio_exists)
    if args.allowed_datasets is not None:
        allowed = set(args.allowed_datasets)
        records = [record for record in records if record.source_dataset in allowed]
    write_jsonl(args.output, records)
    print(f"wrote {len(records)} validated ASR manifest rows to {args.output}")


if __name__ == "__main__":
    main()
