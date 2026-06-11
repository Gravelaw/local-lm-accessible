from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from scripts.verify_model_checksums import verify_manifest

app = FastAPI(title="local-lm ASR service")

SUPPORTED_LANGUAGES = {
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
Transcriber = Callable[[str], Any]
_TRANSCRIBER_CACHE: tuple[Path, Transcriber] | None = None
_TRANSCRIBER_LOADER: Callable[[Path], Transcriber] | None = None


class TranscriptionRequest(BaseModel):
    audio_filepath: str = Field(min_length=1)
    language: str = Field(default="en", min_length=2)
    region: str = Field(default="unknown", min_length=1)
    country: str = Field(default="unknown", min_length=1)
    modality: Literal["audio"] = "audio"
    task: Literal["speech_to_text", "asr"] = "speech_to_text"
    allow_experimental: bool = False

    @model_validator(mode="after")
    def require_supported_or_explicit_experiment(self) -> TranscriptionRequest:
        self.language = self.language.lower()
        self.region = self.region.lower()
        self.country = self.country.lower()
        language = self.language
        region = self.region
        experimental = language not in SUPPORTED_LANGUAGES or (
            region in EXPERIMENTAL_REGIONS and language != "en"
        )
        if experimental and not self.allow_experimental:
            raise ValueError(
                "unsupported or experimental language requires allow_experimental=true"
            )
        return self


class TranscriptionResponse(BaseModel):
    text: str
    language: str
    model: str
    local_only: bool
    experimental: bool
    unsupported_language: bool
    warnings: list[str] = Field(default_factory=list)
    status: Literal["stub", "ok"]
    model_ready: bool = False


@app.get("/health")
def health() -> dict[str, object]:
    model_ready, model_warning = _model_ready()
    return {
        "status": "ok" if model_ready else "model_missing",
        "local_only": True,
        "service": "parakeet",
        "model": "nvidia/parakeet-tdt-0.6b-v3",
        "model_ready": model_ready,
        "warnings": [] if model_warning is None else [model_warning],
    }


@app.post("/transcribe")
def transcribe(request: TranscriptionRequest) -> TranscriptionResponse:
    path = Path(request.audio_filepath)
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=400, detail="audio_filepath must point to a local file")

    model_path, model_warning = _verified_asr_model_path()
    model_ready = model_path is not None
    unsupported_language = request.language.lower() not in SUPPORTED_LANGUAGES
    experimental = unsupported_language or (
        request.region.lower() in EXPERIMENTAL_REGIONS and request.language.lower() != "en"
    )
    warnings = []
    if unsupported_language:
        warnings.append("Language is not supported by Parakeet v3; result is experimental.")
    elif experimental:
        warnings.append(
            "Language/region combination is experimental until local eval proves usable."
        )
    if not model_ready and model_warning is not None:
        warnings.append(model_warning)
    status: Literal["stub", "ok"] = "stub"
    text = f"LOCAL_STUB_TRANSCRIPTION_FOR:{path.name}"
    if model_path is not None:
        try:
            text = _transcribe_with_verified_model(path, model_path)
            status = "ok"
        except RuntimeError as exc:
            warnings.append(str(exc))
    return TranscriptionResponse(
        text=text,
        language=request.language.lower(),
        model="nvidia/parakeet-tdt-0.6b-v3",
        local_only=True,
        experimental=experimental,
        unsupported_language=unsupported_language,
        warnings=warnings,
        status=status,
        model_ready=model_ready,
    )


def _model_ready() -> tuple[bool, str | None]:
    model_path, warning = _verified_asr_model_path()
    return model_path is not None, warning


def _verified_asr_model_path() -> tuple[Path | None, str | None]:
    try:
        verified = verify_manifest("asr")
    except (FileNotFoundError, KeyError, ValueError) as exc:
        return None, f"Parakeet artifact is not checksum-verified locally: {exc}"
    if not verified:
        return None, "Parakeet artifact is not checksum-verified locally"
    return Path(str(verified[0]["path"])), None


def _transcribe_with_verified_model(audio_path: Path, model_path: Path) -> str:
    transcriber = _get_transcriber(model_path)
    result = transcriber(str(audio_path))
    return _extract_transcription_text(result)


def _get_transcriber(model_path: Path) -> Transcriber:
    global _TRANSCRIBER_CACHE
    if _TRANSCRIBER_CACHE is not None and _TRANSCRIBER_CACHE[0] == model_path:
        return _TRANSCRIBER_CACHE[1]
    loader = _TRANSCRIBER_LOADER or _default_transcriber_loader
    transcriber = loader(model_path)
    _TRANSCRIBER_CACHE = (model_path, transcriber)
    return transcriber


def _default_transcriber_loader(model_path: Path) -> Transcriber:
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise RuntimeError(
            "Transformers ASR runtime is not installed; install training/runtime extras "
            "to run Parakeet locally."
        ) from exc
    try:
        return pipeline(
            "automatic-speech-recognition",
            model=str(model_path),
            tokenizer=str(model_path),
            model_kwargs={"local_files_only": True},
        )
    except Exception as exc:
        raise RuntimeError(f"Could not load local Parakeet ASR pipeline: {exc}") from exc


def _extract_transcription_text(result: Any) -> str:
    if isinstance(result, str):
        text = result
    elif isinstance(result, dict):
        text = str(result.get("text", "")).strip()
    else:
        text = ""
    if not text:
        raise RuntimeError("Local Parakeet ASR pipeline returned no transcription text.")
    return text
