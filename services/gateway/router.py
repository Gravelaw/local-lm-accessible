from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from services.gateway.schemas import (
    RouteDecision,
    RouteRequest,
    RouteTarget,
    RuntimeConfig,
    TaskName,
)

AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
PDF_EXTENSIONS = {".pdf"}
EXCEL_INTENTS = {"excel", "xlsx", "spreadsheet", "table", "export", "csv"}
DESCRIBE_INTENTS = {"describe", "description", "accessibility", "alt text", "what is in"}
TRANSLATE_IMAGE_INTENTS = {"translate", "read text", "ocr", "image text"}
WEB_INTENTS = {"url", "web", "website", "link", "http://", "https://"}
WIKIPEDIA_INTENTS = {"wikipedia", "wiki"}


class LocalRouter:
    def __init__(self, config: dict[str, Any]) -> None:
        self.runtime = RuntimeConfig.model_validate(config.get("runtime", {}))
        self.services = {
            service_name: RouteTarget.model_validate(
                {
                    "provider": service_config.get("provider", "local"),
                    "model_key": service_name,
                    "endpoint": service_config["endpoint"],
                }
            )
            for service_name, service_config in config.get("services", {}).items()
        }
        route_items = config.get("routes", {})
        if not route_items:
            raise ValueError("at least one local route is required")
        self.routes = {
            task: RouteTarget.model_validate(route_config)
            for task, route_config in route_items.items()
        }

    def route(self, task: str) -> RouteTarget:
        try:
            return self.routes[task]
        except KeyError as exc:
            raise KeyError(f"no route configured for task: {task}") from exc

    def decide(self, request: RouteRequest) -> RouteDecision:
        task, reason = self._classify(request)
        target = self.route(task.value)
        return RouteDecision(
            task=task,
            provider=target.provider,
            model_key=target.model_key,
            endpoint=target.endpoint,
            privacy_mode=self.runtime.privacy_mode,
            allow_web=task == TaskName.SUMMARIZE_URL and request.allow_web,
            reason=reason,
        )

    def health_services(self) -> list[dict[str, object]]:
        return [
            {
                "name": name,
                "endpoint": target.endpoint,
                "local_only": True,
                "optional": name == "omni",
                "configured": True,
            }
            for name, target in sorted(self.services.items())
        ]

    def _classify(self, request: RouteRequest) -> tuple[TaskName, str]:
        intent = request.intent.casefold()
        file_path = request.file_path or ""
        suffix = Path(file_path).suffix.casefold()
        mime_type = (request.mime_type or "").casefold()
        url = request.url or ""
        url_lower = url.casefold()

        if suffix in AUDIO_EXTENSIONS or mime_type.startswith("audio/"):
            return TaskName.SPEECH_TO_TEXT, "audio input routes to local ASR"
        if (
            suffix in IMAGE_EXTENSIONS | PDF_EXTENSIONS
            or mime_type.startswith("image/")
            or mime_type == "application/pdf"
        ) and _contains_any(intent, EXCEL_INTENTS):
            return TaskName.DOCUMENT_TO_EXCEL, "document plus export intent routes to Excel"
        if (suffix in IMAGE_EXTENSIONS or mime_type.startswith("image/")) and _contains_any(
            intent, DESCRIBE_INTENTS
        ):
            return TaskName.DESCRIBE_IMAGE, "image plus accessibility intent routes to vision"
        if (suffix in IMAGE_EXTENSIONS or mime_type.startswith("image/")) and _contains_any(
            intent, TRANSLATE_IMAGE_INTENTS
        ):
            return TaskName.TRANSLATE_IMAGE_TEXT, "image text or translation intent routes to OCR"
        if _contains_any(intent, WIKIPEDIA_INTENTS) or "wikipedia.org" in url_lower:
            return TaskName.SUMMARIZE_WIKIPEDIA, "Wikipedia intent routes to offline wiki summary"
        if url or _contains_any(intent, WEB_INTENTS):
            return TaskName.SUMMARIZE_URL, "URL intent routes to web summary gate"
        return TaskName.GENERAL_LOCAL_ASSISTANT, "default local assistant route"


def _contains_any(text: str, needles: set[str]) -> bool:
    return any(needle in text for needle in needles)


def load_routes(path: Path) -> LocalRouter:
    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError("routes config must be a mapping")
    return LocalRouter(config)
