from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel, Field, HttpUrl, field_validator

from services.gateway.schemas import LOCAL_HOSTS


class LocalModelError(RuntimeError):
    """Raised when a configured local model service is unavailable or malformed."""


class LocalModelClient(BaseModel):
    model_key: str = Field(min_length=1)
    endpoint: HttpUrl
    timeout_seconds: float = Field(default=30.0, gt=0.0, le=300.0)

    @field_validator("endpoint")
    @classmethod
    def require_loopback_endpoint(cls, value: HttpUrl) -> HttpUrl:
        host = value.host or ""
        if host not in LOCAL_HOSTS:
            raise ValueError("model clients must use loopback endpoints")
        return value

    def completion_url(self) -> str:
        return f"{str(self.endpoint).rstrip('/')}/completion"

    def chat_completions_url(self) -> str:
        return f"{str(self.endpoint).rstrip('/')}/v1/chat/completions"

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        if not prompt.strip():
            raise LocalModelError("prompt is required for local generation")
        payload = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "cache_prompt": False,
        }
        chat_payload = {
            "model": self.model_key,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        api_path = "/completion"
        try:
            raw = self._post_json_url(self.completion_url(), payload)
        except LocalModelError as completion_error:
            if not _can_retry_chat_completions(completion_error):
                raise
            api_path = "/v1/chat/completions"
            try:
                raw = self._post_json_url(self.chat_completions_url(), chat_payload)
            except LocalModelError as chat_error:
                raise LocalModelError(
                    f"local model service unavailable for {self.model_key}"
                ) from chat_error

        text = _extract_generated_text(raw)
        if not text and api_path == "/completion":
            api_path = "/v1/chat/completions"
            try:
                raw = self._post_json_url(self.chat_completions_url(), chat_payload)
            except LocalModelError as chat_error:
                raise LocalModelError(
                    f"local model service returned no text for {self.model_key}"
                ) from chat_error
            text = _extract_generated_text(raw)
        if not text:
            raise LocalModelError(f"local model service returned no text for {self.model_key}")
        return {
            "text": text,
            "raw": raw,
            "model_key": self.model_key,
            "endpoint": str(self.endpoint),
            "api_path": api_path,
            "local_only": True,
        }

    def post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not path.startswith("/"):
            raise LocalModelError("local model client path must start with /")
        raw = self._post_json_url(f"{str(self.endpoint).rstrip('/')}{path}", payload)
        if not isinstance(raw, dict):
            raise LocalModelError(
                f"local model service returned malformed JSON for {self.model_key}"
            )
        raw.setdefault("local_only", True)
        return raw

    def _post_json_url(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise LocalModelError(
                f"local model service HTTP {exc.code} for {self.model_key}"
            ) from exc
        except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise LocalModelError(f"local model service unavailable for {self.model_key}") from exc
        if not isinstance(raw, dict):
            raise LocalModelError(
                f"local model service returned malformed JSON for {self.model_key}"
            )
        return raw

    def stub_response(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "model_key": self.model_key,
            "endpoint": str(self.endpoint),
            "local_only": True,
            "request_received": bool(payload),
        }


def default_clients() -> dict[str, LocalModelClient]:
    return {
        "text": LocalModelClient(model_key="text", endpoint="http://127.0.0.1:8081"),
        "vision": LocalModelClient(model_key="vision", endpoint="http://127.0.0.1:8082"),
        "omni": LocalModelClient(model_key="omni", endpoint="http://127.0.0.1:8083"),
        "asr": LocalModelClient(model_key="asr", endpoint="http://127.0.0.1:8090"),
    }


def _extract_generated_text(payload: dict[str, Any]) -> str:
    for key in ("content", "response", "text"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value

    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            text = first.get("text")
            if isinstance(text, str) and text.strip():
                return text
            message = first.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    return content
    return ""


def _can_retry_chat_completions(error: LocalModelError) -> bool:
    cause = error.__cause__
    return isinstance(cause, HTTPError) and cause.code in {404, 405}
