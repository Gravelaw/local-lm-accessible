from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from services.gateway.schemas import RuntimeConfig

DISALLOWED_REMOTE_HOSTS = {
    "amazonaws.com",
    "azure.com",
    "googleapis.com",
    "api.openai.com",
}


def _walk_values(value: Any) -> list[str]:
    if isinstance(value, dict):
        values: list[str] = []
        for item in value.values():
            values.extend(_walk_values(item))
        return values
    if isinstance(value, list):
        values = []
        for item in value:
            values.extend(_walk_values(item))
        return values
    if isinstance(value, str):
        return [value]
    return []


def test_all_runtime_configs_are_local_only() -> None:
    for config_path in Path("configs").glob("*.yaml"):
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        runtime = config.get("runtime")
        if runtime is not None:
            RuntimeConfig.model_validate(runtime)

        all_text = " ".join(_walk_values(config)).lower()
        assert not any(host in all_text for host in DISALLOWED_REMOTE_HOSTS), config_path


def test_model_manifest_is_local_only_and_under_parameter_cap() -> None:
    manifest = json.loads(Path("models/manifest.json").read_text(encoding="utf-8"))

    assert manifest["local_only"] is True
    assert manifest["privacy_mode"] == "strict"
    total_params = sum(
        model["base_params"] + model["adapter_params"] for model in manifest["models"]
    )
    assert total_params < 32_000_000_000
    assert {model["vendor"] for model in manifest["models"]} <= set(manifest["allowed_vendors"])
    assert {model["port"] for model in manifest["models"]} == {8081, 8082, 8083, 8090}
    for model in manifest["models"]:
        for field in {
            "model_name",
            "model_id",
            "vendor",
            "base_params",
            "adapter_params",
            "quantization",
            "local_path",
            "sha256",
            "license",
            "runtime",
            "port",
            "tasks",
            "supported_languages",
            "unsupported_languages",
            "commercial_status",
            "notes",
        }:
            assert field in model
