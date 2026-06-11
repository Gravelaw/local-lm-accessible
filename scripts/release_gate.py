from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.verify_model_checksums import MANIFEST_PATH, load_manifest, verify_model  # noqa: E402

MODELS_CONFIG_PATH = ROOT / "configs" / "models.yaml"
REQUIRED_MODEL_KEYS = ("text", "vision", "asr")
PLACEHOLDER_MARKERS = ("placeholder", "todo", "tbd")
UNRESOLVED_LICENSE_MARKERS = ("verify", "review_required", "unreviewed", "unknown", "tbd")


class ReleaseGateError(ValueError):
    def __init__(self, summary: dict[str, Any]) -> None:
        failures = [str(item) for item in summary.get("failures", [])]
        super().__init__("release gate failed: " + "; ".join(failures))
        self.summary = summary
        self.failures = tuple(failures)


def run_release_gate(
    *,
    manifest_path: Path = MANIFEST_PATH,
    models_config_path: Path = MODELS_CONFIG_PATH,
    required_keys: tuple[str, ...] = REQUIRED_MODEL_KEYS,
    verify_checksums: bool = True,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    models_config = _load_yaml(models_config_path)
    failures: list[str] = []

    if manifest.get("local_only") is not True:
        failures.append("manifest local_only must be true")
    if manifest.get("privacy_mode") != "strict":
        failures.append("manifest privacy_mode must be strict")

    models_by_key = {str(model["key"]): model for model in manifest["models"]}
    missing_required = [key for key in required_keys if key not in models_by_key]
    if missing_required:
        failures.append(f"required models missing from manifest: {', '.join(missing_required)}")

    required_models = [models_by_key[key] for key in required_keys if key in models_by_key]
    total_params = sum(
        int(model.get("base_params", 0)) + int(model.get("adapter_params", 0))
        for model in required_models
    )
    configured_limit = int(
        float(models_config["model_policy"]["max_v1_total_parameters_b"]) * 1_000_000_000
    )
    if total_params > configured_limit:
        failures.append(
            f"required deployed parameters exceed budget: {total_params} > {configured_limit}"
        )

    for model in required_models:
        key = str(model["key"])
        if not str(model.get("sha256", "")).strip():
            failures.append(f"required model {key} is missing sha256")
        if _has_placeholder_text(model):
            failures.append(f"required model {key} contains placeholder text")
        if _has_unresolved_license_review(model):
            failures.append(f"required model {key} has unresolved license or commercial review")
        if int(model.get("base_params", 0)) <= 0:
            failures.append(f"required model {key} must have base_params > 0")

    if verify_checksums:
        for model in required_models:
            key = str(model["key"])
            if not str(model.get("sha256", "")).strip():
                continue
            try:
                verify_model(model)
            except (FileNotFoundError, ValueError) as exc:
                failures.append(f"required model {key} failed checksum verification: {exc}")

    config_models = models_config.get("models", {})
    if not isinstance(config_models, dict):
        failures.append("configs/models.yaml models must be a mapping")
        config_models = {}

    for key, config_model in config_models.items():
        if not bool(config_model.get("enabled", False)):
            continue
        manifest_model = models_by_key.get(str(key))
        if manifest_model is None:
            failures.append(f"enabled config model {key} missing from manifest")
            continue
        _check_model_config_consistency(str(key), manifest_model, config_model, failures)

    summary = {
        "status": "ok",
        "local_only": True,
        "required_models": list(required_keys),
        "total_required_parameters": total_params,
        "parameter_budget": configured_limit,
        "manifest_models": sorted(models_by_key),
        "checksum_verification": verify_checksums,
    }
    if failures:
        summary["status"] = "failed"
        summary["failures"] = failures
        summary["next_actions"] = _release_next_actions(failures)
        raise ReleaseGateError(summary)

    return summary


def _release_next_actions(failures: list[str]) -> list[str]:
    actions: list[str] = []
    failure_text = " ".join(failures).casefold()
    if "required model asr is missing sha256" in failure_text:
        actions.extend(
            [
                "Stage Parakeet with: python3 scripts/download_models.py "
                "--model asr --download --allow-large-download",
                "Record its checksum with: python3 scripts/verify_model_checksums.py "
                "--model asr --write-manifest-checksum",
            ]
        )
    if "failed checksum verification" in failure_text or "missing local artifact" in failure_text:
        actions.append("Verify staged artifacts with: python3 scripts/verify_model_checksums.py")
    if "unresolved license" in failure_text or "placeholder text" in failure_text:
        actions.append("Review models/manifest.json license, commercial_status, and notes fields.")
    if not actions:
        actions.append(
            "Inspect models/manifest.json and configs/models.yaml against release requirements."
        )
    return actions


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as config_file:
        payload = yaml.safe_load(config_file)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return payload


def _has_placeholder_text(model: dict[str, Any]) -> bool:
    fields = (
        "model_name",
        "model_id",
        "quantization",
        "local_path",
        "license",
        "runtime",
        "commercial_status",
        "notes",
    )
    values = " ".join(str(model.get(field, "")) for field in fields).casefold()
    return any(marker in values for marker in PLACEHOLDER_MARKERS)


def _has_unresolved_license_review(model: dict[str, Any]) -> bool:
    values = " ".join(
        str(model.get(field, "")) for field in ("license", "commercial_status")
    ).casefold()
    return any(marker in values for marker in UNRESOLVED_LICENSE_MARKERS)


def _check_model_config_consistency(
    key: str,
    manifest_model: dict[str, Any],
    config_model: dict[str, Any],
    failures: list[str],
) -> None:
    if str(config_model.get("source")) != str(manifest_model["vendor"]):
        failures.append(f"model {key} source/vendor mismatch")
    if str(config_model.get("id")) != str(manifest_model["model_id"]):
        failures.append(f"model {key} id mismatch")
    endpoint = str(config_model.get("endpoint", ""))
    if endpoint and not endpoint.rstrip("/").endswith(f":{manifest_model['port']}"):
        failures.append(f"model {key} endpoint port mismatch")
    configured_params = int(round(float(config_model.get("parameters_b", 0)) * 1_000_000_000))
    manifest_params = int(manifest_model.get("base_params", 0))
    tolerance = max(1_000_000, int(manifest_params * 0.001))
    if abs(configured_params - manifest_params) > tolerance:
        failures.append(
            f"model {key} config parameters_b does not match manifest base_params"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate local-lm release readiness metadata.")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--models-config", type=Path, default=MODELS_CONFIG_PATH)
    args = parser.parse_args()

    try:
        summary = run_release_gate(
            manifest_path=args.manifest,
            models_config_path=args.models_config,
        )
    except ReleaseGateError as exc:
        print(json.dumps(exc.summary, indent=2, sort_keys=True))
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
