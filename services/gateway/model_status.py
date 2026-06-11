from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "models" / "manifest.json"


def model_readiness_by_key(manifest_path: Path = MANIFEST_PATH) -> dict[str, dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    statuses: dict[str, dict[str, Any]] = {}
    for model in manifest.get("models", []):
        key = str(model["key"])
        required = key != "omni"
        artifact_statuses = [_artifact_status(model)]
        artifact_statuses.extend(
            _artifact_status(artifact) for artifact in model.get("additional_artifacts", [])
        )
        artifact_present = all(status["present"] for status in artifact_statuses)
        checksum_configured = all(status["checksum_configured"] for status in artifact_statuses)
        ready = artifact_present and checksum_configured
        warnings = []
        if required and not artifact_present:
            warnings.append("required artifact is not present locally")
        if required and not checksum_configured:
            warnings.append("required checksum is not configured")
        statuses[key] = {
            "model_id": model["model_id"],
            "required": required,
            "runtime": model["runtime"],
            "artifact_present": artifact_present,
            "checksum_configured": checksum_configured,
            "ready": ready,
            "artifacts": artifact_statuses,
            "warnings": warnings,
        }
    return statuses


def _artifact_status(artifact: dict[str, Any]) -> dict[str, Any]:
    local_path = ROOT / str(artifact["local_path"])
    present = local_path.exists()
    if local_path.is_dir():
        present = any(item.is_file() for item in local_path.rglob("*"))
    return {
        "name": str(artifact.get("name", "model")),
        "local_path": str(local_path),
        "present": present,
        "checksum_configured": bool(str(artifact.get("sha256", "")).strip()),
    }
